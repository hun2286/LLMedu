import os
import re
import asyncio
import torch
from typing import Optional, Union, Any, cast, List
from functools import partial
from dotenv import load_dotenv

# OpenAI 및 LangChain 관련
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 1. 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 읽기)
load_dotenv()

# ==========================================
# [내부 유틸리티 함수] 기존 rag_utils 기능을 통합
# ==========================================
def load_pdf_safe(file_path: str) -> List[Document]:
    """PDF 로드 (간이 버전: PyMuPDF 등 라이브러리 필요 시 설치)"""
    from langchain_community.document_loaders import PyMuPDFLoader
    try:
        loader = PyMuPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        print(f"파일 로드 에러 ({file_path}): {e}")
        return []

def extract_keywords(text: str) -> List[str]:
    """질문에서 주요 키워드 추출 (간이 구현)"""
    # 불용어를 제외하고 2글자 이상의 단어만 추출
    words = re.findall(r'[가-힣a-zA-Z0-9]{2,}', text)
    return words

def title_matches_keywords(title: str, keywords: List[str]) -> bool:
    """파일명에 키워드가 포함되는지 확인"""
    if not keywords:
        return True
    # 키워드 중 하나라도 제목에 포함되어 있으면 True
    return any(kw.lower() in title.lower() for kw in keywords)

# ==========================================
# [프롬프트 구성]
# ==========================================
def _build_gpt_prompt(context: str, question: str, source_list_text: str) -> list:
    system_prompt = f"""당신은 제공된 문서만을 기반으로 질문에 답하는 전문 분석가입니다.
당신의 모든 답변은 아래의 엄격한 지침을 따라야 합니다.

[핵심 지침]
1. **정보 근거:** 답변은 반드시 제공된 [문서 내용]에 근거해야 합니다. 문서에 내용이 부족하면 "정보 없음"이라고 답변하세요.
2. **답변 형식 (Markdown):** 제목(##), 목록(*), 볼드체(**) 등을 사용하여 구조화하십시오.
3. **내용 상세화:** 단순 요약을 넘어 구체적 사례와 맥락을 포함하여 완결성 있게 설명하십시오.
4. **가독성:** 의미 단위마다 줄 바꿈을 충분히 활용하고 짧은 목록(*)으로 쪼개서 작성하십시오.
5. **[필수] 인라인 출처 태깅:** 답변의 각 문단 또는 주요 정보 뒤에는 해당 정보의 문서 번호([1], [2] 등)를 붙이십시오.

[제공된 문서 번호 목록]
{source_list_text}"""

    user_content = f"문서 내용:\n{context}\n\n질문:\n{question}"
    return [
        ("system", system_prompt),
        ("human", user_content)
    ]

# ==========================================
# [RAG 서비스 클래스]
# ==========================================
class RAGService:
    def __init__(self):
        # 환경 설정 (settings 객체 대신 직접 지정)
        self.PDF_FOLDER = "./pdfs"        # PDF 파일들이 있는 폴더
        self.PERSIST_DIR = "./database/Cultural_db" # DB 저장 경로
        self.MODEL_NAME = "gpt-4o"       # 사용할 GPT 모델
        
        self.llm: ChatOpenAI | None = None
        self.embedding_model: OpenAIEmbeddings | None = None
        self.vectorstore: Chroma | None = None
        self.retriever: Any | None = None

    def _load_all_pdfs_recursive(self, root_folder: str) -> list[Document]:
        all_docs = []
        if not os.path.exists(root_folder):
            os.makedirs(root_folder)
            return []
            
        for dirpath, _, filenames in os.walk(root_folder):
            pdf_files = [f for f in filenames if f.lower().endswith(".pdf")]
            for pdf_file in pdf_files:
                pdf_path = os.path.join(dirpath, pdf_file)
                docs = load_pdf_safe(pdf_path)
                if docs:
                    all_docs.extend(docs)
        return all_docs

    def initialize_sync(self):
        print("Windows GPT RAG 시스템 초기화 중...")

        # 1. 임베딩 모델
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # 2. VectorDB 로드 및 생성
        if not os.path.exists(self.PERSIST_DIR) or not os.listdir(self.PERSIST_DIR):
            print(f"'{self.PDF_FOLDER}'에서 문서를 읽어 DB를 새로 생성합니다...")
            docs = self._load_all_pdfs_recursive(self.PDF_FOLDER)

            if docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = text_splitter.split_documents(docs)
                self.vectorstore = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embedding_model,
                    persist_directory=self.PERSIST_DIR
                )
                print(f"임베딩 완료 (청크 수: {len(split_docs)})")
            else:
                print("⚠️ 폴더 내에 PDF 파일이 없어 빈 DB로 시작합니다.")
                self.vectorstore = None
        else:
            print("기존 DB를 로드합니다.")
            self.vectorstore = Chroma(
                persist_directory=self.PERSIST_DIR,
                embedding_function=self.embedding_model
            )

        # 3. Retriever
        if self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        # 4. LLM
        self.llm = ChatOpenAI(model=self.MODEL_NAME, temperature=0.3)
        print("모든 시스템이 준비되었습니다.")

    async def rag_answer(self, question: str) -> str:
        if not question.strip(): return "질문을 입력해주세요."
        if not self.retriever: return "참조할 문서 DB가 비어있습니다."

        # 비동기 실행을 위해 스레드 활용
        return await asyncio.to_thread(self._rag_worker, question)

    def _rag_worker(self, question: str) -> str:
        # 검색 및 키워드 필터링
        keywords = extract_keywords(question)
        docs = self.retriever.invoke(question)
        
        filtered_docs = [d for d in docs if title_matches_keywords(d.metadata.get("source", ""), keywords)]
        if not filtered_docs: filtered_docs = docs

        # 컨텍스트 생성
        context_chunks = []
        sources_map = []
        for i, doc in enumerate(filtered_docs):
            content = doc.page_content.strip()
            # 간단한 전처리
            content = re.sub(r'정밀실측조사보고서?', '', content)
            context_chunks.append(f"[{i+1}] {content}")
            sources_map.append(os.path.basename(doc.metadata.get('source', '알 수 없음')))

        context_text = "\n\n".join(context_chunks)
        source_list_text = "\n".join([f"[{i+1}]: {name}" for i, name in enumerate(sources_map)])
        
        if not context_text: return "관련 정보를 찾을 수 없습니다."

        # LLM 호출
        messages = _build_gpt_prompt(context_text, question, source_list_text)
        response = self.llm.invoke(messages)
        answer = response.content.strip()

        # 후처리 및 출처 정리
        used_tags = set(re.findall(r'\[(\d+)\]', answer))
        final_body = re.sub(r'\[\d+\]', '', answer).strip()
        
        final_sources = []
        for tag in used_tags:
            idx = int(tag) - 1
            if 0 <= idx < len(sources_map):
                final_sources.append(f"[출처: {sources_map[idx]}]")
        
        final_sources = sorted(list(set(final_sources)))
        
        output = final_body + "\n\n" + "-"*50 + "\n" + "\n".join(final_sources)
        return output

# ==========================================
# 실행 예시
# ==========================================
if __name__ == "__main__":
    service = RAGService()
    service.initialize_sync()
    
    async def main():
        ans = await service.rag_answer("질문을 여기에 입력하세요")
        print(ans)
    
    # asyncio.run(main()) # 테스트 시 주석 해제