# 스캔본 제외하고 276개 pdf rag작업 해보는 코드
# 스캔본은 텍스트 길이 판단해서 작으면 반환값을 목록에 넣어서 알려줌 (스캔본은 청크 생략)
# 텍스트 pdf는 이걸 최종으로 이제 스캔본 처리하는거 해야함

import os
import fitz
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\user\Desktop\pdfs\20251106"
persist_dir = "./mk_pdf_chroma_db3"

# LLM / 임베딩 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=8000,
    openai_api_key=api_key
)

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# PDF → 마크다운 변환
def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                blocks = page.get_text("blocks")
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
                page_text = ""
                for block in blocks:
                    text = block[4].strip()
                    if not text:
                        continue
                    num_words = len(text.split())
                    num_lines = text.count("\n") + 1
                    if num_words <= 5 and num_lines <= 2:
                        page_text += f"# {text}\n\n"
                    elif num_words <= 15:
                        page_text += f"## {text}\n\n"
                    else:
                        page_text += f"{text}\n\n"
                md_text += f"# Page {i + 1}\n\n{page_text}"

        return md_text
    except Exception:
        print(f"[오류] PDF 변환 실패 : {pdf_path}") 
        return ""

def load_pdf_safe(pdf_path, min_text_len=20):
    """
    PDF를 읽어서 Document로 변환.
    텍스트 길이가 min_text_len보다 작으면 None 반환 (스캔본 판단)
    """
    md_text = pdf_to_markdown(pdf_path)
    if md_text and len(md_text.strip()) >= min_text_len:
        return [Document(page_content=md_text,
                         metadata={"source": os.path.splitext(os.path.basename(pdf_path))[0]})]
    else:
        return None  # 텍스트가 너무 적으면 실패로 간주

def load_all_pdfs(pdf_folder):
    all_docs = []
    failed_pdfs = []  # 텍스트 부족 PDF 목록
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    print(f"총 PDF 파일 수: {len(pdf_files)}\n")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            docs = load_pdf_safe(pdf_path)
            if not docs:  # 텍스트 부족 → 청크 생성 불가
                failed_pdfs.append(pdf_file)
                print(f"{pdf_file} → 텍스트 부족 / 청크 생성 불가")
            else:
                all_docs.extend(docs)
                print(f"{pdf_file} 처리 완료 ({len(docs)} 문서)")
        except Exception as e:
            print(f"{pdf_file} 처리 중 오류: {e}")
            failed_pdfs.append(pdf_file)
    
    print(f"\n총 Document 수: {len(all_docs)}")
    if failed_pdfs:
        print("\n--- 텍스트 변환 실패 / 청크 생성 불가 PDF 목록 ---")
        for f in failed_pdfs:
            print(f"- {f}")
    
    return all_docs

# Vector DB 구축 (DB 재사용)
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("DB 새로 생성")
    docs = load_all_pdfs(pdf_folder)
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=300)
        split_docs = text_splitter.split_documents(docs)
        print(f"총 청크 수: {len(split_docs)}")
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        vectorstore.persist()
    else:
        print("청크 생성 가능한 문서가 없습니다. DB 생성 생략.")
        vectorstore = None
else:
    print("기존 DB를 재사용")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

if vectorstore:
    # MMR 검색 기반 Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
else:
    retriever = None

# RAG 기반 답변 생성
def rag_answer(question):
    if not retriever:
        return "DB가 없습니다. 먼저 텍스트가 있는 PDF를 처리하세요."

    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    context_texts = []
    for doc in retriever_docs:
        content = doc.page_content.strip()
        if not content:
            continue
        context_texts.append(content)
        
    context = "\n\n".join(context_texts)
    
    # 🔹 출처 표시 시 필터링 (표지, 차례, 목차 등 제거)
    skip_keywords = ["표지", "차례", "목차", "contents", "서문", "머리말", "발간사"]
    sources_used = []
    for doc in retriever_docs:
        content = doc.page_content.strip()
        # 키워드 포함 또는 텍스트 짧으면 제외
        if any(k in content[:300].lower() for k in skip_keywords) or len(content) < 100:
            continue
        source_name = doc.metadata.get("source", "출처 없음")
        if source_name not in sources_used:
            sources_used.append(source_name)

    # 최소 4개 확보 위해 유사도 순으로 추가
    if len(sources_used) < 4:
        all_sources = [doc.metadata.get("source", "출처 없음") for doc in retriever_docs]
        for s in all_sources:
            if len(sources_used) >= 4:
                break
            if s not in sources_used:
                sources_used.append(s)

    sources_to_show = sources_used[:5]


    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 제공된 문서 내용만 활용해서 답변하세요.
            - 문서에 없는 내용은 절대 추가하지 말고, 없으면 '정보 없음'이라고 표시하세요.
            - 답변은 항목별로 구분하고, 각 항목마다 충분한 설명과 예시를 포함하세요.
            - 최소 300단어 이상 작성하고, 가능한 한 문서 내용을 풍부하게 통합하세요.
            - 여러 문서의 내용을 종합해 한 문단 이상의 자세한 답변을 작성하세요.
            - 질문에 등장하는 단어를 각각 구분하여 정확히 답변하세요.
            - 가능한 한 문서 내 문맥과 키워드에 기반하여 정확하게 답변하세요.
            """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    answer = response.content

    # 후처리: 제목에 번호 붙이고, 출처 맨 아래
    lines = answer.split("\n")
    final_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            final_lines.append("")
            continue
        # '###' 제거
        stripped = stripped.lstrip("#").strip()
        # 숫자 + 점 제거 (예: 1. 2. 3. …)
        stripped = re.sub(r"^\d+\.\s*", "", stripped)
        # **굵은글씨** 제거
        stripped = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
        # *기울임* 제거
        stripped = re.sub(r"\*(.*?)\*", r"\1", stripped)
        final_lines.append(stripped)

    if sources_to_show:
        final_lines.append("")
        final_lines.append("-"*50)
        for s in sources_to_show:
            final_lines.append(f"[출처: {s}]")

    return "\n".join(final_lines)

# 실행
if __name__ == "__main__":
    print("=" * 60)
    print("RAG 질의응답 시스템")
    print("=" * 60)

    while True:
        query = input("\n질문을 입력하세요 (exit 입력 시 종료): ").strip()
        if query.lower() == "exit":
            print("프로그램 종료.")
            break

        answer = rag_answer(query)
        print("\n[답변]:\n" + answer)
        print("-" * 50)
