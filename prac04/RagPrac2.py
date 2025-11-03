# 최종 코드 rag_answer 함수 사용 llm질의형과 다름

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, Document
import re
import shutil

# -----------------------------
# 1. 환경 설정
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pdf_path = r"C:\Users\BGR_NC_2_NOTE\Downloads\(대한민국생물지)한국의곤충 제12권 35호 바구미류 VIII.pdf"
persist_dir = "./pdf_chroma_db"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# -----------------------------
# 2. PDF 로드 & 분할
# -----------------------------
loader = PyPDFLoader(pdf_path)
documents = loader.load()

def split_documents(docs):
    chunks = []
    for doc in docs:
        text = doc.page_content
        parts = re.split(r'\n\d+\.\s+', text)  # 항목 단위 분할
        for part in parts:
            part = part.strip()
            if part:
                chunks.append(Document(page_content=part, metadata=doc.metadata))
    return chunks

splits = split_documents(documents)

# -----------------------------
# 3. 전체 문서 chunk 추가
# -----------------------------
whole_text = "\n".join([doc.page_content for doc in documents])

pdf_name = os.path.basename(pdf_path)
pdf_title = os.path.splitext(pdf_name)[0]  

whole_doc = Document(
    page_content=whole_text,
    metadata={"type": "전체_내용", "제목": pdf_title}
)
all_docs = splits + [whole_doc]

# 기존 DB 삭제 후 새로 생성
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# -----------------------------
# 4. RAG + LLM 함수 (PDF 근거 기반)
# -----------------------------
def rag_answer(question):
    # 검색어 자동 처리
    search_query = question if re.search(r'(정보|제목|내용)', question) else question + " 정보"

    retriever_docs = retriever.invoke(search_query)
    docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever_docs]
    doc_context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content="""
        당신은 PDF 문서를 참고하여 질문에 답변하는 전문가입니다.
        문서에 있는 정보만 사용하고, 항목별로 구조화하세요.
        문서에 없는 정보는 '정보 없음'으로 표시합니다.
        """),
        HumanMessage(content=f"문서 참고:\n{doc_context}\n\n질문:\n{question}")
    ]
    response = llm.invoke(messages)
    return response.content

# -----------------------------
# 5. 테스트 실행
# -----------------------------
if __name__ == "__main__":
    print("PDF 기반 RAG 연속 질문 시스템입니다. 'exit' 입력 시 종료\n")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break
        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)

# 여러 pdf를 통합해서 답변하는 것 실습 <<
# nas 데이터로 실습  ***
# 이미지 데이터 처리 caption으로 해야하는데 
# 실제 질문해서 답변해주면서 이미지 출처까지 나오게

# RAG로 텍스트 데이터 찾아오는 서비스 이름 짓기 ** 
# 자연어로 찾아주는 서비스 이름 짓기

# vlm vision language model
# 일단 제일 중요한건 여러 pdf 통합해서 rag 해서 답변나오는거 << 제일 중요 
# 이미지 디스크랩션, 이미지 라벨링 하는 방법 
# metadata 정보 입력 하는 방법 