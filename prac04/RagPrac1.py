# 최종 코드 llm 질의형 함수 포함

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

def split_bio_documents(docs):
    bio_docs = []
    for doc in docs:
        text = doc.page_content
        parts = re.split(r'\n\d+\.\s+', text)
        for part in parts:
            part = part.strip()
            if part:
                bio_docs.append(Document(page_content=part, metadata=doc.metadata))
    return bio_docs

splits = split_bio_documents(documents)

# -----------------------------
# 3. 전체 문서 chunk 추가
# -----------------------------
whole_text = "\n".join([doc.page_content for doc in documents])
whole_doc = Document(
    page_content=whole_text,
    metadata={"type": "전체_내용", "제목": "대한민국생물지 제12권 제35호 (바구미류 VIII)"}
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
# 4. LLM 직접 질의형 함수
# -----------------------------
def llm_direct_answer(question):
    prompt = f"""
    당신은 PDF 문서를 참고하는 전문가입니다.
    아래 질문에 답변하세요. 문서가 길더라도 모든 내용을 참고하세요.
    질문: {question}
    답변 형식:
      - 이름: ○○
      - 몸길이: ○○ mm
      - 분포: ○○
    정보가 없으면 '정보 없음'으로 표시
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# -----------------------------``
# 5. RAG + LLM 하이브리드 함수
# -----------------------------
def rag_answer(question):
    # 검색어 자동 처리
    if not re.search(r'(정보|제목|내용)', question):
        search_query = question + " 정보"
    else:
        search_query = question

    retriever_docs = retriever.invoke(search_query)
    docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever_docs]
    doc_context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        SystemMessage(content="""
        당신은 PDF 문서를 참고하여 질문에 답변하는 전문가입니다.
        문서에 있는 정보만 사용하고, 항목별로 구조화하세요.
        """),
        HumanMessage(content=f"문서 참고:\n{doc_context}\n\n질문:\n{question}")
    ]
    response = llm.invoke(messages)
    return response.content

# -----------------------------
# 6. 테스트
# -----------------------------
if __name__ == "__main__":
    print("PDF 기반 RAG 연속 질문 시스템입니다. 'exit'을 입력하면 종료됩니다.\n")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)
