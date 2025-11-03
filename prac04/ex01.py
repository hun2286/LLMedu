# 0000 정보라고 검색하면 잘 출력

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage, Document
import re

# -----------------------------
# 1. 사용자 설정
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다!")

pdf_path = r"C:\Users\BGR_NC_2_NOTE\Downloads\(대한민국생물지)한국의곤충 제12권 35호 바구미류 VIII.pdf"

# LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=api_key
)

# Embedding 모델
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

persist_dir = "./pdf_chroma_db"

# -----------------------------
# 2. PDF 로드
# -----------------------------
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# -----------------------------
# 2-1. 생물 단위로 문서 분할
# -----------------------------
def split_bio_documents(documents):
    bio_docs = []
    for doc in documents:
        text = doc.page_content
        # 번호 + 생물 이름 단위로 split (예: "17. 일본꼬마거위벌레")
        parts = re.split(r'\n\d+\.\s+', text)
        for part in parts:
            part = part.strip()
            if part:
                bio_docs.append(Document(page_content=part, metadata=doc.metadata))
    return bio_docs

splits = split_bio_documents(documents)

# -----------------------------
# 3. 벡터 스토어 생성
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory=persist_dir
)

# -----------------------------
# 4. Retriever 설정
# -----------------------------
SEARCH_TYPE = "similarity"
retriever = vectorstore.as_retriever(search_type=SEARCH_TYPE, search_kwargs={"k": 5})

# -----------------------------
# 5. RAG + 히스토리 연속 질문 함수
# -----------------------------
HISTORY_LIMIT = 5  # 최근 n개 대화만 유지
chat_history = []

def rag_answer(question):
    # 문서 기반 RAG 검색
    retriever_docs = retriever.invoke(question)
    docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever_docs]
    doc_context = "\n\n".join([doc.page_content for doc in docs])

    # 최근 히스토리 가져오기
    recent_history = chat_history[-HISTORY_LIMIT:]
    history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in recent_history])

    # LLM 프롬프트 구성
    messages = [
        SystemMessage(content="""
            당신은 PDF 문서를 참고하여 질문에 답변하는 전문가입니다. 
            다음 규칙을 반드시 지켜주세요:

            1. 문서에 있는 정보만 사용하여 답변합니다. 문서에 없는 정보는 절대 추측하지 않습니다. 알 수 없는 정보는 '정보 없음'이라고 표시합니다.
            2. 질문에 생물 이름만 포함되어도, 해당 생물 chunk 내 존재하는 모든 주요 속성(이름, 몸길이, 분포 등)을 반드시 제공하십시오.
            3. 답변 형식은 항상 명확하게 구분합니다. 예를 들어:
            - 이름: ○○○
            - 몸길이: ○○ mm
            - 분포: ○○
            필요 없는 항목은 '정보 없음'으로 처리합니다.
            4. 관련 문서 내용 일부를 인용할 때는 간단히 출처를 표시합니다.
            5. 이전 대화 내용도 참고하여 일관성을 유지합니다.
            6. 불필요한 장황한 설명 없이 핵심 정보만 간결하게 제공합니다.
            """),
        
    HumanMessage(content=f"이전 대화:\n{history_text}\n\n문서 참고:\n{doc_context}\n\n새 질문:\n{question}")
]
    # LLM 호출
    response = llm.invoke(messages)
    answer = response.content

    # 히스토리에 저장
    chat_history.append({"question": question, "answer": answer})
    return answer

# -----------------------------
# 6. 테스트 (연속 질문 가능)
# -----------------------------
if __name__ == "__main__":
    print("PDF 기반 RAG 연속 질문 시스템입니다. 종료하려면 'exit'를 입력하세요.\n")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        answer = rag_answer(query)
        print("\n[답변]:", answer)
        print("-" * 50)
