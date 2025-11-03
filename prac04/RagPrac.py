import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage, Document

# -----------------------------
# 1. 사용자 설정
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다!")

pdf_path = r"C:\Users\BGR_NC_2_NOTE\Downloads\(대한민국생물지)한국의곤충 제12권 35호 바구미류 VIII.pdf"
persist_dir = "./pdf_chroma_db"

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

# -----------------------------
# 2. PDF 로드
# -----------------------------
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# -----------------------------
# 3. 문서 분할 함수
# -----------------------------
def split_bio_documents(documents):
    bio_docs = []
    for doc in documents:
        text = doc.page_content
        # 예: "17. 일본꼬마거위벌레" 형태로 split
        parts = re.split(r'\n\d+\.\s+', text)
        for part in parts:
            part = part.strip()
            if part:
                bio_docs.append(Document(page_content=part, metadata=doc.metadata))
    return bio_docs

splits = split_bio_documents(documents)

# -----------------------------
# 4. 전체 문서 chunk 추가 (문서 제목 검색용)
# -----------------------------
whole_text = "\n".join([doc.page_content for doc in documents])
whole_doc = Document(
    page_content=whole_text,
    metadata={
        "type": "전체_내용",
        "제목": "대한민국생물지 제12권 제35호 (바구미류 VIII)"
    }
)

# 기존 DB 폴더 삭제 (중복 방지)
import shutil
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

# 전체 문서 포함하여 벡터스토어 생성
all_docs = splits + [whole_doc]
vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

print("✅ 벡터스토어 생성 완료 (생물별 chunk + 전체 문서 포함)")

# -----------------------------
# 5. Retriever 설정
# -----------------------------
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# -----------------------------
# 6. RAG + 히스토리 연속 질문 함수
# -----------------------------
HISTORY_LIMIT = 5
chat_history = []

def rag_answer(question):
    search_query = question.strip()

    # 생물 이름만 입력 시 자동으로 "정보" 추가
    if not re.search(r'(정보|제목|내용)', search_query):
        search_query += " 정보"

    # 문서 검색
    retriever_docs = retriever.invoke(search_query)
    docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever_docs]
    doc_context = "\n\n".join([doc.page_content for doc in docs])

    # 최근 대화 기록
    recent_history = chat_history[-HISTORY_LIMIT:]
    history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in recent_history])

    # 프롬프트 구성
    messages = [
        SystemMessage(content="""
            당신은 PDF 문서를 참고하여 질문에 답변하는 전문가입니다. 
            다음 규칙을 반드시 지켜주세요:

            1. 문서에 있는 정보만 사용합니다. 문서에 없는 정보는 '정보 없음'이라고 표시합니다.
            2. 질문이 생물 이름일 경우 해당 생물의 주요 속성(이름, 몸길이, 분포)을 제공합니다.
            3. 질문이 '문서 제목'일 경우 문서 전체 메타데이터나 표지 내용을 사용하여 제목을 알려줍니다.
            4. 형식은 다음과 같습니다:
               - 이름: ○○
               - 몸길이: ○○ mm
               - 분포: ○○
            5. 장황한 설명 없이 핵심 정보만 제공합니다.
        """),
        HumanMessage(content=f"이전 대화:\n{history_text}\n\n문서 참고:\n{doc_context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    answer = response.content

    chat_history.append({"question": question, "answer": answer})
    return answer

# -----------------------------
# 7. 실행
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
