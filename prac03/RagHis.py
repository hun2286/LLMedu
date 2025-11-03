import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings

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
# 2. PDF 로드 & 분할
# -----------------------------
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=300)
splits = text_splitter.split_documents(documents)

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
    from langchain.schema import Document

    # 문서 기반 RAG 검색
    retriever_docs = retriever.invoke(question)
    docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever_docs]
    doc_context = "\n\n".join([doc.page_content for doc in docs])

    # 최근 히스토리 가져오기
    recent_history = chat_history[-HISTORY_LIMIT:]
    history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in recent_history])

    # LLM 프롬프트 구성
    messages = [
        SystemMessage(content="당신은 PDF 문서를 참고해서 질문에 답변하는 전문가입니다. 이전 대화도 기억합니다."),
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
