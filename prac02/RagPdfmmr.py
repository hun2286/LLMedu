import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 1. 사용자 설정
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다!")

# PDF 경로 직접 지정
pdf_path = r"C:\Users\BGR_NC_2_NOTE\Downloads\(대한민국생물지)한국의곤충 제12권 35호 바구미류 VIII.pdf"

# LLM 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=api_key
)

# Embedding 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# VectorStore 경로
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
SEARCH_TYPE = "mmr"  # or "similarity"
retriever = vectorstore.as_retriever(
    search_type=SEARCH_TYPE,
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
)

# -----------------------------
# 5. RAG 직접 구현 함수
# -----------------------------
def rag_answer(question):
    # Retriever에서 관련 문서 가져오기
    from langchain.schema import Document
    retriever_docs = retriever.invoke(question)
    docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever_docs]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # LLM 호출
    prompt = f"아래 문서를 참고해서 질문에 답변해주세요.\n\n[문서]\n{context}\n\n[질문]\n{question}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# -----------------------------
# 6. 테스트
# -----------------------------
if __name__ == "__main__":
    print("PDF 기반 RAG 시스템입니다. 종료하려면 'exit'를 입력하세요.\n")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        answer = rag_answer(query)
        print("\n[답변]:", answer)
        print("-" * 50) 
