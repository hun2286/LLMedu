import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 사용자 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다!")

# PDF 경로
pdf_path = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\2006_악기장_03_Ⅰ_국악기 제작의 역사와 악기장_10.pdf"

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

persist_dir = "./pdf_chroma_db"

# 2. PDF 텍스트 추출 (PyMuPDF)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            texts.append(Document(page_content=text, metadata={"page": i + 1}))
    doc.close()
    return texts

documents = extract_text_from_pdf(pdf_path)

# 3. 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
splits = text_splitter.split_documents(documents)

# 4. 벡터 스토어 생성
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory=persist_dir
)
vectorstore.persist()

# 5. Retriever 설정
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 6. RAG 함수
def rag_answer(question):
    retriever_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retriever_docs])

    prompt = f"""
    다음은 참고 문서입니다. 반드시 문서 내용을 바탕으로 답변하세요.
    문서 내용 :
    {context}
    
    문서에 관련 내용이 없으면 '문서에 정보 없음'이라고 답변하세요.
    질문 : {question}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# 7. 실행
if __name__ == "__main__":
    print("PyMuPDF 기반 PDF RAG 시스템입니다. 종료하려면 'exit'를 입력하세요.\n")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        answer = rag_answer(query)
        print("\n[답변]:", answer)
        print("-" * 50)
