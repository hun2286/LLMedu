# pymupdf 사용 

import os
import shutil
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# PDF 경로
pdf_path = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\2008_종묘제례_05_Ⅲ_종묘제례의 구성_64.pdf"
persist_dir = "./single_pdf_chroma_db"

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# PDF 텍스트 추출 (PyMuPDF 사용)
def load_pdf_text(pdf_path):
    docs = []
    pdf_name = os.path.basename(pdf_path)
    pdf_title = os.path.splitext(pdf_name)[0]

    pdf = fitz.open(pdf_path)
    for i, page in enumerate(pdf):
        text = page.get_text("text")  # 깨짐 없는 텍스트 추출
        if text and text.strip():
            docs.append(Document(
                page_content=text.strip(),
                metadata={"page": i + 1, "source": pdf_title}
            ))
    pdf.close()
    return docs

# PDF 로드
docs = load_pdf_text(pdf_path)
print(f"총 페이지에서 추출된 Document 수: {len(docs)}")

# 청킹
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
split_docs = text_splitter.split_documents(docs)
print(f"총 청크 수: {len(split_docs)}")

# Chroma DB 생성 (기존 DB 삭제)
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# RAG + LLM
def rag_answer(question):
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    context_texts = []
    for doc in retriever_docs:
        source = doc.metadata.get("source", "출처 없음")
        page = doc.metadata.get("page", "페이지 없음")
        content = doc.page_content
        context_texts.append(f"[{source} / {page}]\n{content}")

    context = "\n\n".join(context_texts)

    messages = [
        SystemMessage(content="""
        당신은 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
        문서 내용만 기반으로 답변하며, 각 항목의 출처를 명시하세요.
        문서에 없는 정보는 '정보 없음'으로 표시하세요.
        """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    return response.content

# 테스트
if __name__ == "__main__":
    print("\nPDF 기반 RAG 시스템입니다. 'exit' 입력 시 종료\n")
    while True:
        query = input("질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break
        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)
