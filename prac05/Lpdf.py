import os
import shutil
import pdfplumber
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# PDF 폴더 경로 지정
pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs"
persist_dir = "./multi_pdf_chroma_db"

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# 폴더 내 모든 PDF 텍스트 추출
def load_all_pdfs(pdf_folder):
    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

    print(f"총 PDF 파일 수: {len(pdf_files)}\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pdf_title = os.path.splitext(pdf_file)[0]

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        all_docs.append(Document(
                            page_content=text.strip(),
                            metadata={"page": i + 1, "source": pdf_title}
                        ))
            print(f"{pdf_file} -> {len(pdf.pages)} 페이지 처리 완료")
        except Exception as e:
            print(f"{pdf_file} 처리 중 오류 발생: {e}")

    print(f"\n총 추출된 Document 수: {len(all_docs)}")
    return all_docs

# 모든 PDF 문서 로드
docs = load_all_pdfs(pdf_folder)

# 청킹
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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
        당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
        문서 내용만 기반으로 답변하며, 각 항목의 출처를 명시하세요.
        문서에 없는 정보는 '정보 없음'으로 표시하세요.
        """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    return response.content

# 테스트
if __name__ == "__main__":
    print("\n질의응답 시스템, 'exit' 입력 시 종료\n")
    while True:
        query = input("질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break
        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)
