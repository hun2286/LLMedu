# pdf 마크다운 변환 후 rag

import os
import glob
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
import pdfplumber
from markdownify import markdownify as md 

# 환경설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(".env 파일에 OPENAI_API_KEY가 없습니다!")

pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\근무일지\pdf목록"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
if not pdf_files:
    raise FileNotFoundError(f"{pdf_folder} 안에 PDF 파일이 없습니다!")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=api_key
)

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

persist_dir = "./pdf_chroma_db"

# PDF 로드 + Markdown 변환 + 분할
all_splits = []
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)

for pdf_path in pdf_files:
    # PDF → 텍스트 추출
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"

    # PDF → Markdown 변환
    md_text = md(text) 
    # 간단 전처리
    paragraphs = [p.strip() for p in md_text.split("\n\n") if len(p.strip()) > 20]  # 추가

    # Document 객체 생성
    documents = [Document(page_content=p, metadata={"source": os.path.basename(pdf_path)}) for p in paragraphs]

    # 청크 분할
    splits = text_splitter.split_documents(documents)
    all_splits.extend(splits)

print(f"총 문서 청크 수: {len(all_splits)}")

# chroma vectorstore 설정
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding_model,
    persist_directory=persist_dir
)

# Retriever 설정
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# RAG
HISTORY_LIMIT = 5
MAX_CHUNK_LENGTH = 2000
chat_history = []

def rag_answer(question):
    retriever_docs = retriever.get_relevant_documents(question)
    docs = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever_docs]
    doc_context = "\n\n".join([f"({doc.metadata['source']})\n{doc.page_content[:MAX_CHUNK_LENGTH]}" for doc in docs])

    recent_history = chat_history[-HISTORY_LIMIT:]
    history_text = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in recent_history])

    messages = [
        SystemMessage(content="당신은 PDF 문서를 참고하여 정확하게 질문에 답변하는 전문가입니다."),
        HumanMessage(content=f"이전 대화:\n{history_text}\n\n참고 문서:\n{doc_context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    answer = response.content

    chat_history.append({"question": question, "answer": answer})
    return answer

# 테스트
if __name__ == "__main__":
    print("폴더 내 모든 PDF(Markdown 변환 후) 기반 RAG 시스템입니다. 'exit' 입력 시 종료.\n")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break
        answer = rag_answer(query)
        print("\n[답변]:", answer)
        print("-" * 50)
