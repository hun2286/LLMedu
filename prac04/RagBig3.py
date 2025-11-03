import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document, SystemMessage, HumanMessage
import re

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_path = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\2006_악기장_03_Ⅰ_국악기 제작의 역사와 악기장_10.pdf"
persist_dir = "./single_pdf_chroma_db"

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

# 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# PDF 로드 및 문장 단위 청킹 (세부 정보 검색 최적화)
def load_and_split_pdf_sentences(pdf_path, max_chunk_size=300):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    pdf_name = os.path.basename(pdf_path)
    pdf_title = os.path.splitext(pdf_name)[0]

    chunks = []

    for doc in documents:
        text = doc.page_content.strip()
        # 문장 단위로 나누기 (".", "?", "!" 기준)
        sentences = re.split(r'(?<=[.?!])\s+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # max_chunk_size 기준으로 잘라서 Document 생성
            for i in range(0, len(sentence), max_chunk_size):
                chunk_text = sentence[i:i + max_chunk_size]
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source": pdf_title,
                        "page": doc.metadata.get("page", "페이지 정보 없음")
                    }
                ))

    # 전체 문서 청크 추가
    full_text = "\n".join([doc.page_content.strip() for doc in documents if doc.page_content.strip()])
    if full_text:
        chunks.append(Document(
            page_content=full_text,
            metadata={"source": pdf_title, "type": "전체_내용"}
        ))

    return chunks

# PDF 청크 생성
all_docs = load_and_split_pdf_sentences(pdf_path)
print(f"총 문서 청크 수: {len(all_docs)}")

# 벡터 DB 생성
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectorstore = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

# 검색 k값 증가 (세부 정보 포함)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 80})

# RAG + LLM 함수
def rag_answer(question):
    # retriever.invoke 유지
    retriever_docs = retriever.invoke(question)

    # 단일 Document 반환 시 리스트 처리
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    context_texts = []
    for doc in retriever_docs:
        source = getattr(doc.metadata, "source", None) or doc.metadata.get("source", "출처 정보 없음")
        page = getattr(doc.metadata, "page", None) or doc.metadata.get("page", "페이지 정보 없음")
        content = getattr(doc, "page_content", str(doc))
        context_texts.append(f"[{source} / {page}]\n{content}")

    context = "\n\n".join(context_texts)

    messages = [
        SystemMessage(content="""
        당신은 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
        문서 내용만 기반으로 답변하며, 각 항목의 출처를 함께 명시하세요.
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
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break
        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)
