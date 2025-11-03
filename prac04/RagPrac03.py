import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_chroma import Chroma
import tiktoken

# -----------------------------
# 1. 환경 설정
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pdf_path = r"C:\Users\BGR_NC_2_NOTE\Downloads\한국의 곤충 제9권11호_애매미충아과.pdf"
persist_dir = "./pdf_chroma_db"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# -----------------------------
# 2. PDF 로드
# -----------------------------
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# -----------------------------
# 3. 토큰 단위로 문서 분할
# -----------------------------
def split_documents_by_tokens(docs, max_tokens=2000):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    chunks = []
    for doc in docs:
        tokens = encoding.encode(doc.page_content)
        for i in range(0, len(tokens), max_tokens):
            sub_tokens = tokens[i:i+max_tokens]
            sub_text = encoding.decode(sub_tokens)
            chunks.append(Document(page_content=sub_text, metadata=doc.metadata))
    return chunks

splits = split_documents_by_tokens(documents, max_tokens=2000)

# -----------------------------
# 4. 전체 문서도 추가
# -----------------------------
whole_text = "\n".join([doc.page_content for doc in documents])
pdf_name = os.path.basename(pdf_path)
pdf_title = os.path.splitext(pdf_name)[0]

whole_doc = Document(
    page_content=whole_text,
    metadata={"type": "전체_내용", "제목": pdf_title}
)
all_docs = splits + [whole_doc]

# -----------------------------
# 5. 기존 DB 삭제 후 새로 생성
# -----------------------------
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectorstore = Chroma(embedding_function=embedding_model, persist_directory=persist_dir)

# 배치 단위로 문서 추가 (토큰 제한 회피)
batch_size = 2
for i in range(0, len(all_docs), batch_size):
    vectorstore.add_documents(all_docs[i:i+batch_size])

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# -----------------------------
# 6. RAG + LLM 함수 (PDF 근거 기반)
# -----------------------------
def rag_answer(question):
    search_query = question if re.search(r'(정보|제목|내용)', question) else question + " 정보"
    retriever_docs = retriever.get_relevant_documents(search_query)
    doc_context = "\n\n".join([doc.page_content for doc in retriever_docs])

    messages = [
        SystemMessage(content="""
        당신은 PDF 문서를 참고하여 질문에 답변하는 전문가입니다.
        문서에 있는 정보만 사용하고, 항목별로 구조화하세요.
        암컷, 수컷 정보가 있다면 구분하여 작성하고,
        문서에 없는 정보는 '정보 없음'으로 표시합니다.
        """),
        HumanMessage(content=f"문서 참고:\n{doc_context}\n\n질문:\n{question}")
    ]
    response = llm.invoke(messages)
    return response.content

# -----------------------------
# 7. 테스트 실행
# -----------------------------
if __name__ == "__main__":
    print("PDF 기반 RAG 연속 질문 시스템입니다. 'exit' 입력 시 종료\n")
    while True:
        query = input("질문을 입력하세요: ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break
        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)
