# 기본 전처리 pdf 생성만 제거 

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage
import fitz

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20151103"
persist_dir = "./pdf_chroma_db"

# LLM / 임베딩
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=6000, openai_api_key=api_key)

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# PDF → 마크다운 변환
def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                blocks = page.get_text("blocks")
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
                md_text += f"# Page {i+1}\n\n"
                for block in blocks:
                    text = block[4].strip()
                    if not text:
                        continue
                    num_words = len(text.split())
                    num_lines = text.count("\n") + 1
                    if num_words <= 5 and num_lines <= 2:
                        md_text += f"# {text}\n\n"
                    elif num_words <= 15:
                        md_text += f"## {text}\n\n"
                    else:
                        md_text += f"{text}\n\n"
        return md_text
    except Exception as e:
        print(f"[오류] PDF 마크다운 변환 실패: {pdf_path}\n{e}")
        return ""

# pdf 예외처리
def load_pdf_safe(pdf_path):
    md_text = pdf_to_markdown(pdf_path)
    if md_text:
        return [Document(page_content=md_text,
                         metadata={"source": os.path.splitext(os.path.basename(pdf_path))[0]})]
    return []

def load_all_pdfs(pdf_folder):
    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    print(f"총 PDF 파일 수: {len(pdf_files)}\n")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        try:
            docs = load_pdf_safe(pdf_path)
            all_docs.extend(docs)
            print(f"{pdf_file} -> {len(docs)} 문서 처리 완료")
        except Exception as e:
            print(f"{pdf_file} 처리 중 오류 발생: {e}")
    print(f"\n총 추출된 Document 수: {len(all_docs)}")
    return all_docs

# 벡터스토어 로드 또는 생성
if os.path.exists(persist_dir) and os.listdir(persist_dir):
    print("기존 벡터 DB를 로드합니다.")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
else:
    print("새 벡터 DB를 생성합니다.")
    docs = load_all_pdfs(pdf_folder)
    print(f"총 PDF 처리 완료 {len(docs)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    split_docs = text_splitter.split_documents(docs)
    print(f"총 청크의 개수 {len(split_docs)}")
    
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    print("벡터 DB 생성 완료")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# RAG 질문-답변
def rag_answer(question):
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    source_dict = {}
    for doc in retriever_docs:
        source = doc.metadata.get("source", "출처 없음")
        if source not in source_dict:
            source_dict[source] = []
        source_dict[source].append(doc.page_content)

    context_texts = []
    for source, contents in source_dict.items():
        combined_content = "\n\n".join(contents)  # 동일 출처 문서 합치기
        context_texts.append(f"[{source}]\n{combined_content}")
        
    context = "\n\n".join(context_texts)

    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 질문에 등장하는 단어를 정확히 구분하여 각각 답변하세요.
            - 문서에 없으면 '정보 없음'으로 표시하세요.
            - 가능한 한 문서 내 문맥과 키워드에 기반하여 정확하게 답변하세요.
            - 각 항목의 출처는 반드시 별도의 줄에 '[출처: ...]' 형태로 표시하세요.
            - 각 항목과 출처 사이에는 빈 줄 한 줄을 넣으세요.
            """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    return response.content

# 실행
if __name__ == "__main__":
    while True:
        query = input("질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            print("프로그램 종료")
            break

        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)
