import os
import fitz
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage

# ========================
# 환경 설정
# ========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20251106" 
persist_base_dir = r"./vector_dbs1"
os.makedirs(persist_base_dir, exist_ok=True)

# LLM / 임베딩 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=8000,
    openai_api_key=api_key
)

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# ========================
# PDF → Document 변환
# ========================
def pdf_to_markdown(pdf_path):
    md_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                blocks = page.get_text("blocks")
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
                md_text += f"# Page {i + 1}\n\n"
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
    except Exception:
        print(f"[오류] PDF 변환 실패 : {pdf_path}\n")
        return ""

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
            print(f"{pdf_file} 처리 완료 ({len(docs)} 문서)")
        except Exception as e:
            print(f"{pdf_file} 처리 중 오류: {e}")
    print(f"\n총 Document 수: {len(all_docs)}")
    return all_docs

# ========================
# 청크 설정 (2개만)
# ========================
chunk_settings = [
    {"size": 1000, "overlap": 200},
    {"size": 1200, "overlap": 300},
]

docs = None
vectorstores = []

# ========================
# DB 생성 + 청크 내용 전체 저장
# ========================
for setting in chunk_settings:
    db_name = f"test_{setting['size']}_{setting['overlap']}"
    db_dir = os.path.join(persist_base_dir, db_name)
    os.makedirs(db_dir, exist_ok=True)

    print(f"\n=== [{db_name}] DB 생성 시작 ===")

    # PDF 로드 (최초 1회)
    if docs is None:
        docs = load_all_pdfs(pdf_folder)

    # 청크 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=setting["size"],
        chunk_overlap=setting["overlap"]
    )
    split_docs = splitter.split_documents(docs)
    print(f"총 {len(split_docs)}개의 청크 생성 완료.")

    # 벡터 DB 생성
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=db_dir
    )
    vectorstore.persist()
    print(f"✅ 벡터 DB 저장 완료: {db_dir}")

    # 청크 전체 내용 저장
    preview_path = os.path.join(db_dir, f"{db_name}_chunks_full.txt")
    with open(preview_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(split_docs):
            f.write(f"--- 청크 {i + 1} ---\n")
            f.write(chunk.page_content.strip() + "\n")
            f.write(f"[출처: {chunk.metadata.get('source', '없음')}]\n")
            f.write("=" * 80 + "\n")
    print(f"📄 청크 전체 내용 저장 완료: {preview_path}")

    vectorstores.append((vectorstore, setting, len(split_docs)))

print("\n✅ 모든 DB 및 청크 내용 저장 완료!")

# ========================
# RAG 질의응답
# ========================
def rag_answer(question, retriever):
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    context_texts = [doc.page_content.strip() for doc in retriever_docs if doc.page_content.strip()]
    context = "\n\n".join(context_texts)
    sources = sorted(set([doc.metadata.get("source", "출처 없음") for doc in retriever_docs]))

    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 문서에 없는 내용은 절대 추가하지 말고, 없으면 '정보 없음'이라고 표시하세요.
            - 답변은 항목별로 구분된 형태로 작성하세요.
            - 각 항목은 한 줄 띄우기로 구분
        """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    answer = response.content

    # 형식 정리
    lines = answer.split("\n")
    final_lines, counter = [], 1
    for line in lines:
        stripped = line.strip()
        if not stripped:
            final_lines.append("")
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
            final_lines.append(f"{counter}. {stripped}")
            counter += 1
        else:
            final_lines.append(stripped)

    if sources:
        final_lines.append("\n---출처---")
        for s in sources:
            final_lines.append(f"[출처: {s}]")

    return "\n".join(final_lines), len(retriever_docs)

# ========================
# 질의응답 실행
# ========================
if __name__ == "__main__":
    while True:
        question = input("\n질문을 입력하세요 (exit 입력 시 종료): ").strip()
        if question.lower() == "exit":
            print("프로그램 종료.")
            break

        for vectorstore, setting, total_chunks in vectorstores:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            answer, retrieved_count = rag_answer(question, retriever)

            print(f"\n=== [{setting['size']}_{setting['overlap']}] 결과 ===")
            print(f"총 청크 수: {total_chunks}")
            print(f"검색된 청크 수: {retrieved_count}")
            print(f"답변 길이: {len(answer)} 글자\n")
            print(answer)
            print("-" * 60)
