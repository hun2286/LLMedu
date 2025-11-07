# 청크사이즈 3개 각각 성능 비교

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
persist_base_dir = r"./vector_dbs"
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

docs = load_all_pdfs(pdf_folder)

# ========================
# 청크 설정 3개 + 직관적 DB 이름
# ========================
chunk_settings = [
    {"size": 800, "overlap": 200},
    {"size": 1000, "overlap": 200},
    {"size": 1200, "overlap": 300},
]

vectorstores = []

for setting in chunk_settings:
    db_name = f"chunk_{setting['size']}_{setting['overlap']}"
    db_dir = os.path.join(persist_base_dir, db_name)
    os.makedirs(db_dir, exist_ok=True)
    
    print(f"\n=== 청크 설정: size={setting['size']}, overlap={setting['overlap']} ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=setting['size'],
        chunk_overlap=setting['overlap']
    )
    split_docs = splitter.split_documents(docs)
    print(f"총 청크 수: {len(split_docs)}")
    
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=db_dir
    )
    vectorstore.persist()
    vectorstores.append((vectorstore, setting, len(split_docs)))  # DB, 설정, 총 청크 수

# ========================
# RAG 질의응답
# ========================
def rag_answer(question, retriever):
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    context_texts = []
    for doc in retriever_docs:
        content = doc.page_content.strip()
        if not content:
            continue
        context_texts.append(content)
        
    context = "\n\n".join(context_texts)
    sources = sorted(set([doc.metadata.get("source", "출처 없음") for doc in retriever_docs]))

    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 문서에 없는 내용은 절대 추가하지 말고, 없으면 '정보 없음'이라고 표시하세요.
            - 답변은 항목별로 구분된 형태로 작성하세요.
            - 각 항목은 한 줄 띄우기로 구분
            - 문서 내 문맥과 키워드에 기반하여 정확하게 답변
            """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}")
    ]

    response = llm.invoke(messages)
    answer = response.content

    # 번호 매기기 + 출처 맨 아래
    lines = answer.split("\n")
    final_lines = []
    counter = 1
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
        final_lines.append("")
        final_lines.append("---출처---")
        for s in sources:
            final_lines.append(f"[출처: {s}]")
    return "\n".join(final_lines), len(retriever_docs)

# ========================
# 질문 입력 & 성능 비교 출력
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
            
            print(f"\n=== 청크 DB ({setting['size']}_{setting['overlap']}) 성능 비교 ===")
            print(f"총 청크 수: {total_chunks}")
            print(f"검색된 청크 수: {retrieved_count}")
            print(f"답변 길이: {len(answer)} 글자")
            print("답변 내용:")
            print(answer)
            print("-"*60)
