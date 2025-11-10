import os
import fitz
import re
from dotenv import load_dotenv
from PIL import Image
import pytesseract

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

persist_base_dir = r"./vector_dbs1"
os.makedirs(persist_base_dir, exist_ok=True)

# tesseract.exe 경로
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# tessdata 경로 (kor.traineddata와 한자 포함 kor_vert 등 필요)
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

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
# PDF → 텍스트 / OCR 변환 (개선)
# ========================
def pdf_to_text_or_ocr(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                page_text = page.get_text()
                text += page_text + "\n"
    except Exception as e:
        print(f"[PyMuPDF 오류] {pdf_path}: {e}")

    # 텍스트 없거나 너무 짧으면 OCR 수행
    if not text.strip() or len(text.strip()) < 20:
        print("[OCR 실행] 텍스트 없음 또는 너무 짧음, 이미지에서 추출 중:", pdf_path)
        try:
            with fitz.open(pdf_path) as pdf:
                for page in pdf:
                    # 고해상도 이미지 변환
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # 흑백 변환
                    img = img.convert("L")

                    # Tesseract 옵션: 한글 + 번체 한자
                    custom_config = r'--oem 3 --psm 6'
                    page_text = pytesseract.image_to_string(img, lang="kor+chi_tra", config=custom_config)

                    # 줄바꿈, 공백 정리
                    lines = page_text.splitlines()
                    cleaned_lines = []

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        # 괄호 내부 한자만 필터링
                        def preserve_parentheses(match):
                            content = match.group(1)
                            # 한자, 숫자, 영어만 허용 (한글 제외)
                            filtered = re.sub(r"[^一-龥0-9a-zA-Z]", "", content)
                            return f"({filtered})"

                        line = re.sub(r"\((.*?)\)", preserve_parentheses, line)

                        # 괄호 밖 깨진 문자 제거 (한글, 영어, 숫자, 기본 구두점)
                        line = re.sub(r"[^가-힣0-9a-zA-Z\s.,:;!?()\[\]\-<>%]", "", line)

                        if len(line) >= 2:
                            cleaned_lines.append(line)

                    page_text = "\n".join(cleaned_lines)
                    text += page_text + "\n"

        except Exception as e:
            print(f"[OCR 오류] {pdf_path}: {e}")

    return text.strip()
def load_pdf_safe(pdf_path):
    content = pdf_to_text_or_ocr(pdf_path)
    if content:
        return [Document(page_content=content,
                         metadata={"source": os.path.splitext(os.path.basename(pdf_path))[0]})]
    return []

# ========================
# 단일 PDF / 폴더 PDF 처리
# ========================
def process_pdf_folder(pdf_folder, chunk_size=1200, chunk_overlap=300):
    all_docs = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    print(f"총 PDF 파일 수: {len(pdf_files)}\n")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        docs = load_pdf_safe(pdf_path)
        all_docs.extend(docs)
        print(f"{pdf_file} 처리 완료 ({len(docs)} 문서)")
    return create_vectorstore(all_docs, chunk_size, chunk_overlap, name_prefix="folder")

def process_single_pdf(pdf_path, chunk_size=1200, chunk_overlap=300):
    docs = load_pdf_safe(pdf_path)
    if not docs:
        print("[경고] PDF에서 텍스트를 추출할 수 없음:", pdf_path)
        return None
    return create_vectorstore(docs, chunk_size, chunk_overlap, name_prefix="single")

# ========================
# 벡터 DB 생성 + 청크 저장
# ========================
def create_vectorstore(docs, chunk_size, chunk_overlap, name_prefix):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    print(f"총 {len(split_docs)}개의 청크 생성 완료.")

    db_name = f"{name_prefix}_vector_db"
    db_dir = os.path.join(persist_base_dir, db_name)
    os.makedirs(db_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=db_dir
    )
    vectorstore.persist()
    print(f"✅ 벡터 DB 저장 완료: {db_dir}")

    # 청크 전체 저장
    preview_path = os.path.join(db_dir, f"{db_name}_chunks_full.txt")
    with open(preview_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(split_docs):
            f.write(f"--- 청크 {i + 1} ---\n")
            f.write(chunk.page_content.strip() + "\n")
            f.write(f"[출처: {chunk.metadata.get('source', '없음')}]\n")
            f.write("="*80 + "\n")
    print(f"📄 청크 전체 내용 저장 완료: {preview_path}")
    return vectorstore

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
# 실행 선택
# ========================
if __name__ == "__main__":
    print("1. PDF 폴더 전체 처리\n2. 단일 PDF 처리")
    choice = input("선택하세요 (1 또는 2): ").strip()

    if choice == "1":
        pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20251106"
        vectorstore = process_pdf_folder(pdf_folder)
    elif choice == "2":
        pdf_path = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20251106\1999_경기도도당굿_05_Ⅱ_경기도 도당굿의 내용_19.pdf"
        vectorstore = process_single_pdf(pdf_path)
    else:
        print("잘못된 선택입니다.")
        exit()

    if vectorstore:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        while True:
            question = input("\n질문을 입력하세요 (exit 입력 시 종료): ").strip()
            if question.lower() == "exit":
                break
            answer, retrieved_count = rag_answer(question, retriever)
            print(f"\n검색된 청크 수: {retrieved_count}")
            print(answer)
            print("-" * 60)
