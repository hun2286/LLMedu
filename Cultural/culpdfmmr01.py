# 출처를 맨아래 모아서 한 번만 표시하는 코드

import os
import fitz
import io
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from PyPDF2 import PdfReader, PdfWriter
# 추가
from collections import defaultdict


# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20251106"
persist_dir = "./mk_pdf_chroma_db2"
output_pdf_path = "result_log.pdf"

# LLM / 임베딩 설정
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    max_tokens=8000,
    openai_api_key=api_key
)

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# 폰트 등록
font_path = r"C:\Users\BGR_NC_2_NOTE\Desktop\Project\LLMedu\fonts"
pdfmetrics.registerFont(TTFont('NanumGothic', os.path.join(font_path, 'NanumGothic.ttf')))
pdfmetrics.registerFont(TTFont('NanumGothicBold', os.path.join(font_path, 'NanumGothicBold.ttf')))

# PDF → 마크다운 변환
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

# Vector DB 구축 (DB 재사용)
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("[DB 없음] 새로 구축합니다.")
    docs = load_all_pdfs(pdf_folder)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    print(f"총 청크 수: {len(split_docs)}")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectorstore.persist()
else:
    print("[DB 있음] 기존 DB를 재사용합니다.")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)

# MMR 검색 기반 Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.5}
)

# RAG 기반 답변 생성
def rag_answer(question):
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    doc_chunks = defaultdict(list)
    for doc in retriever_docs:
        src = doc.metadata.get("source", "출처 없음")
        if len(doc_chunks[src]) < 3:  # 문서당 최대 3청크 허용
            doc_chunks[src].append(doc)

    retriever_docs = [d for docs in doc_chunks.values() for d in docs]

    # context 구성
    context_texts = [doc.page_content.strip() for doc in retriever_docs if doc.page_content.strip()]
    context = "\n\n".join(context_texts)

    # 출처 정리
    # sources = sorted(set([doc.metadata.get("source", "출처 없음") for doc in retriever_docs]))
    # sources_text = ", ".join(sources)

    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 답변은 **항목별로 구분된 형태**로 작성하세요.
            - 각 항목은 한 줄 띄우기로 구분하세요.
            - 질문에 대해 가능한 한 풍부하게 설명하세요.
            - 여러 문서의 내용을 통합하여 한 문단 이상의 답변을 작성하세요.
            - 문서에 없으면 '정보 없음'으로 표시하세요.
            - **본문에는 출처를 표시하지 마세요.**
            - 대신 모든 출처는 답변 맨 마지막에 한 번만 모아서 표시하세요.
        """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}\n\n출처 : {sources_text}")
    ]

    response = llm.invoke(messages)

    sources = sorted(set([doc.metadata.get("source", "출처 없음") for doc in retriever_docs]))
    sources_text = "\n".join([f"[출처: {src}]" for src in sources])

    final_answer = response.content.strip() + "\n\n" + sources_text
    return final_answer

# PDF 저장 관련
pdf_writer = PdfWriter()
current_canvas = None
current_packet = None
y_position = None

def append_to_pdf(question, answer):
    global pdf_writer, current_canvas, current_packet, y_position

    # 중복 출처 제거
    unique_lines = []
    seen_sources = set()
    for line in answer.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("[출처:"):
            if line not in seen_sources:
                unique_lines.append(line)
                seen_sources.add(line)
        else:
            unique_lines.append(line)
    answer = "\n".join(unique_lines)

    width, height = A4
    margin_top, margin_bottom, margin_left, margin_right = 25, 25, 20, 20
    nanum_style = ParagraphStyle('Nanum', fontName='NanumGothic', fontSize=11, leading=15)
    nanum_bold_style = ParagraphStyle('NanumBold', fontName='NanumGothicBold', fontSize=13, leading=17)

    if current_canvas is None:
        current_packet = io.BytesIO()
        current_canvas = canvas.Canvas(current_packet, pagesize=A4)
        y_position = height - margin_top

    can = current_canvas
    y = y_position

    # 질문
    para = Paragraph(f"질문: {question}", nanum_bold_style)
    w, h = para.wrap(width - margin_left - margin_right, y)
    if y - h < margin_bottom:
        can.showPage()
        y = height - margin_top
    para.drawOn(can, margin_left, y - h)
    y -= h + 10

    # 답변
    for line in answer.split("\n"):
        if not line.strip():
            continue
        if line.startswith("[출처:"):
            para = Paragraph(line, nanum_style)
        elif line.startswith("# "):
            para = Paragraph(line[2:].strip(), nanum_bold_style)
        elif line.startswith("## "):
            para = Paragraph(line[3:].strip(), nanum_bold_style)
        else:
            para = Paragraph(line.strip(), nanum_style)

        w, h = para.wrap(width - margin_left - margin_right, y)
        if y - h < margin_bottom:
            can.showPage()
            y = height - margin_top
        para.drawOn(can, margin_left, y - h)
        y -= h + 5

    y_position = y - 15

def save_pdf(path=output_pdf_path):
    global pdf_writer, current_canvas, current_packet
    if current_canvas is None:
        print("저장할 PDF가 없습니다.")
        return
    current_canvas.save()
    current_packet.seek(0)
    new_pdf = PdfReader(current_packet)
    for page in new_pdf.pages:
        pdf_writer.add_page(page)
    with open(path, "wb") as f:
        pdf_writer.write(f)
    print(f"PDF 최종 저장 완료: {path}")

# 실행
if __name__ == "__main__":
    print("=" * 60)
    print("RAG PDF 질의응답 시스템 (다문서 검색 개선버전)")
    print("=" * 60)

    while True:
        query = input("\n질문을 입력하세요 (exit 입력 시 종료): ").strip()
        if query.lower() == "exit":
            if current_canvas:
                save_choice = input("PDF 저장하시겠습니까? (y/n): ").strip().lower()
                if save_choice == "y":
                    save_pdf()
            print("프로그램 종료.")
            break

        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)

        if "정보 없음" in answer:
            print("정보 없음 포함 — 저장 생략\n")
            continue

        save_choice = input("이 답변을 PDF에 저장하시겠습니까? (y/n): ").strip().lower()
        if save_choice == "y":
            append_to_pdf(query, answer)
            print("저장 완료")
        else:
            print("저장하지 않고 다음으로 넘어갑니다.")


