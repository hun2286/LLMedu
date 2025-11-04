# 기본 RAG 처리에 질의응답 pdf로 생성 (페이지 단위 Document) 흠 조금 이상한데

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

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20151103"
persist_dir = "./doc_chroma_db"
output_pdf_path = "result_log1.pdf"

# LLM / Embedding
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.2, 
    max_tokens=6000,
    openai_api_key=api_key
)

embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# 폰트 등록
font_path = r"C:\Users\BGR_NC_2_NOTE\Desktop\LLMedu\fonts"
pdfmetrics.registerFont(TTFont('NanumGothic', os.path.join(font_path, 'NanumGothic.ttf')))
pdfmetrics.registerFont(TTFont('NanumGothicBold', os.path.join(font_path, 'NanumGothicBold.ttf')))

# PDF → 페이지 단위 Markdown 변환
def pdf_to_page_markdowns(pdf_path):
    """PDF를 페이지 단위로 마크다운 문자열 리스트로 변환"""
    page_texts = []
    try:
        with fitz.open(pdf_path) as pdf:
            for i, page in enumerate(pdf):
                page_md = f"# Page {i+1}\n\n"
                blocks = page.get_text("blocks")
                blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

                for block in blocks:
                    text = block[4].strip()
                    if not text:
                        continue
                    num_words = len(text.split())
                    num_lines = text.count("\n") + 1

                    if num_words <= 5 and num_lines <= 2:
                        page_md += f"# {text}\n\n"
                    elif num_words <= 15:
                        page_md += f"## {text}\n\n"
                    else:
                        page_md += f"{text}\n\n"

                if page_md.strip():
                    page_texts.append(page_md)

        return page_texts
    except Exception as e:
        print(f"[오류] PDF 변환 실패: {pdf_path}\n{e}")
        return []

# 페이지별 Document 생성
def load_pdf_safe(pdf_path):
    docs = []
    page_texts = pdf_to_page_markdowns(pdf_path)
    for i, text in enumerate(page_texts):
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.splitext(os.path.basename(pdf_path))[0],
                    "page": i + 1
                }
            )
        )
    return docs

# 폴더 내 모든 PDF 로드
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# RAG 기반 답변
def rag_answer(question):
    retriever_docs = retriever.invoke(question)
    if isinstance(retriever_docs, Document):
        retriever_docs = [retriever_docs]

    context_texts = []
    for doc in retriever_docs:
        source = doc.metadata.get("source", "출처 없음")
        page = doc.metadata.get("page", "N/A")
        context_texts.append(f"[{source} p.{page}]\n{doc.page_content}")

    context = "\n\n".join(context_texts)
    
    # 출처 별도 정리
    sources = sorted(set([doc.metadata.get("source", "출처 없음") for doc in retriever_docs]))
    sources_text = ", ".join(sources)
    
    messages = [
        SystemMessage(content="""
            당신은 여러 PDF 문서를 참고하여 질문에 답하는 전문가입니다.
            - 질문에 등장하는 단어를 각각 구분하여 정확히 답변하세요.
            - 문서에 없으면 '정보 없음'으로 표시하세요.
            - 가능한 한 문서 내 문맥과 키워드에 기반하여 정확하게 답변하세요.
            - 각 항목의 출처는 반드시 '[출처: 문서명 p.페이지]' 형태로 표시하세요.
        """),
        HumanMessage(content=f"문서 내용:\n{context}\n\n질문:\n{question}\n\n출처 : {sources_text}")
    ]

    response = llm.invoke(messages)
    return response.content

# PDF 저장 (질의응답 로그)
pdf_writer = PdfWriter()
current_canvas = None
current_packet = None
y_position = None

def append_to_pdf(question, answer):
    global pdf_writer, current_canvas, current_packet, y_position

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

    para = Paragraph(f"질문: {question}", nanum_bold_style)
    w, h = para.wrap(width - margin_left - margin_right, y)
    if y - h < margin_bottom:
        can.showPage()
        y = height - margin_top
    para.drawOn(can, margin_left, y - h)
    y -= h + 10

    lines = answer.split("\n")
    for line in lines:
        if not line.strip():
            continue
        if line.startswith("[출처:"):
            para = Paragraph(line, nanum_style)
        elif line.startswith("# "):
            para = Paragraph(line[2:].strip(), nanum_bold_style)
        elif line.startswith("## "):
            para = Paragraph(line[3:].strip(), nanum_bold_style)
        else:
            para = Paragraph(line, nanum_style)

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
    print("페이지 단위 RAG PDF 질의응답 시스템")
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
            print("저장 완료\n")
        else:
            print("저장하지 않고 다음으로 넘어갑니다.\n")
