import os
import shutil
import fitz
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage, HumanMessage
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from PyPDF2 import PdfReader, PdfWriter
from datetime import datetime
import io

# 환경 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

pdf_folder = r"C:\Users\BGR_NC_2_NOTE\Desktop\pdfs\20151103"
persist_dir = "./multi_pdf_chroma_db"
output_pdf_path = "answers_log.pdf"

# LLM / 임베딩
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens = 2000, openai_api_key=api_key)
embedding_model = HuggingFaceEmbeddings(
    model_name="bespin-global/klue-sroberta-base-continue-learning-by-mnr"
)

# 폰트 등록
font_path = r"C:\Users\BGR_NC_2_NOTE\Desktop\LLMedu\fonts"
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

# PDF 로드 및 청킹
docs = load_all_pdfs(pdf_folder)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
split_docs = text_splitter.split_documents(docs)
print(f"총 청크 수: {len(split_docs)}")

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
        content = doc.page_content
        context_texts.append(f"[{source}]\n{content}")

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

# PDF 저장 관련
pdf_writer = PdfWriter()
current_canvas = None
current_packet = None
y_position = None

# PDF 출력 함수
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
    margin_top, margin_bottom = 25, 25
    margin_left, margin_right = 20, 20

    nanum_style = ParagraphStyle('Nanum', fontName='NanumGothic', fontSize=11, leading=14)
    nanum_bold_style = ParagraphStyle('NanumBold', fontName='NanumGothicBold', fontSize=12, leading=15)

    if current_canvas is None:
        current_packet = io.BytesIO()
        current_canvas = canvas.Canvas(current_packet, pagesize=A4)
        y_position = height - margin_top

    can = current_canvas
    y = y_position

    # 날짜
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    can.setFont("NanumGothicBold", 12)
    can.drawString(margin_left, y, f"[{now}]")
    y -= 25

    # 질문
    para = Paragraph(f"질문: {question}", nanum_bold_style)
    w, h = para.wrap(width - margin_left - margin_right, y)
    if y - h < margin_bottom:
        can.showPage()
        y = height - margin_top
    para.drawOn(can, margin_left, y - h)
    y -= h + 10

    # 답변 (줄 단위로 처리)
    lines = answer.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[출처:"):
            para = Paragraph(line, nanum_style)
            w, h = para.wrap(width - margin_left - margin_right, y)
            if y - h < margin_bottom:
                can.showPage()
                y = height - margin_top
            para.drawOn(can, margin_left, y - h)
            y -= h + 10
            continue

        if line.startswith("# "):
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

    y_position = y

    # 질문-답변 사이 여백
    if y - 20 < margin_bottom:
        can.showPage()
        y_position = height - margin_top
    else:
        y_position -= 20

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
    while True:
        print("-" * 50)
        query = input("질문을 입력하세요 (exit 입력 시 종료): ")
        if query.lower() == "exit":
            if current_canvas:
                save_choice = input("작성된 PDF를 저장하시겠습니까? (y/n): ").strip().lower()
                if save_choice == "y":
                    save_pdf()
            print("프로그램 종료")
            break

        answer = rag_answer(query)
        print("\n[답변]:\n", answer)
        print("-" * 50)

        if "정보 없음" in answer:
            print("'정보 없음'이 포함된 답변이므로 저장하지 않습니다.\n")
            continue

        save_choice = input("이 답변을 PDF에 저장하시겠습니까? (y/n): ").strip().lower()
        if save_choice == "y":
            append_to_pdf(query, answer)
        else:
            print("저장하지 않고 넘어갑니다.\n")
