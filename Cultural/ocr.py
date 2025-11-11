import fitz
from PIL import Image
import pytesseract
import os

# PDF 경로
pdf_path = r"C:\Users\user\Desktop\pdfs\scanpdf목록\1999_경기도도당굿_05_Ⅱ_경기도 도당굿의 내용_19.pdf"
# 저장할 텍스트 경로
txt_path = r"C:\Users\user\Desktop\pdfs\example_ocr1.txt"

# Tesseract 경로
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

def ocr_pdf(pdf_path, max_pages=3):  # max_pages 매개변수 추가
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            if page_num > max_pages:  # 지정한 페이지 이상이면 중단
                break 
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = img.convert("L")
            page_text = pytesseract.image_to_string(img, lang="kor", config="--oem 3 --psm 6")
            text += f"--- 페이지 {page_num} ---\n"
            text += page_text + "\n\n"
    return text

# OCR 실행 (처음 2페이지만)
ocr_result = ocr_pdf(pdf_path, max_pages=3)

# 텍스트 파일로 저장
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(ocr_result)

print(f"OCR 결과 저장 완료: {txt_path}")