# 이미지 pdf 찾기 

import os
import fitz

pdf_folder = r"C:\Users\user\Desktop\pdfs\20251106"
min_text_len = 20  # 텍스트 길이 기준

failed_pdfs = []

pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
print(f"총 PDF 파일 수: {len(pdf_files)}\n")

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    try:
        with fitz.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4].strip()
                    if text:
                        full_text += text + "\n"
        if len(full_text.strip()) < min_text_len:
            failed_pdfs.append(pdf_file)
            print(f"{pdf_file} → 텍스트 부족 / 청크 생성 불가")
    except Exception as e:
        failed_pdfs.append(pdf_file)
        print(f"{pdf_file} → 오류: {e}")

if failed_pdfs:
    print("\n--- 텍스트 변환 실패 / 청크 생성 불가 PDF 목록 ---")
    for f in failed_pdfs:
        print(f"- {f}")
else:
    print("모든 PDF에서 텍스트 추출 가능")
