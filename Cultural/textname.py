import os

def extract_pdf_filenames(root_folder, output_txt):
    pdf_filenames = []

    # 폴더 전체 탐색 (하위 폴더 포함)
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                filename_without_ext = os.path.splitext(file)[0]   # ← 확장자 제거
                pdf_filenames.append(filename_without_ext)

    # 중복 제거 + 정렬(Optional)
    pdf_filenames = sorted(list(set(pdf_filenames)))

    # txt 파일로 저장
    with open(output_txt, "w", encoding="utf-8") as f:
        for name in pdf_filenames:
            f.write(name + "\n")

    print(f"총 {len(pdf_filenames)}개의 파일명을 {output_txt} 에 저장했습니다.")


# 실행 파트
if __name__ == "__main__":
    root_folder = r"C:\Users\user\Desktop\pdfs\textpdf자료\문화유산\사적"
    output_txt = "pdf_filenames.txt"

    extract_pdf_filenames(root_folder, output_txt)
