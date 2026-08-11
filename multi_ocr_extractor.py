# save_as_separate_json.py
import os
import json
from glob import glob
from pdf2image import convert_from_path
from pytesseract import image_to_string
from tqdm import tqdm
from PyPDF2 import PdfReader

PDF_DIR = os.path.expanduser("~/pdf/sw2.5")
OCR_DIR = os.path.expanduser("./ocr")
LANG = "jpn"

os.makedirs(OCR_DIR, exist_ok=True)

def ocr_single_pdf_one_by_one(pdf_path):
    bookname = os.path.splitext(os.path.basename(pdf_path))[0]
    output_file = os.path.join(OCR_DIR, f"{bookname}.json")

    # 処理済ならスキップ（必要に応じて消す）
    if os.path.exists(output_file):
        print(f"⏩ スキップ済：{bookname}")
        return

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    results = []

    for i in tqdm(range(num_pages), desc=f"OCR中: {bookname}", unit="page"):
        images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
        if not images:
            continue
        text = image_to_string(images[0], lang=LANG)
        results.append({
            "book": bookname,
            "page": i + 1,
            "text": text.strip()
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 完了：{bookname} → {output_file}")

if __name__ == "__main__":
    pdf_files = glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"📁 PDFファイルが見つかりません：{PDF_DIR}")
    else:
        for pdf in pdf_files:
            try:
                ocr_single_pdf_one_by_one(pdf)
            except Exception as e:
                print(f"⚠️ エラー発生：{pdf} → {str(e)}")

