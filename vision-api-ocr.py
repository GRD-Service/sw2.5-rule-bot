import os
import time
import json
from pdf2image import convert_from_path
from google.cloud import vision

# --------------------------
# 設定パート
# --------------------------
PDF_FOLDER = '../pdf/sw2.5/'
IMAGE_FOLDER = './images/'
OUTPUT_FOLDER = './ocr/'
CREDENTIALS_PATH = './auth/ocrtest-221103-e9fc363e1b4b.json'  # 認証ファイルパス

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

vision_client = vision.ImageAnnotatorClient()

# --------------------------
# テキストクリーニング関数
# --------------------------
def safe_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

# --------------------------
# PDFファイルリスト取得
# --------------------------
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]

for pdf_file in pdf_files:
    print(f"\n[INFO] Processing {pdf_file}...")
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)

    from pdf2image import pdfinfo_from_path
    pdf_info = pdfinfo_from_path(pdf_path)
    total_pages = pdf_info['Pages']

    result_texts = []

    book_name = os.path.splitext(pdf_file)[0]  # 拡張子なしのファイル名

    for page_num in range(1, total_pages + 1):
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        image = images[0]

        image_path = os.path.join(IMAGE_FOLDER, f"{book_name}_{page_num}.jpg")
        image.save(image_path, 'JPEG')

        with open(image_path, 'rb') as img_file:
            content = img_file.read()

        image_vision = vision.Image(content=content)

        response = vision_client.document_text_detection(
            image=image_vision,
            image_context={"language_hints": ["ja"]}
        )

        full_text = response.full_text_annotation.text if response.full_text_annotation.text else ''

        result_texts.append({
            "book": book_name,
            "page": page_num,
            "text": safe_text(full_text.strip())
        })

        print(f"Processed page {page_num}")

    # --------------------------
    # ローカルに保存
    # --------------------------
    output_path = os.path.join(OUTPUT_FOLDER, f"{book_name}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_texts, f, ensure_ascii=False, indent=2)

    print(f"Saved OCR JSON to {output_path}")

    # （任意）画像一時ファイルを削除する場合
    # for page_num in range(1, total_pages + 1):
    #     os.remove(os.path.join(IMAGE_FOLDER, f"{book_name}_{page_num}.jpg"))

