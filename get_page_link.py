import os
import urllib.parse
import json
import re

# OCRディレクトリとPDF保存先
OCR_DIR = os.getenv("OCR_DIR", "./ocr")
PDF_DIR = os.getenv("PDF_DIR", "/var/www/html/pdfjs/docs")

# PDFファイルへのパスを自動構築
book_to_filename = {}

for filename in os.listdir(OCR_DIR):
    if not filename.endswith(".json"):
        continue
    filepath = os.path.join(OCR_DIR, filename)
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                book = entry.get("book")
                if book and book not in book_to_filename:
                    matches = [f for f in os.listdir(PDF_DIR) if book in f and f.endswith(".pdf")]
                    if matches:
                        book_to_filename[book] = matches[0]
    except Exception as e:
        print(f"⚠️ エラー: {filename} - {e}")

# BOOK_URLSを構築
BOOK_URLS = {
    book: f"/pdfjs/docs/{pdf}"
    for book, pdf in book_to_filename.items()
}

PDFJS_BASE = os.getenv("PDFJS_BASE_URL", "http://sw-rule-www.grd-svc.com/pdfjs/web/viewer.html")


def get_page_link(book: str, page: int, book_urls: dict = None) -> str:
    """指定された書籍とページからPDF.jsの出典リンクを生成"""
    urls = book_urls or BOOK_URLS
    path = urls.get(book)
    if not path:
        return f"{book} - p.{page}"
    encoded_path = urllib.parse.quote(path)
    
    # 画像リンクを構成
    base_image_url = os.getenv("IMAGE_BASE_URL", "https://sw-rule-www.grd-svc.com/image")
    encoded_book = urllib.parse.quote(book, safe='')
    page_str = f"P{page:05d}"
    image_link = f"{base_image_url}/{encoded_book}/{page_str}.jpg"

    # PDFリンクと画像リンクを返す
    pdf_link = f"{PDFJS_BASE}?file={encoded_path}#page={page}"
    return pdf_link, image_link


def get_page_and_image_links(book: str, page: int, book_urls: dict = None) -> tuple[str, str]:
    urls = book_urls or BOOK_URLS
    path = urls.get(book)
    if not path:
        return None, None

    encoded_path = urllib.parse.quote(path)
    pdf_link = f"{PDFJS_BASE}?file={encoded_path}#page={page}"

    base_image_url = os.getenv("IMAGE_BASE_URL", "https://sw-rule-www.grd-svc.com/image")
    encoded_book = urllib.parse.quote(book, safe='')
    page_str = f"P{page:05d}"
    image_link = f"{base_image_url}/{encoded_book}/{page_str}.jpg"

    return pdf_link, image_link


def auto_link_answer_text(answer: str, book_name_map: dict = None) -> str:
    """回答中の `(カテゴリ / 書籍名 - p.数字)` パターンをリンクに変換（括弧付き）"""
    if book_name_map is None:
        book_name_map = {}

    def replacer(match):
        full = match.group(0)
        book = match.group(1).strip()
        page = int(match.group(2))
        resolved_book = (
            book_name_map.get(book)
            or book_name_map.get(book.split(" / ")[-1])
            or book
        )
        
        # PDFリンクと画像リンクを生成
        pdf_link, image_link = get_page_link(resolved_book, page)
        if not pdf_link or not image_link:
            return f"[{book} - p.{page}]"
        return f"[{book} - p.{page}]({image_link}) [📄 PDFで開く]({pdf_link})"

    # 正規表現で (書籍名 - p.123) や (カテゴリ / 書籍名 - p.123) を抽出
    pattern = r"[\(（]([^\)）]+?)\s*-\s*p\.\s*(\d+)[\)）]"
    return re.sub(pattern, replacer, answer)


def auto_image_link_answer_text(answer: str, book_name_map: dict = None) -> str:
    from get_page_link import get_page_and_image_links

    if book_name_map is None:
        book_name_map = {}

    def replacer(match):
        book = match.group(1).strip()
        page = int(match.group(2))
        resolved_book = book_name_map.get(book) or book
        pdf_link, image_link = get_page_and_image_links(resolved_book, page)
        if not pdf_link or not image_link:
            return f"[{book} - p.{page}]"
        return f"（<a href='{image_link}' target='_blank'>{book} - p.{page}</a> [<a href='{pdf_link}' target='_blank'>PDF</a>]）"

    pattern = r"[\(（]([^\)）]+?)\s*-\s*p\.\s*(\d+)[\)）]"
    return re.sub(pattern, replacer, answer)


def render_streamlit_demo():
    import streamlit as st
    book = st.selectbox("書籍を選択", list(BOOK_URLS.keys()))
    page = st.number_input("ページ番号", min_value=1, step=1)

    if st.button("出典リンクを生成"):
        link = get_page_link(book, page)
        st.markdown(link, unsafe_allow_html=True)


if __name__ == "__main__":
    render_streamlit_demo()
    
