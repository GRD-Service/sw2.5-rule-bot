import os
import urllib.parse
import json
import re


# ============================================================
# OCRディレクトリとPDF保存先
# ============================================================

OCR_DIR = os.getenv(
    "OCR_DIR",
    "./ocr",
)

PDF_DIR = os.getenv(
    "PDF_DIR",
    "/var/www/html/pdfjs/docs",
)


# ============================================================
# PDFファイルへのパスを自動構築
# ============================================================

book_to_filename = {}

for filename in os.listdir(OCR_DIR):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(
        OCR_DIR,
        filename,
    )

    try:
        with open(
            filepath,
            encoding="utf-8",
        ) as f:
            data = json.load(f)

            for entry in data:
                book = entry.get("book")

                if (
                    book
                    and book not in book_to_filename
                ):
                    matches = [
                        f
                        for f in os.listdir(PDF_DIR)
                        if (
                            book in f
                            and f.endswith(".pdf")
                        )
                    ]

                    if matches:
                        book_to_filename[book] = (
                            matches[0]
                        )

    except Exception as e:
        print(
            f"⚠️ エラー: "
            f"{filename} - {e}"
        )


# ============================================================
# BOOK_URLSを構築
# ============================================================

BOOK_URLS = {
    book: f"/pdfjs/docs/{pdf}"
    for book, pdf
    in book_to_filename.items()
}


PDFJS_BASE = os.getenv(
    "PDFJS_BASE_URL",
    "http://sw-rule-www.grd-svc.com/"
    "pdfjs/web/viewer.html",
)


# ============================================================
# Link helpers
# ============================================================

def get_page_link(
    book: str,
    page: int,
    book_urls: dict = None,
) -> tuple[str | None, str | None]:
    """
    指定された書籍とページから
    PDF.jsリンクと画像リンクを生成する。

    PDF.jsの通信関連設定はviewer側で管理するため、
    URLにはfileとpageだけを指定する。
    """

    urls = book_urls or BOOK_URLS
    path = urls.get(book)

    if not path:
        return None, None

    encoded_path = urllib.parse.quote(
        path
    )

    base_image_url = os.getenv(
        "IMAGE_BASE_URL",
        "https://sw-rule-www.grd-svc.com/image",
    )

    encoded_book = urllib.parse.quote(
        book,
        safe="",
    )

    page_str = f"P{page:05d}"

    image_link = (
        f"{base_image_url}/"
        f"{encoded_book}/"
        f"{page_str}.jpg"
    )

    pdf_link = (
        f"{PDFJS_BASE}"
        f"?file={encoded_path}"
        f"#page={page}"
    )

    return pdf_link, image_link


def get_page_and_image_links(
    book: str,
    page: int,
    book_urls: dict = None,
) -> tuple[str | None, str | None]:
    """
    指定された書籍・PDF内部ページ番号から、
    PDF.jsリンクとJPEG画像リンクを生成する。
    """

    urls = book_urls or BOOK_URLS
    path = urls.get(book)

    if not path:
        return None, None

    encoded_path = urllib.parse.quote(
        path
    )

    pdf_link = (
        f"{PDFJS_BASE}"
        f"?file={encoded_path}"
        f"#page={page}"
    )

    base_image_url = os.getenv(
        "IMAGE_BASE_URL",
        "https://sw-rule-www.grd-svc.com/image",
    )

    encoded_book = urllib.parse.quote(
        book,
        safe="",
    )

    page_str = f"P{page:05d}"

    image_link = (
        f"{base_image_url}/"
        f"{encoded_book}/"
        f"{page_str}.jpg"
    )

    return pdf_link, image_link


def get_citation_links(
    citation: dict,
) -> tuple[str | None, str | None]:
    """
    構造化citationからリンクを生成する。

    表示:
        citation["page"]
        = 書籍に印刷されたページ番号

    リンク:
        citation["pdf_page"]
        = PDF/JPEG内部ページ番号

    古いAPI responseとの互換性のため、
    pdf_pageがなければpageへfallbackする。
    """

    book = citation.get(
        "book"
    )

    pdf_page = citation.get(
        "pdf_page",
        citation.get(
            "page"
        ),
    )

    if (
        not book
        or pdf_page is None
    ):
        return None, None

    try:
        pdf_page = int(
            pdf_page
        )

    except (
        TypeError,
        ValueError,
    ):
        return None, None

    return get_page_and_image_links(
        book,
        pdf_page,
    )


def get_citation_label(
    citation: dict,
) -> str:
    """
    構造化citationを表示用文字列へ変換する。
    """

    book = citation.get(
        "book",
        "不明",
    )

    page = citation.get(
        "page",
        "?",
    )

    return f"{book} - p.{page}"


# ============================================================
# Legacy answer text link helpers
# ============================================================

def auto_link_answer_text(
    answer: str,
    book_name_map: dict = None,
) -> str:
    """
    回答中の
    `(カテゴリ / 書籍名 - p.数字)`
    パターンをリンクに変換（括弧付き）。
    """

    if book_name_map is None:
        book_name_map = {}

    def replacer(match):
        book = match.group(1).strip()
        page = int(
            match.group(2)
        )

        resolved_book = (
            book_name_map.get(book)
            or book_name_map.get(
                book.split(" / ")[-1]
            )
            or book
        )

        pdf_link, image_link = (
            get_page_link(
                resolved_book,
                page,
            )
        )

        if (
            not pdf_link
            or not image_link
        ):
            return (
                f"[{book} - p.{page}]"
            )

        return (
            f"[{book} - p.{page}]"
            f"({image_link}) "
            f"[📄 PDFで開く]"
            f"({pdf_link})"
        )

    # (書籍名 - p.123)
    # (カテゴリ / 書籍名 - p.123)
    pattern = (
        r"[\(（]"
        r"([^\)）]+?)"
        r"\s*-\s*"
        r"p\.\s*"
        r"(\d+)"
        r"[\)）]"
    )

    return re.sub(
        pattern,
        replacer,
        answer,
    )


def auto_image_link_answer_text(
    answer: str,
    book_name_map: dict = None,
) -> str:
    if book_name_map is None:
        book_name_map = {}

    def replacer(match):
        book = match.group(1).strip()

        page = int(
            match.group(2)
        )

        resolved_book = (
            book_name_map.get(book)
            or book
        )

        pdf_link, image_link = (
            get_page_and_image_links(
                resolved_book,
                page,
            )
        )

        if (
            not pdf_link
            or not image_link
        ):
            return (
                f"[{book} - p.{page}]"
            )

        return (
            f"（"
            f"<a href='{image_link}' "
            f"target='_blank'>"
            f"{book} - p.{page}"
            f"</a> "
            f"[<a href='{pdf_link}' "
            f"target='_blank'>"
            f"PDF"
            f"</a>]"
            f"）"
        )

    pattern = (
        r"[\(（]"
        r"([^\)）]+?)"
        r"\s*-\s*"
        r"p\.\s*"
        r"(\d+)"
        r"[\)）]"
    )

    return re.sub(
        pattern,
        replacer,
        answer,
    )


# ============================================================
# Streamlit demo
# ============================================================

def render_streamlit_demo():
    import streamlit as st

    book = st.selectbox(
        "書籍を選択",
        list(
            BOOK_URLS.keys()
        ),
    )

    page = st.number_input(
        "ページ番号",
        min_value=1,
        step=1,
    )

    if st.button(
        "出典リンクを生成"
    ):
        pdf_link, image_link = (
            get_page_link(
                book,
                page,
            )
        )

        if (
            pdf_link
            and image_link
        ):
            st.markdown(
                f"[画像]({image_link}) / "
                f"[PDF]({pdf_link})"
            )

        else:
            st.warning(
                "該当書籍のリンクを"
                "生成できませんでした。"
            )


if __name__ == "__main__":
    render_streamlit_demo()