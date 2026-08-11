import os
import json
from glob import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

OCR_DIR = os.getenv("OCR_DIR", "./ocr")
INDEX_DIR = os.getenv("INDEX_DIR", "./vector_index")
CATEGORY_PATH = os.getenv("BOOK_CATEGORY_PATH", "./book/book_categories.json")

os.makedirs(INDEX_DIR, exist_ok=True)

# カテゴリデータの読み込み
try:
    with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
        book_categories = json.load(f)
except Exception as e:
    print(f"⚠️ カテゴリファイルの読み込みに失敗しました: {e}")
    book_categories = {}

# 書籍名 → カテゴリのマッピングを構築
book_to_category = {
    book: category for category, books in book_categories.items() for book in books
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

embedding = OpenAIEmbeddings()
docs = []

for path in glob(os.path.join(OCR_DIR, "*.json")):
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"⚠️ 無視されました（リスト形式でない）: {path}")
                continue
        except Exception as e:
            print(f"⚠️ JSON読み込みエラー: {path} - {e}")
            continue

    for item in data:
        if not isinstance(item, dict):
            print(f"⚠️ 無視されました（辞書形式でない要素）: {item}")
            continue

        book = item.get("book")
        page = item.get("page")
        category = book_to_category.get(book, "不明")

        meta = {
            "book": book,
            "page": page,
            "category": category
        }

        chunks = text_splitter.split_text(item["text"])
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=meta))

print(f"📄 チャンク総数: {len(docs)}")

db = FAISS.from_documents(docs, embedding)
db.save_local(INDEX_DIR)

print(f"✅ ベクトルインデックスを保存しました：{INDEX_DIR}")

