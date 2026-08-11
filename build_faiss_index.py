import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

OCR_DIR = Path(
    os.getenv("OCR_DIR", "./ocr")
)

BOOK_CATEGORIES_FILE = Path(
    os.getenv(
        "BOOK_CATEGORY_PATH",
        "./book/book_categories.json"
    )
)

FAISS_INDEX_DIR = Path(
    os.getenv("INDEX_DIR", "./vector_index")
)

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================================
# book_categories.json の読み込み
# ============================================================

def load_book_categories(path: Path) -> dict[str, str]:
    """
    book_categories.json を読み込み、
    書籍名 -> カテゴリ名 の辞書を作成する。
    """

    if not path.exists():
        raise FileNotFoundError(
            f"book_categories.json が見つかりません: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "book_categories.json のルートはdictである必要があります。"
        )

    book_to_category: dict[str, str] = {}

    for category_name, category_info in data.items():

        if not isinstance(category_info, dict):
            print(
                f"WARNING: カテゴリ情報がdictではありません: "
                f"{category_name}"
            )
            continue

        books = category_info.get("books", [])

        if not isinstance(books, list):
            print(
                f"WARNING: books がlistではありません: "
                f"{category_name}"
            )
            continue

        for book_entry in books:

            if not isinstance(book_entry, dict):
                print(
                    f"WARNING: 書籍情報がdictではありません: "
                    f"{category_name}: {book_entry}"
                )
                continue

            book_name = book_entry.get("name")

            if not book_name:
                print(
                    f"WARNING: name がありません: "
                    f"{category_name}: {book_entry}"
                )
                continue

            if book_name in book_to_category:
                print(
                    f"WARNING: 書籍が複数カテゴリにあります: "
                    f"{book_name}"
                )

            book_to_category[book_name] = category_name

    return book_to_category

# ============================================================
# OCR JSON の読み込み
# ============================================================

def load_documents(
    ocr_dir: Path,
    book_to_category: dict[str, str],
) -> list[Document]:
    """
    OCR_DIR 以下のJSONファイルを読み込み、
    LangChain Document に変換する。

    OCR JSON想定形式:

    [
        {
            "book": "ソード・ワールド2.5 ルールブック1",
            "page": 42,
            "text": "..."
        },
        ...
    ]

    metadata:

    {
        "book": "...",
        "page": 42,
        "category": "基本ルールブック",
        "source": "..."
    }
    """

    if not ocr_dir.exists():
        raise FileNotFoundError(
            f"OCRディレクトリが見つかりません: {ocr_dir}"
        )

    documents: list[Document] = []

    json_files = sorted(ocr_dir.rglob("*.json"))

    if not json_files:
        raise RuntimeError(
            f"JSONファイルが見つかりません: {ocr_dir}"
        )

    print(f"OCR JSON files: {len(json_files)}")

    unknown_books: set[str] = set()

    for json_path in json_files:
        print(f"Loading: {json_path}")

        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"ERROR: JSON読み込み失敗: {json_path}")
            print(f"  {e}")
            continue

        if not isinstance(data, list):
            print(
                f"WARNING: JSONルートがlistではありません: "
                f"{json_path}"
            )
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue

            book = entry.get("book")
            page = entry.get("page")
            text = entry.get("text")

            if not book:
                print(
                    f"WARNING: book がありません: "
                    f"{json_path}"
                )
                continue

            if page is None:
                print(
                    f"WARNING: page がありません: "
                    f"{json_path} / {book}"
                )
                continue

            if not text:
                continue

            text = str(text).strip()

            if not text:
                continue

            category = book_to_category.get(book)

            if category is None:
                unknown_books.add(book)
                category = "未分類"

            metadata = {
                "book": book,
                "page": page,
                "category": category,
                "source": str(json_path),
            }

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

    print()
    print(f"Loaded page documents: {len(documents)}")

    if unknown_books:
        print()
        print("WARNING: book_categories.json にない書籍:")
        for book in sorted(unknown_books):
            print(f"  - {book}")

    return documents


# ============================================================
# チャンク分割
# ============================================================

def split_documents(
    documents: list[Document],
) -> list[Document]:
    """
    ページ単位Documentをチャンク分割する。

    metadata は LangChain の splitter によって
    各チャンクへ引き継がれる。
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "。",
            "、",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    # 同一ページ内のチャンクを識別できるようにする
    page_chunk_counter: dict[tuple, int] = {}

    for chunk in chunks:
        key = (
            chunk.metadata.get("book"),
            chunk.metadata.get("page"),
        )

        chunk_index = page_chunk_counter.get(key, 0)

        chunk.metadata["chunk"] = chunk_index

        page_chunk_counter[key] = chunk_index + 1

    print(f"Generated chunks: {len(chunks)}")

    return chunks


# ============================================================
# metadata 確認
# ============================================================

def print_metadata_summary(
    documents: list[Document],
) -> None:
    """
    インデックス作成前にカテゴリ・書籍別の件数を表示する。
    """

    category_counts: dict[str, int] = {}
    book_counts: dict[str, int] = {}

    for doc in documents:
        category = doc.metadata.get(
            "category",
            "未分類",
        )
        book = doc.metadata.get(
            "book",
            "不明",
        )

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

        book_counts[book] = (
            book_counts.get(book, 0) + 1
        )

    print()
    print("=" * 60)
    print("Category summary")
    print("=" * 60)

    for category, count in sorted(
        category_counts.items()
    ):
        print(f"{category}: {count}")

    print()
    print("=" * 60)
    print("Book summary")
    print("=" * 60)

    for book, count in sorted(
        book_counts.items()
    ):
        print(f"{book}: {count}")

    print()


# ============================================================
# FAISSインデックス作成
# ============================================================

def build_faiss_index(
    chunks: list[Document],
    output_dir: Path,
) -> None:
    """
    Document群からEmbeddingを作成し、
    FAISSインデックスとして保存する。
    """

    if not chunks:
        raise RuntimeError(
            "インデックス化するDocumentがありません。"
        )

    print()
    print("=" * 60)
    print("Creating embeddings / FAISS index")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(
        str(output_dir)
    )

    print()
    print(f"FAISS index saved: {output_dir}")


# ============================================================
# インデックス内容の簡易確認
# ============================================================

def verify_index(
    output_dir: Path,
) -> None:
    """
    保存したFAISSインデックスを再読み込みして、
    metadata が保持されていることを簡易確認する。
    """

    print()
    print("=" * 60)
    print("Verifying FAISS index")
    print("=" * 60)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    vectorstore = FAISS.load_local(
        str(output_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    docstore_dict = vectorstore.docstore._dict

    print(
        f"Documents in index: "
        f"{len(docstore_dict)}"
    )

    # 最初の数件を表示
    for i, doc in enumerate(
        docstore_dict.values()
    ):
        print()
        print(f"[sample {i + 1}]")
        print(f"metadata: {doc.metadata}")

        preview = (
            doc.page_content
            .replace("\n", " ")
            [:120]
        )

        print(f"text: {preview}")

        if i >= 4:
            break


# ============================================================
# main
# ============================================================

def main() -> None:
    print("=" * 60)
    print("SW2.5 FAISS Index Builder")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # 1. 書籍 -> カテゴリ対応表
    # --------------------------------------------------------

    print("Loading book categories...")

    book_to_category = load_book_categories(
        BOOK_CATEGORIES_FILE
    )

    print(
        f"Registered books: "
        f"{len(book_to_category)}"
    )

    print()

    # --------------------------------------------------------
    # 2. OCR JSON読み込み
    # --------------------------------------------------------

    print("Loading OCR documents...")

    page_documents = load_documents(
        OCR_DIR,
        book_to_category,
    )

    if not page_documents:
        raise RuntimeError(
            "OCR Documentを1件も読み込めませんでした。"
        )

    # --------------------------------------------------------
    # 3. ページDocumentのmetadata確認
    # --------------------------------------------------------

    print_metadata_summary(
        page_documents
    )

    # --------------------------------------------------------
    # 4. チャンク分割
    # --------------------------------------------------------

    print("Splitting documents...")

    chunks = split_documents(
        page_documents
    )

    if not chunks:
        raise RuntimeError(
            "チャンクを生成できませんでした。"
        )

    # --------------------------------------------------------
    # 5. チャンクmetadata確認
    # --------------------------------------------------------

    print_metadata_summary(
        chunks
    )

    # --------------------------------------------------------
    # 6. FAISS構築
    # --------------------------------------------------------

    build_faiss_index(
        chunks,
        FAISS_INDEX_DIR,
    )

    # --------------------------------------------------------
    # 7. 保存結果確認
    # --------------------------------------------------------

    verify_index(
        FAISS_INDEX_DIR
    )

    print()
    print("=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()