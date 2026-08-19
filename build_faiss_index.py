import json
import os
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Environment
# ============================================================

load_dotenv()

OCR_DIR = Path(
    os.getenv(
        "OCR_DIR",
        "./ocr",
    )
)

BOOK_CATEGORIES_FILE = Path(
    os.getenv(
        "BOOK_CATEGORY_PATH",
        "./metadata/book_categories.json",
    )
)

PAGE_MAP_DIR = Path(
    os.getenv(
        "PAGE_MAP_DIR",
        "./metadata/page_maps",
    )
)

FAISS_INDEX_DIR = Path(
    os.getenv(
        "INDEX_DIR",
        "./vector_index",
    )
)

OFFICIAL_ERRATA_CHUNKS = Path(
    os.getenv(
        "OFFICIAL_ERRATA_CHUNKS",
        "/data/official/rag/groupsne_sw25_errata/official_errata_chunks.jsonl",
    )
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

CHUNK_SIZE = int(
    os.getenv(
        "INDEX_CHUNK_SIZE",
        "500",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "INDEX_CHUNK_OVERLAP",
        "100",
    )
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "100",
    )
)

EMBEDDING_BATCH_SLEEP = float(
    os.getenv(
        "EMBEDDING_BATCH_SLEEP",
        "4.0",
    )
)

PAGE_MAP_REQUIRED = (
    os.getenv(
        "PAGE_MAP_REQUIRED",
        "1",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

OFFICIAL_ERRATA_REQUIRED = (
    os.getenv(
        "OFFICIAL_ERRATA_REQUIRED",
        "1",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)


# ============================================================
# JSON helpers
# ============================================================

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _safe_metadata_value(value):
    if value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, list):
        return [
            _safe_metadata_value(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _safe_metadata_value(item)
            for key, item in value.items()
        }

    return str(value)


# ============================================================
# Book categories
# ============================================================

def load_book_categories(
    path: Path,
) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            "book_categories.json が"
            f"見つかりません: {path}"
        )

    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError(
            "book_categories.json の"
            "ルートはdictである必要があります。"
        )

    book_to_category: dict[str, str] = {}

    for category_name, category_info in data.items():
        if not isinstance(category_info, dict):
            print(
                "WARNING: "
                "カテゴリ情報がdictではありません: "
                f"{category_name}"
            )
            continue

        books = category_info.get(
            "books",
            [],
        )

        if not isinstance(books, list):
            print(
                "WARNING: books がlistではありません: "
                f"{category_name}"
            )
            continue

        for book_entry in books:
            if not isinstance(book_entry, dict):
                print(
                    "WARNING: 書籍情報がdictではありません: "
                    f"{category_name}: {book_entry}"
                )
                continue

            book_name = book_entry.get("name")

            if not book_name:
                print(
                    "WARNING: name がありません: "
                    f"{category_name}: {book_entry}"
                )
                continue

            if book_name in book_to_category:
                print(
                    "WARNING: 書籍が複数カテゴリにあります: "
                    f"{book_name}"
                )

            book_to_category[book_name] = category_name

    return book_to_category


# ============================================================
# Page maps
# ============================================================

def load_page_maps(
    page_map_dir: Path,
) -> dict[str, dict[int, int | None]]:
    if not page_map_dir.exists():
        if PAGE_MAP_REQUIRED:
            raise FileNotFoundError(
                "page map directory が"
                f"見つかりません: {page_map_dir}"
            )

        print(
            "WARNING: page map directory not found: "
            f"{page_map_dir}"
        )
        return {}

    json_files = sorted(
        page_map_dir.glob("*.json")
    )

    if not json_files:
        if PAGE_MAP_REQUIRED:
            raise RuntimeError(
                "page map JSON が"
                f"見つかりません: {page_map_dir}"
            )
        return {}

    print(
        "Page map files: "
        f"{len(json_files)}"
    )

    result: dict[
        str,
        dict[int, int | None],
    ] = {}

    for path in json_files:
        try:
            data = load_json(path)
        except Exception as exc:
            raise RuntimeError(
                "page map JSONの読み込みに失敗: "
                f"{path}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "page map root must be dict: "
                f"{path}"
            )

        book = data.get("book")

        if not book:
            raise ValueError(
                "page map にbookがありません: "
                f"{path}"
            )

        if book in result:
            raise ValueError(
                "duplicate page map book: "
                f"{book}"
            )

        status = data.get("status")

        if status not in {
            "AUTO_OK",
            "MANUAL_OVERRIDE",
        }:
            print(
                "WARNING: "
                f"page map status={status}: {book}"
            )

        mapping: dict[int, int | None] = {}

        for item in data.get(
            "mappings",
            [],
        ):
            if not isinstance(item, dict):
                continue

            pdf_page = item.get("pdf_page")
            logical_page = item.get("logical_page")

            if pdf_page is None:
                continue

            try:
                pdf_page = int(pdf_page)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if logical_page is not None:
                try:
                    logical_page = int(logical_page)
                except (
                    TypeError,
                    ValueError,
                ):
                    logical_page = None

            mapping[pdf_page] = logical_page

        if not mapping:
            raise ValueError(
                "page mapにmappingsがありません: "
                f"{path}"
            )

        result[book] = mapping

    return result


def resolve_logical_page(
    *,
    book: str,
    pdf_page: int,
    page_maps: dict[str, dict[int, int | None]],
) -> int | None:
    book_map = page_maps.get(book)

    if book_map is None:
        return None

    return book_map.get(pdf_page)


# ============================================================
# OCR documents
# ============================================================

def load_documents(
    ocr_dir: Path,
    book_to_category: dict[str, str],
    page_maps: dict[str, dict[int, int | None]],
) -> list[Document]:
    if not ocr_dir.exists():
        raise FileNotFoundError(
            "OCRディレクトリが"
            f"見つかりません: {ocr_dir}"
        )

    json_files = sorted(
        ocr_dir.rglob("*.json")
    )

    if not json_files:
        raise RuntimeError(
            "OCR JSONが"
            f"見つかりません: {ocr_dir}"
        )

    print(
        "OCR JSON files: "
        f"{len(json_files)}"
    )

    documents: list[Document] = []
    unknown_books: set[str] = set()
    missing_page_map_books: set[str] = set()
    missing_page_mappings: list[
        tuple[str, int]
    ] = []

    logical_page_count = 0
    logical_page_none_count = 0

    for json_path in json_files:
        print(
            f"Loading: {json_path}"
        )

        try:
            data = load_json(json_path)
        except Exception as exc:
            print(
                "ERROR: JSON読み込み失敗: "
                f"{json_path}"
            )
            print(
                f"  {exc}"
            )
            continue

        if not isinstance(data, list):
            print(
                "WARNING: JSONルートが"
                "listではありません: "
                f"{json_path}"
            )
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue

            book = entry.get("book")
            raw_page = entry.get("page")
            text = entry.get("text")

            if not book:
                print(
                    "WARNING: book がありません: "
                    f"{json_path}"
                )
                continue

            if raw_page is None:
                print(
                    "WARNING: page がありません: "
                    f"{json_path} / {book}"
                )
                continue

            try:
                pdf_page = int(raw_page)
            except (
                TypeError,
                ValueError,
            ):
                print(
                    "WARNING: page が整数ではありません: "
                    f"{book}: {raw_page}"
                )
                continue

            text = str(
                text or ""
            ).strip()

            if not text:
                continue

            category = book_to_category.get(book)

            if category is None:
                unknown_books.add(book)
                category = "未分類"

            if book not in page_maps:
                missing_page_map_books.add(book)
                logical_page = None
            else:
                logical_page = resolve_logical_page(
                    book=book,
                    pdf_page=pdf_page,
                    page_maps=page_maps,
                )

                if pdf_page not in page_maps[book]:
                    missing_page_mappings.append(
                        (
                            book,
                            pdf_page,
                        )
                    )

            if logical_page is None:
                logical_page_none_count += 1
            else:
                logical_page_count += 1

            metadata = {
                "book": book,
                # legacy compatibility
                "page": pdf_page,
                "pdf_page": pdf_page,
                "logical_page": logical_page,
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
    print(
        "Loaded page documents: "
        f"{len(documents)}"
    )
    print(
        "Pages with logical_page: "
        f"{logical_page_count}"
    )
    print(
        "Pages without logical_page: "
        f"{logical_page_none_count}"
    )

    if unknown_books:
        print()
        print(
            "WARNING: "
            "book_categories.jsonにない書籍:"
        )
        for book in sorted(unknown_books):
            print(
                f"  - {book}"
            )

    if missing_page_map_books:
        print()
        print(
            "ERROR: page mapがない書籍:"
        )
        for book in sorted(
            missing_page_map_books
        ):
            print(
                f"  - {book}"
            )

        if PAGE_MAP_REQUIRED:
            raise RuntimeError(
                "page mapが存在しない書籍があります。"
            )

    if missing_page_mappings:
        print()
        print(
            "ERROR: page map内に"
            "PDFページの対応がない箇所があります:"
        )

        for book, pdf_page in (
            missing_page_mappings[:50]
        ):
            print(
                f"  - {book}: PDF {pdf_page}"
            )

        if len(missing_page_mappings) > 50:
            print(
                "  ... "
                f"{len(missing_page_mappings) - 50}"
                " more"
            )

        if PAGE_MAP_REQUIRED:
            raise RuntimeError(
                "page mapの対応漏れがあります。"
            )

    return documents


# ============================================================
# Book chunk splitting
# ============================================================

def split_documents(
    documents: list[Document],
) -> list[Document]:
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

    chunks = splitter.split_documents(
        documents
    )

    page_chunk_counter: dict[
        tuple,
        int,
    ] = {}

    for chunk in chunks:
        key = (
            chunk.metadata.get("book"),
            chunk.metadata.get("pdf_page"),
        )

        chunk_index = page_chunk_counter.get(
            key,
            0,
        )

        chunk.metadata["chunk"] = chunk_index

        page_chunk_counter[key] = (
            chunk_index + 1
        )

    print(
        "Generated chunks: "
        f"{len(chunks)}"
    )

    return chunks


def mark_book_chunks(
    chunks: list[Document],
) -> None:
    for chunk in chunks:
        chunk.metadata.setdefault(
            "source_class",
            "book",
        )
        chunk.metadata.setdefault(
            "source_type",
            "BOOK",
        )


# ============================================================
# Official errata documents
# ============================================================

def load_official_errata_documents(
    path: Path,
) -> list[Document]:
    """
    build_official_errata_chunks.py が生成したJSONLを
    FAISS投入用Documentへ変換する。

    公式訂正は既に1レコード=1検索チャンクなので再分割しない。
    """
    if not path.exists():
        if OFFICIAL_ERRATA_REQUIRED:
            raise FileNotFoundError(
                "official errata chunks JSONL が見つかりません: "
                f"{path}"
            )

        print(
            "WARNING: official errata chunks JSONL not found: "
            f"{path}"
        )
        return []

    documents: list[Document] = []
    invalid_count = 0
    skipped_count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_no, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "official errata JSONL parse failed: "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                invalid_count += 1
                continue

            text = str(
                row.get("text") or ""
            ).strip()
            metadata = row.get("metadata")

            if not text or not isinstance(
                metadata,
                dict,
            ):
                invalid_count += 1
                continue

            if metadata.get(
                "rag_eligible"
            ) is not True:
                skipped_count += 1
                continue

            if metadata.get(
                "source_class"
            ) != "official_correction":
                invalid_count += 1
                continue

            operation = metadata.get("operation")

            if operation not in {
                "replace",
                "append",
                "delete",
            }:
                invalid_count += 1
                continue

            target_page = metadata.get(
                "target_page"
            )

            if target_page is not None:
                try:
                    target_page = int(
                        target_page
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    target_page = None

            normalized_metadata = {
                str(key): _safe_metadata_value(
                    value
                )
                for key, value in metadata.items()
            }

            normalized_metadata.update(
                {
                    "source_class": "official_correction",
                    "book": (
                        metadata.get("source_key")
                        or "official_correction"
                    ),
                    "page": target_page,
                    "pdf_page": None,
                    "logical_page": target_page,
                    "category": "公式エラッタ",
                    "source": (
                        metadata.get("source_url")
                        or str(path)
                    ),
                    "chunk": 0,
                    "official_chunk_id": row.get("id"),
                }
            )

            documents.append(
                Document(
                    page_content=text,
                    metadata=normalized_metadata,
                )
            )

    print()
    print(
        "Official errata documents"
    )
    print(
        "=" * 60
    )
    print(
        f"JSONL: {path}"
    )
    print(
        f"Loaded: {len(documents)}"
    )
    print(
        f"Skipped: {skipped_count}"
    )
    print(
        f"Invalid: {invalid_count}"
    )

    operation_counts = Counter(
        doc.metadata.get(
            "operation",
            "unknown",
        )
        for doc in documents
    )

    print(
        "Operations: "
        f"{dict(sorted(operation_counts.items()))}"
    )

    if invalid_count:
        raise RuntimeError(
            "official errata JSONL に無効なレコードがあります: "
            f"{invalid_count}"
        )

    return documents


# ============================================================
# Summaries
# ============================================================

def print_metadata_summary(
    documents: list[Document],
) -> None:
    category_counts: dict[
        str,
        int,
    ] = {}
    book_counts: dict[
        str,
        int,
    ] = {}

    logical_count = 0
    no_logical_count = 0

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
            category_counts.get(
                category,
                0,
            )
            + 1
        )
        book_counts[book] = (
            book_counts.get(
                book,
                0,
            )
            + 1
        )

        if doc.metadata.get(
            "logical_page"
        ) is None:
            no_logical_count += 1
        else:
            logical_count += 1

    print()
    print(
        "=" * 60
    )
    print(
        "Page metadata summary"
    )
    print(
        "=" * 60
    )
    print(
        "logical_page available: "
        f"{logical_count}"
    )
    print(
        "logical_page unavailable: "
        f"{no_logical_count}"
    )

    print()
    print(
        "=" * 60
    )
    print(
        "Category summary"
    )
    print(
        "=" * 60
    )
    for category, count in sorted(
        category_counts.items()
    ):
        print(
            f"{category}: {count}"
        )

    print()
    print(
        "=" * 60
    )
    print(
        "Book summary"
    )
    print(
        "=" * 60
    )
    for book, count in sorted(
        book_counts.items()
    ):
        print(
            f"{book}: {count}"
        )
    print()


def print_source_class_summary(
    documents: list[Document],
) -> None:
    counts = Counter(
        str(
            doc.metadata.get(
                "source_class"
            )
            or "unknown"
        )
        for doc in documents
    )

    print()
    print(
        "Source class summary"
    )
    print(
        "=" * 60
    )

    for source_class, count in sorted(
        counts.items()
    ):
        print(
            f"{source_class}: {count}"
        )

    print()


# ============================================================
# FAISS build
# ============================================================

def build_faiss_index(
    chunks: list[Document],
    output_dir: Path,
) -> None:
    if not chunks:
        raise RuntimeError(
            "インデックス化する"
            "Documentがありません。"
        )

    print()
    print(
        "=" * 60
    )
    print(
        "Creating embeddings / FAISS index"
    )
    print(
        "=" * 60
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        chunk_size=EMBEDDING_BATCH_SIZE,
        max_retries=20,
    )

    total = len(chunks)
    vectorstore = None

    total_batches = (
        total
        + EMBEDDING_BATCH_SIZE
        - 1
    ) // EMBEDDING_BATCH_SIZE

    for start in range(
        0,
        total,
        EMBEDDING_BATCH_SIZE,
    ):
        end = min(
            start + EMBEDDING_BATCH_SIZE,
            total,
        )

        batch = chunks[start:end]

        batch_no = (
            start // EMBEDDING_BATCH_SIZE
        ) + 1

        print(
            "Embedding batch "
            f"{batch_no}/{total_batches} "
            f"({start + 1}-{end}/{total})"
        )

        batch_store = FAISS.from_documents(
            documents=batch,
            embedding=embeddings,
        )

        if vectorstore is None:
            vectorstore = batch_store
        else:
            vectorstore.merge_from(
                batch_store
            )

        if end < total:
            time.sleep(
                EMBEDDING_BATCH_SLEEP
            )

    if vectorstore is None:
        raise RuntimeError(
            "FAISS indexを生成できませんでした。"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(
        str(output_dir)
    )

    print()
    print(
        "FAISS index saved: "
        f"{output_dir}"
    )


# ============================================================
# Verification
# ============================================================

def verify_index(
    output_dir: Path,
) -> None:
    print()
    print(
        "=" * 60
    )
    print(
        "Verifying FAISS index"
    )
    print(
        "=" * 60
    )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        chunk_size=100,
    )

    vectorstore = FAISS.load_local(
        str(output_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    docstore_dict = (
        vectorstore.docstore._dict
    )

    print(
        "Documents in index: "
        f"{len(docstore_dict)}"
    )

    logical_count = 0
    missing_logical_count = 0
    source_class_counts = Counter()

    for doc in docstore_dict.values():
        if doc.metadata.get(
            "logical_page"
        ) is None:
            missing_logical_count += 1
        else:
            logical_count += 1

        source_class_counts[
            str(
                doc.metadata.get(
                    "source_class"
                )
                or "unknown"
            )
        ] += 1

    print(
        "logical_page available: "
        f"{logical_count}"
    )
    print(
        "logical_page unavailable: "
        f"{missing_logical_count}"
    )

    print(
        "Source classes in saved index: "
        f"{dict(sorted(source_class_counts.items()))}"
    )

    for i, doc in enumerate(
        docstore_dict.values()
    ):
        print()
        print(
            f"[sample {i + 1}]"
        )
        print(
            "metadata: "
            f"{doc.metadata}"
        )

        preview = (
            doc.page_content
            .replace(
                "\n",
                " ",
            )
            [:120]
        )

        print(
            f"text: {preview}"
        )

        if i >= 4:
            break


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 60
    )
    print(
        "SW2.5 FAISS Index Builder + Official Errata"
    )
    print(
        "=" * 60
    )
    print()

    print(
        "Loading book categories..."
    )
    book_to_category = load_book_categories(
        BOOK_CATEGORIES_FILE
    )
    print(
        "Registered books: "
        f"{len(book_to_category)}"
    )
    print()

    print(
        "Loading page maps..."
    )
    page_maps = load_page_maps(
        PAGE_MAP_DIR
    )
    print(
        "Registered page maps: "
        f"{len(page_maps)}"
    )
    print()

    print(
        "Loading OCR documents..."
    )
    page_documents = load_documents(
        ocr_dir=OCR_DIR,
        book_to_category=book_to_category,
        page_maps=page_maps,
    )

    if not page_documents:
        raise RuntimeError(
            "OCR Documentを1件も読み込めませんでした。"
        )

    print_metadata_summary(
        page_documents
    )

    print(
        "Splitting book documents..."
    )
    book_chunks = split_documents(
        page_documents
    )

    if not book_chunks:
        raise RuntimeError(
            "書籍チャンクを生成できませんでした。"
        )

    mark_book_chunks(
        book_chunks
    )

    official_documents = (
        load_official_errata_documents(
            OFFICIAL_ERRATA_CHUNKS
        )
    )

    if (
        OFFICIAL_ERRATA_REQUIRED
        and not official_documents
    ):
        raise RuntimeError(
            "公式エラッタDocumentを"
            "1件も読み込めませんでした。"
        )

    all_chunks = (
        book_chunks
        + official_documents
    )

    print()
    print(
        "Book chunks: "
        f"{len(book_chunks)}"
    )
    print(
        "Official errata chunks: "
        f"{len(official_documents)}"
    )
    print(
        "Total index documents: "
        f"{len(all_chunks)}"
    )

    print_source_class_summary(
        all_chunks
    )

    # Book-only summary remains useful because official documents
    # intentionally do not have a real PDF page.
    print_metadata_summary(
        book_chunks
    )

    build_faiss_index(
        all_chunks,
        FAISS_INDEX_DIR,
    )

    verify_index(
        FAISS_INDEX_DIR
    )

    print()
    print(
        "=" * 60
    )
    print(
        "Done"
    )
    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
