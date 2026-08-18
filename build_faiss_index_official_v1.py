from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

import build_faiss_index as base


OFFICIAL_ERRATA_CHUNKS = Path(
    os.getenv(
        "OFFICIAL_ERRATA_CHUNKS",
        "/data/official/rag/groupsne_sw25_errata/official_errata_chunks.jsonl",
    )
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


def _safe_metadata_value(value):
    """Keep JSON-compatible metadata values only."""
    if value is None or isinstance(value, (str, int, float, bool)):
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


def load_official_errata_documents(
    path: Path,
) -> list[Document]:
    """
    build_official_errata_chunks.py が生成した JSONL を、
    FAISS投入用Documentへ変換する。

    公式エラッタは既に1レコード=1検索チャンクへ整形済みなので、
    RecursiveCharacterTextSplitterでは再分割しない。
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
        for line_no, raw_line in enumerate(handle, start=1):
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

            text = str(row.get("text") or "").strip()
            metadata = row.get("metadata")

            if not text or not isinstance(metadata, dict):
                invalid_count += 1
                continue

            if metadata.get("rag_eligible") is not True:
                skipped_count += 1
                continue

            if metadata.get("source_class") != "official_correction":
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

            target_page = metadata.get("target_page")
            if target_page is not None:
                try:
                    target_page = int(target_page)
                except (TypeError, ValueError):
                    target_page = None

            normalized_metadata = {
                str(key): _safe_metadata_value(value)
                for key, value in metadata.items()
            }

            # qa_api.pyとの段階的互換用。
            # official_correctionは書籍PDFそのものではないので、
            # pdf_pageは設定しない。logical_pageは訂正対象ページを保持する。
            normalized_metadata.update(
                {
                    "source_class": "official_correction",
                    "book": metadata.get("source_key") or "official_correction",
                    "page": target_page,
                    "pdf_page": None,
                    "logical_page": target_page,
                    "category": "公式エラッタ",
                    "source": metadata.get("source_url") or str(path),
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
    print("Official errata documents")
    print("=" * 60)
    print(f"JSONL: {path}")
    print(f"Loaded: {len(documents)}")
    print(f"Skipped: {skipped_count}")
    print(f"Invalid: {invalid_count}")

    operation_counts = Counter(
        doc.metadata.get("operation", "unknown")
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


def mark_book_chunks(
    chunks: list[Document],
) -> None:
    """Existing OCR chunks are explicitly marked as book documents."""
    for chunk in chunks:
        chunk.metadata.setdefault(
            "source_class",
            "book",
        )
        chunk.metadata.setdefault(
            "source_type",
            "BOOK",
        )


def print_source_class_summary(
    documents: list[Document],
) -> None:
    counts = Counter(
        str(doc.metadata.get("source_class") or "unknown")
        for doc in documents
    )

    print()
    print("Source class summary")
    print("=" * 60)
    for source_class, count in sorted(counts.items()):
        print(f"{source_class}: {count}")
    print()


def main() -> None:
    print("=" * 60)
    print("SW2.5 FAISS Index Builder + Official Errata")
    print("=" * 60)
    print()

    print("Loading book categories...")
    book_to_category = base.load_book_categories(
        base.BOOK_CATEGORIES_FILE
    )
    print(f"Registered books: {len(book_to_category)}")
    print()

    print("Loading page maps...")
    page_maps = base.load_page_maps(
        base.PAGE_MAP_DIR
    )
    print(f"Registered page maps: {len(page_maps)}")
    print()

    print("Loading OCR documents...")
    page_documents = base.load_documents(
        ocr_dir=base.OCR_DIR,
        book_to_category=book_to_category,
        page_maps=page_maps,
    )

    if not page_documents:
        raise RuntimeError(
            "OCR Documentを1件も読み込めませんでした。"
        )

    base.print_metadata_summary(
        page_documents
    )

    print("Splitting book documents...")
    book_chunks = base.split_documents(
        page_documents
    )
    if not book_chunks:
        raise RuntimeError(
            "書籍チャンクを生成できませんでした。"
        )

    mark_book_chunks(
        book_chunks
    )

    official_documents = load_official_errata_documents(
        OFFICIAL_ERRATA_CHUNKS
    )

    if OFFICIAL_ERRATA_REQUIRED and not official_documents:
        raise RuntimeError(
            "公式エラッタDocumentを1件も読み込めませんでした。"
        )

    all_chunks = (
        book_chunks
        + official_documents
    )

    print()
    print(f"Book chunks: {len(book_chunks)}")
    print(f"Official errata chunks: {len(official_documents)}")
    print(f"Total index documents: {len(all_chunks)}")

    print_source_class_summary(
        all_chunks
    )

    # Existing metadata summary remains useful for book chunks.
    base.print_metadata_summary(
        book_chunks
    )

    base.build_faiss_index(
        all_chunks,
        base.FAISS_INDEX_DIR,
    )

    base.verify_index(
        base.FAISS_INDEX_DIR
    )

    print()
    print("=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
