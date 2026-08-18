from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


NORMALIZE_VERSION = 2

SOURCE_TYPE_ERRATA = "ERRATA"
SOURCE_TYPE_FAQ = "FAQ"
SOURCE_TYPE_ADDITIONAL = "ADDITIONAL_DATA"
SOURCE_TYPE_OTHER = "OTHER"

ROW_START_RE = re.compile(
    r"^(?P<star>[★☆]?)\s*(?P<page>[0-9]{1,4})\s+(?P<rest>.+?)\s*$"
)

PDF_PAGE_HEADER_RE = re.compile(
    r"^\s*p\.\s*[0-9]+\s+ページ\s+場所\s+誤\s+正\s*$",
    re.IGNORECASE,
)

PDF_PAGE_MARKER_RE = re.compile(
    r"^\s*p\.\s*[0-9]+\s*$",
    re.IGNORECASE,
)

HEADER_PATTERNS = (
    re.compile(r"^\s*ページ\s+場所\s+誤\s+正\s*$"),
    re.compile(r"^\s*ページ\s+誤\s+正\s*$"),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    tmp.replace(path)


def classify_source(filename: str) -> str:
    lower = filename.lower()

    if "faq" in lower:
        return SOURCE_TYPE_FAQ

    if "additional" in lower:
        return SOURCE_TYPE_ADDITIONAL

    if "_eratta" in lower or "-eratta" in lower:
        return SOURCE_TYPE_ERRATA

    return SOURCE_TYPE_OTHER


def source_key_from_filename(filename: str) -> str:
    stem = Path(filename).stem

    return re.sub(
        r"(?i)(?:_eratta|_faq|_additional(?:_20)?data)$",
        "",
        stem,
    )


def clean_line(line: str) -> str:
    return (
        line.replace("\u3000", " ")
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .rstrip()
    )


def is_header_line(line: str) -> bool:
    if not line:
        return True

    if PDF_PAGE_HEADER_RE.match(line):
        return True

    if PDF_PAGE_MARKER_RE.match(line):
        return True

    return any(
        pattern.match(line)
        for pattern in HEADER_PATTERNS
    )


def iter_page_lines(page_record: dict) -> list[str]:
    text = str(page_record.get("text") or "")

    lines: list[str] = []

    for raw_line in text.splitlines():
        line = clean_line(raw_line)

        if is_header_line(line):
            continue

        lines.append(line)

    return lines


def looks_like_row_start(
    line: str,
) -> tuple[int, bool, str] | None:
    """
    行頭の対象ページ番号を検出する。

    v1の「前レコード以上」という単調増加制約は撤廃。
    GroupSNEのエラッタでは、112頁→113頁→112頁のように
    一時的に戻るケースがあるため。

    誤検出防止は、
      - ASCII数字のみ
      - 1..1500
      - 数値や記号だけの残り文字列を除外
    に留める。
    """

    match = ROW_START_RE.match(line)

    if not match:
        return None

    target_page = int(
        match.group("page")
    )

    if not 1 <= target_page <= 1500:
        return None

    rest = match.group("rest").strip()

    if not rest:
        return None

    if re.fullmatch(
        r"[\d\s＋+\-－−.、,／/（）()]+",
        rest,
    ):
        return None

    starred = bool(
        match.group("star")
    )

    return (
        target_page,
        starred,
        rest,
    )


def make_record(
    *,
    record_index: int,
    target_page: int,
    starred: bool,
    source_pdf_pages: list[int],
    raw_lines: list[str],
) -> dict:
    raw_text = "\n".join(
        line
        for line in raw_lines
        if line
    ).strip()

    return {
        "record_index": record_index,
        "target_page": target_page,
        "starred": starred,
        "source_pdf_pages": sorted(
            set(source_pdf_pages)
        ),
        "location": None,
        "before": None,
        "after": None,
        "raw_text": raw_text,
        "parse_status": "RAW_BLOCK",
    }


def parse_errata_pages(
    pages: list[dict],
) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    unparsed: list[dict] = []

    current_target_page: int | None = None
    current_starred = False
    current_lines: list[str] = []
    current_pdf_pages: list[int] = []

    preamble_lines: list[str] = []
    preamble_pdf_pages: list[int] = []

    def flush_current() -> None:
        nonlocal current_target_page
        nonlocal current_starred
        nonlocal current_lines
        nonlocal current_pdf_pages

        if current_target_page is None:
            return

        records.append(
            make_record(
                record_index=len(records) + 1,
                target_page=current_target_page,
                starred=current_starred,
                source_pdf_pages=current_pdf_pages,
                raw_lines=current_lines,
            )
        )

        current_target_page = None
        current_starred = False
        current_lines = []
        current_pdf_pages = []

    for page in pages:
        pdf_page = int(
            page.get("pdf_page")
            or 0
        )

        for line in iter_page_lines(page):
            start = looks_like_row_start(
                line
            )

            if start is not None:
                (
                    target_page,
                    starred,
                    rest,
                ) = start

                flush_current()

                current_target_page = (
                    target_page
                )
                current_starred = (
                    starred
                )
                current_lines = [
                    rest
                ]
                current_pdf_pages = [
                    pdf_page
                ]
                continue

            if current_target_page is not None:
                current_lines.append(
                    line
                )
                current_pdf_pages.append(
                    pdf_page
                )

            elif line:
                preamble_lines.append(
                    line
                )
                preamble_pdf_pages.append(
                    pdf_page
                )

    flush_current()

    preamble_text = "\n".join(
        preamble_lines
    ).strip()

    if preamble_text:
        unparsed.append(
            {
                "reason": (
                    "PREAMBLE_OR_NON_NUMERIC_TARGET"
                ),
                "source_pdf_pages": sorted(
                    set(
                        preamble_pdf_pages
                    )
                ),
                "raw_text": (
                    preamble_text
                ),
            }
        )

    return (
        records,
        unparsed,
    )


def normalize_document(
    path: Path,
) -> dict:
    extracted = load_json(
        path
    )

    if not isinstance(
        extracted,
        dict,
    ):
        raise ValueError(
            f"JSON root must be object: {path}"
        )

    document = (
        extracted.get(
            "document"
        )
        or {}
    )

    if not isinstance(
        document,
        dict,
    ):
        raise ValueError(
            f"document must be object: {path}"
        )

    pages = extracted.get(
        "pages"
    )

    if not isinstance(
        pages,
        list,
    ):
        raise ValueError(
            f"pages must be array: {path}"
        )

    filename = path.name
    source_type = classify_source(
        filename
    )
    source_key = source_key_from_filename(
        filename
    )

    base = {
        "version": (
            NORMALIZE_VERSION
        ),
        "normalized_at": (
            utc_now_iso()
        ),
        "source_id": (
            extracted.get(
                "source"
            )
            or {}
        ).get(
            "id"
        ),
        "source_name": (
            extracted.get(
                "source"
            )
            or {}
        ).get(
            "name"
        ),
        "source_type": (
            source_type
        ),
        "source_key": (
            source_key
        ),
        "target_book": None,
        "source_url": (
            document.get(
                "url"
            )
        ),
        "source_sha256": (
            document.get(
                "sha256"
            )
        ),
        "source_last_modified": (
            document.get(
                "last_modified"
            )
        ),
        "extracted_file": str(
            path
        ),
        "extract_version": (
            extracted.get(
                "version"
            )
        ),
        "page_count": (
            extracted.get(
                "page_count"
            )
        ),
    }

    if source_type == SOURCE_TYPE_ERRATA:
        records, unparsed = (
            parse_errata_pages(
                pages
            )
        )

        base.update(
            {
                "record_count": len(
                    records
                ),
                "unparsed_block_count": len(
                    unparsed
                ),
                "records": records,
                "unparsed_blocks": unparsed,
            }
        )

        return base

    full_text_parts: list[
        dict
    ] = []

    for page in pages:
        text = str(
            page.get(
                "text"
            )
            or ""
        ).strip()

        if text:
            full_text_parts.append(
                {
                    "pdf_page": (
                        page.get(
                            "pdf_page"
                        )
                    ),
                    "extract_method": (
                        page.get(
                            "extract_method"
                        )
                    ),
                    "text": (
                        text
                    ),
                }
            )

    base.update(
        {
            "record_count": 0,
            "unparsed_block_count": 0,
            "records": [],
            "unparsed_blocks": [],
            "pages": (
                full_text_parts
            ),
            "parse_status": (
                "CLASSIFIED_ONLY"
            ),
        }
    )

    return base


def iter_input_files(
    input_dir: Path,
) -> list[Path]:
    return [
        path
        for path in sorted(
            input_dir.glob(
                "*.json"
            )
        )
        if path.name
        != "extraction_manifest.json"
    ]


def build_report(
    *,
    input_dir: Path,
    output_dir: Path,
    results: list[dict],
    errors: list[dict],
) -> dict:
    type_counts = Counter(
        result.get(
            "source_type",
            SOURCE_TYPE_OTHER,
        )
        for result in results
    )

    total_records = sum(
        int(
            result.get(
                "record_count"
            )
            or 0
        )
        for result in results
    )

    total_unparsed = sum(
        int(
            result.get(
                "unparsed_block_count"
            )
            or 0
        )
        for result in results
    )

    return {
        "version": (
            NORMALIZE_VERSION
        ),
        "generated_at": (
            utc_now_iso()
        ),
        "input_dir": str(
            input_dir
        ),
        "output_dir": str(
            output_dir
        ),
        "document_count": len(
            results
        ),
        "error_count": len(
            errors
        ),
        "source_type_counts": dict(
            sorted(
                type_counts.items()
            )
        ),
        "errata_record_count": (
            total_records
        ),
        "unparsed_block_count": (
            total_unparsed
        ),
        "documents": [
            {
                "input_file": (
                    result.get(
                        "_input_file"
                    )
                ),
                "output_file": (
                    result.get(
                        "_output_file"
                    )
                ),
                "source_type": (
                    result.get(
                        "source_type"
                    )
                ),
                "source_key": (
                    result.get(
                        "source_key"
                    )
                ),
                "record_count": (
                    result.get(
                        "record_count"
                    )
                ),
                "unparsed_block_count": (
                    result.get(
                        "unparsed_block_count"
                    )
                ),
            }
            for result in results
        ],
        "errors": (
            errors
        ),
    }


def parse_args(
    argv: Iterable[
        str
    ] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify extracted official SW2.5 documents and "
            "split ERRATA PDFs into one raw correction block per record."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/data/official/extracted/groupsne_sw25_errata"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/official/normalized/groupsne_sw25_errata"
        ),
    )

    return parser.parse_args(
        list(
            argv
        )
        if argv is not None
        else None
    )


def main(
    argv: Iterable[
        str
    ] | None = None,
) -> int:
    args = parse_args(
        argv
    )

    if not args.input_dir.exists():
        print(
            "Input directory not found: "
            f"{args.input_dir}",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_files = iter_input_files(
        args.input_dir
    )

    print(
        f"Input documents: "
        f"{len(input_files)}"
    )

    results: list[
        dict
    ] = []

    errors: list[
        dict
    ] = []

    for index, path in enumerate(
        input_files,
        start=1,
    ):
        print(
            f"[{index}/"
            f"{len(input_files)}] "
            f"{path.name}"
        )

        try:
            normalized = normalize_document(
                path
            )

            output_path = (
                args.output_dir
                / (
                    f"{path.stem}"
                    ".normalized.json"
                )
            )

            write_json_atomic(
                output_path,
                normalized,
            )

            normalized[
                "_input_file"
            ] = str(
                path
            )

            normalized[
                "_output_file"
            ] = str(
                output_path
            )

            results.append(
                normalized
            )

            print(
                "  "
                f"type="
                f"{normalized['source_type']} "
                f"records="
                f"{normalized.get('record_count', 0)} "
                f"unparsed="
                f"{normalized.get('unparsed_block_count', 0)}"
            )

        except Exception as exc:
            errors.append(
                {
                    "input_file": (
                        str(
                            path
                        )
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

            print(
                "  ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}",
                file=sys.stderr,
            )

    report = build_report(
        input_dir=(
            args.input_dir
        ),
        output_dir=(
            args.output_dir
        ),
        results=(
            results
        ),
        errors=(
            errors
        ),
    )

    report_path = (
        args.output_dir
        / "normalization_manifest.json"
    )

    write_json_atomic(
        report_path,
        report,
    )

    print()
    print(
        "Normalization manifest: "
        f"{report_path}"
    )
    print(
        f"Documents: "
        f"{report['document_count']}"
    )
    print(
        f"Errors: "
        f"{report['error_count']}"
    )
    print(
        f"Source types: "
        f"{report['source_type_counts']}"
    )
    print(
        f"ERRATA records: "
        f"{report['errata_record_count']}"
    )
    print(
        f"Unparsed blocks: "
        f"{report['unparsed_block_count']}"
    )

    return (
        1
        if errors
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
