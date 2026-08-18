from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader


MANIFEST_VERSION = 1
EXTRACT_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return data


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    os.replace(
        temp_path,
        path,
    )


def safe_filename(value: str) -> str:
    result = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in value
    )
    return result or "_"


def resolve_pdf_path(
    *,
    raw_root: Path,
    source_id: str,
    document: dict,
) -> Path:
    """
    manifestのlocal_pathを優先して使う。

    Docker移行前など、別環境で生成されたmanifestを持ち込んだ場合は、
    URL末尾からraw_root配下のファイルも探す。
    """

    local_path = document.get(
        "local_path"
    )

    if isinstance(
        local_path,
        str,
    ) and local_path:
        path = Path(
            local_path
        )

        if path.exists():
            return path

    url = document.get(
        "url"
    )

    if not isinstance(
        url,
        str,
    ) or not url:
        raise ValueError(
            "PDF document has neither usable local_path nor url"
        )

    filename = safe_filename(
        url.rsplit(
            "/",
            1,
        )[-1]
    )

    candidate = (
        raw_root
        / source_id
        / "products"
        / "sw"
        / "eratta"
        / "pdf"
        / filename
    )

    if candidate.exists():
        return candidate

    matches = list(
        (
            raw_root
            / source_id
        ).rglob(
            filename
        )
    )

    if len(
        matches
    ) == 1:
        return matches[
            0
        ]

    raise FileNotFoundError(
        "PDF file not found for manifest entry: "
        f"url={url}, local_path={local_path}"
    )


def extract_pdf(
    *,
    pdf_path: Path,
    document: dict,
    source_info: dict,
) -> dict:
    reader = PdfReader(
        str(
            pdf_path
        )
    )

    pages: list[
        dict
    ] = []

    empty_pages: list[
        int
    ] = []

    total_chars = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        extraction_error = None

        try:
            text = (
                page.extract_text()
                or ""
            )

        except Exception as exc:
            text = ""
            extraction_error = (
                f"{type(exc).__name__}: {exc}"
            )

        text = (
            text.replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
            .strip()
        )

        if not text:
            empty_pages.append(
                page_number
            )

        total_chars += len(
            text
        )

        page_record = {
            "pdf_page": (
                page_number
            ),
            "char_count": len(
                text
            ),
            "text": text,
        }

        if extraction_error:
            page_record[
                "extraction_error"
            ] = extraction_error

        pages.append(
            page_record
        )

    return {
        "version": (
            EXTRACT_VERSION
        ),
        "source": {
            "id": source_info.get(
                "id"
            ),
            "name": source_info.get(
                "name"
            ),
        },
        "document": {
            "url": document.get(
                "url"
            ),
            "requested_url": document.get(
                "requested_url"
            ),
            "content_type": document.get(
                "content_type"
            ),
            "sha256": document.get(
                "sha256"
            ),
            "etag": document.get(
                "etag"
            ),
            "last_modified": document.get(
                "last_modified"
            ),
            "retrieved_at": document.get(
                "retrieved_at"
            ),
            "raw_path": str(
                pdf_path
            ),
        },
        "extracted_at": (
            utc_now_iso()
        ),
        "page_count": len(
            pages
        ),
        "empty_page_count": len(
            empty_pages
        ),
        "empty_pages": (
            empty_pages
        ),
        "total_chars": (
            total_chars
        ),
        "pages": pages,
    }


def build_output_path(
    *,
    output_root: Path,
    source_id: str,
    pdf_path: Path,
) -> Path:
    return (
        output_root
        / source_id
        / (
            safe_filename(
                pdf_path.stem
            )
            + ".json"
        )
    )


def extract_manifest(
    *,
    manifest_path: Path,
    raw_root: Path,
    output_root: Path,
    force: bool,
) -> dict:
    manifest = load_json(
        manifest_path
    )

    source_info = (
        manifest.get(
            "source"
        )
        or {}
    )

    source_id = source_info.get(
        "id"
    )

    if not isinstance(
        source_id,
        str,
    ) or not source_id:
        raise ValueError(
            "manifest.source.id is missing"
        )

    documents = manifest.get(
        "documents"
    )

    if not isinstance(
        documents,
        list,
    ):
        raise ValueError(
            "manifest.documents must be an array"
        )

    pdf_documents = [
        document
        for document in documents
        if (
            isinstance(
                document,
                dict,
            )
            and document.get(
                "content_type"
            )
            == "application/pdf"
        )
    ]

    results: list[
        dict
    ] = []

    errors: list[
        dict
    ] = []

    print(
        f"Source: {source_id}"
    )
    print(
        f"PDF documents: {len(pdf_documents)}"
    )

    for index, document in enumerate(
        pdf_documents,
        start=1,
    ):
        url = document.get(
            "url"
        )

        print(
            f"[{index}/{len(pdf_documents)}] {url}"
        )

        try:
            pdf_path = resolve_pdf_path(
                raw_root=raw_root,
                source_id=source_id,
                document=document,
            )

            output_path = build_output_path(
                output_root=output_root,
                source_id=source_id,
                pdf_path=pdf_path,
            )

            if (
                output_path.exists()
                and not force
            ):
                try:
                    existing = load_json(
                        output_path
                    )

                    existing_sha = (
                        existing
                        .get(
                            "document",
                            {}
                        )
                        .get(
                            "sha256"
                        )
                    )

                    if (
                        existing_sha
                        and existing_sha
                        == document.get(
                            "sha256"
                        )
                    ):
                        print(
                            f"  unchanged: {output_path}"
                        )

                        results.append(
                            {
                                "url": url,
                                "raw_path": str(
                                    pdf_path
                                ),
                                "output_path": str(
                                    output_path
                                ),
                                "sha256": document.get(
                                    "sha256"
                                ),
                                "status": (
                                    "unchanged"
                                ),
                                "page_count": existing.get(
                                    "page_count",
                                    0,
                                ),
                                "empty_page_count": existing.get(
                                    "empty_page_count",
                                    0,
                                ),
                                "total_chars": existing.get(
                                    "total_chars",
                                    0,
                                ),
                            }
                        )

                        continue

                except Exception:
                    pass

            extracted = extract_pdf(
                pdf_path=pdf_path,
                document=document,
                source_info=source_info,
            )

            write_json_atomic(
                output_path,
                extracted,
            )

            print(
                "  written: "
                f"{output_path} "
                f"(pages={extracted['page_count']}, "
                f"empty={extracted['empty_page_count']}, "
                f"chars={extracted['total_chars']})"
            )

            results.append(
                {
                    "url": url,
                    "raw_path": str(
                        pdf_path
                    ),
                    "output_path": str(
                        output_path
                    ),
                    "sha256": document.get(
                        "sha256"
                    ),
                    "status": (
                        "written"
                    ),
                    "page_count": extracted[
                        "page_count"
                    ],
                    "empty_page_count": extracted[
                        "empty_page_count"
                    ],
                    "total_chars": extracted[
                        "total_chars"
                    ],
                }
            )

        except Exception as exc:
            error = {
                "url": url,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

            errors.append(
                error
            )

            print(
                f"  ERROR: {error['error']}",
                file=sys.stderr,
            )

    report = {
        "version": (
            MANIFEST_VERSION
        ),
        "source_id": (
            source_id
        ),
        "generated_at": (
            utc_now_iso()
        ),
        "manifest_path": str(
            manifest_path
        ),
        "pdf_document_count": len(
            pdf_documents
        ),
        "success_count": len(
            results
        ),
        "error_count": len(
            errors
        ),
        "documents": (
            results
        ),
        "errors": (
            errors
        ),
    }

    report_path = (
        output_root
        / source_id
        / "extraction_manifest.json"
    )

    write_json_atomic(
        report_path,
        report,
    )

    print()
    print(
        f"Extraction manifest: {report_path}"
    )
    print(
        f"Success: {len(results)}"
    )
    print(
        f"Errors: {len(errors)}"
    )

    return report


def parse_args(
    argv: Iterable[
        str
    ] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text layers from official SW2.5 PDFs "
            "into page-oriented JSON files."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/data/official/raw/groupsne_sw25_errata/manifest.json"
        ),
        help=(
            "Path to fetch_official_sources.py manifest.json"
        ),
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/data/official/raw"
        ),
        help=(
            "Root directory containing downloaded official raw files"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/official/extracted"
        ),
        help=(
            "Root directory for extracted JSON files"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-extract PDFs even when the source SHA-256 is unchanged"
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

    if not args.manifest.exists():
        print(
            "Manifest not found: "
            f"{args.manifest}",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        report = extract_manifest(
            manifest_path=args.manifest,
            raw_root=args.raw_root,
            output_root=args.output_dir,
            force=args.force,
        )

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    if report.get(
        "error_count",
        0,
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
