from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PyPDF2 import PdfReader


MANIFEST_VERSION = 2
EXTRACT_VERSION = 2

DEFAULT_OCR_DPI = 300
DEFAULT_OCR_LANGUAGE = "jpn+eng"
DEFAULT_OCR_PSM = 6
DEFAULT_PDF_TEXT_MIN_CHARS = 1


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

    with temp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    os.replace(temp_path, path)


def normalize_text(text: str | None) -> str:
    return (
        (text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
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
    """Resolve a downloaded PDF referenced by fetch_official_sources manifest."""

    local_path = document.get("local_path")
    if isinstance(local_path, str) and local_path:
        path = Path(local_path)
        if path.exists():
            return path

    url = document.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(
            "PDF document has neither usable local_path nor url"
        )

    filename = safe_filename(url.rsplit("/", 1)[-1])

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

    matches = list((raw_root / source_id).rglob(filename))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        "PDF file not found for manifest entry: "
        f"url={url}, local_path={local_path}"
    )


def ensure_ocr_commands() -> None:
    missing = [
        command
        for command in ("pdftoppm", "tesseract")
        if shutil.which(command) is None
    ]
    if missing:
        raise RuntimeError(
            "OCR fallback requires commands not found in PATH: "
            + ", ".join(missing)
        )


def ocr_pdf_page(
    *,
    pdf_path: Path,
    page_number: int,
    dpi: int,
    language: str,
    psm: int,
) -> str:
    """Render one PDF page with pdftoppm and OCR it with Tesseract."""

    ensure_ocr_commands()

    with tempfile.TemporaryDirectory(prefix="sw25-official-ocr-") as tmp:
        tmp_dir = Path(tmp)
        output_prefix = tmp_dir / "page"

        render_cmd = [
            "pdftoppm",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            "-r",
            str(dpi),
            "-png",
            str(pdf_path),
            str(output_prefix),
        ]

        render = subprocess.run(
            render_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if render.returncode != 0:
            raise RuntimeError(
                "pdftoppm failed: " + render.stderr.strip()
            )

        image_path = output_prefix.with_suffix(".png")
        if not image_path.exists():
            raise RuntimeError(
                f"pdftoppm did not create expected image: {image_path}"
            )

        ocr_cmd = [
            "tesseract",
            str(image_path),
            "stdout",
            "-l",
            language,
            "--psm",
            str(psm),
        ]

        ocr = subprocess.run(
            ocr_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if ocr.returncode != 0:
            raise RuntimeError(
                "tesseract failed: " + ocr.stderr.strip()
            )

        return normalize_text(ocr.stdout)


def extract_pdf(
    *,
    pdf_path: Path,
    document: dict,
    source_info: dict,
    pdf_text_min_chars: int,
    ocr_dpi: int,
    ocr_language: str,
    ocr_psm: int,
) -> dict:
    reader = PdfReader(str(pdf_path))

    pages: list[dict] = []
    empty_pages: list[int] = []
    pdf_text_page_count = 0
    ocr_page_count = 0
    total_chars = 0

    for page_number, page in enumerate(reader.pages, start=1):
        pdf_text_error = None
        ocr_error = None

        try:
            pdf_text = normalize_text(page.extract_text())
        except Exception as exc:
            pdf_text = ""
            pdf_text_error = f"{type(exc).__name__}: {exc}"

        text = pdf_text
        extract_method = "pdf_text"

        if len(pdf_text) < pdf_text_min_chars:
            try:
                ocr_text = ocr_pdf_page(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    dpi=ocr_dpi,
                    language=ocr_language,
                    psm=ocr_psm,
                )
            except Exception as exc:
                ocr_text = ""
                ocr_error = f"{type(exc).__name__}: {exc}"

            if ocr_text:
                text = ocr_text
                extract_method = "tesseract_ocr"
                ocr_page_count += 1
            elif pdf_text:
                text = pdf_text
                extract_method = "pdf_text"
                pdf_text_page_count += 1
            else:
                text = ""
                extract_method = "empty"
        else:
            pdf_text_page_count += 1

        if not text:
            empty_pages.append(page_number)

        total_chars += len(text)

        page_record = {
            "pdf_page": page_number,
            "extract_method": extract_method,
            "char_count": len(text),
            "text": text,
        }

        if pdf_text_error:
            page_record["pdf_text_error"] = pdf_text_error
        if ocr_error:
            page_record["ocr_error"] = ocr_error

        pages.append(page_record)

    return {
        "version": EXTRACT_VERSION,
        "source": {
            "id": source_info.get("id"),
            "name": source_info.get("name"),
        },
        "document": {
            "url": document.get("url"),
            "requested_url": document.get("requested_url"),
            "content_type": document.get("content_type"),
            "sha256": document.get("sha256"),
            "etag": document.get("etag"),
            "last_modified": document.get("last_modified"),
            "retrieved_at": document.get("retrieved_at"),
            "raw_path": str(pdf_path),
        },
        "extraction": {
            "pdf_text_min_chars": pdf_text_min_chars,
            "ocr_engine": "tesseract",
            "ocr_language": ocr_language,
            "ocr_dpi": ocr_dpi,
            "ocr_psm": ocr_psm,
        },
        "extracted_at": utc_now_iso(),
        "page_count": len(pages),
        "pdf_text_page_count": pdf_text_page_count,
        "ocr_page_count": ocr_page_count,
        "empty_page_count": len(empty_pages),
        "empty_pages": empty_pages,
        "total_chars": total_chars,
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
        / (safe_filename(pdf_path.stem) + ".json")
    )


def can_reuse_existing(
    *,
    output_path: Path,
    source_sha256: str | None,
    force: bool,
) -> dict | None:
    if force or not output_path.exists():
        return None

    try:
        existing = load_json(output_path)
    except Exception:
        return None

    if existing.get("version") != EXTRACT_VERSION:
        return None

    existing_sha = existing.get("document", {}).get("sha256")
    if not existing_sha or existing_sha != source_sha256:
        return None

    return existing


def extract_manifest(
    *,
    manifest_path: Path,
    raw_root: Path,
    output_root: Path,
    force: bool,
    pdf_text_min_chars: int,
    ocr_dpi: int,
    ocr_language: str,
    ocr_psm: int,
) -> dict:
    manifest = load_json(manifest_path)
    source_info = manifest.get("source") or {}
    source_id = source_info.get("id")

    if not isinstance(source_id, str) or not source_id:
        raise ValueError("manifest.source.id is missing")

    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("manifest.documents must be an array")

    pdf_documents = [
        document
        for document in documents
        if isinstance(document, dict)
        and document.get("content_type") == "application/pdf"
    ]

    results: list[dict] = []
    errors: list[dict] = []

    total_pdf_text_pages = 0
    total_ocr_pages = 0
    total_empty_pages = 0

    print(f"Source: {source_id}")
    print(f"PDF documents: {len(pdf_documents)}")
    print(f"PDF text minimum chars: {pdf_text_min_chars}")
    print(f"OCR: tesseract language={ocr_language}, dpi={ocr_dpi}, psm={ocr_psm}")

    for index, document in enumerate(pdf_documents, start=1):
        url = document.get("url")
        print(f"[{index}/{len(pdf_documents)}] {url}")

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

            existing = can_reuse_existing(
                output_path=output_path,
                source_sha256=document.get("sha256"),
                force=force,
            )

            if existing is not None:
                status = "unchanged"
                extracted = existing
                print(
                    "  unchanged: "
                    f"{output_path} "
                    f"(pages={extracted.get('page_count', 0)}, "
                    f"pdf_text={extracted.get('pdf_text_page_count', 0)}, "
                    f"ocr={extracted.get('ocr_page_count', 0)}, "
                    f"empty={extracted.get('empty_page_count', 0)}, "
                    f"chars={extracted.get('total_chars', 0)})"
                )
            else:
                status = "written"
                extracted = extract_pdf(
                    pdf_path=pdf_path,
                    document=document,
                    source_info=source_info,
                    pdf_text_min_chars=pdf_text_min_chars,
                    ocr_dpi=ocr_dpi,
                    ocr_language=ocr_language,
                    ocr_psm=ocr_psm,
                )
                write_json_atomic(output_path, extracted)
                print(
                    "  written: "
                    f"{output_path} "
                    f"(pages={extracted['page_count']}, "
                    f"pdf_text={extracted['pdf_text_page_count']}, "
                    f"ocr={extracted['ocr_page_count']}, "
                    f"empty={extracted['empty_page_count']}, "
                    f"chars={extracted['total_chars']})"
                )

            total_pdf_text_pages += int(extracted.get("pdf_text_page_count", 0))
            total_ocr_pages += int(extracted.get("ocr_page_count", 0))
            total_empty_pages += int(extracted.get("empty_page_count", 0))

            results.append(
                {
                    "url": url,
                    "raw_path": str(pdf_path),
                    "output_path": str(output_path),
                    "sha256": document.get("sha256"),
                    "status": status,
                    "page_count": extracted.get("page_count", 0),
                    "pdf_text_page_count": extracted.get("pdf_text_page_count", 0),
                    "ocr_page_count": extracted.get("ocr_page_count", 0),
                    "empty_page_count": extracted.get("empty_page_count", 0),
                    "total_chars": extracted.get("total_chars", 0),
                }
            )

        except Exception as exc:
            error = {
                "url": url,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(f"  ERROR: {error['error']}", file=sys.stderr)

    report = {
        "version": MANIFEST_VERSION,
        "extract_version": EXTRACT_VERSION,
        "source_id": source_id,
        "generated_at": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "pdf_document_count": len(pdf_documents),
        "success_count": len(results),
        "error_count": len(errors),
        "pdf_text_page_count": total_pdf_text_pages,
        "ocr_page_count": total_ocr_pages,
        "empty_page_count": total_empty_pages,
        "documents": results,
        "errors": errors,
    }

    report_path = output_root / source_id / "extraction_manifest.json"
    write_json_atomic(report_path, report)

    print()
    print(f"Extraction manifest: {report_path}")
    print(f"Success: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"PDF text pages: {total_pdf_text_pages}")
    print(f"OCR pages: {total_ocr_pages}")
    print(f"Empty pages: {total_empty_pages}")

    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from official SW2.5 PDFs. "
            "Pages without usable PDF text automatically fall back to "
            "pdftoppm + Tesseract OCR."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/data/official/raw/groupsne_sw25_errata/manifest.json"
        ),
        help="Path to fetch_official_sources.py manifest.json",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data/official/raw"),
        help="Root directory containing downloaded official raw files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/official/extracted"),
        help="Root directory for extracted JSON files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract PDFs even when source SHA-256 is unchanged",
    )
    parser.add_argument(
        "--pdf-text-min-chars",
        type=int,
        default=int(
            os.getenv(
                "OFFICIAL_PDF_TEXT_MIN_CHARS",
                str(DEFAULT_PDF_TEXT_MIN_CHARS),
            )
        ),
        help=(
            "OCR a page when PyPDF2 extracts fewer than this many "
            "characters. Default: 1 (only empty pages)."
        ),
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=int(
            os.getenv("OFFICIAL_OCR_DPI", str(DEFAULT_OCR_DPI))
        ),
        help="Rasterization DPI for OCR fallback",
    )
    parser.add_argument(
        "--ocr-language",
        default=os.getenv(
            "OFFICIAL_OCR_LANGUAGE",
            DEFAULT_OCR_LANGUAGE,
        ),
        help="Tesseract language expression",
    )
    parser.add_argument(
        "--ocr-psm",
        type=int,
        default=int(
            os.getenv("OFFICIAL_OCR_PSM", str(DEFAULT_OCR_PSM))
        ),
        help="Tesseract page segmentation mode",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    if args.pdf_text_min_chars < 0:
        print("--pdf-text-min-chars must be >= 0", file=sys.stderr)
        return 2
    if args.ocr_dpi <= 0:
        print("--ocr-dpi must be > 0", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        report = extract_manifest(
            manifest_path=args.manifest,
            raw_root=args.raw_root,
            output_root=args.output_dir,
            force=args.force,
            pdf_text_min_chars=args.pdf_text_min_chars,
            ocr_dpi=args.ocr_dpi,
            ocr_language=args.ocr_language,
            ocr_psm=args.ocr_psm,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if report.get("error_count", 0):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
