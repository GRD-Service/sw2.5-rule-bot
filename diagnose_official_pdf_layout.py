from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPORT_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    tmp.replace(path)


def run_pdftotext_bbox(
    *,
    pdf_path: Path,
    output_html: Path,
) -> None:
    command = [
        "pdftotext",
        "-bbox-layout",
        "-enc",
        "UTF-8",
        str(pdf_path),
        str(output_html),
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "pdftotext failed: "
            f"returncode={completed.returncode}, "
            f"stderr={completed.stderr.strip()}"
        )


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def parse_float(
    element: ET.Element,
    name: str,
) -> float | None:
    value = element.attrib.get(name)

    if value is None:
        return None

    try:
        return float(value)

    except ValueError:
        return None


def parse_bbox_html(path: Path) -> list[dict]:
    tree = ET.parse(path)
    root = tree.getroot()

    pages: list[dict] = []

    for page_index, page_el in enumerate(
        (
            element
            for element in root.iter()
            if strip_namespace(element.tag) == "page"
        ),
        start=1,
    ):
        page_width = parse_float(page_el, "width")
        page_height = parse_float(page_el, "height")

        lines: list[dict] = []

        for line_index, line_el in enumerate(
            (
                element
                for element in page_el.iter()
                if strip_namespace(element.tag) == "line"
            ),
            start=1,
        ):
            words: list[dict] = []

            for word_el in line_el:
                if strip_namespace(word_el.tag) != "word":
                    continue

                text = "".join(word_el.itertext()).strip()

                if not text:
                    continue

                word = {
                    "text": text,
                    "x_min": parse_float(word_el, "xMin"),
                    "y_min": parse_float(word_el, "yMin"),
                    "x_max": parse_float(word_el, "xMax"),
                    "y_max": parse_float(word_el, "yMax"),
                }

                words.append(word)

            if not words:
                continue

            x_values_min = [
                word["x_min"]
                for word in words
                if word["x_min"] is not None
            ]
            x_values_max = [
                word["x_max"]
                for word in words
                if word["x_max"] is not None
            ]
            y_values_min = [
                word["y_min"]
                for word in words
                if word["y_min"] is not None
            ]
            y_values_max = [
                word["y_max"]
                for word in words
                if word["y_max"] is not None
            ]

            line_text = " ".join(
                word["text"]
                for word in words
            )

            lines.append(
                {
                    "line_index": line_index,
                    "text": line_text,
                    "x_min": (
                        min(x_values_min)
                        if x_values_min
                        else None
                    ),
                    "y_min": (
                        min(y_values_min)
                        if y_values_min
                        else None
                    ),
                    "x_max": (
                        max(x_values_max)
                        if x_values_max
                        else None
                    ),
                    "y_max": (
                        max(y_values_max)
                        if y_values_max
                        else None
                    ),
                    "words": words,
                }
            )

        pages.append(
            {
                "pdf_page": page_index,
                "width": page_width,
                "height": page_height,
                "line_count": len(lines),
                "lines": lines,
            }
        )

    return pages


def find_header_candidates(
    pages: list[dict],
) -> list[dict]:
    """
    「ページ / 場所 / 誤 / 正」に相当しそうな語のX座標を拾う。
    完全一致ではなく、診断用途として各ページの上部から候補を出す。
    """

    candidates: list[dict] = []

    header_terms = {
        "ページ",
        "場所",
        "誤",
        "正",
    }

    for page in pages:
        for line in page.get("lines", []):
            words = line.get("words", [])

            matched = [
                word
                for word in words
                if word.get("text") in header_terms
            ]

            if len(matched) < 2:
                continue

            candidates.append(
                {
                    "pdf_page": page.get("pdf_page"),
                    "line_index": line.get("line_index"),
                    "line_text": line.get("text"),
                    "matched_words": matched,
                }
            )

    return candidates


def parse_args(
    argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose PDF text layout using Poppler pdftotext -bbox-layout."
        )
    )

    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            "/data/official/raw/groupsne_sw25_errata/"
            "products/sw/eratta/pdf/SW2.5_1_eratta.pdf"
        ),
        help="Input PDF path",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/official/diagnostics/"
            "SW2.5_1_eratta.layout.json"
        ),
        help="Output JSON path",
    )

    return parser.parse_args(
        list(argv)
        if argv is not None
        else None
    )


def main(
    argv: Iterable[str] | None = None,
) -> int:
    args = parse_args(argv)

    if not args.pdf.exists():
        print(
            f"PDF not found: {args.pdf}",
            file=sys.stderr,
        )
        return 2

    try:
        with tempfile.TemporaryDirectory(
            prefix="sw25_bbox_"
        ) as temp_dir:
            html_path = (
                Path(temp_dir)
                / "bbox.html"
            )

            run_pdftotext_bbox(
                pdf_path=args.pdf,
                output_html=html_path,
            )

            pages = parse_bbox_html(
                html_path
            )

        report = {
            "version": REPORT_VERSION,
            "generated_at": utc_now_iso(),
            "pdf_path": str(args.pdf),
            "page_count": len(pages),
            "header_candidates": find_header_candidates(
                pages
            ),
            "pages": pages,
        }

        write_json_atomic(
            args.output,
            report,
        )

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        f"PDF: {args.pdf}"
    )
    print(
        f"Pages: {report['page_count']}"
    )
    print(
        f"Header candidates: "
        f"{len(report['header_candidates'])}"
    )

    for candidate in report["header_candidates"]:
        print(
            "  "
            f"page={candidate['pdf_page']} "
            f"line={candidate['line_index']} "
            f"text={candidate['line_text']}"
        )

        for word in candidate["matched_words"]:
            print(
                "    "
                f"{word['text']}: "
                f"x={word['x_min']:.2f}-"
                f"{word['x_max']:.2f}, "
                f"y={word['y_min']:.2f}-"
                f"{word['y_max']:.2f}"
            )

    print(
        f"Output: {args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
