from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BUILD_VERSION = 7

SOURCE_TYPE_ERRATA = "ERRATA"
SOURCE_TYPE_FAQ = "FAQ"
SOURCE_TYPE_ADDITIONAL = "ADDITIONAL_DATA"
SOURCE_TYPE_OTHER = "OTHER"

PAGE_TOKEN_RE = re.compile(r"^[★☆]?[0-9]{1,4}$")
FOOTER_RE = re.compile(r"^p\.\s*[0-9]+$", re.IGNORECASE)

# SW2.5エラッタPDFで確認した基準位置。
# ページ幅595.32pt前後を前提にせず、割合へ変換して使う。
PAGE_COL_MAX_RATIO = 125.0 / 595.32
LOCATION_COL_MAX_RATIO = 270.0 / 595.32
BEFORE_COL_MAX_RATIO = 415.0 / 595.32

Y_TOLERANCE = 2.2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temp_path.replace(path)


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


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def parse_float(element: ET.Element, name: str) -> float | None:
    value = element.attrib.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def run_pdftotext_bbox(pdf_path: Path, output_html: Path) -> bool:
    completed = run_command(
        [
            "pdftotext",
            "-bbox-layout",
            "-enc",
            "UTF-8",
            str(pdf_path),
            str(output_html),
        ]
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "pdftotext failed: "
            f"returncode={completed.returncode}, "
            f"stderr={completed.stderr.strip()}"
        )

    return output_html.exists() and output_html.stat().st_size > 0


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
        page_width = parse_float(page_el, "width") or 595.32
        page_height = parse_float(page_el, "height") or 841.92

        items: list[dict] = []

        for line_el in (
            element
            for element in page_el.iter()
            if strip_namespace(element.tag) == "line"
        ):
            words: list[dict] = []

            for word_el in line_el:
                if strip_namespace(word_el.tag) != "word":
                    continue

                text = "".join(word_el.itertext()).strip()
                if not text:
                    continue

                x_min = parse_float(word_el, "xMin")
                y_min = parse_float(word_el, "yMin")
                x_max = parse_float(word_el, "xMax")
                y_max = parse_float(word_el, "yMax")

                if None in (x_min, y_min, x_max, y_max):
                    continue

                words.append(
                    {
                        "text": text,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    }
                )

            if not words:
                continue

            items.append(
                {
                    "text": " ".join(word["text"] for word in words),
                    "x_min": min(word["x_min"] for word in words),
                    "y_min": min(word["y_min"] for word in words),
                    "x_max": max(word["x_max"] for word in words),
                    "y_max": max(word["y_max"] for word in words),
                    "words": words,
                }
            )

        pages.append(
            {
                "pdf_page": page_index,
                "width": page_width,
                "height": page_height,
                "items": items,
                "extract_method": "pdftotext_bbox",
            }
        )

    return pages


def run_tesseract_tsv_on_page(
    *,
    pdf_path: Path,
    page_number: int,
    temp_dir: Path,
) -> dict:
    image_prefix = temp_dir / f"page_{page_number:04d}"
    image_path = image_prefix.with_suffix(".png")

    render = run_command(
        [
            "pdftoppm",
            "-f",
            str(page_number),
            "-singlefile",
            "-png",
            "-r",
            "200",
            str(pdf_path),
            str(image_prefix),
        ]
    )

    if render.returncode != 0 or not image_path.exists():
        raise RuntimeError(
            "pdftoppm failed: "
            f"page={page_number}, stderr={render.stderr.strip()}"
        )

    tsv_base = temp_dir / f"ocr_{page_number:04d}"

    ocr = run_command(
        [
            "tesseract",
            str(image_path),
            str(tsv_base),
            "-l",
            "jpn+eng",
            "--psm",
            "6",
            "tsv",
        ]
    )

    tsv_path = tsv_base.with_suffix(".tsv")

    if ocr.returncode != 0 or not tsv_path.exists():
        raise RuntimeError(
            "tesseract failed: "
            f"page={page_number}, stderr={ocr.stderr.strip()}"
        )

    rows = tsv_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    if not rows:
        return {
            "pdf_page": page_number,
            "width": 1.0,
            "height": 1.0,
            "items": [],
            "extract_method": "tesseract_tsv",
        }

    header = rows[0].split("\t")
    index = {name: i for i, name in enumerate(header)}

    words: list[dict] = []
    image_width = None
    image_height = None

    for row in rows[1:]:
        fields = row.split("\t")
        if len(fields) < len(header):
            continue

        text = fields[index["text"]].strip()
        if not text:
            continue

        try:
            left = float(fields[index["left"]])
            top = float(fields[index["top"]])
            width = float(fields[index["width"]])
            height = float(fields[index["height"]])
            page_num = int(fields[index["page_num"]])
            level = int(fields[index["level"]])
        except (ValueError, KeyError):
            continue

        if page_num != 1 or level != 5:
            continue

        image_width = max(image_width or 0.0, left + width)
        image_height = max(image_height or 0.0, top + height)

        words.append(
            {
                "text": text,
                "x_min": left,
                "y_min": top,
                "x_max": left + width,
                "y_max": top + height,
            }
        )

    # 近いY座標のOCR wordを行へまとめる。
    words.sort(key=lambda item: (item["y_min"], item["x_min"]))

    items: list[dict] = []
    current: list[dict] = []
    current_y = None

    for word in words:
        if current_y is None or abs(word["y_min"] - current_y) <= 8.0:
            current.append(word)
            if current_y is None:
                current_y = word["y_min"]
            else:
                current_y = sum(w["y_min"] for w in current) / len(current)
            continue

        items.append(make_item_from_words(current))
        current = [word]
        current_y = word["y_min"]

    if current:
        items.append(make_item_from_words(current))

    return {
        "pdf_page": page_number,
        "width": float(image_width or 1.0),
        "height": float(image_height or 1.0),
        "items": items,
        "extract_method": "tesseract_tsv",
    }


def make_item_from_words(words: list[dict]) -> dict:
    words = sorted(words, key=lambda item: item["x_min"])

    return {
        "text": " ".join(word["text"] for word in words),
        "x_min": min(word["x_min"] for word in words),
        "y_min": min(word["y_min"] for word in words),
        "x_max": max(word["x_max"] for word in words),
        "y_max": max(word["y_max"] for word in words),
        "words": words,
    }


def bbox_page_has_meaningful_text(page: dict) -> bool:
    text = "".join(
        item.get("text", "")
        for item in page.get("items", [])
    )
    return len(text.strip()) >= 10


def extract_layout_pages(pdf_path: Path) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="sw25_layout_") as temp_name:
        temp_dir = Path(temp_name)
        bbox_path = temp_dir / "bbox.html"

        run_pdftotext_bbox(
            pdf_path,
            bbox_path,
        )

        pages = parse_bbox_html(
            bbox_path
        )

        if not pages:
            raise RuntimeError(
                f"No pages extracted from: {pdf_path}"
            )

        result: list[dict] = []

        for page in pages:
            if bbox_page_has_meaningful_text(page):
                result.append(page)
                continue

            result.append(
                run_tesseract_tsv_on_page(
                    pdf_path=pdf_path,
                    page_number=page["pdf_page"],
                    temp_dir=temp_dir,
                )
            )

        return result


def classify_column(
    *,
    x_min: float,
    page_width: float,
) -> str:
    ratio = x_min / page_width

    if ratio < PAGE_COL_MAX_RATIO:
        return "page"
    if ratio < LOCATION_COL_MAX_RATIO:
        return "location"
    if ratio < BEFORE_COL_MAX_RATIO:
        return "before"
    return "after"


def normalize_text(text: str) -> str:
    return (
        text.replace("\u3000", " ")
        .replace("\u00a0", " ")
        .strip()
    )


def is_page_number_text(text: str) -> bool:
    normalized = normalize_text(text).replace(" ", "")
    return bool(PAGE_TOKEN_RE.fullmatch(normalized))


def parse_target_page(text: str) -> tuple[int, bool]:
    normalized = normalize_text(text).replace(" ", "")
    starred = normalized.startswith(("★", "☆"))
    number = normalized.lstrip("★☆")
    return int(number), starred


def is_header_or_footer(item: dict, page_width: float) -> bool:
    text = normalize_text(item.get("text", ""))

    if not text:
        return True

    if FOOTER_RE.fullmatch(text):
        return True

    if text in {"ページ", "場所", "誤", "正"}:
        return True

    # タイトル・更新日は列解析対象外。
    if item.get("y_min", 0.0) < 100.0 and page_width > 100:
        return True

    return False


def group_items_by_y(items: list[dict]) -> list[list[dict]]:
    """
    同一Y帯の要素を1つの視覚行としてまとめる。
    Popplerでは4列が別lineになるため、ここで再結合する。
    """

    sorted_items = sorted(
        items,
        key=lambda item: (
            item.get("y_min", 0.0),
            item.get("x_min", 0.0),
        ),
    )

    groups: list[list[dict]] = []

    for item in sorted_items:
        y = float(item.get("y_min", 0.0))

        if not groups:
            groups.append([item])
            continue

        previous = groups[-1]
        average_y = sum(
            float(member.get("y_min", 0.0))
            for member in previous
        ) / len(previous)

        if abs(y - average_y) <= Y_TOLERANCE:
            previous.append(item)
        else:
            groups.append([item])

    for group in groups:
        group.sort(
            key=lambda item: float(
                item.get("x_min", 0.0)
            )
        )

    return groups


def append_cell_line(cell: list[str], text: str) -> None:
    text = normalize_text(text)
    if text:
        cell.append(text)


def join_cell_lines(lines: list[str]) -> str | None:
    if not lines:
        return None

    result = ""

    for part in lines:
        if not result:
            result = part
            continue

        # 日本語PDFの折返しは基本的に空白を入れず連結。
        if result.endswith(("、", "。", "」", "）", "〉", "》", "】")):
            result += part
        else:
            result += part

    result = result.strip()
    return result or None



APPEND_INSTRUCTION_PATTERNS = (
    "※末尾に以下の一文を追記",
    "末尾に以下の一文を追記",
    "以下の文を追記",
    "以下の一文を追記",
    "※追記",
    "を追記",
)

SUSPICIOUS_ENDINGS = (
    "修",
    "の",
    "に",
    "を",
    "へ",
    "が",
    "は",
    "で",
    "と",
    "（",
    "「",
    "〈",
    "《",
    "【",
)


def looks_truncated(text: str | None) -> bool:
    if not text:
        return False

    value = text.strip()

    if len(value) < 2:
        return False

    if value.endswith(SUSPICIOUS_ENDINGS):
        return True

    # 開き括弧だけが多い場合も途中切れの可能性。
    bracket_pairs = (
        ("（", "）"),
        ("「", "」"),
        ("〈", "〉"),
        ("《", "》"),
        ("【", "】"),
    )

    for opening, closing in bracket_pairs:
        if value.count(opening) > value.count(closing):
            return True

    return False


def split_trailing_note(text: str | None) -> tuple[str | None, str | None]:
    """
    after末尾の編集注記を本文から分離する。

    例:
        〇蛇の身体※アイコンを追記
    ->
        value = 〇蛇の身体
        note = ※アイコンを追記
    """
    if not text:
        return text, None

    value = text.strip()
    marker_index = value.find("※")

    if marker_index <= 0:
        return value, None

    main = value[:marker_index].rstrip()
    note = value[marker_index:].strip()

    return (
        main or None,
        note or None,
    )


def classify_operation(record: dict) -> tuple[str, dict]:
    """
    ERRATA操作種別を安全側で正規化する。

    優先順位:
      1. internal table -> complex
      2. before/after両方あり -> replace
      3. 片側のみ + 明示的追記指示 + 追記本文あり -> append
      4. その他 -> complex
    """

    before = record.get("before")
    after = record.get("after")
    location = record.get("location")
    internal_table = bool(
        record.get("internal_table_detected")
    )

    if internal_table:
        return "complex", {}

    if before is not None and after is not None:
        normalized_after, note = split_trailing_note(
            str(after)
        )

        return "replace", {
            "normalized_after": normalized_after,
            "note": note,
        }

    before_text = str(before or "")
    after_text = str(after or "")

    append_marker = None
    append_source = None

    for marker in APPEND_INSTRUCTION_PATTERNS:
        if marker in before_text:
            append_marker = marker
            append_source = before_text
            break

        if marker in after_text:
            append_marker = marker
            append_source = after_text
            break

    if append_marker is not None:
        instruction, _, remainder = append_source.partition(
            append_marker
        )

        if not instruction.strip():
            instruction = append_marker
            remainder = append_source[len(append_marker):]
        else:
            instruction = (
                instruction
                + append_marker
            ).strip()

        append_text = remainder.strip()

        if append_text:
            return "append", {
                "append_instruction": instruction or None,
                "append_text": append_text,
                "append_location": location,
            }

    return "complex", {}


def apply_operation_classification(record: dict) -> None:
    operation, extra = classify_operation(
        record
    )

    record["operation"] = operation

    record["append_instruction"] = extra.get(
        "append_instruction"
    )
    record["append_text"] = extra.get(
        "append_text"
    )
    record["append_location"] = extra.get(
        "append_location"
    )
    record["note"] = extra.get(
        "note"
    )

    if operation == "replace":
        normalized_after = extra.get(
            "normalized_after"
        )

        if normalized_after is not None:
            record["after"] = normalized_after

def assess_record_quality(record: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []

    location = record.get("location")
    before = record.get("before")
    after = record.get("after")

    if not location:
        reasons.append("MISSING_LOCATION")

    if before is None or after is None:
        reasons.append("MISSING_BEFORE_OR_AFTER")

    # 「追加Ｄ」のような項目名を誤検知しないため、
    # locationは見ず、before/afterの指示文だけを見る。
    before_text = str(before or "")
    after_text = str(after or "")
    instruction_text = before_text + "\n" + after_text

    if any(
        marker in instruction_text
        for marker in APPEND_INSTRUCTION_PATTERNS
    ):
        reasons.append("APPEND_INSTRUCTION_PATTERN")

    if record.get("internal_table_detected"):
        reasons.append("INTERNAL_TABLE_PATTERN")

    if looks_truncated(before):
        reasons.append("BEFORE_LOOKS_TRUNCATED")

    if looks_truncated(after):
        reasons.append("AFTER_LOOKS_TRUNCATED")

    status = (
        "LAYOUT_COMPLEX"
        if reasons
        else "LAYOUT_PARSED"
    )

    return status, reasons


def group_has_internal_table_pattern(
    *,
    column_items: dict[str, list[dict]],
) -> bool:
    """
    同一視覚行・同一論理列に複数の独立要素が存在する場合だけ、
    セル内部の小表候補とみなす。

    単なる折り返しは別Y行になるため、ここでは検出されない。
    """
    for column_name in ("location", "before", "after"):
        items = column_items.get(column_name, [])

        if len(items) >= 2:
            # 近接した別wordではなく、Poppler上で独立itemになっているもの。
            # 明確に離れた要素が同一列に並ぶ場合を小表候補とする。
            sorted_items = sorted(
                items,
                key=lambda item: float(item.get("x_min", 0.0)),
            )

            previous = None

            for item in sorted_items:
                if previous is not None:
                    gap = (
                        float(item.get("x_min", 0.0))
                        - float(previous.get("x_max", 0.0))
                    )

                    if gap >= 6.0:
                        return True

                previous = item

    return False


def parse_errata_page(
    page: dict,
    *,
    next_record_index: int,
) -> tuple[list[dict], int, list[dict]]:
    page_width = float(page.get("width") or 595.32)
    pdf_page = int(page.get("pdf_page") or 0)

    items = [
        item
        for item in page.get("items", [])
        if not is_header_or_footer(item, page_width)
    ]

    groups = group_items_by_y(items)

    records: list[dict] = []
    diagnostics: list[dict] = []

    current: dict | None = None

    def flush() -> None:
        nonlocal current
        nonlocal next_record_index

        if current is None:
            return

        record = {
            "record_index": next_record_index,
            "target_page": current["target_page"],
            "starred": current["starred"],
            "source_pdf_pages": [pdf_page],
            "location": join_cell_lines(current["location"]),
            "before": join_cell_lines(current["before"]),
            "after": join_cell_lines(current["after"]),
            "raw_text": current["raw_text"],
            "parse_status": "LAYOUT_PARSED",
            "quality_reasons": [],
            "internal_table_detected": bool(
                current.get("internal_table_detected")
            ),
            "extract_method": page.get("extract_method"),
        }

        apply_operation_classification(
            record
        )

        quality_status, quality_reasons = assess_record_quality(
            record
        )

        # 完成したappendだけ、before/after欠落を許容する。
        if (
            record.get("operation") == "append"
            and record.get("append_text")
        ):
            quality_reasons = [
                reason
                for reason in quality_reasons
                if reason not in {
                    "MISSING_BEFORE_OR_AFTER",
                    "APPEND_INSTRUCTION_PATTERN",
                }
            ]

            if not quality_reasons:
                quality_status = "LAYOUT_PARSED"

        record["parse_status"] = quality_status
        record["quality_reasons"] = quality_reasons

        if record["parse_status"] == "LAYOUT_COMPLEX":
            record["operation"] = "complex"

        records.append(record)
        next_record_index += 1
        current = None

    for group in groups:
        page_tokens: list[dict] = []
        column_items: dict[str, list[dict]] = {
            "location": [],
            "before": [],
            "after": [],
        }

        for item in group:
            column = classify_column(
                x_min=float(item["x_min"]),
                page_width=page_width,
            )

            if (
                column == "page"
                and is_page_number_text(
                    item.get("text", "")
                )
            ):
                page_tokens.append(item)
                continue

            if column in column_items:
                column_items[column].append(item)

        if page_tokens:
            # 同一Y帯に複数ページ番号が現れるのは通常想定外。
            # 先頭を採用し、残りはdiagnosticへ。
            flush()

            token = page_tokens[0]
            target_page, starred = parse_target_page(
                token["text"]
            )

            current = {
                "target_page": target_page,
                "starred": starred,
                "location": [],
                "before": [],
                "after": [],
                "raw_text": [],
                "internal_table_detected": False,
            }

            if len(page_tokens) > 1:
                diagnostics.append(
                    {
                        "pdf_page": pdf_page,
                        "type": "MULTIPLE_PAGE_TOKENS_SAME_Y",
                        "tokens": [
                            token.get("text")
                            for token in page_tokens
                        ],
                    }
                )

        if current is not None:
            if group_has_internal_table_pattern(
                column_items=column_items,
            ):
                current["internal_table_detected"] = True

        if current is None:
            # 表前文など。診断として保持するがレコード化しない。
            texts = [
                item.get("text", "")
                for item in group
                if normalize_text(
                    item.get("text", "")
                )
            ]

            if texts:
                diagnostics.append(
                    {
                        "pdf_page": pdf_page,
                        "type": "UNASSIGNED_ROW",
                        "y_min": min(
                            float(item.get("y_min", 0.0))
                            for item in group
                        ),
                        "text": " | ".join(texts),
                    }
                )
            continue

        raw_parts: list[str] = []

        for column_name in ("location", "before", "after"):
            for item in column_items[column_name]:
                text = normalize_text(
                    item.get("text", "")
                )
                if not text:
                    continue

                append_cell_line(
                    current[column_name],
                    text,
                )

                raw_parts.append(
                    f"{column_name}:{text}"
                )

        if raw_parts:
            current["raw_text"].append(
                " | ".join(raw_parts)
            )

    flush()

    for record in records:
        record["raw_text"] = "\n".join(
            record["raw_text"]
        )

    return (
        records,
        next_record_index,
        diagnostics,
    )


def parse_errata_document(
    *,
    pdf_path: Path,
    source_url: str | None,
    source_sha256: str | None,
    source_last_modified: str | None,
    source_id: str | None,
    source_name: str | None,
) -> dict:
    pages = extract_layout_pages(
        pdf_path
    )

    records: list[dict] = []
    diagnostics: list[dict] = []
    next_record_index = 1

    extract_methods = Counter()

    for page in pages:
        extract_methods[
            page.get(
                "extract_method",
                "unknown",
            )
        ] += 1

        page_records, next_record_index, page_diagnostics = (
            parse_errata_page(
                page,
                next_record_index=next_record_index,
            )
        )

        records.extend(
            page_records
        )
        diagnostics.extend(
            page_diagnostics
        )

    return {
        "version": BUILD_VERSION,
        "built_at": utc_now_iso(),
        "source_id": source_id,
        "source_name": source_name,
        "source_type": SOURCE_TYPE_ERRATA,
        "source_key": source_key_from_filename(
            pdf_path.name
        ),
        "target_book": None,
        "source_url": source_url,
        "source_sha256": source_sha256,
        "source_last_modified": source_last_modified,
        "raw_pdf": str(pdf_path),
        "page_count": len(pages),
        "extract_method_counts": dict(
            sorted(
                extract_methods.items()
            )
        ),
        "record_count": len(records),
        "parse_status_counts": dict(
            sorted(
                Counter(
                    record.get(
                        "parse_status",
                        "UNKNOWN",
                    )
                    for record in records
                ).items()
            )
        ),
        "operation_counts": dict(
            sorted(
                Counter(
                    record.get(
                        "operation",
                        "unknown",
                    )
                    for record in records
                ).items()
            )
        ),
        "records": records,
        "diagnostic_count": len(diagnostics),
        "diagnostics": diagnostics,
    }


def find_pdf_path(
    *,
    raw_root: Path,
    source_id: str,
    manifest_entry: dict,
) -> Path:
    local_path = manifest_entry.get("local_path")

    if isinstance(local_path, str) and local_path:
        candidate = Path(local_path)
        if candidate.exists():
            return candidate

    url = manifest_entry.get("url")

    if not isinstance(url, str) or not url:
        raise ValueError(
            "Manifest entry missing usable local_path and url"
        )

    filename = url.rsplit("/", 1)[-1]
    filename = filename.replace("%20", "_20")

    direct = (
        raw_root
        / source_id
        / "products"
        / "sw"
        / "eratta"
        / "pdf"
        / filename
    )

    if direct.exists():
        return direct

    matches = list(
        (raw_root / source_id).rglob(
            Path(filename).name
        )
    )

    if len(matches) == 1:
        return matches[0]

    # URLデコード前の保存名差異を吸収するためstemで探索。
    stem = Path(filename).stem.lower()

    fuzzy = [
        path
        for path in (
            raw_root
            / source_id
        ).rglob("*.pdf")
        if path.stem.lower() == stem
    ]

    if len(fuzzy) == 1:
        return fuzzy[0]

    raise FileNotFoundError(
        f"PDF not found for URL: {url}"
    )


def parse_args(
    argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized official SW2.5 sources using "
            "layout-aware PDF extraction."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/data/official/raw/groupsne_sw25_errata/manifest.json"
        ),
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "/data/official/raw"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/official/layout_normalized/groupsne_sw25_errata"
        ),
    )

    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help=(
            "Process only matching PDF filenames. "
            "May be specified multiple times."
        ),
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

    if not args.manifest.exists():
        print(
            f"Manifest not found: {args.manifest}",
            file=sys.stderr,
        )
        return 2

    manifest = load_json(
        args.manifest
    )

    source_info = manifest.get("source") or {}
    source_id = source_info.get("id")
    source_name = source_info.get("name")

    if not isinstance(source_id, str) or not source_id:
        print(
            "manifest.source.id is missing",
            file=sys.stderr,
        )
        return 2

    documents = manifest.get("documents")

    if not isinstance(documents, list):
        print(
            "manifest.documents must be array",
            file=sys.stderr,
        )
        return 2

    only = {
        value.lower()
        for value in args.only
    }

    results: list[dict] = []
    errors: list[dict] = []

    pdf_entries = [
        entry
        for entry in documents
        if (
            isinstance(entry, dict)
            and entry.get("content_type") == "application/pdf"
        )
    ]

    print(
        f"PDF documents: {len(pdf_entries)}"
    )

    for index, entry in enumerate(
        pdf_entries,
        start=1,
    ):
        try:
            pdf_path = find_pdf_path(
                raw_root=args.raw_root,
                source_id=source_id,
                manifest_entry=entry,
            )
        except Exception as exc:
            errors.append(
                {
                    "url": entry.get("url"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[{index}/{len(pdf_entries)}] ERROR locating PDF: {exc}",
                file=sys.stderr,
            )
            continue

        if only and pdf_path.name.lower() not in only:
            continue

        source_type = classify_source(
            pdf_path.name
        )

        print(
            f"[{index}/{len(pdf_entries)}] "
            f"{pdf_path.name} type={source_type}"
        )

        if source_type != SOURCE_TYPE_ERRATA:
            print(
                "  skipped: Phase 1 handles ERRATA only"
            )
            continue

        try:
            normalized = parse_errata_document(
                pdf_path=pdf_path,
                source_url=entry.get("url"),
                source_sha256=entry.get("sha256"),
                source_last_modified=entry.get("last_modified"),
                source_id=source_id,
                source_name=source_name,
            )

            output_path = (
                args.output_dir
                / f"{pdf_path.stem}.layout.normalized.json"
            )

            write_json_atomic(
                output_path,
                normalized,
            )

            print(
                "  "
                f"records={normalized['record_count']} "
                f"statuses={normalized['parse_status_counts']} "
                f"operations={normalized['operation_counts']} "
                f"diagnostics={normalized['diagnostic_count']} "
                f"methods={normalized['extract_method_counts']}"
            )

            parsed_count = normalized[
                "parse_status_counts"
            ].get(
                "LAYOUT_PARSED",
                0,
            )

            complex_count = normalized[
                "parse_status_counts"
            ].get(
                "LAYOUT_COMPLEX",
                0,
            )

            record_count = normalized[
                "record_count"
            ]

            complex_ratio = (
                complex_count
                / record_count
                if record_count
                else 0.0
            )

            results.append(
                {
                    "input_pdf": str(pdf_path),
                    "output_file": str(output_path),
                    "source_type": source_type,
                    "record_count": record_count,
                    "parsed_count": parsed_count,
                    "complex_count": complex_count,
                    "complex_ratio": round(
                        complex_ratio,
                        4,
                    ),
                    "parse_status_counts": normalized[
                        "parse_status_counts"
                    ],
                    "operation_counts": normalized[
                        "operation_counts"
                    ],
                    "diagnostic_count": normalized["diagnostic_count"],
                    "extract_method_counts": normalized[
                        "extract_method_counts"
                    ],
                }
            )

        except Exception as exc:
            errors.append(
                {
                    "pdf": str(pdf_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            print(
                f"  ERROR: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    total_records = sum(
        item.get(
            "record_count",
            0,
        )
        for item in results
    )

    total_parsed = sum(
        item.get(
            "parsed_count",
            0,
        )
        for item in results
    )

    total_complex = sum(
        item.get(
            "complex_count",
            0,
        )
        for item in results
    )

    total_diagnostics = sum(
        item.get(
            "diagnostic_count",
            0,
        )
        for item in results
    )

    total_operations = Counter()
    total_extract_methods = Counter()

    for item in results:
        total_operations.update(
            item.get(
                "operation_counts",
                {}
            )
        )
        total_extract_methods.update(
            item.get(
                "extract_method_counts",
                {}
            )
        )

    by_complex_ratio = sorted(
        results,
        key=lambda item: (
            item.get(
                "complex_ratio",
                0.0,
            ),
            item.get(
                "complex_count",
                0,
            ),
        ),
        reverse=True,
    )

    by_diagnostics = sorted(
        results,
        key=lambda item: item.get(
            "diagnostic_count",
            0,
        ),
        reverse=True,
    )

    ocr_documents = [
        item
        for item in results
        if item.get(
            "extract_method_counts",
            {}
        ).get(
            "tesseract_tsv",
            0,
        )
        > 0
    ]

    report = {
        "version": BUILD_VERSION,
        "generated_at": utc_now_iso(),
        "document_count": len(results),
        "error_count": len(errors),
        "summary": {
            "record_count": total_records,
            "parsed_count": total_parsed,
            "complex_count": total_complex,
            "complex_ratio": round(
                (
                    total_complex
                    / total_records
                    if total_records
                    else 0.0
                ),
                4,
            ),
            "diagnostic_count": total_diagnostics,
            "operation_counts": dict(
                sorted(
                    total_operations.items()
                )
            ),
            "extract_method_counts": dict(
                sorted(
                    total_extract_methods.items()
                )
            ),
        },
        "review_priority": {
            "highest_complex_ratio": [
                {
                    "input_pdf": item[
                        "input_pdf"
                    ],
                    "record_count": item[
                        "record_count"
                    ],
                    "complex_count": item[
                        "complex_count"
                    ],
                    "complex_ratio": item[
                        "complex_ratio"
                    ],
                }
                for item in by_complex_ratio[:10]
            ],
            "highest_diagnostic_count": [
                {
                    "input_pdf": item[
                        "input_pdf"
                    ],
                    "diagnostic_count": item[
                        "diagnostic_count"
                    ],
                }
                for item in by_diagnostics[:10]
            ],
            "ocr_documents": [
                {
                    "input_pdf": item[
                        "input_pdf"
                    ],
                    "extract_method_counts": item[
                        "extract_method_counts"
                    ],
                }
                for item in ocr_documents
            ],
        },
        "documents": results,
        "errors": errors,
    }

    report_path = (
        args.output_dir
        / "layout_normalization_manifest.json"
    )

    write_json_atomic(
        report_path,
        report,
    )

    print()
    print(
        f"Manifest: {report_path}"
    )
    print(
        f"Documents: {report['document_count']}"
    )
    print(
        f"Errors: {report['error_count']}"
    )
    print(
        f"Records: {report['summary']['record_count']}"
    )
    print(
        "Parsed / Complex: "
        f"{report['summary']['parsed_count']} / "
        f"{report['summary']['complex_count']} "
        f"(complex_ratio="
        f"{report['summary']['complex_ratio']})"
    )
    print(
        f"Operations: "
        f"{report['summary']['operation_counts']}"
    )
    print(
        f"Diagnostics: "
        f"{report['summary']['diagnostic_count']}"
    )
    print(
        f"Extract methods: "
        f"{report['summary']['extract_method_counts']}"
    )

    if report[
        "review_priority"
    ][
        "highest_complex_ratio"
    ]:
        print()
        print(
            "Highest complex ratio:"
        )
        for item in report[
            "review_priority"
        ][
            "highest_complex_ratio"
        ]:
            print(
                "  "
                f"{Path(item['input_pdf']).name}: "
                f"{item['complex_count']}/"
                f"{item['record_count']} "
                f"({item['complex_ratio']:.1%})"
            )

    if report[
        "review_priority"
    ][
        "ocr_documents"
    ]:
        print()
        print(
            "OCR fallback documents:"
        )
        for item in report[
            "review_priority"
        ][
            "ocr_documents"
        ]:
            print(
                "  "
                f"{Path(item['input_pdf']).name}: "
                f"{item['extract_method_counts']}"
            )

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
