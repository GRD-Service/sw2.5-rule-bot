from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ============================================================
# Version
# ============================================================

SCHEMA_VERSION = 2
CHECKPOINT_VERSION = 1


# ============================================================
# Helpers: environment
# ============================================================

def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(
        name,
        "1" if default else "0",
    )

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================
# Paths
# ============================================================

OCR_DIR = Path(
    os.getenv(
        "OCR_DIR",
        "/data/ocr",
    )
)

IMAGE_DIR = Path(
    os.getenv(
        "IMAGE_DIR",
        "/data/image",
    )
)

PAGE_MAP_DIR = Path(
    os.getenv(
        "PAGE_MAP_DIR",
        "/data/metadata/page_maps",
    )
)

NAVIGATION_DIR = Path(
    os.getenv(
        "NAVIGATION_DIR",
        "/data/metadata/navigation",
    )
)

NAVIGATION_WORK_DIR = Path(
    os.getenv(
        "NAVIGATION_WORK_DIR",
        "/data/metadata/navigation_work",
    )
)


# ============================================================
# Model / request settings
# ============================================================

NAVIGATION_MODEL = os.getenv(
    "NAVIGATION_MODEL",
    "gpt-4.1-mini",
)

NAVIGATION_REQUEST_TIMEOUT = float(
    os.getenv(
        "NAVIGATION_REQUEST_TIMEOUT",
        "120",
    )
)

NAVIGATION_MAX_RETRIES = int(
    os.getenv(
        "NAVIGATION_MAX_RETRIES",
        "2",
    )
)

NAVIGATION_MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "NAVIGATION_MAX_OUTPUT_TOKENS",
        "12000",
    )
)


# ============================================================
# Image settings
# ============================================================

NAVIGATION_IMAGE_MAX_SIDE = int(
    os.getenv(
        "NAVIGATION_IMAGE_MAX_SIDE",
        "2400",
    )
)

NAVIGATION_JPEG_QUALITY = int(
    os.getenv(
        "NAVIGATION_JPEG_QUALITY",
        "88",
    )
)

NAVIGATION_TILE_ROWS = int(
    os.getenv(
        "NAVIGATION_TILE_ROWS",
        "2",
    )
)

NAVIGATION_TILE_COLS = int(
    os.getenv(
        "NAVIGATION_TILE_COLS",
        "2",
    )
)

NAVIGATION_TILE_OVERLAP = float(
    os.getenv(
        "NAVIGATION_TILE_OVERLAP",
        "0.10",
    )
)


# ============================================================
# Detection settings
# ============================================================

TOC_SCAN_END_RATIO = float(
    os.getenv(
        "TOC_SCAN_END_RATIO",
        "0.30",
    )
)

TOC_MAX_PAGES = int(
    os.getenv(
        "TOC_MAX_PAGES",
        "10",
    )
)

INDEX_SCAN_START_RATIO = float(
    os.getenv(
        "INDEX_SCAN_START_RATIO",
        "0.60",
    )
)

INDEX_MAX_PAGES = int(
    os.getenv(
        "INDEX_MAX_PAGES",
        "12",
    )
)

MAX_INDEX_HEADING_LENGTH = int(
    os.getenv(
        "MAX_INDEX_HEADING_LENGTH",
        "30",
    )
)

FORCE_REBUILD = env_bool(
    "NAVIGATION_FORCE_REBUILD",
    False,
)


# ============================================================
# Patterns
# ============================================================

PAGE_IMAGE_PATTERN = re.compile(
    r"^P0*(\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)

INDEX_HEADING_PATTERN = re.compile(
    r"^(.{0,30}?索引)$"
)

TOC_INDEX_PATTERN = re.compile(
    r"^\s*(.{0,30}?索引)"
    r"[\s.．…⋯・･\-―ー]*"
    r"([0-9]{1,4})\s*$"
)

PURE_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*([0-9]{1,4})\s*$"
)

TOC_HEADING_PATTERNS = (
    re.compile(r"目次", re.IGNORECASE),
    re.compile(r"もくじ", re.IGNORECASE),
    re.compile(r"contents?", re.IGNORECASE),
)

TOC_ENTRY_LIKE_PATTERN = re.compile(
    r".{1,80}"
    r"[\s.．…⋯・･\-―ー]+"
    r"\d{1,4}\s*$"
)

TOC_SECTION_HINT_PATTERN = re.compile(
    r"("
    r"第[一二三四五六七八九十0-9]+部"
    r"|第[一二三四五六七八九十0-9]+章"
    r"|はじめに"
    r"|序章"
    r"|終章"
    r"|chapter"
    r")",
    re.IGNORECASE,
)


# ============================================================
# Structured output: TOC
# ============================================================

class TocEntry(BaseModel):
    title: str = Field(
        description=(
            "目次に実際に印刷されている項目名。"
            "画像から確認できない文字は推測しない。"
        )
    )

    logical_page: int | None = Field(
        default=None,
        description=(
            "目次に印刷されている書籍ページ番号。"
            "PDFページ番号ではない。"
        ),
    )

    level: int = Field(
        default=1,
        description=(
            "見た目上の階層。"
            "章などの大項目=1、"
            "その下=2、さらに下=3。"
            "不明なら1。"
        ),
    )


class TocPageExtraction(BaseModel):
    is_toc_page: bool = Field(
        description=(
            "画像が目次ページ、または目次の続きならtrue。"
            "タイル画像の場合、目次項目が含まれていればtrue。"
        )
    )

    entries: list[TocEntry] = Field(
        default_factory=list
    )


# ============================================================
# Structured output: INDEX
# ============================================================

class IndexEntry(BaseModel):
    term: str = Field(
        description=(
            "索引に実際に印刷されている索引項目名。"
            "画像から確認できない文字は推測しない。"
        )
    )

    logical_pages: list[int] = Field(
        default_factory=list,
        description=(
            "その項目に対応して印刷されている書籍ページ番号。"
            "PDFページ番号ではない。"
        ),
    )


class IndexSection(BaseModel):
    index_type: str = Field(
        description=(
            "画像に書かれている索引分類名。"
            "例: 索引、一般索引、魔法索引、"
            "戦闘特技索引、魔物索引、流派索引など。"
            "未知の分類でも表記をそのまま返す。"
        )
    )

    entries: list[IndexEntry] = Field(
        default_factory=list
    )


class IndexPageExtraction(BaseModel):
    is_index_page: bool = Field(
        description=(
            "画像が索引ページ、索引の続き、"
            "またはタイル内に索引項目が含まれていればtrue。"
        )
    )

    sections: list[IndexSection] = Field(
        default_factory=list
    )


# ============================================================
# Prompts
# ============================================================

TOC_SYSTEM_PROMPT = """
あなたは日本語TRPGルールブックの目次を構造化する担当です。

画像には書籍の目次ページ、目次の続き、
またはその一部分が含まれている可能性があります。

規則:

1. 画像に実際に印刷されている項目だけを抽出する。
2. 項目名を知識や文脈から補完しない。
3. ページ番号は書籍に印刷されたページ番号であり、
   PDFページ番号ではない。
4. 複数カラム、縦書き、横書きがあり得る。
5. OCRの行順より画像上の配置を優先する。
6. ページ自体のノンブルを目次の参照ページと誤認しない。
7. 見た目の階層をlevelとして1～3程度で表現する。
8. 判別できないページ番号はnullにする。
9. 判読不能な文字を推測しない。
10. タイル画像の場合でも、目次項目が含まれていれば
    is_toc_page=true とする。
11. タイル境界で項目名またはページ番号が欠けている場合は、
    推測せず、その項目を省略する。
"""


INDEX_SYSTEM_PROMPT = """
あなたは日本語TRPGルールブックの索引を構造化する担当です。

画像には書籍の索引ページ、索引の続き、
またはその一部分が含まれている可能性があります。

規則:

1. 画像に実際に印刷されている索引項目だけを抽出する。
2. 項目名を知識や文脈から補完・修正・推測しない。
3. ページ番号は書籍に印刷されたページ番号であり、
   PDFページ番号ではない。
4. 複数カラム、縦書き、横書きがあり得る。
5. OCRの行順より画像上の配置を優先する。
6. 「ア行」「カ行」「人物」「組織」「地名」など、
   単なる分類見出しは索引項目にしない。
7. ページ自体のノンブルを参照ページと誤認しない。
8. 索引分類には固定された一覧はない。
9. 「一般索引」「魔法索引」「流派索引」など、
   画像の分類名をそのままindex_typeへ入れる。
10. 単に「索引」とだけ書かれていれば
    index_type="索引" とする。
11. タイル画像の場合でも索引項目があれば
    is_index_page=true とする。
12. タイル境界で項目名またはページ番号が欠けている場合は、
    推測せず、その項目を省略する。
13. 正確性を最優先する。
"""


# ============================================================
# JSON helpers
# ============================================================

def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def atomic_write_json(
    path: Path,
    data,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# OCR
# ============================================================

def load_ocr(
    path: Path,
) -> tuple[str, list[dict]]:
    data = load_json(
        path
    )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            f"OCR JSON root must be list: {path}"
        )

    if not data:
        raise ValueError(
            f"OCR JSON is empty: {path}"
        )

    book_name = None
    pages = []

    for entry in data:
        if not isinstance(
            entry,
            dict,
        ):
            continue

        current_book = entry.get(
            "book"
        )

        if not current_book:
            continue

        if book_name is None:
            book_name = current_book

        elif current_book != book_name:
            raise ValueError(
                f"Multiple books in OCR JSON: {path}"
            )

        try:
            pdf_page = int(
                entry["page"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        text = entry.get(
            "text",
            "",
        )

        if not isinstance(
            text,
            str,
        ):
            text = str(
                text
            )

        pages.append(
            {
                "pdf_page": pdf_page,
                "text": text,
            }
        )

    pages.sort(
        key=lambda item: item[
            "pdf_page"
        ]
    )

    if not book_name:
        raise ValueError(
            f"No book name: {path}"
        )

    return (
        book_name,
        pages,
    )


# ============================================================
# Page map
# ============================================================

def load_page_map(
    book_name: str,
) -> dict:
    path = (
        PAGE_MAP_DIR
        / f"{book_name}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Page map not found: {path}"
        )

    data = load_json(
        path
    )

    if data.get(
        "book"
    ) != book_name:
        raise ValueError(
            f"Page map book mismatch: {path}"
        )

    return data


def build_page_lookup(
    page_map: dict,
) -> tuple[
    dict[int, int],
    dict[int, int],
]:
    logical_to_pdf = {}
    pdf_to_logical = {}

    for item in page_map.get(
        "mappings",
        [],
    ):
        pdf_page = item.get(
            "pdf_page"
        )

        logical_page = item.get(
            "logical_page"
        )

        if (
            pdf_page is None
            or logical_page is None
        ):
            continue

        try:
            pdf_page = int(
                pdf_page
            )

            logical_page = int(
                logical_page
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        logical_to_pdf[
            logical_page
        ] = pdf_page

        pdf_to_logical[
            pdf_page
        ] = logical_page

    return (
        logical_to_pdf,
        pdf_to_logical,
    )


# ============================================================
# Normalization
# ============================================================

def normalize_heading(
    value: str,
) -> str:
    value = value.strip()

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    return value.strip(
        "・･.…⋯-―ー"
    )


def normalize_term_key(
    value: str,
) -> str:
    return re.sub(
        r"\s+",
        "",
        value.strip(),
    )


def normalize_index_type(
    value: str,
) -> str:
    value = normalize_heading(
        value
    )

    if not value:
        return "索引"

    if value.endswith(
        "索引"
    ):
        return value

    return "索引"


# ============================================================
# Detect TOC seed pages
# ============================================================

def is_toc_heading(
    line: str,
) -> bool:
    value = normalize_heading(
        line
    )

    if not value:
        return False

    # 長すぎる本文中の「目次」言及を除外する。
    if len(value) > 40:
        return False

    return any(
        pattern.search(
            value
        )
        for pattern in TOC_HEADING_PATTERNS
    )


def find_toc_seed_pages(
    pages: list[dict],
) -> list[int]:
    if not pages:
        return []

    max_pdf_page = max(
        item["pdf_page"]
        for item in pages
    )

    scan_end = max(
        1,
        int(
            max_pdf_page
            * TOC_SCAN_END_RATIO
        ),
    )

    seeds = []

    for page in pages:
        pdf_page = page[
            "pdf_page"
        ]

        if pdf_page > scan_end:
            continue

        text = page[
            "text"
        ]

        lines = [
            line
            for line in text.splitlines()
            if line.strip()
        ]

        heading_found = any(
            is_toc_heading(
                line
            )
            for line in lines
        )

        toc_like = looks_like_toc_page(
            text
        )

        if (
            heading_found
            or toc_like
        ):
            seeds.append(
                pdf_page
            )

    return sorted(
        set(
            seeds
        )
    )

def looks_like_toc_page(
    text: str,
) -> bool:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    numbered_entries = 0
    section_hints = 0

    for line in lines:
        if TOC_ENTRY_LIKE_PATTERN.fullmatch(
            line
        ):
            numbered_entries += 1

        if TOC_SECTION_HINT_PATTERN.search(
            line
        ):
            section_hints += 1

    # 通常の目次
    if numbered_entries >= 5:
        return True

    # OCRで「……」などが壊れた場合のfallback
    if (
        numbered_entries >= 2
        and section_hints >= 2
    ):
        return True

    return False


# ============================================================
# Index seed detection
# ============================================================

def detect_index_headings(
    text: str,
) -> list[str]:
    result = []

    for raw_line in text.splitlines():
        value = normalize_heading(
            raw_line
        )

        if not value:
            continue

        if len(
            value
        ) > MAX_INDEX_HEADING_LENGTH:
            continue

        match = INDEX_HEADING_PATTERN.fullmatch(
            value
        )

        if not match:
            continue

        result.append(
            match.group(1)
        )

    return list(
        dict.fromkeys(
            result
        )
    )


def find_tail_index_seed_pages(
    pages: list[dict],
) -> list[int]:
    if not pages:
        return []

    max_pdf_page = max(
        item["pdf_page"]
        for item in pages
    )

    scan_start = max(
        1,
        int(
            max_pdf_page
            * INDEX_SCAN_START_RATIO
        ),
    )

    seeds = []

    for page in pages:
        pdf_page = page[
            "pdf_page"
        ]

        if pdf_page < scan_start:
            continue

        if detect_index_headings(
            page["text"]
        ):
            seeds.append(
                pdf_page
            )

    return sorted(
        set(
            seeds
        )
    )


def find_raw_toc_index_seed_pages(
    pages: list[dict],
    logical_to_pdf: dict[int, int],
) -> list[int]:
    """
    LLMで目次を解析できなかった場合のfallback。
    OCRの「○○索引 ... 123」から直接取得する。
    """

    if not pages:
        return []

    max_pdf_page = max(
        item["pdf_page"]
        for item in pages
    )

    scan_end = max(
        1,
        int(
            max_pdf_page
            * TOC_SCAN_END_RATIO
        ),
    )

    result = []

    for page in pages:
        if page[
            "pdf_page"
        ] > scan_end:
            continue

        lines = [
            line.strip()
            for line in page[
                "text"
            ].splitlines()
            if line.strip()
        ]

        for index, line in enumerate(
            lines
        ):
            match = TOC_INDEX_PATTERN.fullmatch(
                line
            )

            if match:
                logical_page = int(
                    match.group(2)
                )

                pdf_page = (
                    logical_to_pdf.get(
                        logical_page
                    )
                )

                if pdf_page is not None:
                    result.append(
                        pdf_page
                    )

                continue

            heading = normalize_heading(
                line
            )

            if not heading.endswith(
                "索引"
            ):
                continue

            if index + 1 >= len(
                lines
            ):
                continue

            number_match = (
                PURE_PAGE_NUMBER_PATTERN.fullmatch(
                    lines[
                        index + 1
                    ]
                )
            )

            if not number_match:
                continue

            logical_page = int(
                number_match.group(1)
            )

            pdf_page = (
                logical_to_pdf.get(
                    logical_page
                )
            )

            if pdf_page is not None:
                result.append(
                    pdf_page
                )

    return sorted(
        set(
            result
        )
    )


# ============================================================
# Image lookup / manipulation
# ============================================================

def build_image_lookup(
    book_name: str,
) -> dict[int, Path]:
    book_dir = (
        IMAGE_DIR
        / book_name
    )

    if not book_dir.exists():
        raise FileNotFoundError(
            f"Book image directory not found: "
            f"{book_dir}"
        )

    result = {}

    for path in book_dir.iterdir():
        if not path.is_file():
            continue

        match = PAGE_IMAGE_PATTERN.match(
            path.name
        )

        if not match:
            continue

        pdf_page = int(
            match.group(1)
        )

        result[
            pdf_page
        ] = path

    return result


def load_image(
    path: Path,
) -> Image.Image:
    with Image.open(
        path
    ) as source:
        image = ImageOps.exif_transpose(
            source
        )

        return image.convert(
            "RGB"
        )


def resize_image(
    image: Image.Image,
    max_side: int,
) -> Image.Image:
    width, height = image.size

    current_max = max(
        width,
        height,
    )

    if current_max <= max_side:
        return image

    scale = (
        max_side
        / current_max
    )

    new_width = max(
        1,
        int(
            width
            * scale
        ),
    )

    new_height = max(
        1,
        int(
            height
            * scale
        ),
    )

    return image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )


def image_to_data_url(
    image: Image.Image,
) -> str:
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=(
            NAVIGATION_JPEG_QUALITY
        ),
        optimize=True,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "ascii"
    )

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


def create_tiles(
    image: Image.Image,
) -> list[dict]:
    rows = max(
        1,
        NAVIGATION_TILE_ROWS,
    )

    cols = max(
        1,
        NAVIGATION_TILE_COLS,
    )

    overlap_ratio = max(
        0.0,
        min(
            NAVIGATION_TILE_OVERLAP,
            0.40,
        ),
    )

    width, height = image.size

    cell_width = (
        width
        / cols
    )

    cell_height = (
        height
        / rows
    )

    overlap_x = int(
        cell_width
        * overlap_ratio
    )

    overlap_y = int(
        cell_height
        * overlap_ratio
    )

    result = []

    for row in range(
        rows
    ):
        for col in range(
            cols
        ):
            x0 = int(
                col
                * cell_width
            )

            y0 = int(
                row
                * cell_height
            )

            x1 = int(
                (col + 1)
                * cell_width
            )

            y1 = int(
                (row + 1)
                * cell_height
            )

            if col > 0:
                x0 -= overlap_x

            if row > 0:
                y0 -= overlap_y

            if col < cols - 1:
                x1 += overlap_x

            if row < rows - 1:
                y1 += overlap_y

            x0 = max(
                0,
                x0,
            )

            y0 = max(
                0,
                y0,
            )

            x1 = min(
                width,
                x1,
            )

            y1 = min(
                height,
                y1,
            )

            result.append(
                {
                    "row": row,
                    "col": col,
                    "image": image.crop(
                        (
                            x0,
                            y0,
                            x1,
                            y1,
                        )
                    ),
                }
            )

    return result


# ============================================================
# Model
# ============================================================

def build_base_model():
    return ChatOpenAI(
        model=NAVIGATION_MODEL,
        temperature=0,
        timeout=(
            NAVIGATION_REQUEST_TIMEOUT
        ),
        max_retries=(
            NAVIGATION_MAX_RETRIES
        ),
        max_tokens=(
            NAVIGATION_MAX_OUTPUT_TOKENS
        ),
    )


def build_models():
    base = build_base_model()

    toc_model = (
        base.with_structured_output(
            TocPageExtraction
        )
    )

    index_model = (
        base.with_structured_output(
            IndexPageExtraction
        )
    )

    return (
        toc_model,
        index_model,
    )


# ============================================================
# Error detection
# ============================================================

def is_length_limit_error(
    exc: Exception,
) -> bool:
    message = str(
        exc
    ).lower()

    patterns = (
        "length limit",
        "max_tokens",
        "maximum output",
        "maximum context",
        "completion_tokens=32768",
        "finish_reason='length'",
        "finish_reason\": \"length",
    )

    return any(
        pattern in message
        for pattern in patterns
    )


# ============================================================
# Checkpoint
# ============================================================

def checkpoint_path(
    book_name: str,
    kind: str,
    pdf_page: int,
) -> Path:
    return (
        NAVIGATION_WORK_DIR
        / book_name
        / kind
        / f"P{pdf_page:05d}.json"
    )


def tile_checkpoint_path(
    book_name: str,
    kind: str,
    pdf_page: int,
    row: int,
    col: int,
) -> Path:
    return (
        NAVIGATION_WORK_DIR
        / book_name
        / kind
        / f"P{pdf_page:05d}_tiles"
        / f"tile_{row:02d}_{col:02d}.json"
    )


def load_checkpoint(
    path: Path,
):
    if FORCE_REBUILD:
        return None

    if not path.exists():
        return None

    try:
        data = load_json(
            path
        )

    except Exception:
        return None

    if data.get(
        "checkpoint_version"
    ) != CHECKPOINT_VERSION:
        return None

    return data


def save_checkpoint(
    path: Path,
    *,
    kind: str,
    pdf_page: int,
    mode: str,
    extraction: dict,
) -> None:
    atomic_write_json(
        path,
        {
            "checkpoint_version": (
                CHECKPOINT_VERSION
            ),
            "kind": kind,
            "pdf_page": pdf_page,
            "mode": mode,
            "extraction": extraction,
        },
    )


# ============================================================
# Invocation
# ============================================================

def invoke_image_model(
    model,
    system_prompt: str,
    book_name: str,
    pdf_page: int,
    logical_page: int | None,
    image: Image.Image,
    ocr_text: str,
    *,
    tile_description: str | None = None,
):
    data_url = image_to_data_url(
        image
    )

    context = (
        f"書籍名: {book_name}\n"
        f"PDFページ: {pdf_page}\n"
        f"推定書籍ページ: {logical_page}\n"
    )

    if tile_description:
        context += (
            f"画像領域: "
            f"{tile_description}\n"
        )

    context += (
        "\n以下はGoogle Vision OCRです。"
        "画像レイアウトを優先してください。\n"
        "--- OCR ---\n"
        f"{ocr_text}\n"
        "--- OCR END ---"
    )

    return model.invoke(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": context,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
    )


# ============================================================
# Merge TOC tiles/pages
# ============================================================

def merge_toc_extractions(
    extractions: list[dict],
) -> dict:
    entries = {}

    is_toc_page = False

    for extraction in extractions:
        if extraction.get(
            "is_toc_page"
        ):
            is_toc_page = True

        for entry in extraction.get(
            "entries",
            [],
        ):
            title = str(
                entry.get(
                    "title",
                    "",
                )
            ).strip()

            if not title:
                continue

            logical_page = entry.get(
                "logical_page"
            )

            if logical_page is not None:
                try:
                    logical_page = int(
                        logical_page
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    logical_page = None

            try:
                level = int(
                    entry.get(
                        "level",
                        1,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                level = 1

            level = max(
                1,
                min(
                    3,
                    level,
                ),
            )

            key = (
                normalize_term_key(
                    title
                ),
                logical_page,
            )

            if key not in entries:
                entries[
                    key
                ] = {
                    "title": title,
                    "logical_page": (
                        logical_page
                    ),
                    "level": level,
                }
            else:
                entries[
                    key
                ][
                    "level"
                ] = min(
                    entries[
                        key
                    ]["level"],
                    level,
                )

    return {
        "is_toc_page": (
            is_toc_page
        ),
        "entries": list(
            entries.values()
        ),
    }


# ============================================================
# Merge INDEX tiles/pages
# ============================================================

def merge_index_extractions(
    extractions: list[dict],
) -> dict:
    merged = {}

    is_index_page = False

    for extraction in extractions:
        if extraction.get(
            "is_index_page"
        ):
            is_index_page = True

        for section in extraction.get(
            "sections",
            [],
        ):
            index_type = (
                normalize_index_type(
                    str(
                        section.get(
                            "index_type",
                            "索引",
                        )
                    )
                )
            )

            for entry in section.get(
                "entries",
                [],
            ):
                term = str(
                    entry.get(
                        "term",
                        "",
                    )
                ).strip()

                if not term:
                    continue

                logical_pages = []

                for page in entry.get(
                    "logical_pages",
                    [],
                ):
                    try:
                        page = int(
                            page
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if page > 0:
                        logical_pages.append(
                            page
                        )

                logical_pages = sorted(
                    set(
                        logical_pages
                    )
                )

                if not logical_pages:
                    continue

                key = (
                    index_type,
                    normalize_term_key(
                        term
                    ),
                )

                if key not in merged:
                    merged[
                        key
                    ] = {
                        "term": term,
                        "index_type": (
                            index_type
                        ),
                        "logical_pages": set(),
                    }

                merged[
                    key
                ][
                    "logical_pages"
                ].update(
                    logical_pages
                )

    sections = defaultdict(
        list
    )

    for item in merged.values():
        sections[
            item["index_type"]
        ].append(
            {
                "term": item[
                    "term"
                ],
                "logical_pages": sorted(
                    item[
                        "logical_pages"
                    ]
                ),
            }
        )

    result_sections = []

    for index_type in sorted(
        sections
    ):
        entries = sections[
            index_type
        ]

        entries.sort(
            key=lambda item: (
                normalize_term_key(
                    item["term"]
                )
            )
        )

        result_sections.append(
            {
                "index_type": (
                    index_type
                ),
                "entries": entries,
            }
        )

    return {
        "is_index_page": (
            is_index_page
        ),
        "sections": (
            result_sections
        ),
    }


# ============================================================
# Generic page extraction
# ============================================================

def extract_page_with_fallback(
    *,
    kind: str,
    model,
    system_prompt: str,
    book_name: str,
    pdf_page: int,
    logical_page: int | None,
    image_path: Path,
    ocr_text: str,
) -> dict:
    cp_path = checkpoint_path(
        book_name,
        kind,
        pdf_page,
    )

    checkpoint = load_checkpoint(
        cp_path
    )

    if checkpoint is not None:
        print(
            f"    checkpoint "
            f"{kind} PDF={pdf_page}",
            flush=True,
        )

        return checkpoint[
            "extraction"
        ]

    image = resize_image(
        load_image(
            image_path
        ),
        NAVIGATION_IMAGE_MAX_SIDE,
    )

    started_at = time.monotonic()

    try:
        extraction_model = (
            invoke_image_model(
                model=model,
                system_prompt=(
                    system_prompt
                ),
                book_name=book_name,
                pdf_page=pdf_page,
                logical_page=(
                    logical_page
                ),
                image=image,
                ocr_text=ocr_text,
            )
        )

        extraction = (
            extraction_model.model_dump()
        )

        elapsed = (
            time.monotonic()
            - started_at
        )

        print(
            f"    completed "
            f"{kind} PDF={pdf_page} "
            f"in {elapsed:.1f}s",
            flush=True,
        )

        save_checkpoint(
            cp_path,
            kind=kind,
            pdf_page=pdf_page,
            mode="whole_page",
            extraction=(
                extraction
            ),
        )

        return extraction

    except Exception as exc:
        elapsed = (
            time.monotonic()
            - started_at
        )

        if not is_length_limit_error(
            exc
        ):
            raise

        print(
            f"    length limit "
            f"{kind} PDF={pdf_page} "
            f"after {elapsed:.1f}s "
            f"-> tile fallback",
            flush=True,
        )

    # --------------------------------------------------------
    # Tile fallback
    # --------------------------------------------------------

    tiles = create_tiles(
        image
    )

    tile_extractions = []

    for tile in tiles:
        row = tile[
            "row"
        ]

        col = tile[
            "col"
        ]

        tile_path = (
            tile_checkpoint_path(
                book_name=book_name,
                kind=kind,
                pdf_page=(
                    pdf_page
                ),
                row=row,
                col=col,
            )
        )

        tile_checkpoint = (
            load_checkpoint(
                tile_path
            )
        )

        if tile_checkpoint is not None:
            print(
                f"      checkpoint "
                f"tile {row},{col}",
                flush=True,
            )

            tile_extractions.append(
                tile_checkpoint[
                    "extraction"
                ]
            )

            continue

        tile_started = (
            time.monotonic()
        )

        tile_result = (
            invoke_image_model(
                model=model,
                system_prompt=(
                    system_prompt
                ),
                book_name=book_name,
                pdf_page=pdf_page,
                logical_page=(
                    logical_page
                ),
                image=tile[
                    "image"
                ],
                ocr_text=ocr_text,
                tile_description=(
                    f"tile row={row}, "
                    f"col={col}"
                ),
            )
        )

        tile_extraction = (
            tile_result.model_dump()
        )

        tile_elapsed = (
            time.monotonic()
            - tile_started
        )

        print(
            f"      completed "
            f"tile {row},{col} "
            f"in {tile_elapsed:.1f}s",
            flush=True,
        )

        save_checkpoint(
            tile_path,
            kind=kind,
            pdf_page=pdf_page,
            mode="tile",
            extraction=(
                tile_extraction
            ),
        )

        tile_extractions.append(
            tile_extraction
        )

    if kind == "toc":
        extraction = (
            merge_toc_extractions(
                tile_extractions
            )
        )

    elif kind == "index":
        extraction = (
            merge_index_extractions(
                tile_extractions
            )
        )

    else:
        raise ValueError(
            f"Unknown extraction kind: "
            f"{kind}"
        )

    save_checkpoint(
        cp_path,
        kind=kind,
        pdf_page=pdf_page,
        mode="tiles",
        extraction=extraction,
    )

    return extraction


# ============================================================
# Sequential page extraction
# ============================================================

def extract_page_sequences(
    *,
    kind: str,
    seed_pages: list[int],
    max_pages: int,
    valid_pdf_pages: set[int],
    page_by_pdf: dict[int, dict],
    pdf_to_logical: dict[int, int],
    image_lookup: dict[int, Path],
    model,
    system_prompt: str,
    book_name: str,
) -> tuple[
    dict[int, dict],
    list[dict],
]:
    """
    seedページから後続ページを順番に解析する。

    一度対象ページを確認した後、
    非対象ページになった時点でそのsequenceを終了する。
    """

    page_results = {}
    errors = []

    for seed in sorted(
        set(
            seed_pages
        )
    ):
        found_positive = False
        initial_misses = 0

        for offset in range(
            max_pages
        ):
            pdf_page = (
                seed
                + offset
            )

            if pdf_page not in (
                valid_pdf_pages
            ):
                break

            if pdf_page in page_results:
                extraction = (
                    page_results[
                        pdf_page
                    ]
                )

            else:
                page_data = (
                    page_by_pdf.get(
                        pdf_page
                    )
                )

                image_path = (
                    image_lookup.get(
                        pdf_page
                    )
                )

                if (
                    page_data is None
                    or image_path is None
                ):
                    errors.append(
                        {
                            "kind": kind,
                            "pdf_page": (
                                pdf_page
                            ),
                            "error": (
                                "page data or "
                                "image missing"
                            ),
                        }
                    )

                    continue

                logical_page = (
                    pdf_to_logical.get(
                        pdf_page
                    )
                )

                print(
                    f"  extracting "
                    f"{kind} "
                    f"PDF={pdf_page} "
                    f"logical={logical_page}",
                    flush=True,
                )

                try:
                    extraction = (
                        extract_page_with_fallback(
                            kind=kind,
                            model=model,
                            system_prompt=(
                                system_prompt
                            ),
                            book_name=(
                                book_name
                            ),
                            pdf_page=(
                                pdf_page
                            ),
                            logical_page=(
                                logical_page
                            ),
                            image_path=(
                                image_path
                            ),
                            ocr_text=(
                                page_data[
                                    "text"
                                ]
                            ),
                        )
                    )

                except Exception as exc:
                    print(
                        f"    ERROR "
                        f"{kind} "
                        f"PDF={pdf_page}: "
                        f"{type(exc).__name__}: "
                        f"{exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                    errors.append(
                        {
                            "kind": kind,
                            "pdf_page": (
                                pdf_page
                            ),
                            "error_type": (
                                type(
                                    exc
                                ).__name__
                            ),
                            "error": str(
                                exc
                            ),
                        }
                    )

                    # このページだけ失敗。
                    # 書籍全体は続行する。
                    continue

                page_results[
                    pdf_page
                ] = extraction

            if kind == "toc":
                positive = bool(
                    extraction.get(
                        "is_toc_page"
                    )
                )
            else:
                positive = bool(
                    extraction.get(
                        "is_index_page"
                    )
                )

            if positive:
                found_positive = True
                initial_misses = 0
                continue

            if found_positive:
                # 連続する目次/索引が終了。
                break

            initial_misses += 1

            # seedがOCR誤認だった場合、
            # 次ページまでは確認する。
            if initial_misses >= 2:
                break

    return (
        page_results,
        errors,
    )


# ============================================================
# Final TOC merge
# ============================================================

def build_final_toc(
    page_results: dict[int, dict],
    logical_to_pdf: dict[int, int],
) -> list[dict]:
    merged = {}

    for source_pdf_page in sorted(
        page_results
    ):
        extraction = page_results[
            source_pdf_page
        ]

        if not extraction.get(
            "is_toc_page"
        ):
            continue

        for entry in extraction.get(
            "entries",
            [],
        ):
            title = str(
                entry.get(
                    "title",
                    "",
                )
            ).strip()

            if not title:
                continue

            logical_page = entry.get(
                "logical_page"
            )

            if logical_page is not None:
                try:
                    logical_page = int(
                        logical_page
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    logical_page = None

            try:
                level = int(
                    entry.get(
                        "level",
                        1,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                level = 1

            level = max(
                1,
                min(
                    3,
                    level,
                ),
            )

            key = (
                normalize_term_key(
                    title
                ),
                logical_page,
            )

            if key not in merged:
                merged[
                    key
                ] = {
                    "title": title,
                    "logical_page": (
                        logical_page
                    ),
                    "pdf_page": (
                        logical_to_pdf.get(
                            logical_page
                        )
                        if logical_page
                        is not None
                        else None
                    ),
                    "level": level,
                    "source_toc_pages": set(),
                }

            merged[
                key
            ][
                "source_toc_pages"
            ].add(
                source_pdf_page
            )

    result = []

    for item in merged.values():
        item[
            "source_toc_pages"
        ] = sorted(
            item[
                "source_toc_pages"
            ]
        )

        result.append(
            item
        )

    return result


# ============================================================
# Final INDEX merge
# ============================================================

def build_final_index(
    page_results: dict[int, dict],
    logical_to_pdf: dict[int, int],
) -> list[dict]:
    merged = {}

    for source_pdf_page in sorted(
        page_results
    ):
        extraction = page_results[
            source_pdf_page
        ]

        if not extraction.get(
            "is_index_page"
        ):
            continue

        for section in extraction.get(
            "sections",
            [],
        ):
            index_type = (
                normalize_index_type(
                    str(
                        section.get(
                            "index_type",
                            "索引",
                        )
                    )
                )
            )

            for entry in section.get(
                "entries",
                [],
            ):
                term = str(
                    entry.get(
                        "term",
                        "",
                    )
                ).strip()

                if not term:
                    continue

                logical_pages = []

                for logical_page in (
                    entry.get(
                        "logical_pages",
                        [],
                    )
                ):
                    try:
                        logical_page = int(
                            logical_page
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if logical_page > 0:
                        logical_pages.append(
                            logical_page
                        )

                logical_pages = sorted(
                    set(
                        logical_pages
                    )
                )

                if not logical_pages:
                    continue

                key = (
                    index_type,
                    normalize_term_key(
                        term
                    ),
                )

                if key not in merged:
                    merged[
                        key
                    ] = {
                        "term": term,
                        "index_type": (
                            index_type
                        ),
                        "logical_pages": set(),
                        "pdf_pages": set(),
                        "source_index_pages": (
                            set()
                        ),
                    }

                target = merged[
                    key
                ]

                target[
                    "logical_pages"
                ].update(
                    logical_pages
                )

                target[
                    "source_index_pages"
                ].add(
                    source_pdf_page
                )

                for logical_page in (
                    logical_pages
                ):
                    pdf_page = (
                        logical_to_pdf.get(
                            logical_page
                        )
                    )

                    if pdf_page is not None:
                        target[
                            "pdf_pages"
                        ].add(
                            pdf_page
                        )

    result = []

    for item in merged.values():
        result.append(
            {
                "term": item[
                    "term"
                ],
                "index_type": item[
                    "index_type"
                ],
                "logical_pages": sorted(
                    item[
                        "logical_pages"
                    ]
                ),
                "pdf_pages": sorted(
                    item[
                        "pdf_pages"
                    ]
                ),
                "source_index_pages": sorted(
                    item[
                        "source_index_pages"
                    ]
                ),
            }
        )

    result.sort(
        key=lambda item: (
            item[
                "index_type"
            ],
            normalize_term_key(
                item[
                    "term"
                ]
            ),
        )
    )

    return result


# ============================================================
# Index seeds from parsed TOC
# ============================================================

def index_seed_pages_from_toc(
    toc_entries: list[dict],
) -> list[int]:
    seeds = []

    for entry in toc_entries:
        title = normalize_heading(
            str(
                entry.get(
                    "title",
                    "",
                )
            )
        )

        if not title.endswith(
            "索引"
        ):
            continue

        pdf_page = entry.get(
            "pdf_page"
        )

        if pdf_page is None:
            continue

        seeds.append(
            int(
                pdf_page
            )
        )

    return sorted(
        set(
            seeds
        )
    )


# ============================================================
# Completion handling
# ============================================================

TERMINAL_STATUSES = {
    "OK",
    "TOC_ONLY",
    "INDEX_ONLY",
    "NO_NAVIGATION",
}


def existing_completed_result(
    book_name: str,
):
    if FORCE_REBUILD:
        return None

    path = (
        NAVIGATION_DIR
        / f"{book_name}.json"
    )

    if not path.exists():
        return None

    try:
        data = load_json(
            path
        )
    except Exception:
        return None

    if data.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        return None

    if data.get(
        "status"
    ) not in TERMINAL_STATUSES:
        return None

    return data


# ============================================================
# Book processing
# ============================================================

def process_book(
    ocr_path: Path,
    toc_model,
    index_model,
) -> dict:
    book_name, pages = (
        load_ocr(
            ocr_path
        )
    )

    existing = (
        existing_completed_result(
            book_name
        )
    )

    if existing is not None:
        print(
            f"SKIP completed: "
            f"{book_name}",
            flush=True,
        )

        return existing

    page_map = load_page_map(
        book_name
    )

    (
        logical_to_pdf,
        pdf_to_logical,
    ) = build_page_lookup(
        page_map
    )

    image_lookup = (
        build_image_lookup(
            book_name
        )
    )

    page_by_pdf = {
        page[
            "pdf_page"
        ]: page
        for page in pages
    }

    valid_pdf_pages = set(
        page_by_pdf
    )

    # ========================================================
    # TOC
    # ========================================================

    toc_seeds = (
        find_toc_seed_pages(
            pages
        )
    )

    toc_page_results = {}
    toc_errors = []

    if toc_seeds:
        (
            toc_page_results,
            toc_errors,
        ) = extract_page_sequences(
            kind="toc",
            seed_pages=toc_seeds,
            max_pages=(
                TOC_MAX_PAGES
            ),
            valid_pdf_pages=(
                valid_pdf_pages
            ),
            page_by_pdf=(
                page_by_pdf
            ),
            pdf_to_logical=(
                pdf_to_logical
            ),
            image_lookup=(
                image_lookup
            ),
            model=toc_model,
            system_prompt=(
                TOC_SYSTEM_PROMPT
            ),
            book_name=(
                book_name
            ),
        )

    toc = build_final_toc(
        page_results=(
            toc_page_results
        ),
        logical_to_pdf=(
            logical_to_pdf
        ),
    )

    # ========================================================
    # INDEX seeds
    # ========================================================

    index_seeds = set(
        find_tail_index_seed_pages(
            pages
        )
    )

    index_seeds.update(
        index_seed_pages_from_toc(
            toc
        )
    )

    index_seeds.update(
        find_raw_toc_index_seed_pages(
            pages=pages,
            logical_to_pdf=(
                logical_to_pdf
            ),
        )
    )

    index_page_results = {}
    index_errors = []

    if index_seeds:
        (
            index_page_results,
            index_errors,
        ) = extract_page_sequences(
            kind="index",
            seed_pages=sorted(
                index_seeds
            ),
            max_pages=(
                INDEX_MAX_PAGES
            ),
            valid_pdf_pages=(
                valid_pdf_pages
            ),
            page_by_pdf=(
                page_by_pdf
            ),
            pdf_to_logical=(
                pdf_to_logical
            ),
            image_lookup=(
                image_lookup
            ),
            model=index_model,
            system_prompt=(
                INDEX_SYSTEM_PROMPT
            ),
            book_name=(
                book_name
            ),
        )

    index_entries = (
        build_final_index(
            page_results=(
                index_page_results
            ),
            logical_to_pdf=(
                logical_to_pdf
            ),
        )
    )

    # ========================================================
    # Status
    # ========================================================

    errors = (
        toc_errors
        + index_errors
    )

    has_toc = bool(
        toc
    )

    has_index = bool(
        index_entries
    )

    if errors:
        status = "WARNING"
        reason = (
            "navigation extracted "
            "with page-level errors"
        )

    elif (
        has_toc
        and has_index
    ):
        status = "OK"
        reason = (
            "toc and index extracted"
        )

    elif has_toc:
        status = "TOC_ONLY"
        reason = (
            "toc extracted; "
            "no index found"
        )

    elif has_index:
        status = "INDEX_ONLY"
        reason = (
            "index extracted; "
            "no toc found"
        )

    else:
        status = "NO_NAVIGATION"
        reason = (
            "no toc or index found"
        )

    result = {
        "schema_version": (
            SCHEMA_VERSION
        ),
        "book": book_name,
        "status": status,
        "reason": reason,

        "page_map": {
            "offset": (
                page_map.get(
                    "offset"
                )
            ),
            "status": (
                page_map.get(
                    "status"
                )
            ),
        },

        "toc_seed_pages": sorted(
            toc_seeds
        ),

        "index_seed_pages": sorted(
            index_seeds
        ),

        "toc_entry_count": len(
            toc
        ),

        "index_entry_count": len(
            index_entries
        ),

        "toc": toc,

        "index": (
            index_entries
        ),

        "errors": errors,
    }

    output_path = (
        NAVIGATION_DIR
        / f"{book_name}.json"
    )

    atomic_write_json(
        output_path,
        result,
    )

    return result


# ============================================================
# Main
# ============================================================

def main() -> int:
    for path, label in (
        (
            OCR_DIR,
            "OCR directory",
        ),
        (
            IMAGE_DIR,
            "Image directory",
        ),
        (
            PAGE_MAP_DIR,
            "Page map directory",
        ),
    ):
        if not path.exists():
            print(
                f"{label} not found: "
                f"{path}",
                file=sys.stderr,
            )

            return 1

    NAVIGATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    NAVIGATION_WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ocr_files = sorted(
        OCR_DIR.glob(
            "*.json"
        )
    )

    if not ocr_files:
        print(
            "No OCR JSON files.",
            file=sys.stderr,
        )

        return 1

    (
        toc_model,
        index_model,
    ) = build_models()

    results = []

    for ocr_path in ocr_files:
        print()
        print(
            "=" * 72
        )

        print(
            f"Processing: "
            f"{ocr_path.name}",
            flush=True,
        )

        try:
            result = process_book(
                ocr_path=ocr_path,
                toc_model=toc_model,
                index_model=(
                    index_model
                ),
            )

        except Exception as exc:
            result = {
                "schema_version": (
                    SCHEMA_VERSION
                ),
                "book": (
                    ocr_path.stem
                ),
                "status": "ERROR",
                "reason": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "toc": [],
                "index": [],
                "errors": [
                    {
                        "error_type": (
                            type(
                                exc
                            ).__name__
                        ),
                        "error": str(
                            exc
                        ),
                    }
                ],
            }

            # ERRORも診断用に保存する。
            output_path = (
                NAVIGATION_DIR
                / f"{ocr_path.stem}.json"
            )

            atomic_write_json(
                output_path,
                result,
            )

        results.append(
            result
        )

        print(
            f"Status: "
            f"{result['status']}",
            flush=True,
        )

        print(
            f"Reason: "
            f"{result['reason']}",
            flush=True,
        )

        print(
            f"TOC entries: "
            f"{len(result.get('toc', []))}",
            flush=True,
        )

        print(
            f"Index entries: "
            f"{len(result.get('index', []))}",
            flush=True,
        )

    summary = defaultdict(
        int
    )

    for result in results:
        summary[
            result["status"]
        ] += 1

    print()
    print(
        "=" * 72
    )

    print(
        "Book navigation summary"
    )

    print(
        "=" * 72
    )

    for status in (
        "OK",
        "TOC_ONLY",
        "INDEX_ONLY",
        "NO_NAVIGATION",
        "WARNING",
        "ERROR",
    ):
        print(
            f"{status:16s}: "
            f"{summary[status]}"
        )

    print(
        f"{'TOTAL':16s}: "
        f"{len(results)}"
    )

    print()

    for result in results:
        if result[
            "status"
        ] in {
            "WARNING",
            "ERROR",
        }:
            print(
                f"[{result['status']}] "
                f"{result['book']}: "
                f"{result['reason']}"
            )

    # ERRORだけ終了コード1。
    # WARNINGは部分データを保持しているので0。
    if summary[
        "ERROR"
    ]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )