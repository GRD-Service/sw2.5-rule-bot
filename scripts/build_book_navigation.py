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
from difflib import SequenceMatcher

from PIL import Image, ImageOps
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ============================================================
# Version
# ============================================================

SCHEMA_VERSION = 5

TOC_CHECKPOINT_VERSION = 3
INDEX_CHECKPOINT_VERSION = 2


# ============================================================
# Environment helpers
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
# Model settings
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

# root page -> 2x2 -> さらに2x2
#
# depth=0: ページ全体
# depth=1: 1/4
# depth=2: 1/16
NAVIGATION_MAX_TILE_DEPTH = int(
    os.getenv(
        "NAVIGATION_MAX_TILE_DEPTH",
        "2",
    )
)


# ============================================================
# TOC settings
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

TOC_SEED_COLLAPSE_DISTANCE = int(
    os.getenv(
        "TOC_SEED_COLLAPSE_DISTANCE",
        "3",
    )
)

# OCRで目次を発見できなかった場合、
# logical page 1 の何ページ前まで確認するか。
TOC_FALLBACK_BEFORE_LOGICAL_ONE = int(
    os.getenv(
        "TOC_FALLBACK_BEFORE_LOGICAL_ONE",
        "8",
    )
)

# logical page 1 の何ページ後まで確認するか。
TOC_FALLBACK_AFTER_LOGICAL_ONE = int(
    os.getenv(
        "TOC_FALLBACK_AFTER_LOGICAL_ONE",
        "20",
    )
)

# page mapにlogical 1がない場合のみ使う。
TOC_FALLBACK_SCAN_PAGES = int(
    os.getenv(
        "TOC_FALLBACK_SCAN_PAGES",
        "24",
    )
)


# ============================================================
# INDEX settings
# ============================================================

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


# ============================================================
# Rebuild
# ============================================================

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

TOC_HEADING_PATTERNS = (
    re.compile(
        r"目次",
        re.IGNORECASE,
    ),
    re.compile(
        r"もくじ",
        re.IGNORECASE,
    ),
    re.compile(
        r"contents?",
        re.IGNORECASE,
    ),
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


# ============================================================
# Structured Output - TOC
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
            "大項目=1、その下=2、さらに下=3。"
            "不明なら1。"
        ),
    )


class TocPageExtraction(BaseModel):
    is_toc_page: bool = Field(
        description=(
            "画像が目次ページ、"
            "目次の続き、または目次タイルならtrue。"
        )
    )

    entries: list[TocEntry] = Field(
        default_factory=list
    )


# ============================================================
# Structured Output - INDEX
# ============================================================

class IndexEntry(BaseModel):
    term: str = Field(
        description=(
            "索引に実際に印刷されている項目名。"
            "画像から確認できない文字は推測しない。"
        )
    )

    logical_pages: list[int] = Field(
        default_factory=list,
        description=(
            "索引項目に対応する書籍上のページ番号。"
            "PDFページ番号ではない。"
        ),
    )


class IndexSection(BaseModel):
    index_type: str = Field(
        description=(
            "画像に印刷されている索引分類名。"
            "未知の分類でも画像の表記をそのまま返す。"
        )
    )

    entries: list[IndexEntry] = Field(
        default_factory=list
    )


class IndexPageExtraction(BaseModel):
    is_index_page: bool = Field(
        description=(
            "画像が索引ページ、"
            "索引の続き、または索引タイルならtrue。"
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

入力画像には書籍の目次ページ、目次の続き、
またはその一部分が含まれている可能性があります。

規則:

1. 画像に実際に印刷されている目次項目だけを抽出する。
2. 項目名を知識や文脈から補完しない。
3. ページ番号は書籍に印刷されたページ番号であり、
   PDFページ番号ではない。
4. 複数カラム、縦書き、横書きがあり得る。
5. OCRの行順より画像上の配置を優先する。
6. ページ自体のノンブルを目次の参照ページと誤認しない。
7. 見た目の階層をlevel=1～3程度で表現する。
8. 判別できないページ番号はnullにする。
9. 判読不能な文字を推測しない。
10. タイル画像の場合も、目次項目が存在すれば
    is_toc_page=true とする。
11. タイル境界で項目名またはページ番号が欠ける場合は
    推測せず、その項目を省略する。
12. 通常本文、章扉、一覧表、データ表は目次ではない。
13. ページ上部や下部に章名や部名があるだけなら目次ではない。
14. 「第一部」「第二部」などがあるだけでは目次判定しない。
15. 項目名と参照ページ番号が多数並ぶ、
    明確な目次レイアウトを重視する。
"""


INDEX_SYSTEM_PROMPT = """
あなたは日本語TRPGルールブックの索引を構造化する担当です。

入力画像には書籍の索引ページ、索引の続き、
またはその一部分が含まれている可能性があります。

規則:

1. 画像に実際に印刷されている索引項目だけを抽出する。
2. 項目名を知識や文脈から補完・修正・推測しない。
3. ページ番号は書籍上のページ番号であり、
   PDFページ番号ではない。
4. 複数カラム、縦書き、横書きがあり得る。
5. OCRの行順より画像上の配置を優先する。
6. 「ア行」「カ行」「人物」「組織」「地名」など、
   単なる分類見出しは項目にしない。
7. ページ自体のノンブルを参照ページと誤認しない。
8. 索引分類には固定一覧はない。
9. 索引分類名は画像の表記をそのまま返す。
10. 単に「索引」ならindex_type="索引"とする。
11. タイル画像の場合も索引項目が存在すれば
    is_index_page=true とする。
12. タイル境界で項目名または番号が欠ける場合は
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
# OCR loading
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
# TOC OCR seed detection
# ============================================================

def is_toc_heading(
    line: str,
) -> bool:
    value = normalize_heading(
        line
    )

    if not value:
        return False

    # 本文中の「目次」への言及を除外。
    if len(
        value
    ) > 40:
        return False

    return any(
        pattern.search(
            value
        )
        for pattern in TOC_HEADING_PATTERNS
    )


def collapse_nearby_seeds(
    seeds: list[int],
    distance: int,
) -> list[int]:
    if not seeds:
        return []

    seeds = sorted(
        set(
            seeds
        )
    )

    result = [
        seeds[0]
    ]

    last_seen = seeds[0]

    for seed in seeds[1:]:
        if (
            seed
            - last_seen
            <= distance
        ):
            last_seen = seed
            continue

        result.append(
            seed
        )

        last_seen = seed

    return result


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

        lines = [
            line
            for line
            in page["text"].splitlines()
            if line.strip()
        ]

        if any(
            is_toc_heading(
                line
            )
            for line in lines
        ):
            seeds.append(
                pdf_page
            )

    return collapse_nearby_seeds(
        seeds,
        TOC_SEED_COLLAPSE_DISTANCE,
    )


# ============================================================
# TOC fallback candidates
# ============================================================

def build_toc_fallback_pages(
    pages: list[dict],
    pdf_to_logical: dict[int, int],
) -> list[int]:
    """
    全書籍に目次がある前提。

    logical page 1を基準にして、
    まずlogical 1そのものを確認。

    次に前付け方向、
    最後に本文側を確認する。

    これにより表紙等が多い書籍でも
    PDF先頭N枚だけで探索終了しない。
    """

    pdf_pages = sorted(
        page["pdf_page"]
        for page in pages
    )

    if not pdf_pages:
        return []

    logical_one_pdf = None

    for pdf_page in pdf_pages:
        if (
            pdf_to_logical.get(
                pdf_page
            ) == 1
        ):
            logical_one_pdf = (
                pdf_page
            )
            break

    if logical_one_pdf is None:
        return pdf_pages[
            :TOC_FALLBACK_SCAN_PAGES
        ]

    pdf_page_set = set(
        pdf_pages
    )

    candidates = []

    def add_candidate(
        pdf_page: int,
    ) -> None:
        if (
            pdf_page in pdf_page_set
            and pdf_page not in candidates
        ):
            candidates.append(
                pdf_page
            )

    # まず本文1ページ目。
    add_candidate(
        logical_one_pdf
    )

    # 次に前付け側。
    for delta in range(
        1,
        TOC_FALLBACK_BEFORE_LOGICAL_ONE
        + 1,
    ):
        add_candidate(
            logical_one_pdf
            - delta
        )

    # 最後に本文側。
    for delta in range(
        1,
        TOC_FALLBACK_AFTER_LOGICAL_ONE
        + 1,
    ):
        add_candidate(
            logical_one_pdf
            + delta
        )

    return candidates


# ============================================================
# INDEX seed detection
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

        match = (
            INDEX_HEADING_PATTERN.fullmatch(
                value
            )
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
            for line
            in page["text"].splitlines()
            if line.strip()
        ]

        for index, line in enumerate(
            lines
        ):
            match = (
                TOC_INDEX_PATTERN.fullmatch(
                    line
                )
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

            if (
                index + 1
                >= len(
                    lines
                )
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
# Image helpers
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
            "Book image directory "
            f"not found: {book_dir}"
        )

    result = {}

    for path in book_dir.iterdir():
        if not path.is_file():
            continue

        match = (
            PAGE_IMAGE_PATTERN.match(
                path.name
            )
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
        image = (
            ImageOps.exif_transpose(
                source
            )
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
                    "image": (
                        image.crop(
                            (
                                x0,
                                y0,
                                x1,
                                y1,
                            )
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

    return (
        base.with_structured_output(
            TocPageExtraction
        ),
        base.with_structured_output(
            IndexPageExtraction
        ),
    )


# ============================================================
# Error handling
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
        "completion_tokens=12000",
        "finish_reason='length'",
        'finish_reason": "length',
    )

    return any(
        pattern in message
        for pattern in patterns
    )


# ============================================================
# Checkpoint
# ============================================================

def checkpoint_version_for_kind(
    kind: str,
) -> int:
    if kind == "toc":
        return (
            TOC_CHECKPOINT_VERSION
        )

    if kind == "index":
        return (
            INDEX_CHECKPOINT_VERSION
        )

    raise ValueError(
        f"Unknown checkpoint kind: {kind}"
    )


def page_checkpoint_path(
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


def region_checkpoint_path(
    book_name: str,
    kind: str,
    pdf_page: int,
    region_id: str,
) -> Path:
    return (
        NAVIGATION_WORK_DIR
        / book_name
        / kind
        / f"P{pdf_page:05d}_regions"
        / f"{region_id}.json"
    )


def load_checkpoint(
    path: Path,
    kind: str,
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
        "kind"
    ) != kind:
        return None

    expected_version = (
        checkpoint_version_for_kind(
            kind
        )
    )

    if data.get(
        "checkpoint_version"
    ) != expected_version:
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
                checkpoint_version_for_kind(
                    kind
                )
            ),
            "kind": kind,
            "pdf_page": pdf_page,
            "mode": mode,
            "extraction": extraction,
        },
    )


# ============================================================
# LLM invocation
# ============================================================

def invoke_image_model(
    *,
    model,
    system_prompt: str,
    book_name: str,
    pdf_page: int,
    logical_page: int | None,
    image: Image.Image,
    ocr_text: str,
    region_description: str | None = None,
):
    data_url = image_to_data_url(
        image
    )

    context = (
        f"書籍名: {book_name}\n"
        f"PDFページ: {pdf_page}\n"
        f"推定書籍ページ: {logical_page}\n"
    )

    if region_description:
        context += (
            "画像領域: "
            f"{region_description}\n"
        )

    if ocr_text:
        context += (
            "\n以下はGoogle Vision OCRです。"
            "OCRの行順より画像レイアウトを"
            "優先してください。\n"
            "--- OCR ---\n"
            f"{ocr_text}\n"
            "--- OCR END ---"
        )

    else:
        context += (
            "\nこの画像は元ページの一部分です。"
            "この画像領域内に完全に表示されている"
            "項目だけを抽出してください。"
            "画像外の内容は推測しないでください。"
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
# Extraction merging
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
                    ][
                        "level"
                    ],
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

                pages = []

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
                        pages.append(
                            page
                        )

                pages = sorted(
                    set(
                        pages
                    )
                )

                if not pages:
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
                        "logical_pages": (
                            set()
                        ),
                    }

                merged[
                    key
                ][
                    "logical_pages"
                ].update(
                    pages
                )

    sections = defaultdict(
        list
    )

    for item in merged.values():
        sections[
            item[
                "index_type"
            ]
        ].append(
            {
                "term": (
                    item[
                        "term"
                    ]
                ),
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
                    item[
                        "term"
                    ]
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


def merge_extractions(
    kind: str,
    extractions: list[dict],
) -> dict:
    if kind == "toc":
        return merge_toc_extractions(
            extractions
        )

    if kind == "index":
        return merge_index_extractions(
            extractions
        )

    raise ValueError(
        f"Unknown kind: {kind}"
    )


# ============================================================
# Recursive region fallback
# ============================================================

def extract_region_recursive(
    *,
    kind: str,
    model,
    system_prompt: str,
    book_name: str,
    pdf_page: int,
    logical_page: int | None,
    image: Image.Image,
    depth: int,
    region_id: str,
) -> dict:
    """
    分割画像専用。

    OCR全文は渡さない。

    それでもlength limitになった場合、
    NAVIGATION_MAX_TILE_DEPTHまで再帰分割する。
    """

    cp_path = region_checkpoint_path(
        book_name=book_name,
        kind=kind,
        pdf_page=pdf_page,
        region_id=region_id,
    )

    checkpoint = load_checkpoint(
        cp_path,
        kind,
    )

    if checkpoint is not None:
        print(
            f"      checkpoint "
            f"region={region_id}",
            flush=True,
        )

        return checkpoint[
            "extraction"
        ]

    started = time.monotonic()

    try:
        result = invoke_image_model(
            model=model,
            system_prompt=(
                system_prompt
            ),
            book_name=(
                book_name
            ),
            pdf_page=pdf_page,
            logical_page=(
                logical_page
            ),
            image=image,
            ocr_text="",
            region_description=(
                f"{region_id}, "
                f"split-depth={depth}"
            ),
        )

        extraction = (
            result.model_dump()
        )

        elapsed = (
            time.monotonic()
            - started
        )

        print(
            f"      completed "
            f"region={region_id} "
            f"depth={depth} "
            f"in {elapsed:.1f}s",
            flush=True,
        )

        save_checkpoint(
            cp_path,
            kind=kind,
            pdf_page=pdf_page,
            mode=(
                f"region-depth-{depth}"
            ),
            extraction=(
                extraction
            ),
        )

        return extraction

    except Exception as exc:
        elapsed = (
            time.monotonic()
            - started
        )

        if not is_length_limit_error(
            exc
        ):
            raise

        if (
            depth
            >= NAVIGATION_MAX_TILE_DEPTH
        ):
            print(
                f"      ERROR length limit "
                f"region={region_id} "
                f"depth={depth} "
                f"after {elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

            raise

        print(
            f"      length limit "
            f"region={region_id} "
            f"depth={depth} "
            f"after {elapsed:.1f}s "
            "-> subdivide",
            flush=True,
        )

    # --------------------------------------------------------
    # Recursively subdivide
    # --------------------------------------------------------

    child_results = []

    for tile in create_tiles(
        image
    ):
        row = tile[
            "row"
        ]

        col = tile[
            "col"
        ]

        child_id = (
            f"{region_id}_"
            f"{row}_{col}"
        )

        child = extract_region_recursive(
            kind=kind,
            model=model,
            system_prompt=(
                system_prompt
            ),
            book_name=(
                book_name
            ),
            pdf_page=pdf_page,
            logical_page=(
                logical_page
            ),
            image=tile[
                "image"
            ],
            depth=(
                depth + 1
            ),
            region_id=(
                child_id
            ),
        )

        child_results.append(
            child
        )

    merged = merge_extractions(
        kind,
        child_results,
    )

    save_checkpoint(
        cp_path,
        kind=kind,
        pdf_page=pdf_page,
        mode=(
            f"subdivided-depth-{depth}"
        ),
        extraction=(
            merged
        ),
    )

    return merged


# ============================================================
# Whole page extraction
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
    cp_path = page_checkpoint_path(
        book_name,
        kind,
        pdf_page,
    )

    checkpoint = load_checkpoint(
        cp_path,
        kind,
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

    started = time.monotonic()

    try:
        result = invoke_image_model(
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

        extraction = (
            result.model_dump()
        )

        elapsed = (
            time.monotonic()
            - started
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
            - started
        )

        if not is_length_limit_error(
            exc
        ):
            raise

        print(
            f"    length limit "
            f"{kind} PDF={pdf_page} "
            f"after {elapsed:.1f}s "
            "-> recursive tile fallback",
            flush=True,
        )

    # --------------------------------------------------------
    # First-level tiles
    # --------------------------------------------------------

    tile_results = []

    for tile in create_tiles(
        image
    ):
        row = tile[
            "row"
        ]

        col = tile[
            "col"
        ]

        region_id = (
            f"d1_{row}_{col}"
        )

        result = (
            extract_region_recursive(
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
                image=tile[
                    "image"
                ],
                depth=1,
                region_id=(
                    region_id
                ),
            )
        )

        tile_results.append(
            result
        )

    extraction = merge_extractions(
        kind,
        tile_results,
    )

    save_checkpoint(
        cp_path,
        kind=kind,
        pdf_page=pdf_page,
        mode="recursive_tiles",
        extraction=(
            extraction
        ),
    )

    return extraction


# ============================================================
# TOC fallback
# ============================================================

def find_toc_seed_by_image_fallback(
    *,
    pages: list[dict],
    page_by_pdf: dict[int, dict],
    pdf_to_logical: dict[int, int],
    image_lookup: dict[int, Path],
    toc_model,
    book_name: str,
) -> tuple[
    list[int],
    dict[int, dict],
    list[dict],
]:
    candidates = build_toc_fallback_pages(
        pages=pages,
        pdf_to_logical=(
            pdf_to_logical
        ),
    )

    page_results = {}
    errors = []

    for pdf_page in candidates:
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
            continue

        logical_page = (
            pdf_to_logical.get(
                pdf_page
            )
        )

        print(
            "  fallback checking "
            f"toc PDF={pdf_page} "
            f"logical={logical_page}",
            flush=True,
        )

        try:
            extraction = (
                extract_page_with_fallback(
                    kind="toc",
                    model=toc_model,
                    system_prompt=(
                        TOC_SYSTEM_PROMPT
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
            errors.append(
                {
                    "kind": "toc",
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

            continue

        page_results[
            pdf_page
        ] = extraction

        if extraction.get(
            "is_toc_page"
        ):
            print(
                "  fallback found "
                f"toc seed PDF={pdf_page}",
                flush=True,
            )

            return (
                [pdf_page],
                page_results,
                errors,
            )

    return (
        [],
        page_results,
        errors,
    )


# ============================================================
# Sequential extraction
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
    initial_page_results: (
        dict[int, dict] | None
    ) = None,
) -> tuple[
    dict[int, dict],
    list[dict],
]:
    page_results = dict(
        initial_page_results
        or {}
    )

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

            elif kind == "index":
                positive = bool(
                    extraction.get(
                        "is_index_page"
                    )
                )

            else:
                raise ValueError(
                    f"Unknown kind: {kind}"
                )

            if positive:
                found_positive = True
                initial_misses = 0
                continue

            if found_positive:
                break

            initial_misses += 1

            if initial_misses >= 2:
                break

    return (
        page_results,
        errors,
    )


# ============================================================
# Final TOC
# ============================================================

def build_final_toc(
    page_results: dict[int, dict],
    logical_to_pdf: dict[int, int],
) -> list[dict]:
    merged = {}

    for source_pdf_page in sorted(
        page_results
    ):
        extraction = (
            page_results[
                source_pdf_page
            ]
        )

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

            logical_page = (
                entry.get(
                    "logical_page"
                )
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


            # ========================================================
            # navigation用途ではページ番号がない目次項目は使わない。
            #
            # ページヘッダー等をLLMが目次項目として拾ったケースも
            # ここで除外される。
            # ========================================================

            if logical_page is None:
                continue

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
                    "source_toc_pages": (
                        set()
                    ),
                }

            else:
                merged[
                    key
                ][
                    "level"
                ] = min(
                    merged[
                        key
                    ][
                        "level"
                    ],
                    level,
                )

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

    result.sort(
        key=lambda item: (
            item[
                "logical_page"
            ],
            item[
                "level"
            ],
        )
    )

    return result


# ============================================================
# Final INDEX
# ============================================================

def build_final_index(
    page_results: dict[int, dict],
    logical_to_pdf: dict[int, int],
    index_types_by_pdf: dict[
        int,
        list[str],
    ],
) -> list[dict]:
    merged = {}

    for source_pdf_page in sorted(
        page_results
    ):
        extraction = (
            page_results[
                source_pdf_page
            ]
        )

        if not extraction.get(
            "is_index_page"
        ):
            continue

        expected_types = (
            index_types_by_pdf.get(
                source_pdf_page,
                [],
            )
        )

        for section in extraction.get(
            "sections",
            [],
        ):
            detected_type = str(
                section.get(
                    "index_type",
                    "索引",
                )
            )

            (
                index_type,
                candidate_index_types,
            ) = correct_index_type_from_toc(
                detected_type=(
                    detected_type
                ),
                expected_types=(
                    expected_types
                ),
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

                        "candidate_index_types": (
                            set()
                        ),

                        "logical_pages": (
                            set()
                        ),

                        "pdf_pages": (
                            set()
                        ),

                        "source_index_pages": (
                            set()
                        ),
                    }

                target = merged[
                    key
                ]

                target[
                    "candidate_index_types"
                ].update(
                    candidate_index_types
                )

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
        output = {
            "term": (
                item[
                    "term"
                ]
            ),

            "index_type": (
                item[
                    "index_type"
                ]
            ),

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

        candidate_types = sorted(
            item[
                "candidate_index_types"
            ]
        )

        # 確定できなかった場合だけ出力する。
        if candidate_types:
            output[
                "candidate_index_types"
            ] = candidate_types

        result.append(
            output
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
# build_index_types_by_pdf
# ============================================================

def build_index_types_by_pdf(
    toc_entries: list[dict],
) -> dict[int, list[str]]:
    """
    目次から、

        PDF 162 -> ["索引", "一般索引"]
        PDF 163 -> [
            "戦闘特技索引",
            "魔法索引",
            "練技・呪歌/終律・騎芸・賦術索引",
        ]

    のような対応表を作る。

    単独の総称「索引」と、より具体的な分類が
    同一ページに存在する場合は具体的分類を優先する。
    """

    mapping: dict[int, list[str]] = defaultdict(
        list
    )

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

        try:
            pdf_page = int(
                pdf_page
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if title not in mapping[
            pdf_page
        ]:
            mapping[
                pdf_page
            ].append(
                title
            )

    result = {}

    for pdf_page, values in (
        mapping.items()
    ):
        specific = [
            value
            for value in values
            if value != "索引"
        ]

        # 「索引」と「一般索引」が同じページにあるなら、
        # 「索引」は総称とみなして落とす。
        if specific:
            result[
                pdf_page
            ] = specific

        else:
            result[
                pdf_page
            ] = values

    return result


# ============================================================
# correct_index_type_from_toc
# ============================================================

def correct_index_type_from_toc(
    detected_type: str,
    expected_types: list[str],
) -> tuple[
    str,
    list[str],
]:
    """
    LLMが返したindex_typeを、
    目次に記載された分類名を使って補正する。

    戻り値:
        (
            corrected_type,
            candidate_index_types,
        )

    candidate_index_typesは、
    一意に決められなかった場合だけ使用する。
    """

    detected = normalize_index_type(
        detected_type
    )

    expected_types = list(
        dict.fromkeys(
            normalize_heading(
                value
            )
            for value in expected_types
            if value
        )
    )

    if not expected_types:
        return (
            detected,
            [],
        )

    # --------------------------------------------------------
    # このPDFページに索引分類が1種類しかない。
    #
    # LLMの分類結果より目次を正とする。
    # --------------------------------------------------------

    if len(
        expected_types
    ) == 1:
        return (
            expected_types[0],
            [],
        )

    # --------------------------------------------------------
    # 完全一致
    # --------------------------------------------------------

    if detected in expected_types:
        return (
            detected,
            [],
        )

    # --------------------------------------------------------
    # 「索引」だけでは複数候補から決められない。
    # --------------------------------------------------------

    if detected == "索引":
        return (
            "索引",
            expected_types,
        )

    # --------------------------------------------------------
    # fuzzy match
    #
    # 例:
    #   闘特技索引
    #      ↓
    #   戦闘特技索引
    # --------------------------------------------------------

    best_type = None
    best_score = 0.0

    for expected in expected_types:
        score = SequenceMatcher(
            None,
            detected,
            expected,
        ).ratio()

        if score > best_score:
            best_score = score
            best_type = expected

    # 分類名は比較的短いため、
    # 0.55程度ならOCRの1～数文字欠落を吸収できる。
    if (
        best_type is not None
        and best_score >= 0.55
    ):
        return (
            best_type,
            [],
        )

    # 無理に分類しない。
    return (
        detected,
        expected_types,
    )


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

        try:
            pdf_page = int(
                pdf_page
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        seeds.append(
            pdf_page
        )

    return sorted(
        set(
            seeds
        )
    )


# ============================================================
# Resolved error cleanup
# ============================================================

def remove_resolved_errors(
    *,
    errors: list[dict],
    toc_page_results: dict[int, dict],
    index_page_results: dict[int, dict],
) -> list[dict]:
    """
    同じページが後のseedから再試行され成功した場合、
    以前のエラーを最終結果に残さない。
    """

    unresolved = []

    for error in errors:
        kind = error.get(
            "kind"
        )

        pdf_page = error.get(
            "pdf_page"
        )

        if kind == "toc":
            if pdf_page in toc_page_results:
                continue

        elif kind == "index":
            if pdf_page in index_page_results:
                continue

        unresolved.append(
            error
        )

    return unresolved


# ============================================================
# Completion handling
# ============================================================

# 全書籍にTOCがある前提なので、
# INDEX_ONLYやNO_NAVIGATIONは完成扱いにしない。
TERMINAL_STATUSES = {
    "OK",
    "TOC_ONLY",
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
            "SKIP completed: "
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
    # TOC - OCR heading first
    # ========================================================

    toc_seeds = (
        find_toc_seed_pages(
            pages
        )
    )

    toc_page_results = {}
    toc_errors = []

    toc_detection_method = (
        "ocr_heading"
        if toc_seeds
        else "image_fallback"
    )

    if toc_seeds:
        (
            toc_page_results,
            toc_errors,
        ) = extract_page_sequences(
            kind="toc",
            seed_pages=(
                toc_seeds
            ),
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

    # ========================================================
    # TOC - fallback if OCR seed missing OR OCR seed was false
    # ========================================================

    has_detected_toc_page = any(
        extraction.get(
            "is_toc_page"
        )
        for extraction
        in toc_page_results.values()
    )

    if not has_detected_toc_page:
        (
            fallback_seeds,
            fallback_results,
            fallback_errors,
        ) = find_toc_seed_by_image_fallback(
            pages=pages,
            page_by_pdf=(
                page_by_pdf
            ),
            pdf_to_logical=(
                pdf_to_logical
            ),
            image_lookup=(
                image_lookup
            ),
            toc_model=(
                toc_model
            ),
            book_name=(
                book_name
            ),
        )

        toc_page_results.update(
            fallback_results
        )

        toc_errors.extend(
            fallback_errors
        )

        if fallback_seeds:
            toc_detection_method = (
                "image_fallback"
            )

            toc_seeds = (
                fallback_seeds
            )

            (
                sequence_results,
                sequence_errors,
            ) = extract_page_sequences(
                kind="toc",
                seed_pages=(
                    fallback_seeds
                ),
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
                initial_page_results=(
                    toc_page_results
                ),
            )

            toc_page_results.update(
                sequence_results
            )

            toc_errors.extend(
                sequence_errors
            )

    toc = build_final_toc(
        page_results=(
            toc_page_results
        ),
        logical_to_pdf=(
            logical_to_pdf
        ),
    )

    index_types_by_pdf = (
        build_index_types_by_pdf(
            toc
        )
    )

    # ========================================================
    # INDEX
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
            index_types_by_pdf=(
                index_types_by_pdf
            ),
        )
    )

    # ========================================================
    # Resolve old page errors if a later retry succeeded
    # ========================================================

    errors = remove_resolved_errors(
        errors=(
            toc_errors
            + index_errors
        ),
        toc_page_results=(
            toc_page_results
        ),
        index_page_results=(
            index_page_results
        ),
    )

    # ========================================================
    # Status
    # ========================================================

    has_toc = bool(
        toc
    )

    has_index = bool(
        index_entries
    )

    # TOC is mandatory in this collection.
    if not has_toc:
        status = "WARNING"

        if has_index:
            reason = (
                "index extracted but "
                "toc was not detected"
            )
        else:
            reason = (
                "toc was not detected"
            )

    elif errors:
        status = "WARNING"
        reason = (
            "navigation extracted "
            "with unresolved page-level errors"
        )

    elif has_index:
        status = "OK"
        reason = (
            "toc and index extracted"
        )

    else:
        status = "TOC_ONLY"
        reason = (
            "toc extracted; "
            "no index found"
        )

    detected_toc_pages = sorted(
        pdf_page
        for pdf_page, extraction
        in toc_page_results.items()
        if extraction.get(
            "is_toc_page"
        )
    )

    detected_index_pages = sorted(
        pdf_page
        for pdf_page, extraction
        in index_page_results.items()
        if extraction.get(
            "is_index_page"
        )
    )

    # ========================================================
    # Output
    # ========================================================

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

        "toc_detection": {
            "method": (
                toc_detection_method
            ),
            "seed_pages": sorted(
                toc_seeds
            ),
            "detected_pages": (
                detected_toc_pages
            ),
        },

        "index_detection": {
            "seed_pages": sorted(
                index_seeds
            ),
            "detected_pages": (
                detected_index_pages
            ),

            "types_by_pdf": {
                str(
                    pdf_page
                ): index_types
                for (
                    pdf_page,
                    index_types
                ) in sorted(
                    index_types_by_pdf.items()
                )
            },
        },

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
            "Processing: "
            f"{ocr_path.name}",
            flush=True,
        )

        try:
            result = process_book(
                ocr_path=(
                    ocr_path
                ),
                toc_model=(
                    toc_model
                ),
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

            output_path = (
                NAVIGATION_DIR
                / (
                    f"{ocr_path.stem}"
                    ".json"
                )
            )

            atomic_write_json(
                output_path,
                result,
            )

        results.append(
            result
        )

        print(
            "Status: "
            f"{result['status']}",
            flush=True,
        )

        print(
            "Reason: "
            f"{result['reason']}",
            flush=True,
        )

        print(
            "TOC entries: "
            f"{len(result.get('toc', []))}",
            flush=True,
        )

        print(
            "Index entries: "
            f"{len(result.get('index', []))}",
            flush=True,
        )

        toc_detection = (
            result.get(
                "toc_detection",
                {},
            )
        )

        if toc_detection:
            print(
                "TOC method: "
                f"{toc_detection.get('method')}",
                flush=True,
            )

            print(
                "TOC seeds: "
                f"{toc_detection.get('seed_pages', [])}",
                flush=True,
            )

            print(
                "TOC pages: "
                f"{toc_detection.get('detected_pages', [])}",
                flush=True,
            )

        index_detection = (
            result.get(
                "index_detection",
                {},
            )
        )

        if index_detection:
            print(
                "Index seeds: "
                f"{index_detection.get('seed_pages', [])}",
                flush=True,
            )

            print(
                "Index pages: "
                f"{index_detection.get('detected_pages', [])}",
                flush=True,
            )

    # ========================================================
    # Summary
    # ========================================================

    summary = defaultdict(
        int
    )

    for result in results:
        summary[
            result[
                "status"
            ]
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

    if summary[
        "ERROR"
    ]:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )