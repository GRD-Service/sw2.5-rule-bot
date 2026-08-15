from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


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

BOOK_INDEX_DIR = Path(
    os.getenv(
        "BOOK_INDEX_DIR",
        "/data/metadata/indexes",
    )
)


# ============================================================
# Settings
# ============================================================

INDEX_EXTRACT_MODEL = os.getenv(
    "INDEX_EXTRACT_MODEL",
    "gpt-4.1-mini",
)

# 巻末側の何%から索引見出しを探索するか。
#
# 0.60なら全ページの60%以降を走査する。
INDEX_SCAN_START_RATIO = float(
    os.getenv(
        "INDEX_SCAN_START_RATIO",
        "0.60",
    )
)

# 発見した索引ページ候補の前後何ページまで
# 画像判定対象に含めるか。
INDEX_PAGE_EXPAND = int(
    os.getenv(
        "INDEX_PAGE_EXPAND",
        "1",
    )
)

# 目次探索対象。
# 通常、目次は書籍前半に存在するため、
# PDF全体の何%までを走査するか。
TOC_SCAN_END_RATIO = float(
    os.getenv(
        "TOC_SCAN_END_RATIO",
        "0.30",
    )
)

# 索引見出しとして許可する最大文字数。
MAX_INDEX_HEADING_LENGTH = int(
    os.getenv(
        "MAX_INDEX_HEADING_LENGTH",
        "30",
    )
)


# ============================================================
# Patterns
# ============================================================

# ------------------------------------------------------------
# 索引ページ上の見出し
#
# 例:
#
#   索引
#   一般索引
#   魔法索引
#   戦闘特技索引
#   魔物索引
#   アイテム索引
#   神格索引
#
# 固定分類は持たない。
# ------------------------------------------------------------

INDEX_PAGE_HEADING_PATTERN = re.compile(
    r"^(.{0,30}?索引)$"
)

# ------------------------------------------------------------
# 目次中の索引記載
#
# 例:
#
#   一般索引 ........ 476
#   魔法索引……477
#   索引 142
#
# ------------------------------------------------------------

TOC_INDEX_PATTERN = re.compile(
    r"^\s*(.{0,30}?索引)"
    r"[\s.．…⋯・･…\-―ー]*"
    r"([0-9]{1,4})\s*$"
)

# OCRで、
#
#   一般索引
#   476
#
# のように分離されるケースにも対応する。
PURE_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*([0-9]{1,4})\s*$"
)

# JPEG filename
PAGE_IMAGE_PATTERN = re.compile(
    r"^P0*(\d+)\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


# ============================================================
# Structured output schema
# ============================================================

class IndexEntry(BaseModel):
    term: str = Field(
        description=(
            "索引に実際に印刷されている索引項目名。"
            "画像から確認できない文字を推測しない。"
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
            "画像に印刷されている索引分類見出し。"
            "例: 索引、一般索引、魔法索引、"
            "戦闘特技索引、魔物索引、"
            "アイテム索引、神格索引など。"
            "未知の分類でも画像表記をそのまま返す。"
            "分類が単に『索引』なら『索引』と返す。"
        )
    )

    entries: list[IndexEntry] = Field(
        default_factory=list
    )


class IndexPageExtraction(BaseModel):
    is_index_page: bool = Field(
        description=(
            "この画像が索引ページ、"
            "または索引ページの続きならtrue。"
        )
    )

    sections: list[IndexSection] = Field(
        default_factory=list
    )


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

        pdf_to_logical[
            pdf_page
        ] = logical_page

        logical_to_pdf[
            logical_page
        ] = pdf_page

    return (
        logical_to_pdf,
        pdf_to_logical,
    )


# ============================================================
# Text normalization
# ============================================================

def normalize_heading(
    value: str,
) -> str:
    value = value.strip()

    # OCRで文字間に空白が入るケースを吸収する。
    value = re.sub(
        r"\s+",
        "",
        value,
    )

    # 見出し末尾などの装飾を除去。
    value = value.strip(
        "・･.…⋯-―ー"
    )

    return value


def normalize_term_key(
    term: str,
) -> str:
    value = term.strip()

    value = re.sub(
        r"\s+",
        "",
        value,
    )

    return value


def normalize_index_type(
    index_type: str,
) -> str:
    value = normalize_heading(
        index_type
    )

    if not value:
        return "索引"

    if value.endswith(
        "索引"
    ):
        return value

    # LLMが「一般」等だけ返した場合でも、
    # 意味を勝手に分類し直さず、
    # 最小限のfallbackとする。
    return "索引"


# ============================================================
# Index heading detection
# ============================================================

def detect_index_headings(
    text: str,
) -> list[str]:
    """
    索引ページそのものに存在する、

        索引
        一般索引
        魔法索引
        アイテム索引

    などの短い見出しを検出する。

    固定分類は持たない。
    """

    headings = []

    for raw_line in text.splitlines():
        normalized = normalize_heading(
            raw_line
        )

        if not normalized:
            continue

        if len(
            normalized
        ) > MAX_INDEX_HEADING_LENGTH:
            continue

        match = (
            INDEX_PAGE_HEADING_PATTERN.fullmatch(
                normalized
            )
        )

        if not match:
            continue

        heading = normalize_heading(
            match.group(1)
        )

        if not heading:
            continue

        headings.append(
            heading
        )

    return list(
        dict.fromkeys(
            headings
        )
    )


# ============================================================
# TOC detection
# ============================================================

def detect_toc_index_entries(
    text: str,
) -> list[dict]:
    """
    目次から、

        一般索引 ...... 476
        魔法索引 ...... 477
        索引 .......... 142

    のような記載を取得する。

    OCRで見出しと数字が別行になったケースも
    限定的に補完する。
    """

    raw_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    results = []

    # --------------------------------------------------------
    # 同一行
    # --------------------------------------------------------

    for line in raw_lines:
        match = TOC_INDEX_PATTERN.fullmatch(
            line
        )

        if not match:
            continue

        heading = normalize_heading(
            match.group(1)
        )

        if not heading.endswith(
            "索引"
        ):
            continue

        try:
            logical_page = int(
                match.group(2)
            )
        except ValueError:
            continue

        if logical_page <= 0:
            continue

        results.append(
            {
                "heading": heading,
                "logical_page": logical_page,
                "source": "same_line",
            }
        )

    # --------------------------------------------------------
    # OCRで、
    #
    #   魔法索引
    #   477
    #
    # と分離された場合
    # --------------------------------------------------------

    for index in range(
        len(raw_lines) - 1
    ):
        heading = normalize_heading(
            raw_lines[index]
        )

        if not heading:
            continue

        if len(
            heading
        ) > MAX_INDEX_HEADING_LENGTH:
            continue

        if not heading.endswith(
            "索引"
        ):
            continue

        number_match = (
            PURE_PAGE_NUMBER_PATTERN.fullmatch(
                raw_lines[
                    index + 1
                ]
            )
        )

        if not number_match:
            continue

        logical_page = int(
            number_match.group(1)
        )

        if logical_page <= 0:
            continue

        results.append(
            {
                "heading": heading,
                "logical_page": logical_page,
                "source": "next_line",
            }
        )

    # 重複除去
    unique = {}

    for item in results:
        key = (
            item["heading"],
            item["logical_page"],
        )

        if key not in unique:
            unique[
                key
            ] = item

    return list(
        unique.values()
    )


# ============================================================
# Candidate page discovery
# ============================================================

def find_toc_index_candidates(
    pages: list[dict],
    logical_to_pdf: dict[int, int],
) -> list[dict]:
    """
    書籍前半の目次から索引ページ番号を探し、
    page mapを使ってPDFページへ変換する。
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

    results = []

    for page in pages:
        pdf_page = page[
            "pdf_page"
        ]

        if pdf_page > scan_end:
            continue

        toc_entries = (
            detect_toc_index_entries(
                page["text"]
            )
        )

        for entry in toc_entries:
            logical_page = entry[
                "logical_page"
            ]

            target_pdf = (
                logical_to_pdf.get(
                    logical_page
                )
            )

            if target_pdf is None:
                continue

            results.append(
                {
                    "heading": entry[
                        "heading"
                    ],
                    "logical_page": (
                        logical_page
                    ),
                    "pdf_page": target_pdf,
                    "toc_pdf_page": (
                        pdf_page
                    ),
                    "source": "toc",
                }
            )

    unique = {}

    for item in results:
        key = (
            item["heading"],
            item["pdf_page"],
        )

        if key not in unique:
            unique[
                key
            ] = item

    return list(
        unique.values()
    )


def find_tail_index_candidates(
    pages: list[dict],
    pdf_to_logical: dict[int, int],
) -> list[dict]:
    """
    書籍後半から実際の「○○索引」見出しを探す。
    """

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

    results = []

    for page in pages:
        pdf_page = page[
            "pdf_page"
        ]

        if pdf_page < scan_start:
            continue

        headings = (
            detect_index_headings(
                page["text"]
            )
        )

        for heading in headings:
            results.append(
                {
                    "heading": heading,
                    "logical_page": (
                        pdf_to_logical.get(
                            pdf_page
                        )
                    ),
                    "pdf_page": pdf_page,
                    "source": "tail_heading",
                }
            )

    return results


def merge_index_candidates(
    toc_candidates: list[dict],
    tail_candidates: list[dict],
) -> list[dict]:
    """
    目次由来と巻末由来の候補を統合する。
    """

    merged = {}

    for item in (
        toc_candidates
        + tail_candidates
    ):
        pdf_page = item[
            "pdf_page"
        ]

        key = pdf_page

        if key not in merged:
            merged[key] = {
                "pdf_page": pdf_page,
                "headings": set(),
                "sources": set(),
                "logical_pages": set(),
            }

        target = merged[
            key
        ]

        heading = item.get(
            "heading"
        )

        if heading:
            target[
                "headings"
            ].add(
                heading
            )

        source = item.get(
            "source"
        )

        if source:
            target[
                "sources"
            ].add(
                source
            )

        logical_page = item.get(
            "logical_page"
        )

        if logical_page is not None:
            target[
                "logical_pages"
            ].add(
                logical_page
            )

    results = []

    for item in merged.values():
        results.append(
            {
                "pdf_page": item[
                    "pdf_page"
                ],
                "headings": sorted(
                    item[
                        "headings"
                    ]
                ),
                "sources": sorted(
                    item[
                        "sources"
                    ]
                ),
                "logical_pages": sorted(
                    item[
                        "logical_pages"
                    ]
                ),
            }
        )

    results.sort(
        key=lambda item: item[
            "pdf_page"
        ]
    )

    return results


def expand_candidate_pages(
    candidates: list[dict],
    valid_pdf_pages: set[int],
) -> list[int]:
    result = set()

    for candidate in candidates:
        seed_page = candidate[
            "pdf_page"
        ]

        for delta in range(
            -INDEX_PAGE_EXPAND,
            INDEX_PAGE_EXPAND + 1,
        ):
            pdf_page = (
                seed_page
                + delta
            )

            if pdf_page in valid_pdf_pages:
                result.add(
                    pdf_page
                )

    return sorted(
        result
    )


# ============================================================
# Image lookup
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

    lookup = {}

    for image_path in (
        book_dir.iterdir()
    ):
        if not image_path.is_file():
            continue

        match = PAGE_IMAGE_PATTERN.match(
            image_path.name
        )

        if not match:
            continue

        page = int(
            match.group(1)
        )

        lookup[
            page
        ] = image_path

    return lookup


def image_to_data_url(
    path: Path,
) -> str:
    mime_type, _ = (
        mimetypes.guess_type(
            str(path)
        )
    )

    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode(
        "ascii"
    )

    return (
        f"data:{mime_type};"
        f"base64,{encoded}"
    )


# ============================================================
# LLM
# ============================================================

def build_index_model():
    model = ChatOpenAI(
        model=INDEX_EXTRACT_MODEL,
        temperature=0,
    )

    return model.with_structured_output(
        IndexPageExtraction
    )


INDEX_SYSTEM_PROMPT = """
あなたは日本語TRPGルールブックの索引ページを
高精度に構造化する担当です。

入力される画像は、書籍の索引ページ候補です。
Google Vision OCRテキストも補助情報として渡されます。

必ず以下の規則を守ってください。

1. 画像に実際に印刷されている索引項目だけを抽出してください。

2. 項目名を一般知識や文脈から補完、修正、推測してはいけません。

3. ページ番号も画像に実際に印刷されている数字だけを使用してください。

4. ページ番号は書籍本体に印刷されているページ番号です。
   PDF上の通しページ番号ではありません。

5. 索引は複数カラムになっていることがあります。
   項目とページ番号の対応は画像上のレイアウトを優先してください。

6. OCRテキストの行順は、
   複数カラムや縦書きのため崩れていることがあります。
   OCRだけを信用して項目と番号を対応させないでください。

7. 「ア行」「カ行」「人物」「組織」「地名」などの
   分類見出しそのものは索引項目として登録しないでください。

8. ページ下部や上部に印刷されている
   ページ自体のノンブルを、
   索引項目の参照ページと誤認しないでください。

9. 索引分類には固定された一覧はありません。
   「一般索引」「魔法索引」「戦闘特技索引」
   「魔物索引」「アイテム索引」「神格索引」
   その他の未知の分類が存在する可能性があります。

10. 索引分類見出しは、
    画像に書かれている表記を可能な限りそのまま返してください。

11. 分類見出しが単純に「索引」の場合は
    index_type="索引" としてください。

12. 複数種類の索引が同じページに存在する場合は、
    sectionsを分けてください。

13. このページが索引ページでも、
    索引ページの続きでもない場合は
    is_index_page=false としてください。

14. 判読できない項目や番号は推測せず、
    その項目を省略してください。

15. 正確性を最優先してください。
"""


def extract_index_page(
    model,
    book_name: str,
    pdf_page: int,
    logical_page: int | None,
    image_path: Path,
    ocr_text: str,
) -> IndexPageExtraction:
    data_url = image_to_data_url(
        image_path
    )

    user_text = (
        f"書籍名: {book_name}\n"
        f"PDFページ: {pdf_page}\n"
        f"推定書籍ページ: {logical_page}\n\n"
        "以下はGoogle Vision OCRの結果です。\n"
        "複数カラムや縦書きにより、"
        "行順が崩れている可能性があります。\n\n"
        "--- OCR ---\n"
        f"{ocr_text}\n"
        "--- OCR END ---"
    )

    result = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    INDEX_SYSTEM_PROMPT
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            },
        ]
    )

    return result


# ============================================================
# Merge extracted entries
# ============================================================

def merge_entries(
    extracted_pages: list[dict],
    logical_to_pdf: dict[int, int],
) -> list[dict]:
    merged = {}

    for page_result in extracted_pages:

        for section in page_result.get(
            "sections",
            [],
        ):
            index_type = (
                normalize_index_type(
                    section[
                        "index_type"
                    ]
                )
            )

            for entry in section.get(
                "entries",
                [],
            ):
                term = str(
                    entry[
                        "term"
                    ]
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

                    if page <= 0:
                        continue

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

                source_pdf_page = (
                    page_result.get(
                        "pdf_page"
                    )
                )

                if source_pdf_page is not None:
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

    results = []

    for item in merged.values():
        results.append(
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

    results.sort(
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

    return results


# ============================================================
# Process book
# ============================================================

def process_book(
    ocr_path: Path,
    model,
) -> dict:
    book_name, pages = load_ocr(
        ocr_path
    )

    page_map = load_page_map(
        book_name
    )

    (
        logical_to_pdf,
        pdf_to_logical,
    ) = build_page_lookup(
        page_map
    )

    valid_pdf_pages = {
        item[
            "pdf_page"
        ]
        for item in pages
    }

    # --------------------------------------------------------
    # 目次由来候補
    # --------------------------------------------------------

    toc_candidates = (
        find_toc_index_candidates(
            pages=pages,
            logical_to_pdf=(
                logical_to_pdf
            ),
        )
    )

    # --------------------------------------------------------
    # 巻末見出し由来候補
    # --------------------------------------------------------

    tail_candidates = (
        find_tail_index_candidates(
            pages=pages,
            pdf_to_logical=(
                pdf_to_logical
            ),
        )
    )

    # --------------------------------------------------------
    # 統合
    # --------------------------------------------------------

    index_candidates = (
        merge_index_candidates(
            toc_candidates=(
                toc_candidates
            ),
            tail_candidates=(
                tail_candidates
            ),
        )
    )

    if not index_candidates:
        result = {
            "book": book_name,
            "status": "NO_INDEX",
            "reason": (
                "no index candidate was found "
                "from TOC or tail headings"
            ),
            "toc_candidates": [],
            "tail_candidates": [],
            "index_candidates": [],
            "index_pages": [],
            "entry_count": 0,
            "entries": [],
        }

        output_path = (
            BOOK_INDEX_DIR
            / f"{book_name}.json"
        )

        atomic_write_json(
            output_path,
            result,
        )

        return result

    candidate_pages = (
        expand_candidate_pages(
            candidates=(
                index_candidates
            ),
            valid_pdf_pages=(
                valid_pdf_pages
            ),
        )
    )

    image_lookup = (
        build_image_lookup(
            book_name
        )
    )

    page_by_number = {
        item[
            "pdf_page"
        ]: item
        for item in pages
    }

    extracted_pages = []
    actual_index_pages = []

    for pdf_page in candidate_pages:
        page_data = (
            page_by_number.get(
                pdf_page
            )
        )

        if page_data is None:
            continue

        image_path = (
            image_lookup.get(
                pdf_page
            )
        )

        if image_path is None:
            print(
                (
                    "WARNING: image not found: "
                    f"{book_name} "
                    f"PDF {pdf_page}"
                ),
                file=sys.stderr,
            )

            continue

        logical_page = (
            pdf_to_logical.get(
                pdf_page
            )
        )

        print(
            (
                f"  extracting "
                f"PDF={pdf_page} "
                f"logical={logical_page}"
            )
        )

        extraction = (
            extract_index_page(
                model=model,
                book_name=book_name,
                pdf_page=pdf_page,
                logical_page=(
                    logical_page
                ),
                image_path=image_path,
                ocr_text=page_data[
                    "text"
                ],
            )
        )

        if not extraction.is_index_page:
            continue

        sections = []

        for section in (
            extraction.sections
        ):
            entries = []

            for entry in (
                section.entries
            ):
                entries.append(
                    {
                        "term": (
                            entry.term
                        ),
                        "logical_pages": (
                            entry.logical_pages
                        ),
                    }
                )

            sections.append(
                {
                    "index_type": (
                        normalize_index_type(
                            section.index_type
                        )
                    ),
                    "entries": entries,
                }
            )

        extracted_pages.append(
            {
                "pdf_page": (
                    pdf_page
                ),
                "logical_page": (
                    logical_page
                ),
                "sections": sections,
            }
        )

        candidate_info = next(
            (
                item
                for item in index_candidates
                if item[
                    "pdf_page"
                ] == pdf_page
            ),
            None,
        )

        actual_index_pages.append(
            {
                "pdf_page": (
                    pdf_page
                ),
                "logical_page": (
                    logical_page
                ),
                "candidate_headings": (
                    candidate_info[
                        "headings"
                    ]
                    if candidate_info
                    else []
                ),
                "candidate_sources": (
                    candidate_info[
                        "sources"
                    ]
                    if candidate_info
                    else []
                ),
            }
        )

    entries = merge_entries(
        extracted_pages=(
            extracted_pages
        ),
        logical_to_pdf=(
            logical_to_pdf
        ),
    )

    if not actual_index_pages:
        status = "NO_INDEX"

        reason = (
            "index candidates existed but "
            "image analysis found no index pages"
        )

    elif not entries:
        status = "WARNING"

        reason = (
            "index pages detected but "
            "no entries could be extracted"
        )

    else:
        status = "OK"

        reason = (
            "index extracted"
        )

    result = {
        "book": book_name,
        "status": status,
        "reason": reason,

        "toc_candidates": (
            toc_candidates
        ),

        "tail_candidates": (
            tail_candidates
        ),

        "index_candidates": (
            index_candidates
        ),

        "index_pages": (
            actual_index_pages
        ),

        "entry_count": len(
            entries
        ),

        "entries": entries,
    }

    output_path = (
        BOOK_INDEX_DIR
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
    if not OCR_DIR.exists():
        print(
            (
                "OCR directory not found: "
                f"{OCR_DIR}"
            ),
            file=sys.stderr,
        )

        return 1

    if not PAGE_MAP_DIR.exists():
        print(
            (
                "Page map directory not found: "
                f"{PAGE_MAP_DIR}"
            ),
            file=sys.stderr,
        )

        return 1

    if not IMAGE_DIR.exists():
        print(
            (
                "Image directory not found: "
                f"{IMAGE_DIR}"
            ),
            file=sys.stderr,
        )

        return 1

    BOOK_INDEX_DIR.mkdir(
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

    model = build_index_model()

    results = []

    for ocr_path in ocr_files:
        print()
        print(
            "=" * 72
        )

        print(
            f"Processing: "
            f"{ocr_path.name}"
        )

        try:
            result = process_book(
                ocr_path,
                model,
            )

        except Exception as exc:
            result = {
                "book": (
                    ocr_path.stem
                ),
                "status": "ERROR",
                "reason": str(
                    exc
                ),
                "entry_count": 0,
                "entries": [],
            }

        results.append(
            result
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Reason: "
            f"{result['reason']}"
        )

        print(
            f"Entries: "
            f"{result.get('entry_count', 0)}"
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
        "Book index summary"
    )

    print(
        "=" * 72
    )

    for status in (
        "OK",
        "NO_INDEX",
        "WARNING",
        "ERROR",
    ):
        print(
            f"{status:12s}: "
            f"{summary[status]}"
        )

    print(
        f"{'TOTAL':12s}: "
        f"{len(results)}"
    )

    print()

    for result in results:
        if result[
            "status"
        ] in (
            "NO_INDEX",
            "WARNING",
            "ERROR",
        ):
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