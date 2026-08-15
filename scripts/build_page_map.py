from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


OCR_DIR = Path(
    os.getenv(
        "OCR_DIR",
        "/data/ocr",
    )
)

PAGE_MAP_DIR = Path(
    os.getenv(
        "PAGE_MAP_DIR",
        "/data/metadata/page_maps",
    )
)

PAGE_OVERRIDE_DIR = Path(
    os.getenv(
        "PAGE_OVERRIDE_DIR",
        "/data/metadata/page_overrides",
    )
)


# ============================================================
# 自動判定パラメータ
# ============================================================

TAIL_LINES = int(
    os.getenv(
        "PAGE_MAP_TAIL_LINES",
        "10",
    )
)

MIN_LOGICAL_PAGE = 1

MAX_LOGICAL_PAGE = int(
    os.getenv(
        "PAGE_MAP_MAX_LOGICAL_PAGE",
        "1000",
    )
)

MIN_MATCH_COUNT = int(
    os.getenv(
        "PAGE_MAP_MIN_MATCH_COUNT",
        "10",
    )
)

HIGH_CONFIDENCE_RATIO = float(
    os.getenv(
        "PAGE_MAP_HIGH_CONFIDENCE_RATIO",
        "0.70",
    )
)

MEDIUM_CONFIDENCE_RATIO = float(
    os.getenv(
        "PAGE_MAP_MEDIUM_CONFIDENCE_RATIO",
        "0.40",
    )
)

MIN_WINNER_MARGIN = float(
    os.getenv(
        "PAGE_MAP_MIN_WINNER_MARGIN",
        "3.0",
    )
)


PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*([0-9]{1,4})\s*$"
)


# ============================================================
# JSON
# ============================================================

def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


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
# OCR JSON
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
            f"OCR root must be list: {path}"
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
            raise ValueError(
                f"Invalid OCR entry: {path}"
            )

        book = entry.get(
            "book"
        )

        if not book:
            raise ValueError(
                f"No book field: {path}"
            )

        if book_name is None:
            book_name = book
        elif book != book_name:
            raise ValueError(
                f"Multiple books in OCR JSON: "
                f"{path}"
            )

        try:
            pdf_page = int(
                entry["page"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Invalid page: {entry}"
            ) from exc

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

    page_numbers = [
        item["pdf_page"]
        for item in pages
    ]

    if len(page_numbers) != len(
        set(page_numbers)
    ):
        raise ValueError(
            f"Duplicate PDF page: {path}"
        )

    return (
        book_name,
        pages,
    )


# ============================================================
# ページ番号候補
# ============================================================

def extract_page_number_candidates(
    text: str,
) -> list[int]:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return []

    tail_lines = lines[
        -TAIL_LINES:
    ]

    candidates = []

    for line in tail_lines:
        match = PAGE_NUMBER_PATTERN.match(
            line
        )

        if not match:
            continue

        logical_page = int(
            match.group(1)
        )

        if not (
            MIN_LOGICAL_PAGE
            <= logical_page
            <= MAX_LOGICAL_PAGE
        ):
            continue

        candidates.append(
            logical_page
        )

    return list(
        dict.fromkeys(
            candidates
        )
    )


# ============================================================
# offset 投票
# ============================================================

def build_offset_votes(
    pages: list[dict],
) -> tuple[
    Counter,
    dict[int, list[int]],
]:
    votes = Counter()

    candidates_by_pdf = {}

    for page in pages:
        pdf_page = page[
            "pdf_page"
        ]

        candidates = (
            extract_page_number_candidates(
                page["text"]
            )
        )

        candidates_by_pdf[
            pdf_page
        ] = candidates

        # 同じPDFページから同じoffsetへ
        # 二重投票しない。
        offsets_on_page = set()

        for logical_page in candidates:
            offset = (
                pdf_page
                - logical_page
            )

            # 印刷ページがPDFページより
            # 大きくなるケースは通常想定しない。
            if offset < 0:
                continue

            offsets_on_page.add(
                offset
            )

        for offset in offsets_on_page:
            votes[
                offset
            ] += 1

    return (
        votes,
        candidates_by_pdf,
    )


# ============================================================
# offset 評価
# ============================================================

def longest_consecutive_run(
    matching_pdf_pages: list[int],
) -> int:
    if not matching_pdf_pages:
        return 0

    pages = sorted(
        matching_pdf_pages
    )

    longest = 1
    current = 1

    for previous, current_page in zip(
        pages,
        pages[1:],
    ):
        if current_page == previous + 1:
            current += 1
        else:
            longest = max(
                longest,
                current,
            )
            current = 1

    return max(
        longest,
        current,
    )


def evaluate_offset(
    pages: list[dict],
    candidates_by_pdf: dict[int, list[int]],
    offset: int,
) -> dict:
    matching_pdf_pages = []

    candidate_pages = 0

    for page in pages:
        pdf_page = page[
            "pdf_page"
        ]

        candidates = candidates_by_pdf.get(
            pdf_page,
            [],
        )

        if candidates:
            candidate_pages += 1

        expected_logical = (
            pdf_page
            - offset
        )

        if expected_logical in candidates:
            matching_pdf_pages.append(
                pdf_page
            )

    match_count = len(
        matching_pdf_pages
    )

    if candidate_pages:
        match_ratio = (
            match_count
            / candidate_pages
        )
    else:
        match_ratio = 0.0

    longest_run = longest_consecutive_run(
        matching_pdf_pages
    )

    return {
        "offset": offset,
        "match_count": match_count,
        "candidate_page_count": (
            candidate_pages
        ),
        "match_ratio": match_ratio,
        "longest_consecutive_run": (
            longest_run
        ),
        "matching_pdf_pages": (
            matching_pdf_pages
        ),
    }


def detect_best_offset(
    pages: list[dict],
) -> tuple[
    dict | None,
    list[dict],
]:
    votes, candidates_by_pdf = (
        build_offset_votes(
            pages
        )
    )

    if not votes:
        return (
            None,
            [],
        )

    ranked_offsets = []

    for offset, _vote_count in (
        votes.most_common()
    ):
        evaluation = evaluate_offset(
            pages=pages,
            candidates_by_pdf=(
                candidates_by_pdf
            ),
            offset=offset,
        )

        ranked_offsets.append(
            evaluation
        )

    ranked_offsets.sort(
        key=lambda item: (
            item["match_count"],
            item[
                "longest_consecutive_run"
            ],
            item["match_ratio"],
        ),
        reverse=True,
    )

    return (
        ranked_offsets[0],
        ranked_offsets,
    )


# ============================================================
# confidence
# ============================================================

def determine_confidence(
    best: dict | None,
    ranked: list[dict],
) -> tuple[
    str,
    str,
]:
    if best is None:
        return (
            "MANUAL_REQUIRED",
            "no page-number candidates",
        )

    match_count = best[
        "match_count"
    ]

    match_ratio = best[
        "match_ratio"
    ]

    longest_run = best[
        "longest_consecutive_run"
    ]

    if match_count < MIN_MATCH_COUNT:
        return (
            "MANUAL_REQUIRED",
            (
                "too few matches: "
                f"{match_count}"
            ),
        )

    if len(ranked) >= 2:
        second_count = ranked[
            1
        ][
            "match_count"
        ]

        if second_count > 0:
            winner_margin = (
                match_count
                / second_count
            )
        else:
            winner_margin = float(
                "inf"
            )
    else:
        winner_margin = float(
            "inf"
        )

    if winner_margin < MIN_WINNER_MARGIN:
        return (
            "WARNING",
            (
                "offset candidates are "
                "too close"
            ),
        )

    if (
        match_ratio
        >= HIGH_CONFIDENCE_RATIO
        or longest_run >= 20
    ):
        return (
            "AUTO_OK",
            "high confidence",
        )

    if (
        match_ratio
        >= MEDIUM_CONFIDENCE_RATIO
        or longest_run >= 10
    ):
        return (
            "WARNING",
            "medium confidence",
        )

    return (
        "MANUAL_REQUIRED",
        "low confidence",
    )


# ============================================================
# 手動 override
# ============================================================

def load_override(
    book_name: str,
) -> dict | None:
    override_path = (
        PAGE_OVERRIDE_DIR
        / f"{book_name}.json"
    )

    if not override_path.exists():
        return None

    data = load_json(
        override_path
    )

    if not isinstance(
        data,
        dict,
    ):
        raise ValueError(
            f"Invalid override: "
            f"{override_path}"
        )

    if data.get(
        "book"
    ) != book_name:
        raise ValueError(
            f"Override book mismatch: "
            f"{override_path}"
        )

    if "offset" not in data:
        raise ValueError(
            f"No offset in override: "
            f"{override_path}"
        )

    try:
        offset = int(
            data["offset"]
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"Invalid override offset: "
            f"{override_path}"
        ) from exc

    if offset < 0:
        raise ValueError(
            f"Negative override offset: "
            f"{offset}"
        )

    return {
        "offset": offset,
        "path": str(
            override_path
        ),
    }


# ============================================================
# page map
# ============================================================

def build_page_map(
    book_name: str,
    pages: list[dict],
    offset: int,
    detection_source: str,
    status: str,
    reason: str,
    evaluation: dict | None,
) -> dict:
    mappings = []

    for page in pages:
        pdf_page = page[
            "pdf_page"
        ]

        logical_page = (
            pdf_page
            - offset
        )

        if logical_page < 1:
            logical_page = None

        mappings.append(
            {
                "pdf_page": pdf_page,
                "logical_page": (
                    logical_page
                ),
                "source": (
                    detection_source
                ),
            }
        )

    return {
        "book": book_name,
        "status": status,
        "reason": reason,
        "offset": offset,
        "detection_source": (
            detection_source
        ),
        "statistics": (
            {
                "match_count": (
                    evaluation[
                        "match_count"
                    ]
                    if evaluation
                    else None
                ),
                "candidate_page_count": (
                    evaluation[
                        "candidate_page_count"
                    ]
                    if evaluation
                    else None
                ),
                "match_ratio": (
                    evaluation[
                        "match_ratio"
                    ]
                    if evaluation
                    else None
                ),
                "longest_consecutive_run": (
                    evaluation[
                        "longest_consecutive_run"
                    ]
                    if evaluation
                    else None
                ),
            }
        ),
        "mappings": mappings,
    }


# ============================================================
# 書籍処理
# ============================================================

def process_book(
    ocr_path: Path,
) -> dict:
    book_name, pages = load_ocr(
        ocr_path
    )

    override = load_override(
        book_name
    )

    best, ranked = detect_best_offset(
        pages
    )

    if override is not None:
        offset = override[
            "offset"
        ]

        evaluation = None

        # overrideしたoffsetについても
        # OCR上の一致率を計算して記録する。
        _votes, candidates_by_pdf = (
            build_offset_votes(
                pages
            )
        )

        evaluation = evaluate_offset(
            pages=pages,
            candidates_by_pdf=(
                candidates_by_pdf
            ),
            offset=offset,
        )

        status = "MANUAL_OVERRIDE"
        reason = (
            f"manual offset override: "
            f"{override['path']}"
        )

        source = "manual_override"

    else:
        status, reason = (
            determine_confidence(
                best,
                ranked,
            )
        )

        if best is None:
            return {
                "book": book_name,
                "status": status,
                "reason": reason,
                "offset": None,
            }

        offset = best[
            "offset"
        ]

        evaluation = best
        source = "auto"

    page_map = build_page_map(
        book_name=book_name,
        pages=pages,
        offset=offset,
        detection_source=source,
        status=status,
        reason=reason,
        evaluation=evaluation,
    )

    output_path = (
        PAGE_MAP_DIR
        / f"{book_name}.json"
    )

    atomic_write_json(
        output_path,
        page_map,
    )

    return page_map


# ============================================================
# 表示
# ============================================================

def print_book_result(
    result: dict,
) -> None:
    book = result[
        "book"
    ]

    status = result[
        "status"
    ]

    offset = result.get(
        "offset"
    )

    print()
    print(
        "=" * 72
    )
    print(
        f"Book:   {book}"
    )
    print(
        f"Status: {status}"
    )

    if offset is not None:
        print(
            f"Offset: {offset}"
        )

    stats = result.get(
        "statistics"
    )

    if stats:
        print(
            f"Matches: "
            f"{stats['match_count']}"
            f"/"
            f"{stats['candidate_page_count']}"
        )

        ratio = stats.get(
            "match_ratio"
        )

        if ratio is not None:
            print(
                f"Match ratio: "
                f"{ratio:.3f}"
            )

        print(
            "Longest consecutive run: "
            f"{stats['longest_consecutive_run']}"
        )

    print(
        f"Reason: {result['reason']}"
    )


# ============================================================
# main
# ============================================================

def main() -> int:
    if not OCR_DIR.exists():
        print(
            f"OCR directory not found: "
            f"{OCR_DIR}",
            file=sys.stderr,
        )
        return 1

    PAGE_MAP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PAGE_OVERRIDE_DIR.mkdir(
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
            f"No OCR JSON files: "
            f"{OCR_DIR}",
            file=sys.stderr,
        )
        return 1

    results = []

    for ocr_path in ocr_files:
        try:
            result = process_book(
                ocr_path
            )

        except Exception as exc:
            result = {
                "book": ocr_path.stem,
                "status": "ERROR",
                "reason": str(
                    exc
                ),
                "offset": None,
            }

        results.append(
            result
        )

        print_book_result(
            result
        )

    counts = Counter(
        result["status"]
        for result in results
    )

    print()
    print(
        "=" * 72
    )
    print(
        "Page map summary"
    )
    print(
        "=" * 72
    )

    for status in (
        "AUTO_OK",
        "WARNING",
        "MANUAL_OVERRIDE",
        "MANUAL_REQUIRED",
        "ERROR",
    ):
        count = counts.get(
            status,
            0,
        )

        print(
            f"{status:16s}: {count}"
        )

    print(
        f"{'TOTAL':16s}: {len(results)}"
    )

    print()

    for result in results:
        if result[
            "status"
        ] in (
            "WARNING",
            "MANUAL_REQUIRED",
            "ERROR",
        ):
            print(
                f"[{result['status']}] "
                f"{result['book']}: "
                f"{result['reason']}"
            )

    return (
        1
        if counts.get(
            "ERROR",
            0,
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )