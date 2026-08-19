from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"{label}: target block not found")
    return text.replace(old, new, 1)


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")

    marker = """# ============================================================
# Search helpers
# ============================================================
"""

    resolver = r'''# ============================================================
# Official correction target resolution
# ============================================================

OFFICIAL_SOURCE_KEY_HINTS = {
    "SW2.5_1": "ルールブック1",
    "SW2.5_2": "ルールブック2",
    "SW2.5_3": "ルールブック3",
    "SW2.5_CBB": "キャラクタービルディングブック",
    "SW2.5_granzale": "冒険の国グランゼール",
    "SW2.5_vicecity": "ヴァイスシティ",
    "SW2.5_epictreasury": "エピックトレジャリー",
    "SW2.5_kingsfall": "鉄道の都キングスフォール",
    "SW2.5_daemonsline": "デモンズライン",
    "SW2.5_monstrouslore": "モンストラスロア",
    "SW2.5_outlaw": "アウトロープロファイルブック",
    "SW2.5_magusarts": "メイガスアーツ",
    "SW2.5_battlemastery": "バトルマスタリー",
    "SW2.5_burlight": "ブルライト博物誌",
    "SW2.5_arcanerelik": "アーケインレリック",
    "SW2.5_raxialife": "ラクシアライフ",
    "SW2.5_travelsinalfreim": "アルフレイム見聞録",
    "SW2.5_barbarous": "バルバロスレイジ",
    "SW2.5_barbarousSaga": "バルバロスサーガ",
    "SW2.5_abyssbreaker": "アビスブレイカー",
    "SW2.5_ursyla": "ウルシラ博物誌",
    "SW2.5_infinite": "インフィニットコロッセオ",
    "SW2.5_tyrant": "タイラントクリプト",
    "SW2.5_star": "星座の町サイレックオード",
}


def resolve_official_target_book(doc) -> str | None:
    source_key = str(doc.metadata.get("source_key") or "").strip()
    hint = OFFICIAL_SOURCE_KEY_HINTS.get(source_key)
    if not hint:
        return None

    matches = [
        book
        for book in book_to_category
        if hint in book
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def official_target_page_documents(doc) -> list:
    book = resolve_official_target_book(doc)
    if not book:
        return []

    target_page = doc.metadata.get("target_page")
    if target_page is None:
        return []

    try:
        target_page = int(target_page)
    except (TypeError, ValueError):
        return []

    return list(
        page_documents_by_logical.get(
            (book, target_page),
            []
        )
    )


def normalize_override_match_text(value: str | None) -> str:
    value = str(value or "")
    value = re.sub(r"\\s+", "", value)
    value = re.sub(
        r"[《》〈〉「」『』\\(\\)（）\\[\\]【】・･/／：:。．,.，!?！？\\-―—～〜~]",
        "",
        value,
    )
    return value.lower()


def override_text_similarity(needle: str | None, haystack: str | None) -> float:
    normalized_needle = normalize_override_match_text(needle)
    normalized_haystack = normalize_override_match_text(haystack)

    if not normalized_needle or not normalized_haystack:
        return 0.0

    if normalized_needle in normalized_haystack:
        return 1.0

    coverage = char_ngram_coverage(
        normalized_needle,
        normalized_haystack,
        2,
    )
    ratio = SequenceMatcher(
        None,
        normalized_needle,
        normalized_haystack,
    ).ratio()

    return max(
        coverage,
        ratio * 0.70,
    )


def resolve_official_override(doc) -> dict:
    target_book = resolve_official_target_book(doc)
    target_page = doc.metadata.get("target_page")
    operation = doc.metadata.get("operation")
    before = doc.metadata.get("before")
    location = doc.metadata.get("location")

    page_docs = official_target_page_documents(doc)

    if not target_book:
        return {
            "official_doc": doc,
            "target_book": None,
            "target_page": target_page,
            "operation": operation,
            "match_status": "NO_BOOK_MAPPING",
            "best_book_doc": None,
            "before_similarity": 0.0,
            "location_similarity": 0.0,
            "match_score": 0.0,
        }

    if target_page is None:
        return {
            "official_doc": doc,
            "target_book": target_book,
            "target_page": None,
            "operation": operation,
            "match_status": "NO_TARGET_PAGE",
            "best_book_doc": None,
            "before_similarity": 0.0,
            "location_similarity": 0.0,
            "match_score": 0.0,
        }

    if not page_docs:
        return {
            "official_doc": doc,
            "target_book": target_book,
            "target_page": target_page,
            "operation": operation,
            "match_status": "TARGET_PAGE_NOT_INDEXED",
            "best_book_doc": None,
            "before_similarity": 0.0,
            "location_similarity": 0.0,
            "match_score": 0.0,
        }

    scored = []

    for book_doc in page_docs:
        page_text = book_doc.page_content or ""

        before_similarity = override_text_similarity(
            before,
            page_text,
        )
        location_similarity = override_text_similarity(
            location,
            page_text,
        )

        if operation == "replace":
            match_score = (
                before_similarity * 0.80
                + location_similarity * 0.20
            )
        elif operation == "delete":
            delete_similarity = override_text_similarity(
                doc.metadata.get("delete_text"),
                page_text,
            )
            match_score = max(
                delete_similarity,
                location_similarity * 0.85,
            )
        elif operation == "append":
            match_score = location_similarity
        else:
            match_score = max(
                before_similarity,
                location_similarity,
            )

        scored.append(
            (
                match_score,
                before_similarity,
                location_similarity,
                book_doc,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    (
        best_score,
        best_before,
        best_location,
        best_doc,
    ) = scored[0]

    if best_score >= 0.80:
        status = "MATCHED_STRONG"
    elif best_score >= 0.50:
        status = "MATCHED_WEAK"
    else:
        status = "PAGE_ONLY"

    return {
        "official_doc": doc,
        "target_book": target_book,
        "target_page": target_page,
        "operation": operation,
        "match_status": status,
        "best_book_doc": best_doc,
        "before_similarity": best_before,
        "location_similarity": best_location,
        "match_score": best_score,
    }


'''

    text = replace_once(
        text,
        marker,
        resolver + marker,
        "resolver insertion",
    )

    compile(
        text,
        str(args.output),
        "exec",
    )

    args.output.write_text(
        text,
        encoding="utf-8",
        newline="\n",
    )

    print(args.output)


if __name__ == "__main__":
    main()
