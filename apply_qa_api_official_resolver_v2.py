
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def replace_between(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    label: str,
) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:start] + replacement + text[end:]


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")

    start_marker = "def official_target_page_documents(doc) -> list:\n"
    end_marker = "# ============================================================\n# Search helpers\n# ============================================================\n"

    replacement = r'''def official_target_page_documents(doc) -> list:
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
    value = re.sub(r"\s+", "", value)
    value = re.sub(
        r"[《》〈〉「」『』\(\)（）\[\]【】・･/／：:。．,.，!?！？\-―—～〜~]",
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


def official_target_page_text(doc) -> str:
    docs = official_target_page_documents(doc)
    if not docs:
        return ""

    docs = sorted(
        docs,
        key=lambda item: int(item.metadata.get("chunk", 0)),
    )

    return "\n".join(
        item.page_content or ""
        for item in docs
    )


def best_official_target_chunk(doc, query_text: str | None):
    docs = official_target_page_documents(doc)
    if not docs:
        return None

    if not query_text:
        return docs[0]

    return max(
        docs,
        key=lambda item: override_text_similarity(
            query_text,
            item.page_content or "",
        ),
    )


def is_short_override_token(value: str | None) -> bool:
    normalized = normalize_override_match_text(value)
    return len(normalized) <= 4


def resolve_official_override(doc) -> dict:
    target_book = resolve_official_target_book(doc)
    target_page = doc.metadata.get("target_page")
    operation = doc.metadata.get("operation")

    before = doc.metadata.get("before")
    after = doc.metadata.get("after")
    location = doc.metadata.get("location")
    delete_text = doc.metadata.get("delete_text")

    page_docs = official_target_page_documents(doc)

    base = {
        "official_doc": doc,
        "target_book": target_book,
        "target_page": target_page,
        "operation": operation,
        "best_book_doc": None,
        "before_similarity": 0.0,
        "after_similarity": 0.0,
        "location_similarity": 0.0,
        "delete_similarity": 0.0,
        "match_score": 0.0,
    }

    if not target_book:
        return {**base, "match_status": "NO_BOOK_MAPPING"}

    if target_page is None:
        return {**base, "match_status": "NO_TARGET_PAGE"}

    if not page_docs:
        return {**base, "match_status": "TARGET_PAGE_NOT_INDEXED"}

    page_text = official_target_page_text(doc)

    before_similarity = override_text_similarity(before, page_text)
    after_similarity = override_text_similarity(after, page_text)
    location_similarity = override_text_similarity(location, page_text)
    delete_similarity = override_text_similarity(delete_text, page_text)

    anchor_text = before or location or delete_text or after
    best_doc = best_official_target_chunk(doc, anchor_text)

    if operation == "replace":
        if after_similarity >= 0.90 and before_similarity < 0.65:
            status = "ALREADY_APPLIED"
            match_score = after_similarity
        else:
            match_score = (
                before_similarity * 0.80
                + location_similarity * 0.20
            )

            if before_similarity >= 0.90 and match_score >= 0.80:
                status = "MATCHED_STRONG"
            elif match_score >= 0.50:
                status = "MATCHED_WEAK"
            else:
                status = "PAGE_ONLY"

    elif operation == "delete":
        if is_short_override_token(delete_text):
            match_score = location_similarity
        else:
            match_score = (
                delete_similarity * 0.65
                + location_similarity * 0.35
            )

        if (
            location_similarity >= 0.80
            and (
                delete_similarity >= 0.80
                or is_short_override_token(delete_text)
            )
        ):
            status = "MATCHED_STRONG"
        elif match_score >= 0.45:
            status = "MATCHED_WEAK"
        else:
            status = "PAGE_ONLY"

    elif operation == "append":
        match_score = location_similarity

        if location_similarity >= 0.85:
            status = "MATCHED_STRONG"
        elif location_similarity >= 0.45:
            status = "MATCHED_WEAK"
        else:
            status = "PAGE_ONLY"

    else:
        match_score = max(
            before_similarity,
            after_similarity,
            location_similarity,
        )

        if match_score >= 0.85:
            status = "MATCHED_STRONG"
        elif match_score >= 0.50:
            status = "MATCHED_WEAK"
        else:
            status = "PAGE_ONLY"

    return {
        **base,
        "best_book_doc": best_doc,
        "before_similarity": before_similarity,
        "after_similarity": after_similarity,
        "location_similarity": location_similarity,
        "delete_similarity": delete_similarity,
        "match_score": match_score,
        "match_status": status,
    }


'''

    text = replace_between(
        text,
        start_marker,
        end_marker,
        replacement,
        "official resolver block",
    )

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
