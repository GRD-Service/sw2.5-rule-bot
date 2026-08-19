
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

    if "import unicodedata\n" not in text:
        text = replace_once(
            text,
            "import re\n",
            "import re\nimport unicodedata\n",
            "unicodedata import",
        )

    old_env = '''OFFICIAL_SEARCH_TOP_K = int(os.getenv("OFFICIAL_SEARCH_TOP_K", "6"))
OFFICIAL_SEARCH_CANDIDATE_K = int(
    os.getenv("OFFICIAL_SEARCH_CANDIDATE_K", "100")
)
'''
    new_env = '''OFFICIAL_SEARCH_TOP_K = int(os.getenv("OFFICIAL_SEARCH_TOP_K", "6"))
OFFICIAL_SEARCH_CANDIDATE_K = int(
    os.getenv("OFFICIAL_SEARCH_CANDIDATE_K", "100")
)
OFFICIAL_RELEVANCE_MIN = float(
    os.getenv("OFFICIAL_RELEVANCE_MIN", "0.58")
)
OFFICIAL_RELEVANCE_RELATIVE = float(
    os.getenv("OFFICIAL_RELEVANCE_RELATIVE", "0.72")
)
'''
    text = replace_once(
        text,
        old_env,
        new_env,
        "official relevance env",
    )

    start_marker = "def select_official_context_items(\n"
    end_marker = "def format_official_context_item(item: dict) -> str:\n"

    replacement = r'''def normalize_official_relevance_text(value: str | None) -> str:
    value = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    )
    value = re.sub(r"\s+", "", value)
    value = re.sub(
        r"[《》〈〉「」『』\(\)（）\[\]【】・･/／：:。．,.，!?！？\-―—～〜~]",
        "",
        value,
    )
    return value.lower()


def official_query_terms(question: str) -> list[str]:
    terms = extract_query_terms(question)
    normalized = []

    for term in terms:
        value = normalize_official_relevance_text(term)
        if value and value not in normalized:
            normalized.append(value)

    whole = normalize_official_relevance_text(
        normalize_search_query(question)
    )
    if whole and whole not in normalized:
        normalized.insert(0, whole)

    return normalized


def official_candidate_haystack(doc) -> str:
    metadata = doc.metadata
    values = [
        doc.page_content or "",
        metadata.get("location") or "",
        metadata.get("before") or "",
        metadata.get("after") or "",
        metadata.get("append_text") or "",
        metadata.get("delete_text") or "",
        metadata.get("source_key") or "",
        metadata.get("source_name") or "",
    ]
    return normalize_official_relevance_text(
        "\n".join(str(value) for value in values if value)
    )


def official_candidate_relevance(
    question: str,
    doc,
) -> dict:
    terms = official_query_terms(question)
    haystack = official_candidate_haystack(doc)

    if not terms or not haystack:
        return {
            "score": 0.0,
            "matched_terms": [],
            "term_scores": [],
            "whole_match": False,
        }

    whole = normalize_official_relevance_text(
        normalize_search_query(question)
    )
    whole_match = bool(
        whole
        and len(whole) >= 4
        and whole in haystack
    )

    term_scores = []
    matched_terms = []

    for term in terms:
        if not term:
            continue

        if term in haystack:
            score = 1.0
        elif len(term) >= 2:
            score = char_ngram_coverage(
                term,
                haystack,
                2,
            )
        else:
            score = 0.0

        term_scores.append(score)

        if score >= 0.72:
            matched_terms.append(term)

    if not term_scores:
        relevance = 0.0
    elif len(term_scores) == 1:
        relevance = term_scores[0]
    else:
        relevance = (
            min(term_scores) * 0.65
            + (
                sum(term_scores)
                / len(term_scores)
            ) * 0.35
        )

    if whole_match:
        relevance = max(
            relevance,
            1.0,
        )

    return {
        "score": relevance,
        "matched_terms": matched_terms,
        "term_scores": term_scores,
        "whole_match": whole_match,
    }


def select_official_context_items(
    *,
    question: str,
    books: Optional[List[str]],
    variants: Optional[list[str]] = None,
) -> list[dict]:
    query_variants = variants or build_query_variants(question)
    merged = {}

    for variant in query_variants:
        results = official_search(
            variant,
            top_k=max(
                OFFICIAL_SEARCH_TOP_K * 3,
                12,
            ),
            candidate_k=OFFICIAL_SEARCH_CANDIDATE_K,
        )

        for item in results:
            doc = item["doc"]

            if not official_doc_allowed_for_books(doc, books):
                continue

            chunk_id = (
                doc.metadata.get("chunk_id")
                or doc.metadata.get("id")
            )
            key = chunk_id or document_key(doc)

            relevance = official_candidate_relevance(
                question,
                doc,
            )
            resolved = resolve_official_override(doc)

            candidate = {
                "doc": doc,
                "resolved": resolved,
                "retrieval_score": float(
                    item.get("retrieval_score", 0.0)
                ),
                "relevance_score": float(
                    relevance["score"]
                ),
                "relevance_terms": relevance["matched_terms"],
                "relevance_term_scores": relevance["term_scores"],
                "whole_query_match": relevance["whole_match"],
                "reason": item.get(
                    "reason",
                    "公式エラッタ検索",
                ),
                "query_variant": variant,
            }

            existing = merged.get(key)

            candidate_sort_key = (
                candidate["relevance_score"],
                candidate["retrieval_score"],
            )
            existing_sort_key = (
                (
                    existing["relevance_score"],
                    existing["retrieval_score"],
                )
                if existing is not None
                else (-1.0, -1.0)
            )

            if candidate_sort_key > existing_sort_key:
                merged[key] = candidate

    candidates = list(
        merged.values()
    )

    if not candidates:
        return []

    candidates.sort(
        key=lambda item: (
            item["relevance_score"],
            item["retrieval_score"],
        ),
        reverse=True,
    )

    best_relevance = candidates[0]["relevance_score"]

    threshold = max(
        OFFICIAL_RELEVANCE_MIN,
        best_relevance * OFFICIAL_RELEVANCE_RELATIVE,
    )

    selected = [
        item
        for item in candidates
        if item["relevance_score"] >= threshold
    ]

    whole_matches = [
        item
        for item in selected
        if item["whole_query_match"]
    ]
    if whole_matches:
        selected = whole_matches

    selected.sort(
        key=lambda item: (
            item["relevance_score"],
            item["retrieval_score"],
        ),
        reverse=True,
    )

    return selected[:OFFICIAL_SEARCH_TOP_K]


'''

    text = replace_between(
        text,
        start_marker,
        end_marker,
        replacement,
        "official context selector",
    )

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
