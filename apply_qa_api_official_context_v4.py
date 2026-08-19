
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

    old = """def official_query_terms(question: str) -> list[str]:
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
"""

    new = """def normalize_official_question_for_terms(question: str) -> str:
    value = unicodedata.normalize(
        "NFKC",
        str(question or ""),
    )

    value = re.sub(
        r"(について|に関して|に関する)\\s*(教えてください|教えて|知りたいです|知りたい)?[。.!！?？]*$",
        "",
        value,
    )
    value = re.sub(
        r"(とは)?\\s*(何ですか|なんですか|何でしょうか|どうなりますか|どうですか)[。.!！?？]*$",
        "",
        value,
    )
    value = re.sub(
        r"(を)?\\s*(教えてください|教えて|知りたいです|知りたい)[。.!！?？]*$",
        "",
        value,
    )
    value = re.sub(
        r"[。.!！?？]+$",
        "",
        value,
    )

    return value.strip()


def official_query_terms(question: str) -> list[str]:
    cleaned_question = normalize_official_question_for_terms(
        question
    )

    terms = extract_query_terms(cleaned_question)
    normalized = []

    for term in terms:
        term = re.sub(
            r"(?:は|が|を|に|で|と|の)$",
            "",
            term,
        ).strip()

        value = normalize_official_relevance_text(term)
        if value and value not in normalized:
            normalized.append(value)

    whole = normalize_official_relevance_text(
        cleaned_question
    )
    if whole and whole not in normalized:
        normalized.insert(0, whole)

    return normalized
"""

    text = replace_once(
        text,
        old,
        new,
        "official query term normalization",
    )

    old2 = """    whole = normalize_official_relevance_text(
        normalize_search_query(question)
    )
"""
    new2 = """    whole = normalize_official_relevance_text(
        normalize_official_question_for_terms(
            question
        )
    )
"""

    text = replace_once(
        text,
        old2,
        new2,
        "official whole query normalization",
    )

    compile(
        text,
        str(args.output),
        "exec",
    )

    args.output.write_text(
        text,
        encoding="utf-8",
        newline="\\n",
    )

    print(args.output)


if __name__ == "__main__":
    main()
