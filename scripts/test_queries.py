#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:8000/ask"

DEFAULT_TESTS = [
    {
        "id": "a",
        "question": "リルドラケンについて教えて",
    },
    {
        "id": "b",
        "question": "リルドラケンの希少種について教えて",
    },
    {
        "id": "c",
        "question": "武器の両手持ちに関する情報を教えて",
    },
    {
        "id": "d",
        "question": "マルチアクションについて教えて",
    },
    {
        "id": "e",
        "question": "槍を使った流派はありますか？",
    },
    {
        "id": "f",
        "question": "経験点テーブル",
    },
]


def call_api(
    api_url: str,
    question: str,
    model: str,
    k: int,
    books: list[str] | None,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "model": model,
        "k": k,
        "mode": "rules_strict",
    }

    if books:
        payload["books"] = books

    response = requests.post(
        api_url,
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def print_result(
    test_id: str,
    question: str,
    result: dict[str, Any],
    show_excerpt: bool,
) -> None:
    print()
    print("=" * 100)
    print(f"[{test_id}] {question}")
    print("=" * 100)

    print()
    print("ANSWER")
    print("-" * 100)
    print(result.get("answer", ""))

    print()
    print("SEARCH INFO")
    print("-" * 100)

    print(
        "context chunks       :",
        result.get("k_used"),
    )

    print(
        "hybrid chunks        :",
        result.get("hybrid_k_used"),
    )

    print(
        "navigation pages     :",
        result.get("navigation_pages_used"),
    )

    # 新版qa_api.pyに存在する場合のみ表示
    optional_stats = [
        (
            "structured pages",
            "structured_pages_used",
        ),
        (
            "reference pages",
            "reference_pages_used",
        ),
        (
            "query variants",
            "query_variants_used",
        ),
    ]

    for label, key in optional_stats:
        if key in result:
            print(
                f"{label:20}:",
                result.get(key),
            )

    token_usage = result.get(
        "token_usage",
        {},
    )

    if token_usage:
        print(
            "prompt tokens        :",
            token_usage.get(
                "prompt_tokens",
                0,
            ),
        )

        print(
            "completion tokens    :",
            token_usage.get(
                "completion_tokens",
                0,
            ),
        )

        print(
            "total tokens         :",
            token_usage.get(
                "total_tokens",
                0,
            ),
        )

    print()
    print("CITATIONS")
    print("-" * 100)

    citations = result.get(
        "citations",
        [],
    )

    if not citations:
        print("(none)")
        return

    for citation in citations:
        citation_id = citation.get(
            "id",
            "?",
        )

        book = citation.get(
            "book",
            "?",
        )

        page = citation.get(
            "page",
            "?",
        )

        pdf_page = citation.get(
            "pdf_page",
            "?",
        )

        reason = citation.get(
            "reason",
            "",
        )

        print(
            f"C{citation_id}: "
            f"{book} "
            f"p.{page} "
            f"(PDF {pdf_page})"
        )

        if reason:
            print(
                f"    reason: {reason}"
            )

        if show_excerpt:
            excerpt = citation.get(
                "excerpt",
                "",
            )

            if excerpt:
                print(
                    f"    excerpt: {excerpt}"
                )

        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "SW2.5 Rule Bot CLI regression tester"
        )
    )

    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
    )

    parser.add_argument(
        "--model",
        default="gpt-5.4-nano",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--test",
        action="append",
        help=(
            "実行するテストID。"
            "複数指定可。例: --test a --test c"
        ),
    )

    parser.add_argument(
        "--question",
        help=(
            "任意の質問を1件だけ実行"
        ),
    )

    parser.add_argument(
        "--book",
        action="append",
        dest="books",
        help=(
            "検索対象書籍を限定。複数指定可"
        ),
    )

    parser.add_argument(
        "--excerpt",
        action="store_true",
        help=(
            "出典本文抜粋も表示"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "APIレスポンスをJSONで表示"
        ),
    )

    args = parser.parse_args()

    if args.question:
        tests = [
            {
                "id": "custom",
                "question": args.question,
            }
        ]

    else:
        selected_ids = set(
            args.test or []
        )

        if selected_ids:
            tests = [
                test
                for test in DEFAULT_TESTS
                if test["id"]
                in selected_ids
            ]

            unknown = (
                selected_ids
                - {
                    test["id"]
                    for test
                    in DEFAULT_TESTS
                }
            )

            if unknown:
                print(
                    "Unknown test IDs:",
                    ", ".join(
                        sorted(
                            unknown
                        )
                    ),
                    file=sys.stderr,
                )

                return 2

        else:
            tests = DEFAULT_TESTS

    failed = 0

    for test in tests:
        try:
            result = call_api(
                api_url=args.api_url,
                question=test["question"],
                model=args.model,
                k=args.k,
                books=args.books,
                timeout=args.timeout,
            )

        except Exception as exc:
            failed += 1

            print()
            print("=" * 100)
            print(
                f"[{test['id']}] "
                f"{test['question']}"
            )
            print("=" * 100)
            print(
                f"ERROR: {exc}"
            )

            continue

        if args.json:
            print(
                json.dumps(
                    {
                        "id": test["id"],
                        "question": (
                            test["question"]
                        ),
                        "result": result,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        else:
            print_result(
                test_id=test["id"],
                question=test[
                    "question"
                ],
                result=result,
                show_excerpt=args.excerpt,
            )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )