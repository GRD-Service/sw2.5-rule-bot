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

    # --------------------------------------------------------
    # 1. Environment settings
    # --------------------------------------------------------
    old = '''HYBRID_CANDIDATE_K = int(os.getenv("HYBRID_CANDIDATE_K", "100"))\n'''
    new = '''HYBRID_CANDIDATE_K = int(os.getenv("HYBRID_CANDIDATE_K", "100"))\n\nOFFICIAL_SEARCH_TOP_K = int(os.getenv("OFFICIAL_SEARCH_TOP_K", "6"))\nOFFICIAL_SEARCH_CANDIDATE_K = int(\n    os.getenv("OFFICIAL_SEARCH_CANDIDATE_K", "100")\n)\n'''
    text = replace_once(text, old, new, "official env settings")

    # --------------------------------------------------------
    # 2. Add official context helpers before /ask
    # --------------------------------------------------------
    marker = '''# ============================================================\n# /ask\n# ============================================================\n'''

    helpers = r'''# ============================================================
# Official correction context
# ============================================================


def official_doc_allowed_for_books(doc, books: Optional[List[str]]) -> bool:
    if not books:
        return True

    target_book = resolve_official_target_book(doc)

    # 対応書籍が解決できる場合は、ユーザーの書籍フィルタに従う。
    if target_book:
        return target_book in books

    # 対応書籍が未収録の場合は、書籍フィルタ指定時には混ぜない。
    return False


def select_official_context_items(
    *,
    question: str,
    books: Optional[List[str]],
    variants: Optional[list[str]] = None,
) -> list[dict]:
    query_variants = variants or build_query_variants(question)

    # primary queryに加え、展開語でも検索する。
    merged = {}

    for variant in query_variants:
        results = official_search(
            variant,
            top_k=OFFICIAL_SEARCH_TOP_K,
            candidate_k=OFFICIAL_SEARCH_CANDIDATE_K,
        )

        for item in results:
            doc = item["doc"]

            if not official_doc_allowed_for_books(doc, books):
                continue

            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id")
            key = chunk_id or document_key(doc)

            resolved = resolve_official_override(doc)

            candidate = {
                "doc": doc,
                "resolved": resolved,
                "retrieval_score": float(item.get("retrieval_score", 0.0)),
                "reason": item.get("reason", "公式エラッタ検索"),
                "query_variant": variant,
            }

            existing = merged.get(key)
            if (
                existing is None
                or candidate["retrieval_score"] > existing["retrieval_score"]
            ):
                merged[key] = candidate

    selected = sorted(
        merged.values(),
        key=lambda item: item["retrieval_score"],
        reverse=True,
    )

    return selected[:OFFICIAL_SEARCH_TOP_K]


def format_official_context_item(item: dict) -> str:
    doc = item["doc"]
    resolved = item["resolved"]

    metadata = doc.metadata
    source_name = metadata.get("source_name") or metadata.get("source_key") or "GroupSNE公式"
    source_key = metadata.get("source_key") or "不明"
    target_page = metadata.get("target_page")
    operation = metadata.get("operation") or "unknown"
    location = metadata.get("location")
    source_url = metadata.get("source_url")

    lines = [
        "[OFFICIAL CORRECTION]",
        f"公式資料: {source_name}",
        f"source_key: {source_key}",
        f"対象ページ: {target_page if target_page is not None else 'ページ指定なし'}",
        f"operation: {operation}",
        f"resolver_status: {resolved.get('match_status')}",
        f"resolver_score: {resolved.get('match_score', 0.0):.4f}",
    ]

    target_book = resolved.get("target_book")
    if target_book:
        lines.append(f"対応書籍: {target_book}")

    if location:
        lines.append(f"対象箇所: {location}")

    if source_url:
        lines.append(f"公式URL: {source_url}")

    # chunk builderで作った自然文をそのまま使う。
    lines.append("公式訂正内容:")
    lines.append(doc.page_content or "")

    status = resolved.get("match_status")
    if status == "ALREADY_APPLIED":
        lines.append(
            "適用状態: 所有しているOCR本文は、すでに訂正後内容を含む可能性が高い。"
        )
    elif status in {"MATCHED_STRONG", "MATCHED_WEAK", "PAGE_ONLY"}:
        lines.append(
            "適用状態: 回答ではこの公式訂正を原本記述より優先する。"
        )
    else:
        lines.append(
            "適用状態: 原本との自動対応は未確定だが、GroupSNE公式訂正として扱う。"
        )

    return "\n".join(lines)


def build_official_context(
    items: list[dict],
) -> str:
    if not items:
        return ""

    return "\n\n".join(
        format_official_context_item(item)
        for item in items
    )


'''

    text = replace_once(
        text,
        marker,
        helpers + marker,
        "official context helper insertion",
    )

    # --------------------------------------------------------
    # 3. Run official search in rules_strict
    # --------------------------------------------------------
    old = '''    hybrid_items = hybrid_search(\n        query=search_question,\n        top_k=initial_k,\n        candidate_k=max_k,\n        books=books,\n        variants=variants,\n    )\n    navigation_pages, _exact_index_match = navigation_search(\n'''

    new = '''    hybrid_items = hybrid_search(\n        query=search_question,\n        top_k=initial_k,\n        candidate_k=max_k,\n        books=books,\n        variants=variants,\n    )\n\n    official_context_items = select_official_context_items(\n        question=search_question,\n        books=books,\n        variants=variants,\n    )\n\n    navigation_pages, _exact_index_match = navigation_search(\n'''

    text = replace_once(
        text,
        old,
        new,
        "official search invocation",
    )

    # --------------------------------------------------------
    # 4. Merge official context immediately before prompt
    # --------------------------------------------------------
    old = '''    context = "\\n\\n".join(context_parts)\n    full_prompt = prompt.format(\n        context=context,\n'''

    new = '''    context = "\\n\\n".join(context_parts)\n\n    official_context = build_official_context(\n        official_context_items\n    )\n\n    if official_context:\n        context = (\n            context\n            + "\\n\\n"\n            + "=== GroupSNE公式エラッタ・追加訂正 ===\\n"\n            + "以下の公式訂正は、対応する原本記述より優先してください。\\n"\n            + "原本本文と矛盾する場合は公式訂正を採用してください。\\n\\n"\n            + official_context\n        )\n\n    full_prompt = prompt.format(\n        context=context,\n'''

    text = replace_once(
        text,
        old,
        new,
        "official context merge",
    )

    # --------------------------------------------------------
    # 5. Add prompt rule describing authority
    # --------------------------------------------------------
    old = '''- コンテキストに存在しない情報を推測して補ってはいけません。\n- 根拠を示す場合は、対応するコンテキストの引用IDを `[C1]` の形式で記載してください。\n'''

    new = '''- コンテキストに存在しない情報を推測して補ってはいけません。\n- `GroupSNE公式エラッタ・追加訂正` が含まれる場合、それは対応する書籍本文より新しい公式訂正として優先してください。\n- 公式訂正が `replace` の場合は訂正後を採用し、`delete` の場合は削除対象を現行ルールとして扱わず、`append` の場合は追加内容を現行ルールへ加えてください。\n- 根拠を示す場合は、対応するコンテキストの引用IDを `[C1]` の形式で記載してください。\n'''

    text = replace_once(
        text,
        old,
        new,
        "official prompt authority rules",
    )

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
