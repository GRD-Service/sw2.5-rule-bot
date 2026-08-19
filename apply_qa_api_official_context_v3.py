
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

    marker = "def format_official_context_item(item: dict) -> str:\n"

    helpers = r'''def official_target_anchor_text(doc) -> str:
    metadata = doc.metadata
    operation = metadata.get("operation")

    if operation == "replace":
        return (
            metadata.get("before")
            or metadata.get("location")
            or metadata.get("after")
            or ""
        )

    if operation == "delete":
        return (
            metadata.get("location")
            or metadata.get("delete_text")
            or ""
        )

    if operation == "append":
        return (
            metadata.get("location")
            or metadata.get("append_text")
            or ""
        )

    return (
        metadata.get("location")
        or metadata.get("before")
        or metadata.get("after")
        or ""
    )


def build_official_target_context_items(
    official_items: list[dict],
    question: str,
) -> list[dict]:
    selected = []
    seen_pages = set()

    for official_item in official_items:
        doc = official_item["doc"]
        resolved = official_item["resolved"]

        target_book = resolved.get("target_book")
        target_page = resolved.get("target_page")

        if not target_book or target_page is None:
            continue

        page_docs = official_target_page_documents(doc)
        if not page_docs:
            continue

        anchor = official_target_anchor_text(doc)

        scored = []
        for page_doc in page_docs:
            anchor_score = override_text_similarity(
                anchor,
                page_doc.page_content or "",
            )
            question_score = chunk_relevance_score(
                page_doc,
                question,
                extra_terms=extract_query_terms(question),
            )

            score = (
                anchor_score * 1000.0
                + question_score
            )
            scored.append(
                (
                    score,
                    anchor_score,
                    page_doc,
                )
            )

        scored.sort(
            key=lambda value: (
                value[0],
                -int(value[2].metadata.get("chunk", 0)),
            ),
            reverse=True,
        )

        if not scored:
            continue

        _best_score, anchor_score, best_doc = scored[0]
        pdf_page = get_pdf_page(best_doc)
        logical_page = get_logical_page(best_doc)

        if pdf_page is None or logical_page is None:
            continue

        page_key = (
            target_book,
            pdf_page,
        )

        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)

        operation = doc.metadata.get("operation") or "unknown"
        location = doc.metadata.get("location") or ""

        reason = (
            f"公式エラッタ対象原本: {target_book} p.{logical_page}"
            f" / operation={operation}"
        )
        if location:
            reason += f" / 対象箇所={location}"

        selected.append(
            {
                "doc": best_doc,
                "mandatory": True,
                "context_score": (
                    2500.0
                    + float(
                        official_item.get(
                            "relevance_score",
                            0.0,
                        )
                    ) * 500.0
                    + anchor_score * 250.0
                ),
                "reason": reason,
                "source": "official_target",
            }
        )

    selected.sort(
        key=lambda item: item["context_score"],
        reverse=True,
    )

    return selected


def merge_official_target_context_items(
    context_items: list[dict],
    official_target_items: list[dict],
) -> list[dict]:
    selected = []
    seen = set()

    for item in official_target_items + context_items:
        key = document_key(item["doc"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)

        if len(selected) >= CONTEXT_MAX_DOCS:
            break

    return selected


def official_target_citation_id(
    item: dict,
    citation_id_map: dict,
) -> int | None:
    resolved = item["resolved"]
    best_doc = resolved.get("best_book_doc")

    if best_doc is None:
        target_docs = official_target_page_documents(item["doc"])
        if target_docs:
            best_doc = target_docs[0]

    if best_doc is None:
        return None

    book = best_doc.metadata.get("book")
    pdf_page = get_pdf_page(best_doc)

    if not book or pdf_page is None:
        return None

    return citation_id_map.get(
        (book, pdf_page)
    )


'''
    if marker not in text:
        raise RuntimeError("official format marker not found")
    text = text.replace(marker, helpers + marker, 1)

    text = replace_once(
        text,
        '''def format_official_context_item(item: dict) -> str:
    doc = item["doc"]
    resolved = item["resolved"]
''',
        '''def format_official_context_item(
    item: dict,
    citation_id_map: Optional[dict] = None,
) -> str:
    doc = item["doc"]
    resolved = item["resolved"]
''',
        "official formatter signature",
    )

    text = replace_once(
        text,
        '''    if source_url:
        lines.append(f"公式URL: {source_url}")

    # chunk builderで作った自然文をそのまま使う。
''',
        '''    if source_url:
        lines.append(f"公式URL: {source_url}")

    target_citation_id = None
    if citation_id_map:
        target_citation_id = official_target_citation_id(
            item,
            citation_id_map,
        )

    if target_citation_id is not None:
        lines.append(
            f"対応する原本引用ID: [C{target_citation_id}]"
        )
        lines.append(
            "この公式訂正を回答根拠として使う場合、"
            f"現段階では対応原本の [C{target_citation_id}] を引用してください。"
        )

    # chunk builderで作った自然文をそのまま使う。
''',
        "official formatter citation",
    )

    text = replace_once(
        text,
        '''    status = resolved.get("match_status")
    if status == "ALREADY_APPLIED":
''',
        '''    if operation == "delete":
        delete_text = metadata.get("delete_text")
        lines.append(
            "現行ルール上の扱い: この訂正は削除指示です。"
            "削除対象が原本に記載されていても、"
            "現行ルールの有効な項目・効果として扱わないでください。"
        )
        if delete_text:
            lines.append(
                f"削除対象の明示: {delete_text}"
            )
    elif operation == "replace":
        lines.append(
            "現行ルール上の扱い: 訂正前ではなく訂正後の内容を採用してください。"
        )
    elif operation == "append":
        lines.append(
            "現行ルール上の扱い: 原本内容にこの追記を加えて解釈してください。"
        )

    status = resolved.get("match_status")
    if status == "ALREADY_APPLIED":
''',
        "official semantics",
    )

    text = replace_once(
        text,
        '''def build_official_context(
    items: list[dict],
) -> str:
    if not items:
        return ""

    return "\\n\\n".join(
        format_official_context_item(item)
        for item in items
    )
''',
        '''def build_official_context(
    items: list[dict],
    citation_id_map: Optional[dict] = None,
) -> str:
    if not items:
        return ""

    return "\\n\\n".join(
        format_official_context_item(
            item,
            citation_id_map=citation_id_map,
        )
        for item in items
    )
''',
        "official context citation map",
    )

    text = replace_once(
        text,
        '''    context_items = merge_chat_memory_items(context_items, memory_items)

    if not context_items:
''',
        '''    context_items = merge_chat_memory_items(context_items, memory_items)

    official_target_items = build_official_target_context_items(
        official_context_items,
        search_question,
    )
    context_items = merge_official_target_context_items(
        context_items,
        official_target_items,
    )

    if not context_items:
''',
        "official target context injection",
    )

    text = replace_once(
        text,
        '''    official_context = build_official_context(
        official_context_items
    )
''',
        '''    official_context = build_official_context(
        official_context_items,
        citation_id_map=citation_id_map,
    )
''',
        "official context citation invocation",
    )

    text = replace_once(
        text,
        '''- `GroupSNE公式エラッタ・追加訂正` が含まれる場合、それは対応する書籍本文より新しい公式訂正として優先してください。
- 公式訂正が `replace` の場合は訂正後を採用し、`delete` の場合は削除対象を現行ルールとして扱わず、`append` の場合は追加内容を現行ルールへ加えてください。
''',
        '''- `GroupSNE公式エラッタ・追加訂正` が含まれる場合、それは対応する書籍本文より新しい公式訂正として優先してください。
- 公式訂正が `replace` の場合は訂正後を採用し、`delete` の場合は削除対象を現行ルールとして扱わず、`append` の場合は追加内容を現行ルールへ加えてください。
- `delete` は「情報が見つからない」という意味ではありません。「公式に削除された」という確定情報として回答してください。
- 公式訂正ブロックに「対応する原本引用ID」がある場合、訂正内容について回答するときはその引用IDを根拠として使用してください。
''',
        "official prompt source pairing",
    )

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
