
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

    old_prompt = '''- `GroupSNE公式エラッタ・追加訂正` が含まれる場合、それは対応する書籍本文より新しい公式訂正として優先してください。
- 公式訂正が `replace` の場合は訂正後を採用し、`delete` の場合は削除対象を現行ルールとして扱わず、`append` の場合は追加内容を現行ルールへ加えてください。
- `delete` は「情報が見つからない」という意味ではありません。「公式に削除された」という確定情報として回答してください。
- 公式訂正ブロックに「対応する原本引用ID」がある場合、訂正内容について回答するときはその引用IDを根拠として使用してください。
'''

    new_prompt = '''- `GroupSNE公式エラッタ・追加訂正` が含まれる場合、それは対応する書籍本文より新しい公式訂正として優先してください。
- 公式訂正が `replace` の場合は訂正後を採用し、`delete` の場合は削除対象を現行ルールとして扱わず、`append` の場合は追加内容を現行ルールへ加えてください。
- `delete` は「情報が見つからない」という意味ではありません。「公式に削除された」という確定情報として回答してください。
- 公式訂正ブロックに「対応する原本引用ID」がある場合、訂正内容について回答するときはその引用IDを根拠として使用してください。
- 質問に直接該当する `[OFFICIAL CORRECTION]` が複数ある場合、それらを恣意的に省略してはいけません。特に、複数の `append` が同一テーマに対して適用される場合は、該当する対象をすべて回答へ反映してください。
- 同じ書籍ページに複数の `[OFFICIAL CORRECTION]` が存在しても、それらは別々の公式訂正です。原本Citationが同じ `[C#]` に集約されていても、公式訂正の対象自体を省略してはいけません。
- 回答中で複数対象を列挙する場合、`GroupSNE公式エラッタ・追加訂正` に含まれる直接該当項目をすべて列挙してください。
'''

    text = replace_once(
        text,
        old_prompt,
        new_prompt,
        "official multi-correction prompt",
    )

    # v4のcontext構築方式に依存せず、既存の説明文だけを強化する。
    old_header = (
        '"以下の公式訂正は、対応する原本記述より優先してください。\\\\n\\\\n"'
    )
    new_header = (
        '"以下の公式訂正は、対応する原本記述より優先してください。'
        '複数件ある場合は、同一ページに属していても個別の訂正として扱い、'
        '質問に直接該当するものを省略しないでください。\\\\n\\\\n"'
    )

    if old_header in text:
        text = text.replace(old_header, new_header, 1)

    compile(text, str(args.output), "exec")
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(args.output)


if __name__ == "__main__":
    main()
