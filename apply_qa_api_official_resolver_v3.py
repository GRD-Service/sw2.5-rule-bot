
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")

    old = '''        if after_similarity >= 0.90 and before_similarity < 0.65:
            status = "ALREADY_APPLIED"
            match_score = after_similarity
        else:
'''

    new = '''        # OCR済み原本が既に訂正版の場合を検出する。
        #
        # 文字認識揺れがあるためafter完全一致だけを要求しない。
        # beforeよりafterが明確に強く、before側が弱い場合は
        # ALREADY_APPLIEDとみなす。
        already_applied = (
            (
                after_similarity >= 0.90
                and before_similarity < 0.65
            )
            or (
                after_similarity >= 0.65
                and before_similarity <= 0.55
                and (
                    after_similarity
                    - before_similarity
                ) >= 0.15
            )
        )

        if already_applied:
            status = "ALREADY_APPLIED"
            match_score = after_similarity
        else:
'''

    if old not in text:
        raise RuntimeError(
            "replace ALREADY_APPLIED block not found"
        )

    text = text.replace(
        old,
        new,
        1,
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
