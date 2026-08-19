from __future__ import annotations

import argparse
from pathlib import Path


SOURCE_KEY_HINTS = {
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

    # Add resolver helpers immediately before Search helpers.
    marker = """# ============================================================
# Search helpers
# ============================================================
"""

    resolver = r