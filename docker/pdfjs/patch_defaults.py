#!/usr/bin/env python3

import re
import sys
from pathlib import Path


def replace_option(text: str, name: str, replacement: str) -> str:
    pattern = re.compile(
        rf"({re.escape(name)}\s*:\s*\{{.*?\bvalue\s*:\s*)"
        rf"(\"[^\"]*\"|-?\d+)"
        rf"(?=\s*,)",
        re.DOTALL,
    )

    text, count = pattern.subn(
        lambda match: match.group(1) + replacement,
        text,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"Could not uniquely patch PDF.js option: {name} "
            f"(matches={count})"
        )

    return text


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: patch_defaults.py "
            "<viewer.mjs> <zoom> <scroll_mode> <spread_mode>"
        )

    viewer_path = Path(sys.argv[1])
    zoom = sys.argv[2]
    scroll_mode = int(sys.argv[3])
    spread_mode = int(sys.argv[4])

    if not viewer_path.is_file():
        raise RuntimeError(f"PDF.js viewer not found: {viewer_path}")

    if scroll_mode not in {-1, 0, 1, 2, 3}:
        raise ValueError(
            f"Invalid PDFJS_DEFAULT_SCROLL_MODE: {scroll_mode}"
        )

    if spread_mode not in {-1, 0, 1, 2}:
        raise ValueError(
            f"Invalid PDFJS_DEFAULT_SPREAD_MODE: {spread_mode}"
        )

    text = viewer_path.read_text(encoding="utf-8")

    text = replace_option(
        text,
        "defaultZoomValue",
        repr(zoom).replace("'", '"'),
    )
    text = replace_option(
        text,
        "scrollModeOnLoad",
        str(scroll_mode),
    )
    text = replace_option(
        text,
        "spreadModeOnLoad",
        str(spread_mode),
    )

    viewer_path.write_text(text, encoding="utf-8")

    print("PDF.js viewer defaults patched:")
    print(f"  defaultZoomValue = {zoom}")
    print(f"  scrollModeOnLoad = {scroll_mode}")
    print(f"  spreadModeOnLoad = {spread_mode}")


if __name__ == "__main__":
    main()