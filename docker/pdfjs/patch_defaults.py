#!/usr/bin/env python3

import re
import sys
from pathlib import Path


def patch_option(
    text: str,
    name: str,
    old_value_pattern: str,
    new_value: str,
) -> str:
    pattern = re.compile(
        rf"({re.escape(name)}\s*:\s*\{{"
        rf".*?"
        rf"\bvalue\s*:\s*)"
        rf"({old_value_pattern})"
        rf"(\s*,)",
        re.DOTALL,
    )

    text, count = pattern.subn(
        lambda m: m.group(1) + new_value + m.group(3),
        text,
        count=1,
    )

    if count != 1:
        raise RuntimeError(
            f"Could not patch {name}: matches={count}"
        )

    print(f"patched {name} -> {new_value}")
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
        raise RuntimeError(
            f"viewer.mjs not found: {viewer_path}"
        )

    if scroll_mode not in (-1, 0, 1, 2, 3):
        raise ValueError(
            f"Invalid scroll mode: {scroll_mode}"
        )

    if spread_mode not in (-1, 0, 1, 2):
        raise ValueError(
            f"Invalid spread mode: {spread_mode}"
        )

    text = viewer_path.read_text(encoding="utf-8")

    # PDF.js factory defaults.
    #
    # These are deliberately changed at the source-default level rather
    # than through AppOptions.set(). Preferences therefore continue to
    # work normally and can override these defaults.

    text = patch_option(
        text,
        "defaultZoomValue",
        r'""',
        f'"{zoom}"',
    )

    text = patch_option(
        text,
        "scrollModeOnLoad",
        r"-1",
        str(scroll_mode),
    )

    text = patch_option(
        text,
        "spreadModeOnLoad",
        r"-1",
        str(spread_mode),
    )

    viewer_path.write_text(
        text,
        encoding="utf-8",
    )

    # Verify the actual result.
    verify = viewer_path.read_text(encoding="utf-8")

    expected = {
        "defaultZoomValue": f'"{zoom}"',
        "scrollModeOnLoad": str(scroll_mode),
        "spreadModeOnLoad": str(spread_mode),
    }

    for name, value in expected.items():
        pattern = re.compile(
            rf"{re.escape(name)}\s*:\s*\{{"
            rf".*?"
            rf"\bvalue\s*:\s*{re.escape(value)}\s*,",
            re.DOTALL,
        )

        if not pattern.search(verify):
            raise RuntimeError(
                f"Verification failed: {name} != {value}"
            )

    print("PDF.js default preference patch verified successfully.")


if __name__ == "__main__":
    main()