from __future__ import annotations

import sys
from pathlib import Path

import pikepdf
from pikepdf import Array, Dictionary, Name


GROUP_SIZE = 32


def rebalance_page_tree(
    source: Path,
    destination: Path,
    group_size: int = GROUP_SIZE,
) -> None:
    print(f"Source: {source}")
    print(f"Destination: {destination}")
    print(f"Group size: {group_size}")

    with pikepdf.Pdf.open(source) as pdf:
        pages = [
            page.obj
            for page in pdf.pages
        ]

        if not pages:
            raise RuntimeError(
                "PDF contains no pages"
            )

        root_pages = pdf.Root.Pages

        print(f"Pages: {len(pages)}")

        intermediate_nodes = []

        for start in range(
            0,
            len(pages),
            group_size,
        ):
            children = pages[
                start:start + group_size
            ]

            node = Dictionary(
                Type=Name.Pages,
                Kids=Array(children),
                Count=len(children),
            )

            node = pdf.make_indirect(node)

            for page in children:
                page.Parent = node

            intermediate_nodes.append(node)

            end = min(
                start + group_size,
                len(pages),
            )

            print(
                f"  group {start + 1}-{end}"
            )

        root_pages.Kids = Array(
            intermediate_nodes
        )

        root_pages.Count = len(pages)

        for node in intermediate_nodes:
            node.Parent = root_pages

        pdf.save(destination)

    print("Done.")


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: rebalance_page_tree.py "
            "input.pdf output.pdf",
            file=sys.stderr,
        )
        return 2

    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])

    if not source.exists():
        print(
            f"ERROR: source does not exist: {source}",
            file=sys.stderr,
        )
        return 1

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rebalance_page_tree(
        source,
        destination,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())