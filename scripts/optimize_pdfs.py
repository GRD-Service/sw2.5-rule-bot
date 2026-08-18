from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fitz
import pikepdf
from PIL import Image
from pikepdf import Array, Dictionary, Name


SOURCE_DIR = Path(
    os.getenv(
        "PDF_SOURCE_DIR",
        "/data/pdf-original",
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "PDF_OUTPUT_DIR",
        "/data/pdf-web",
    )
)

TARGET_WIDTH = int(
    os.getenv(
        "PDF_OPTIMIZER_WIDTH",
        "1400",
    )
)

JPEG_QUALITY = int(
    os.getenv(
        "PDF_OPTIMIZER_JPEG_QUALITY",
        "68",
    )
)

PAGE_TREE_GROUP_SIZE = int(
    os.getenv(
        "PDF_OPTIMIZER_PAGE_TREE_GROUP_SIZE",
        "32",
    )
)

STATE_FILE = OUTPUT_DIR / ".optimizer-state.json"

# Version 2:
# - hierarchical Page Tree
# - linearization validation
# - Page Tree validation
OPTIMIZER_VERSION = 2


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}

    try:
        with STATE_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception as exc:
        print(
            f"Warning: failed to read state file: {exc}"
        )

    return {}


def save_state(state: dict) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = STATE_FILE.with_suffix(
        ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    os.replace(
        temp_path,
        STATE_FILE,
    )


def source_signature(
    path: Path,
) -> dict:
    stat = path.stat()

    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "width": TARGET_WIDTH,
        "jpeg_quality": JPEG_QUALITY,
        "page_tree_group_size": PAGE_TREE_GROUP_SIZE,
        "optimizer_version": OPTIMIZER_VERSION,
    }


def needs_rebuild(
    source: Path,
    destination: Path,
    state: dict,
) -> bool:
    if not destination.exists():
        return True

    previous = state.get(
        source.name
    )

    if previous is None:
        return True

    return (
        previous
        != source_signature(source)
    )


def render_page_to_jpeg(
    page: fitz.Page,
) -> bytes:
    rect = page.rect

    if rect.width <= 0:
        raise ValueError(
            "Invalid page width"
        )

    scale = TARGET_WIDTH / rect.width

    # 現行挙動を維持する。
    #
    # TARGET_WIDTH の扱いについては別途改善候補だが、
    # Page Tree 最適化との比較条件を変えないため、
    # 今回は変更しない。
    scale = min(
        scale,
        1.0,
    )

    matrix = fitz.Matrix(
        scale,
        scale,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
        colorspace=fitz.csRGB,
    )

    image = Image.frombytes(
        "RGB",
        (
            pixmap.width,
            pixmap.height,
        ),
        pixmap.samples,
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=JPEG_QUALITY,
        optimize=True,
        progressive=False,
        subsampling=2,
    )

    return buffer.getvalue()


def create_image_pdf(
    source: Path,
    destination: Path,
) -> int:
    source_doc = fitz.open(
        source
    )

    output_doc = fitz.open()

    try:
        total_pages = source_doc.page_count

        for index in range(
            total_pages
        ):
            page = source_doc.load_page(
                index
            )

            rect = page.rect

            jpeg_data = (
                render_page_to_jpeg(
                    page
                )
            )

            output_page = (
                output_doc.new_page(
                    width=rect.width,
                    height=rect.height,
                )
            )

            output_page.insert_image(
                output_page.rect,
                stream=jpeg_data,
                keep_proportion=False,
            )

            if (
                index == 0
                or (index + 1) % 25 == 0
                or index + 1 == total_pages
            ):
                print(
                    f"  rendered "
                    f"{index + 1}/{total_pages}"
                )

        output_doc.save(
            destination,
            garbage=4,
            deflate=True,
            clean=True,
        )

        return total_pages

    finally:
        output_doc.close()
        source_doc.close()


def rebalance_page_tree(
    source: Path,
    destination: Path,
    group_size: int,
) -> None:
    if group_size < 2:
        raise ValueError(
            "Page Tree group size must be >= 2"
        )

    with pikepdf.Pdf.open(
        source
    ) as pdf:
        pages = [
            page.obj
            for page in pdf.pages
        ]

        if not pages:
            raise RuntimeError(
                "PDF contains no pages"
            )

        root_pages = pdf.Root.Pages

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

            node = pdf.make_indirect(
                node
            )

            for page in children:
                page.Parent = node

            intermediate_nodes.append(
                node
            )

        root_pages.Kids = Array(
            intermediate_nodes
        )

        root_pages.Count = len(
            pages
        )

        for node in intermediate_nodes:
            node.Parent = root_pages

        pdf.save(
            destination
        )

    print(
        "  Page Tree:"
        f" {len(pages)} pages"
        f" -> {len(intermediate_nodes)} groups"
        f" (max {group_size} pages/group)"
    )


def get_page_count(
    path: Path,
) -> int:
    doc = fitz.open(
        path
    )

    try:
        return doc.page_count
    finally:
        doc.close()


def linearize_pdf(
    source: Path,
    destination: Path,
) -> None:
    result = subprocess.run(
        [
            "qpdf",
            "--linearize",
            "--object-streams=generate",
            str(source),
            str(destination),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.stdout:
        print(
            result.stdout.rstrip()
        )

    # qpdf:
    #   0 = success
    #   2 = error
    #   3 = success with warnings
    if result.returncode not in (
        0,
        3,
    ):
        raise RuntimeError(
            "qpdf linearization failed "
            f"with exit status "
            f"{result.returncode}"
        )

    if result.returncode == 3:
        print(
            "  qpdf completed with warnings; "
            "continuing to validation..."
        )


def run_qpdf_check(
    args: list[str],
    description: str,
) -> None:
    result = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.stdout:
        print(
            result.stdout.rstrip()
        )

    if result.returncode not in (
        0,
        3,
    ):
        raise RuntimeError(
            f"{description} failed "
            f"with exit status "
            f"{result.returncode}"
        )


def validate_page_tree(
    path: Path,
    expected_pages: int,
    group_size: int,
) -> None:
    with pikepdf.Pdf.open(
        path
    ) as pdf:
        root_pages = pdf.Root.Pages

        root_count = int(
            root_pages.Count
        )

        if root_count != expected_pages:
            raise RuntimeError(
                "Page Tree root count mismatch: "
                f"{root_count} != "
                f"{expected_pages}"
            )

        root_kids = list(
            root_pages.Kids
        )

        if expected_pages <= group_size:
            expected_groups = 1
        else:
            expected_groups = (
                expected_pages
                + group_size
                - 1
            ) // group_size

        if len(root_kids) != expected_groups:
            raise RuntimeError(
                "Page Tree group count mismatch: "
                f"{len(root_kids)} != "
                f"{expected_groups}"
            )

        counted_pages = 0

        for index, node in enumerate(
            root_kids
        ):
            if node.Type != Name.Pages:
                raise RuntimeError(
                    "Page Tree root child "
                    f"{index} is not /Pages"
                )

            node_count = int(
                node.Count
            )

            if (
                node_count < 1
                or node_count > group_size
            ):
                raise RuntimeError(
                    "Invalid Page Tree group size: "
                    f"{node_count}"
                )

            children = list(
                node.Kids
            )

            if len(children) != node_count:
                raise RuntimeError(
                    "Page Tree /Count and /Kids "
                    "length mismatch"
                )

            for page in children:
                if page.Type != Name.Page:
                    raise RuntimeError(
                        "Intermediate /Pages node "
                        "contains non-/Page child"
                    )

                try:
                    parent = page.Parent
                except AttributeError as exc:
                    raise RuntimeError(
                        "Page object has no /Parent"
                    ) from exc

                if parent.objgen != node.objgen:
                    raise RuntimeError(
                        "Page /Parent does not point "
                        "to expected intermediate node"
                    )

            counted_pages += node_count

        if counted_pages != expected_pages:
            raise RuntimeError(
                "Page Tree total count mismatch: "
                f"{counted_pages} != "
                f"{expected_pages}"
            )

    print(
        "  Page Tree validation: OK"
    )


def validate_pdf(
    original: Path,
    optimized: Path,
    expected_pages: int,
) -> None:
    if not optimized.exists():
        raise RuntimeError(
            "Optimized PDF was not created"
        )

    if optimized.stat().st_size <= 0:
        raise RuntimeError(
            "Optimized PDF is empty"
        )

    original_pages = get_page_count(
        original
    )

    optimized_pages = get_page_count(
        optimized
    )

    if original_pages != expected_pages:
        raise RuntimeError(
            "Original page count changed "
            "during processing: "
            f"{original_pages} != "
            f"{expected_pages}"
        )

    if optimized_pages != expected_pages:
        raise RuntimeError(
            "Optimized PDF page count mismatch: "
            f"{optimized_pages} != "
            f"{expected_pages}"
        )

    run_qpdf_check(
        [
            "qpdf",
            "--check",
            str(optimized),
        ],
        "qpdf PDF check",
    )

    run_qpdf_check(
        [
            "qpdf",
            "--check-linearization",
            str(optimized),
        ],
        "qpdf linearization check",
    )

    validate_page_tree(
        optimized,
        expected_pages,
        PAGE_TREE_GROUP_SIZE,
    )


def optimize_pdf(
    source: Path,
    destination: Path,
) -> None:
    print()
    print(
        "=" * 70
    )

    print(
        f"Optimizing: {source.name}"
    )

    print(
        "=" * 70
    )

    source_size = (
        source.stat().st_size
    )

    start_time = time.monotonic()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=".pdf-opt-",
    ) as temp_dir:
        temp_dir_path = Path(
            temp_dir
        )

        image_pdf = (
            temp_dir_path
            / "image.pdf"
        )

        balanced_pdf = (
            temp_dir_path
            / "balanced.pdf"
        )

        linearized_pdf = (
            temp_dir_path
            / "linearized.pdf"
        )

        expected_pages = (
            create_image_pdf(
                source,
                image_pdf,
            )
        )

        print(
            "  balancing Page Tree..."
        )

        rebalance_page_tree(
            image_pdf,
            balanced_pdf,
            PAGE_TREE_GROUP_SIZE,
        )

        print(
            "  linearizing..."
        )

        linearize_pdf(
            balanced_pdf,
            linearized_pdf,
        )

        print(
            "  validating..."
        )

        validate_pdf(
            source,
            linearized_pdf,
            expected_pages,
        )

        # 全検証が完了した場合だけ、
        # 公開用PDFをatomic replaceする。
        #
        # TemporaryDirectoryはdestinationと
        # 同じfilesystem上に作成しているため、
        # os.replace()はatomic。
        os.replace(
            linearized_pdf,
            destination,
        )

    elapsed = (
        time.monotonic()
        - start_time
    )

    output_size = (
        destination.stat().st_size
    )

    ratio = (
        output_size
        / source_size
        * 100
    )

    print(
        f"  pages: {expected_pages}"
    )

    print(
        f"  original: "
        f"{source_size / 1024 / 1024:.1f} MiB"
    )

    print(
        f"  optimized: "
        f"{output_size / 1024 / 1024:.1f} MiB"
    )

    print(
        f"  ratio: {ratio:.1f}%"
    )

    print(
        f"  elapsed: {elapsed:.1f}s"
    )


def main() -> int:
    print(
        "=" * 70
    )

    print(
        "SW2.5 PDF.js PDF Optimizer"
    )

    print(
        "=" * 70
    )

    print(
        f"Source: {SOURCE_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Target width: {TARGET_WIDTH}px"
    )

    print(
        f"JPEG quality: {JPEG_QUALITY}"
    )

    print(
        "Page Tree group size: "
        f"{PAGE_TREE_GROUP_SIZE}"
    )

    print(
        f"Optimizer version: "
        f"{OPTIMIZER_VERSION}"
    )

    if PAGE_TREE_GROUP_SIZE < 2:
        print(
            "ERROR: "
            "PDF_OPTIMIZER_PAGE_TREE_GROUP_SIZE "
            "must be >= 2",
            file=sys.stderr,
        )

        return 1

    if not SOURCE_DIR.exists():
        print(
            f"ERROR: source directory "
            f"does not exist: {SOURCE_DIR}",
            file=sys.stderr,
        )

        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        SOURCE_DIR.glob("*.pdf"),
        key=lambda path: path.name,
    )

    if not pdf_files:
        print(
            "ERROR: no PDF files found",
            file=sys.stderr,
        )

        return 1

    state = load_state()

    converted = 0
    skipped = 0
    failed = 0

    for source in pdf_files:
        destination = (
            OUTPUT_DIR
            / source.name
        )

        if not needs_rebuild(
            source,
            destination,
            state,
        ):
            print(
                f"SKIP: {source.name}"
            )

            skipped += 1
            continue

        try:
            optimize_pdf(
                source,
                destination,
            )

            state[
                source.name
            ] = source_signature(
                source
            )

            save_state(
                state
            )

            converted += 1

        except Exception as exc:
            failed += 1

            print(
                f"ERROR: {source.name}: "
                f"{exc}",
                file=sys.stderr,
            )

    print()

    print(
        "=" * 70
    )

    print(
        "Summary"
    )

    print(
        "=" * 70
    )

    print(
        f"Converted: {converted}"
    )

    print(
        f"Skipped:   {skipped}"
    )

    print(
        f"Failed:    {failed}"
    )

    if failed:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )