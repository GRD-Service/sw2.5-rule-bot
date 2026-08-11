from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from google.api_core.exceptions import GoogleAPICallError
from google.cloud import vision


IMAGE_DIR = Path(
    os.getenv(
        "OCR_IMAGE_DIR",
        "/data/image",
    )
)

OCR_DIR = Path(
    os.getenv(
        "OCR_OUTPUT_DIR",
        "/data/ocr",
    )
)

OCR_WORK_DIR = Path(
    os.getenv(
        "OCR_WORK_DIR",
        "/data/ocr-work",
    )
)

LANGUAGE_HINTS = [
    value.strip()
    for value in os.getenv(
        "GOOGLE_VISION_LANGUAGE_HINTS",
        "ja",
    ).split(",")
    if value.strip()
]

MAX_RETRIES = int(
    os.getenv(
        "OCR_MAX_RETRIES",
        "5",
    )
)

RETRY_BASE_SECONDS = float(
    os.getenv(
        "OCR_RETRY_BASE_SECONDS",
        "2",
    )
)

REQUEST_INTERVAL_SECONDS = float(
    os.getenv(
        "OCR_REQUEST_INTERVAL_SECONDS",
        "0.1",
    )
)

REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "OCR_REQUEST_TIMEOUT_SECONDS",
        "180",
    )
)

PAGE_PATTERN = re.compile(
    r"^P(\d+)\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)


def load_json(
    path: Path,
):
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except Exception:
        return None


def atomic_write_json(
    path: Path,
    data,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temp_path,
        path,
    )


def get_image_signature(
    image_path: Path,
) -> dict:
    stat = image_path.stat()

    return {
        "filename": image_path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def collect_page_images(
    book_dir: Path,
) -> list[tuple[int, Path]]:
    pages = []

    for image_path in book_dir.iterdir():

        if not image_path.is_file():
            continue

        match = PAGE_PATTERN.match(
            image_path.name
        )

        if not match:
            continue

        page_number = int(
            match.group(1)
        )

        pages.append(
            (
                page_number,
                image_path,
            )
        )

    pages.sort(
        key=lambda item: item[0]
    )

    return pages


def validate_page_sequence(
    book_name: str,
    pages: list[tuple[int, Path]],
) -> None:

    if not pages:
        raise RuntimeError(
            f"No JPEG pages found: {book_name}"
        )

    page_numbers = [
        page_number
        for page_number, _
        in pages
    ]

    if len(page_numbers) != len(
        set(page_numbers)
    ):
        raise RuntimeError(
            f"Duplicate page numbers: {book_name}"
        )

    if page_numbers[0] != 1:
        raise RuntimeError(
            f"First page is not page 1: "
            f"{book_name}: "
            f"{page_numbers[0]}"
        )

    expected = list(
        range(
            1,
            page_numbers[-1] + 1,
        )
    )

    if page_numbers != expected:

        missing = sorted(
            set(expected)
            - set(page_numbers)
        )

        raise RuntimeError(
            f"Missing page images: "
            f"{book_name}: "
            f"{missing}"
        )


def final_json_is_complete(
    output_file: Path,
    book_name: str,
    pages: list[tuple[int, Path]],
) -> bool:
    """
    既存の完成済みOCR JSONが、
    現在のJPEGページ構成と一致するか確認する。

    一致していればGoogle Vision APIを一切呼ばず、
    書籍全体をSKIPする。
    """

    if not output_file.exists():
        return False

    data = load_json(
        output_file
    )

    if not isinstance(
        data,
        list,
    ):
        return False

    if len(data) != len(pages):
        return False

    expected_pages = [
        page_number
        for page_number, _
        in pages
    ]

    actual_pages = []

    for entry in data:

        if not isinstance(
            entry,
            dict,
        ):
            return False

        if entry.get("book") != book_name:
            return False

        page = entry.get(
            "page"
        )

        try:
            page = int(
                page
            )
        except (
            TypeError,
            ValueError,
        ):
            return False

        if "text" not in entry:
            return False

        if not isinstance(
            entry.get("text"),
            str,
        ):
            return False

        actual_pages.append(
            page
        )

    if actual_pages != expected_pages:
        return False

    return True


def work_file_is_current(
    work_file: Path,
    image_path: Path,
    page_number: int,
    book_name: str,
) -> bool:

    if not work_file.exists():
        return False

    data = load_json(
        work_file
    )

    if not isinstance(
        data,
        dict,
    ):
        return False

    if data.get(
        "book"
    ) != book_name:
        return False

    if data.get(
        "page"
    ) != page_number:
        return False

    if data.get(
        "image_signature"
    ) != get_image_signature(
        image_path
    ):
        return False

    if "text" not in data:
        return False

    return True


def perform_ocr(
    client: vision.ImageAnnotatorClient,
    image_path: Path,
) -> str:

    content = image_path.read_bytes()

    image = vision.Image(
        content=content
    )

    image_context = vision.ImageContext(
        language_hints=LANGUAGE_HINTS
    )

    last_exception = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:
            response = (
                client.document_text_detection(
                    image=image,
                    image_context=image_context,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            )

            if response.error.message:
                raise RuntimeError(
                    response.error.message
                )

            text = (
                response
                .full_text_annotation
                .text
            )

            return (
                text.strip()
                if text
                else ""
            )

        except (
            GoogleAPICallError,
            RuntimeError,
        ) as exc:

            last_exception = exc

            if attempt >= MAX_RETRIES:
                break

            wait_seconds = (
                RETRY_BASE_SECONDS
                * (2 ** (attempt - 1))
            )

            print(
                f"    retry "
                f"{attempt}/{MAX_RETRIES} "
                f"after {wait_seconds:.1f}s: "
                f"{exc}"
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"Vision OCR failed: "
        f"{last_exception}"
    )


def process_page(
    client: vision.ImageAnnotatorClient,
    book_name: str,
    page_number: int,
    image_path: Path,
    work_file: Path,
) -> bool:

    if work_file_is_current(
        work_file,
        image_path,
        page_number,
        book_name,
    ):

        print(
            f"  SKIP page {page_number}: "
            f"{image_path.name}"
        )

        return False

    print(
        f"  OCR page {page_number}: "
        f"{image_path.name}"
    )

    text = perform_ocr(
        client,
        image_path,
    )

    page_data = {
        "book": book_name,
        "page": page_number,
        "text": text,
        "image_signature": (
            get_image_signature(
                image_path
            )
        ),
    }

    atomic_write_json(
        work_file,
        page_data,
    )

    if REQUEST_INTERVAL_SECONDS > 0:

        time.sleep(
            REQUEST_INTERVAL_SECONDS
        )

    return True


def build_final_json(
    book_name: str,
    pages: list[tuple[int, Path]],
    work_dir: Path,
    output_file: Path,
) -> None:

    results = []

    for page_number, image_path in pages:

        work_file = (
            work_dir
            / f"P{page_number:05d}.json"
        )

        if not work_file_is_current(
            work_file,
            image_path,
            page_number,
            book_name,
        ):
            raise RuntimeError(
                f"Incomplete OCR data: "
                f"{book_name} "
                f"page {page_number}"
            )

        work_data = load_json(
            work_file
        )

        results.append(
            {
                "book": book_name,
                "page": page_number,
                "text": work_data.get(
                    "text",
                    "",
                ),
            }
        )

    atomic_write_json(
        output_file,
        results,
    )


def process_book(
    client: vision.ImageAnnotatorClient,
    book_dir: Path,
) -> tuple[int, int, bool]:

    book_name = book_dir.name

    pages = collect_page_images(
        book_dir
    )

    validate_page_sequence(
        book_name,
        pages,
    )

    output_file = (
        OCR_DIR
        / f"{book_name}.json"
    )

    print()
    print(
        "=" * 70
    )
    print(
        f"Book: {book_name}"
    )
    print(
        f"Pages: {len(pages)}"
    )
    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 既存の完成済みOCR JSONが正常なら、
    # Google Vision APIには一切アクセスしない。
    # --------------------------------------------------------

    if final_json_is_complete(
        output_file,
        book_name,
        pages,
    ):

        print(
            "  SKIP BOOK: "
            "completed OCR JSON already exists"
        )

        print(
            f"  output: {output_file}"
        )

        return (
            0,
            len(pages),
            True,
        )

    work_dir = (
        OCR_WORK_DIR
        / book_name
    )

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed = 0
    skipped = 0

    for page_number, image_path in pages:

        work_file = (
            work_dir
            / f"P{page_number:05d}.json"
        )

        changed = process_page(
            client,
            book_name,
            page_number,
            image_path,
            work_file,
        )

        if changed:
            processed += 1
        else:
            skipped += 1

    print(
        "  building final JSON..."
    )

    build_final_json(
        book_name,
        pages,
        work_dir,
        output_file,
    )

    print(
        f"  output: {output_file}"
    )

    return (
        processed,
        skipped,
        False,
    )


def find_book_directories() -> list[Path]:

    books = []

    for path in IMAGE_DIR.iterdir():

        if not path.is_dir():
            continue

        if collect_page_images(
            path
        ):
            books.append(
                path
            )

    books.sort(
        key=lambda path: path.name
    )

    return books


def main() -> int:

    print(
        "=" * 70
    )
    print(
        "SW2.5 Google Cloud Vision OCR"
    )
    print(
        "=" * 70
    )

    print(
        f"Image directory: "
        f"{IMAGE_DIR}"
    )

    print(
        f"OCR work directory: "
        f"{OCR_WORK_DIR}"
    )

    print(
        f"OCR output directory: "
        f"{OCR_DIR}"
    )

    print(
        "Vision mode: "
        "DOCUMENT_TEXT_DETECTION"
    )

    print(
        "Language hints: "
        f"{LANGUAGE_HINTS}"
    )

    print(
        "Google Cloud Storage: NOT USED"
    )

    if not IMAGE_DIR.exists():

        print(
            f"ERROR: image directory "
            f"does not exist: "
            f"{IMAGE_DIR}",
            file=sys.stderr,
        )

        return 1

    OCR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OCR_WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    books = find_book_directories()

    if not books:

        print(
            "ERROR: no book directories "
            "containing JPEG pages found",
            file=sys.stderr,
        )

        return 1

    print(
        f"Books found: "
        f"{len(books)}"
    )

    # 認証確認も兼ねてクライアントを生成するが、
    # 完成済み書籍についてはOCR APIは呼ばれない。
    client = (
        vision.ImageAnnotatorClient()
    )

    processed_pages = 0
    skipped_pages = 0
    completed_books = 0
    newly_built_books = 0
    failed_books = 0

    for book_dir in books:

        try:

            (
                processed,
                skipped,
                already_complete,
            ) = process_book(
                client,
                book_dir,
            )

            processed_pages += (
                processed
            )

            skipped_pages += (
                skipped
            )

            if already_complete:
                completed_books += 1
            else:
                newly_built_books += 1

        except Exception as exc:

            failed_books += 1

            print(
                f"ERROR: "
                f"{book_dir.name}: "
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
        f"OCR pages:        "
        f"{processed_pages}"
    )

    print(
        f"Skipped pages:    "
        f"{skipped_pages}"
    )

    print(
        f"Existing books:   "
        f"{completed_books}"
    )

    print(
        f"New/updated books:"
        f" {newly_built_books}"
    )

    print(
        f"Failed books:     "
        f"{failed_books}"
    )

    if failed_books:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )