from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BUILD_VERSION = 2

SUPPORTED_OPERATIONS = {
    "replace",
    "append",
    "delete",
}


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def write_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write(
            "\n"
        )


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )
            handle.write(
                "\n"
            )


SOURCE_TITLE_MAP = {
    "SW2.5_1": "ソード・ワールド2.5 ルールブックI",
    "SW2.5_2": "ソード・ワールド2.5 ルールブックII",
    "SW2.5_3": "ソード・ワールド2.5 ルールブックIII",
    "SW2.5_DX": "ソード・ワールド2.5 ルールブックDX",
    "SW2.5_CBB": "ソード・ワールド2.5 キャラクタービルディングブック",
    "SW2.5_arcanerelik": "ソード・ワールド2.5 アーケインレリック",
    "SW2.5_magusarts": "ソード・ワールド2.5 メイガスアーツ",
    "SW2.5_monstrouslore": "ソード・ワールド2.5 モンストラスロア",
    "SW2.5_epictreasury": "ソード・ワールド2.5 エピックトレジャリー",
    "SW2.5_battlemastery": "ソード・ワールド2.5 バトルマスタリー",
    "SW2.5_barbarous": "ソード・ワールド2.5 バルバロスレイジ",
    "SW2.5_outlaw": "ソード・ワールド2.5 アウトロープロファイルブック",
    "SW2.5_daemonsline": "ソード・ワールド2.5 デモンズライン",
    "SW2.5_kingsfall": "ソード・ワールド2.5 キングスフォール",
    "SW2.5_granzale": "ソード・ワールド2.5 グランゼール",
    "SW2.5_vicecity": "ソード・ワールド2.5 ヴァイスシティ",
    "SW2.5_raxialife": "ソード・ワールド2.5 ラクシアライフ",
    "SW2.5_travelsinalfreim": "ソード・ワールド2.5 アルフレイム見聞録",
    "SW2.5_star": "ソード・ワールド2.5 星をつかむ迷宮",
    "SW2.5_tyrant": "ソード・ワールド2.5 タイラント",
    "SW2.5_infinite": "ソード・ワールド2.5 インフィニットコロッセオ",
    "SW2.5_abyssbreaker": "ソード・ワールド2.5 アビスブレイカー",
    "SW2.5_ursyla": "ソード・ワールド2.5 ウルシラ地方",
    "SW2.5_leondar": "ソード・ワールド2.5 レオンダール地方",
    "SW2.5_barbarousSaga": "ソード・ワールド2.5 バルバロスサーガ",
}


def normalize_source_title(
    source_key: str | None,
) -> str:
    if not source_key:
        return "ソード・ワールド2.5"

    mapped = SOURCE_TITLE_MAP.get(
        source_key
    )

    if mapped:
        return mapped

    return source_key.replace(
        "_",
        " ",
    )


def page_label(
    target_page: int | str | None,
) -> str:
    if target_page is None:
        return "ページ指定なし"

    return f"{target_page}頁"


def build_replace_text(
    *,
    title: str,
    target_page: int | str | None,
    location: str | None,
    before: str | None,
    after: str | None,
    note: str | None,
) -> str:
    parts = [
        f"{title} {page_label(target_page)}の公式エラッタ。",
    ]

    if location:
        parts.append(
            f"対象箇所は「{location}」。"
        )

    if before:
        parts.append(
            f"訂正前は「{before}」。"
        )

    if after:
        parts.append(
            f"訂正後は「{after}」。"
        )

    if note:
        parts.append(
            f"注記: {note}"
        )

    return "\n".join(
        parts
    )


def build_append_text(
    *,
    title: str,
    target_page: int | str | None,
    location: str | None,
    append_instruction: str | None,
    append_text: str | None,
    note: str | None,
) -> str:
    parts = [
        f"{title} {page_label(target_page)}の公式追加・追記。",
    ]

    if location:
        parts.append(
            f"対象箇所は「{location}」。"
        )

    if append_instruction:
        parts.append(
            f"追記指示: {append_instruction}"
        )

    if append_text:
        parts.append(
            f"追記内容: {append_text}"
        )

    if note:
        parts.append(
            f"注記: {note}"
        )

    return "\n".join(
        parts
    )


def build_delete_text(
    *,
    title: str,
    target_page: int | str | None,
    location: str | None,
    delete_instruction: str | None,
    delete_text: str | None,
    note: str | None,
) -> str:
    parts = [
        f"{title} {page_label(target_page)}の公式削除エラッタ。",
    ]

    if location:
        parts.append(
            f"対象箇所は「{location}」。"
        )

    if delete_instruction:
        parts.append(
            f"削除指示: {delete_instruction}"
        )

    if delete_text:
        parts.append(
            f"削除対象: {delete_text}"
        )

    if note:
        parts.append(
            f"注記: {note}"
        )

    return "\n".join(
        parts
    )


def build_chunk_text(
    *,
    title: str,
    record: dict,
) -> str:
    operation = record.get(
        "operation"
    )

    common = {
        "title": title,
        "target_page": record.get(
            "target_page"
        ),
        "location": record.get(
            "location"
        ),
        "note": record.get(
            "note"
        ),
    }

    if operation == "replace":
        return build_replace_text(
            **common,
            before=record.get(
                "before"
            ),
            after=record.get(
                "after"
            ),
        )

    if operation == "append":
        return build_append_text(
            **common,
            append_instruction=record.get(
                "append_instruction"
            ),
            append_text=record.get(
                "append_text"
            ),
        )

    if operation == "delete":
        return build_delete_text(
            **common,
            delete_instruction=record.get(
                "delete_instruction"
            ),
            delete_text=record.get(
                "delete_text"
            ),
        )

    raise ValueError(
        f"Unsupported operation: {operation}"
    )


def split_recovery_metadata(
    note: str | None,
) -> tuple[str | None, str | None]:
    if note == "rescued_from_diagnostic":
        return None, "diagnostic_rescue"

    return note, None


def build_chunk(
    *,
    document: dict,
    record: dict,
    source_file: Path,
) -> dict:
    source_key = document.get(
        "source_key"
    )

    title = normalize_source_title(
        source_key
    )

    operation = record.get(
        "operation"
    )

    official_note, recovery_method = split_recovery_metadata(
        record.get(
            "note"
        )
    )

    record_for_text = dict(
        record
    )
    record_for_text[
        "note"
    ] = official_note

    record_index = record.get(
        "record_index"
    )

    chunk_id = (
        f"{source_key or source_file.stem}"
        f":errata:{record_index}"
    )

    text = build_chunk_text(
        title=title,
        record=record_for_text,
    )

    metadata = {
        "chunk_id": chunk_id,
        "source_class": "official_correction",
        "source_type": document.get(
            "source_type",
            "ERRATA",
        ),
        "official": True,
        "authority": "GroupSNE",
        "authority_priority": 300,
        "source_id": document.get(
            "source_id"
        ),
        "source_name": document.get(
            "source_name"
        ),
        "source_key": source_key,
        "source_url": document.get(
            "source_url"
        ),
        "source_sha256": document.get(
            "source_sha256"
        ),
        "source_last_modified": document.get(
            "source_last_modified"
        ),
        "source_file": str(
            source_file
        ),
        "record_index": record_index,
        "target_page": record.get(
            "target_page"
        ),
        "location": record.get(
            "location"
        ),
        "operation": operation,
        "before": record.get(
            "before"
        ),
        "after": record.get(
            "after"
        ),
        "append_instruction": record.get(
            "append_instruction"
        ),
        "append_text": record.get(
            "append_text"
        ),
        "append_location": record.get(
            "append_location"
        ),
        "delete_instruction": record.get(
            "delete_instruction"
        ),
        "delete_text": record.get(
            "delete_text"
        ),
        "delete_location": record.get(
            "delete_location"
        ),
        "note": official_note,
        "recovery_method": recovery_method,
        "parse_status": record.get(
            "parse_status"
        ),
        "extract_method": record.get(
            "extract_method"
        ),
        "source_pdf_pages": record.get(
            "source_pdf_pages"
        ),
        "rag_eligible": True,
        "override_candidate": operation
        in {
            "replace",
            "delete",
        },
        "supplement_candidate": operation
        == "append",
    }

    return {
        "id": chunk_id,
        "text": text,
        "metadata": metadata,
    }


def iter_input_files(
    input_dir: Path,
) -> list[Path]:
    return sorted(
        path
        for path in input_dir.glob(
            "*.layout.normalized.json"
        )
        if path.is_file()
    )


def build_official_chunks(
    *,
    input_dir: Path,
    exclude_source_keys: set[str],
) -> tuple[
    list[dict],
    list[dict],
]:
    chunks: list[dict] = []
    documents: list[dict] = []

    for path in iter_input_files(
        input_dir
    ):
        document = load_json(
            path
        )

        source_key = str(
            document.get(
                "source_key"
            )
            or ""
        )

        if source_key in exclude_source_keys:
            documents.append(
                {
                    "source_file": str(
                        path
                    ),
                    "source_key": source_key,
                    "status": "EXCLUDED",
                    "chunk_count": 0,
                    "reason": "excluded_source_key",
                }
            )
            continue

        records = document.get(
            "records"
        )

        if not isinstance(
            records,
            list,
        ):
            documents.append(
                {
                    "source_file": str(
                        path
                    ),
                    "source_key": source_key,
                    "status": "INVALID",
                    "chunk_count": 0,
                    "reason": "records_not_array",
                }
            )
            continue

        document_chunks: list[dict] = []
        skipped = Counter()

        for record in records:
            if not isinstance(
                record,
                dict,
            ):
                skipped[
                    "record_not_object"
                ] += 1
                continue

            if (
                record.get(
                    "parse_status"
                )
                != "LAYOUT_PARSED"
            ):
                skipped[
                    "not_layout_parsed"
                ] += 1
                continue

            operation = record.get(
                "operation"
            )

            if operation not in SUPPORTED_OPERATIONS:
                skipped[
                    f"unsupported_operation:{operation}"
                ] += 1
                continue

            document_chunks.append(
                build_chunk(
                    document=document,
                    record=record,
                    source_file=path,
                )
            )

        chunks.extend(
            document_chunks
        )

        documents.append(
            {
                "source_file": str(
                    path
                ),
                "source_key": source_key,
                "status": "BUILT",
                "record_count": len(
                    records
                ),
                "chunk_count": len(
                    document_chunks
                ),
                "skipped_counts": dict(
                    sorted(
                        skipped.items()
                    )
                ),
            }
        )

    return chunks, documents


def parse_args(
    argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build RAG-ready official SW2.5 errata chunks "
            "from layout-normalized JSON files."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/data/official/layout_normalized/groupsne_sw25_errata"
        ),
    )

    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "/data/official/rag/groupsne_sw25_errata/"
            "official_errata_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path(
            "/data/official/rag/groupsne_sw25_errata/"
            "official_errata_chunks_manifest.json"
        ),
    )

    parser.add_argument(
        "--exclude-source-key",
        action="append",
        default=[
            "SW2.5_water",
        ],
        help=(
            "Exclude a source_key from RAG chunk generation. "
            "May be specified multiple times."
        ),
    )

    return parser.parse_args(
        list(argv)
        if argv is not None
        else None
    )


def main(
    argv: Iterable[str] | None = None,
) -> int:
    args = parse_args(
        argv
    )

    if not args.input_dir.exists():
        print(
            f"Input directory not found: {args.input_dir}"
        )
        return 2

    excluded = {
        value.strip()
        for value in args.exclude_source_key
        if value.strip()
    }

    chunks, documents = build_official_chunks(
        input_dir=args.input_dir,
        exclude_source_keys=excluded,
    )

    write_jsonl(
        args.output_jsonl,
        chunks,
    )

    operation_counts = Counter(
        chunk["metadata"]["operation"]
        for chunk in chunks
    )

    source_counts = Counter(
        chunk["metadata"]["source_key"]
        for chunk in chunks
    )

    manifest = {
        "version": BUILD_VERSION,
        "generated_at": utc_now_iso(),
        "input_dir": str(
            args.input_dir
        ),
        "output_jsonl": str(
            args.output_jsonl
        ),
        "chunk_count": len(
            chunks
        ),
        "document_count": len(
            documents
        ),
        "excluded_source_keys": sorted(
            excluded
        ),
        "operation_counts": dict(
            sorted(
                operation_counts.items()
            )
        ),
        "source_counts": dict(
            sorted(
                source_counts.items()
            )
        ),
        "documents": documents,
    }

    write_json(
        args.manifest_out,
        manifest,
    )

    print(
        f"Chunks: {len(chunks)}"
    )
    print(
        f"Operations: {dict(sorted(operation_counts.items()))}"
    )
    print(
        f"Sources: {len(source_counts)}"
    )
    print(
        f"JSONL: {args.output_jsonl}"
    )
    print(
        f"Manifest: {args.manifest_out}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
