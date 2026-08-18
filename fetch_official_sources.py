from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests


DEFAULT_CONFIG_PATH = Path("metadata/official_sources.json")
DEFAULT_OUTPUT_DIR = Path("official/raw")
MANIFEST_VERSION = 1


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value.strip())
                break


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    name: str
    index_url: str
    allowed_hosts: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...]
    allowed_content_types: tuple[str, ...]
    max_depth: int
    request_timeout_seconds: float
    request_interval_seconds: float
    user_agent: str


@dataclass(frozen=True)
class QueueItem:
    url: str
    depth: int
    discovered_from: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def load_source_configs(path: Path) -> list[SourceConfig]:
    data = load_json(path)
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("official_sources.json must contain a non-empty 'sources' array")

    result: list[SourceConfig] = []
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError("Each source entry must be an object")

        required = ("id", "name", "index_url", "allowed_hosts", "allowed_path_prefixes")
        missing = [key for key in required if not item.get(key)]
        if missing:
            raise ValueError(f"Missing required source fields {missing}: {item}")

        allowed_content_types = item.get(
            "allowed_content_types",
            ["text/html", "application/pdf"],
        )
        result.append(
            SourceConfig(
                source_id=str(item["id"]),
                name=str(item["name"]),
                index_url=str(item["index_url"]),
                allowed_hosts=tuple(str(x).lower() for x in item["allowed_hosts"]),
                allowed_path_prefixes=tuple(str(x) for x in item["allowed_path_prefixes"]),
                allowed_content_types=tuple(str(x).lower() for x in allowed_content_types),
                max_depth=max(0, int(item.get("max_depth", 2))),
                request_timeout_seconds=max(1.0, float(item.get("request_timeout_seconds", 30))),
                request_interval_seconds=max(0.0, float(item.get("request_interval_seconds", 0.5))),
                user_agent=str(
                    item.get(
                        "user_agent",
                        "SW25RuleBotOfficialSourceFetcher/1.0 (+local RAG ingestion)",
                    )
                ),
            )
        )
    return result


def canonicalize_url(raw_url: str, base_url: str | None = None) -> str | None:
    url = urljoin(base_url, raw_url) if base_url else raw_url
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if scheme not in {"http", "https"} or not hostname:
        return None

    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def is_allowed_url(url: str, config: SourceConfig) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme.lower() not in {"http", "https"}:
        return False
    if host not in config.allowed_hosts:
        return False
    return any(parts.path.startswith(prefix) for prefix in config.allowed_path_prefixes)


def normalize_content_type(value: str | None) -> str:
    if not value:
        return "application/octet-stream"
    return value.split(";", 1)[0].strip().lower()


def infer_content_type(url: str) -> str:
    guessed, _ = mimetypes.guess_type(urlsplit(url).path)
    return (guessed or "application/octet-stream").lower()


def is_allowed_content_type(content_type: str, config: SourceConfig) -> bool:
    return content_type in config.allowed_content_types


def safe_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned or "_"


def output_path_for_url(root: Path, config: SourceConfig, url: str, content_type: str) -> Path:
    parts = urlsplit(url)
    raw_parts = [safe_segment(p) for p in parts.path.split("/") if p]
    if not raw_parts:
        raw_parts = ["index"]

    last = raw_parts[-1]
    suffix = Path(last).suffix.lower()
    if content_type == "text/html" and suffix not in {".html", ".htm"}:
        if parts.path.endswith("/"):
            raw_parts.append("index.html")
        else:
            raw_parts[-1] = last + ".html"
    elif content_type == "application/pdf" and suffix != ".pdf":
        raw_parts[-1] = last + ".pdf"

    if parts.query:
        query_hash = hashlib.sha256(parts.query.encode("utf-8")).hexdigest()[:10]
        p = Path(raw_parts[-1])
        raw_parts[-1] = f"{p.stem}__q_{query_hash}{p.suffix}"

    return root / safe_segment(config.source_id) / Path(*raw_parts)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_links(content: bytes, encoding: str | None, page_url: str) -> list[str]:
    enc = encoding or "utf-8"
    try:
        text = content.decode(enc, errors="replace")
    except LookupError:
        text = content.decode("utf-8", errors="replace")

    parser = LinkExtractor()
    parser.feed(text)

    result: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        canonical = canonicalize_url(href, base_url=page_url)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def load_existing_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": MANIFEST_VERSION, "documents": []}
    try:
        return load_json(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load existing manifest: {path}: {exc}") from exc


def document_map(manifest: dict) -> dict[str, dict]:
    docs = manifest.get("documents", [])
    if not isinstance(docs, list):
        return {}
    result: dict[str, dict] = {}
    for doc in docs:
        if isinstance(doc, dict) and isinstance(doc.get("url"), str):
            result[doc["url"]] = doc
    return result


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(temp_path, path)


def fetch_source(
    config: SourceConfig,
    output_root: Path,
    *,
    dry_run: bool,
    force: bool,
) -> dict:
    source_root = output_root
    manifest_path = source_root / safe_segment(config.source_id) / "manifest.json"
    existing_manifest = load_existing_manifest(manifest_path)
    previous = document_map(existing_manifest)

    session = requests.Session()
    session.headers.update({"User-Agent": config.user_agent, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.1"})

    index_url = canonicalize_url(config.index_url)
    if not index_url or not is_allowed_url(index_url, config):
        raise ValueError(f"index_url is outside allowlist: {config.index_url}")

    queue: deque[QueueItem] = deque([QueueItem(index_url, 0, None)])
    queued: set[str] = {index_url}
    processed: set[str] = set()
    documents: list[dict] = []
    errors: list[dict] = []

    while queue:
        item = queue.popleft()
        if item.url in processed:
            continue
        processed.add(item.url)

        print(f"[{config.source_id}] depth={item.depth} GET {item.url}")
        try:
            response = session.get(
                item.url,
                timeout=config.request_timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            errors.append({
                "url": item.url,
                "discovered_from": item.discovered_from,
                "error": str(exc),
                "retrieved_at": utc_now_iso(),
            })
            print(f"  ERROR: {exc}", file=sys.stderr)
            continue

        final_url = canonicalize_url(response.url)
        if not final_url or not is_allowed_url(final_url, config):
            msg = f"redirected outside allowlist: {response.url}"
            errors.append({
                "url": item.url,
                "final_url": response.url,
                "discovered_from": item.discovered_from,
                "error": msg,
                "retrieved_at": utc_now_iso(),
            })
            print(f"  SKIP: {msg}", file=sys.stderr)
            continue

        content_type = normalize_content_type(response.headers.get("Content-Type"))
        if content_type == "application/octet-stream":
            content_type = infer_content_type(final_url)

        if not is_allowed_content_type(content_type, config):
            print(f"  SKIP content-type={content_type}")
            continue

        content = response.content
        digest = sha256_bytes(content)
        out_path = output_path_for_url(source_root, config, final_url, content_type)
        previous_doc = previous.get(final_url)
        unchanged = bool(
            previous_doc
            and previous_doc.get("sha256") == digest
            and out_path.exists()
        )

        if dry_run:
            write_status = "dry-run"
        elif unchanged and not force:
            write_status = "unchanged"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)
            write_status = "written"

        print(f"  {write_status}: {out_path} ({len(content)} bytes, {content_type})")

        record = {
            "source_id": config.source_id,
            "url": final_url,
            "requested_url": item.url,
            "discovered_from": item.discovered_from,
            "depth": item.depth,
            "content_type": content_type,
            "status_code": response.status_code,
            "sha256": digest,
            "size_bytes": len(content),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "retrieved_at": utc_now_iso(),
            "local_path": str(out_path),
        }
        documents.append(record)

        if content_type == "text/html" and item.depth < config.max_depth:
            for link in extract_links(content, response.encoding, final_url):
                if not is_allowed_url(link, config):
                    continue
                if link in processed or link in queued:
                    continue
                queued.add(link)
                queue.append(QueueItem(link, item.depth + 1, final_url))

        if config.request_interval_seconds > 0:
            time.sleep(config.request_interval_seconds)

    manifest = {
        "version": MANIFEST_VERSION,
        "source": {
            "id": config.source_id,
            "name": config.name,
            "index_url": index_url,
            "allowed_hosts": list(config.allowed_hosts),
            "allowed_path_prefixes": list(config.allowed_path_prefixes),
            "max_depth": config.max_depth,
        },
        "generated_at": utc_now_iso(),
        "document_count": len(documents),
        "error_count": len(errors),
        "documents": sorted(documents, key=lambda x: x["url"]),
        "errors": errors,
    }

    if not dry_run:
        write_json_atomic(manifest_path, manifest)
        print(f"  manifest: {manifest_path}")

    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch allowlisted official SW2.5 source pages/PDFs for later RAG normalization."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", action="append", default=[], help="Only fetch the specified source id; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and inspect without writing raw files or manifests")
    parser.add_argument("--force", action="store_true", help="Rewrite files even when SHA-256 is unchanged")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    configs = load_source_configs(args.config)
    selected = set(args.source)
    if selected:
        configs = [cfg for cfg in configs if cfg.source_id in selected]
        missing = selected - {cfg.source_id for cfg in configs}
        if missing:
            print(f"Unknown source id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_errors = 0
    for config in configs:
        manifest = fetch_source(
            config,
            args.output_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
        total_errors += int(manifest.get("error_count", 0))

    if total_errors:
        print(f"Completed with {total_errors} fetch error(s).", file=sys.stderr)
        return 1
    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
