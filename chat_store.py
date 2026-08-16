from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
import uuid


DB_PATH = Path(os.getenv("USER_DB_PATH", "/data/app/sw25.sqlite3"))


class ChatStoreError(Exception):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def init_chat_store() -> None:
    with _connect() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL COLLATE NOCASE,
                title TEXT NOT NULL DEFAULT '新しいチャット',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_email)
                    REFERENCES users(email)
                    ON UPDATE CASCADE
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chats_user_updated
                ON chats(user_email, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat_id
                ON messages(chat_id, id);

            CREATE TABLE IF NOT EXISTS chat_sources (
                chat_id TEXT NOT NULL,
                book TEXT NOT NULL,
                pdf_page INTEGER NOT NULL,
                logical_page INTEGER,
                first_seen_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                PRIMARY KEY (chat_id, book, pdf_page),
                FOREIGN KEY (chat_id)
                    REFERENCES chats(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chat_sources_chat
                ON chat_sources(chat_id, last_used_at DESC);
            """
        )


def _row_to_chat(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def create_chat(user_email: str, title: str = "") -> dict:
    email = str(user_email or "").strip().lower()
    if not email:
        raise ChatStoreError("ユーザーのメールアドレスがありません。")
    title = str(title or "").strip() or "新しいチャット"
    now = _utc_now()
    chat_id = str(uuid.uuid4())
    with _connect() as connection:
        connection.execute(
            "INSERT INTO chats (id, user_email, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, email, title, now, now),
        )
    return get_chat(chat_id, email)


def list_chats(user_email: str, limit: int = 100) -> list[dict]:
    email = str(user_email or "").strip().lower()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, user_email, title, created_at, updated_at
            FROM chats
            WHERE user_email = ? COLLATE NOCASE
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (email, max(1, int(limit))),
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat(chat_id: str, user_email: str) -> dict | None:
    email = str(user_email or "").strip().lower()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT id, user_email, title, created_at, updated_at
            FROM chats
            WHERE id = ? AND user_email = ? COLLATE NOCASE
            """,
            (chat_id, email),
        ).fetchone()
    return _row_to_chat(row)


def get_messages(chat_id: str, user_email: str) -> list[dict]:
    if get_chat(chat_id, user_email) is None:
        raise ChatStoreError("チャットが見つかりません。")
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id, role, content, metadata_json, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except (TypeError, ValueError):
            item["metadata"] = {}
        result.append(item)
    return result


def get_sources(chat_id: str, user_email: str) -> list[dict]:
    if get_chat(chat_id, user_email) is None:
        raise ChatStoreError("チャットが見つかりません。")
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT book, pdf_page, logical_page, first_seen_at, last_used_at
            FROM chat_sources
            WHERE chat_id = ?
            ORDER BY last_used_at DESC, first_seen_at DESC
            """,
            (chat_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def rename_chat(chat_id: str, user_email: str, title: str) -> dict:
    if get_chat(chat_id, user_email) is None:
        raise ChatStoreError("チャットが見つかりません。")
    title = str(title or "").strip() or "新しいチャット"
    now = _utc_now()
    with _connect() as connection:
        connection.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, chat_id),
        )
    return get_chat(chat_id, user_email)


def delete_chat(chat_id: str, user_email: str) -> None:
    if get_chat(chat_id, user_email) is None:
        raise ChatStoreError("チャットが見つかりません。")
    with _connect() as connection:
        connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def save_turn(
    chat_id: str,
    user_email: str,
    question: str,
    answer: str,
    *,
    metadata: dict | None = None,
    citations: list[dict] | None = None,
) -> None:
    chat = get_chat(chat_id, user_email)
    if chat is None:
        raise ChatStoreError("チャットが見つかりません。")

    now = _utc_now()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    citations = citations or []

    with _connect() as connection:
        message_count = connection.execute(
            "SELECT COUNT(*) AS count FROM messages WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()["count"]

        connection.execute(
            "INSERT INTO messages (chat_id, role, content, metadata_json, created_at) VALUES (?, 'user', ?, '{}', ?)",
            (chat_id, question, now),
        )
        connection.execute(
            "INSERT INTO messages (chat_id, role, content, metadata_json, created_at) VALUES (?, 'assistant', ?, ?, ?)",
            (chat_id, answer, metadata_json, now),
        )

        for citation in citations:
            book = str(citation.get("book") or "").strip()
            pdf_page = citation.get("pdf_page")
            logical_page = citation.get("page")
            if not book or pdf_page is None:
                continue
            try:
                pdf_page = int(pdf_page)
                logical_page = int(logical_page) if logical_page is not None else None
            except (TypeError, ValueError):
                continue
            connection.execute(
                """
                INSERT INTO chat_sources (
                    chat_id, book, pdf_page, logical_page, first_seen_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, book, pdf_page) DO UPDATE SET
                    logical_page = COALESCE(excluded.logical_page, chat_sources.logical_page),
                    last_used_at = excluded.last_used_at
                """,
                (chat_id, book, pdf_page, logical_page, now, now),
            )

        title = chat["title"]
        if message_count == 0 and title == "新しいチャット":
            compact = " ".join(str(question or "").split())
            title = compact[:40] + ("…" if len(compact) > 40 else "")
            if not title:
                title = "新しいチャット"

        connection.execute(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, chat_id),
        )
