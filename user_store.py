from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import sqlite3


USER_DB_PATH = Path(
    os.getenv("USER_DB_PATH", "/data/app/sw25.sqlite3")
)
INITIAL_ADMIN_EMAIL = os.getenv("INITIAL_ADMIN_EMAIL", "").strip().lower()
INITIAL_ADMIN_DISPLAY_NAME = os.getenv(
    "INITIAL_ADMIN_DISPLAY_NAME",
    "",
).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _connect() -> sqlite3.Connection:
    USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        USER_DB_PATH,
        timeout=5.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def init_user_store() -> None:
    with _connect() as connection:
        # WAL is persistent for the database and improves concurrent read/write behavior.
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY COLLATE NOCASE,
                display_name TEXT NOT NULL DEFAULT '',
                is_admin INTEGER NOT NULL DEFAULT 0 CHECK (is_admin IN (0, 1)),
                is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_seen_at TEXT
            )
            """
        )

        count = connection.execute(
            "SELECT COUNT(*) AS count FROM users"
        ).fetchone()["count"]

        if count == 0 and INITIAL_ADMIN_EMAIL:
            now = _utc_now()
            display_name = (
                INITIAL_ADMIN_DISPLAY_NAME
                or INITIAL_ADMIN_EMAIL.split("@", 1)[0]
            )
            connection.execute(
                """
                INSERT INTO users (
                    email,
                    display_name,
                    is_admin,
                    is_active,
                    created_at,
                    updated_at
                ) VALUES (?, ?, 1, 1, ?, ?)
                """,
                (
                    INITIAL_ADMIN_EMAIL,
                    display_name,
                    now,
                    now,
                ),
            )


def get_user(email: str) -> dict | None:
    normalized = _normalize_email(email)
    if not normalized:
        return None

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT
                email,
                display_name,
                is_admin,
                is_active,
                created_at,
                updated_at,
                last_seen_at
            FROM users
            WHERE email = ? COLLATE NOCASE
            """,
            (normalized,),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["is_admin"] = bool(result["is_admin"])
    result["is_active"] = bool(result["is_active"])
    return result


def touch_user_last_seen(email: str) -> None:
    normalized = _normalize_email(email)
    if not normalized:
        return

    now = _utc_now()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE users
            SET last_seen_at = ?, updated_at = ?
            WHERE email = ? COLLATE NOCASE
            """,
            (now, now, normalized),
        )
