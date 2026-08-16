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


class UserStoreError(Exception):
    pass


class UserAlreadyExistsError(UserStoreError):
    pass


class UserNotFoundError(UserStoreError):
    pass


class LastAdminError(UserStoreError):
    pass


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


def _row_to_user(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    result["is_admin"] = bool(result["is_admin"])
    result["is_active"] = bool(result["is_active"])
    return result


def init_user_store() -> None:
    with _connect() as connection:
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

    return _row_to_user(row)


def list_users() -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
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
            ORDER BY
                is_admin DESC,
                is_active DESC,
                display_name COLLATE NOCASE,
                email COLLATE NOCASE
            """
        ).fetchall()
    return [_row_to_user(row) for row in rows]


def create_user(
    email: str,
    display_name: str = "",
    *,
    is_admin: bool = False,
    is_active: bool = True,
) -> dict:
    normalized = _normalize_email(email)
    if not normalized or "@" not in normalized:
        raise UserStoreError("有効なメールアドレスを指定してください。")

    display_name = str(display_name or "").strip()
    if not display_name:
        display_name = normalized.split("@", 1)[0]

    now = _utc_now()
    try:
        with _connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    email,
                    display_name,
                    is_admin,
                    is_active,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized,
                    display_name,
                    int(bool(is_admin)),
                    int(bool(is_active)),
                    now,
                    now,
                ),
            )
    except sqlite3.IntegrityError as exc:
        raise UserAlreadyExistsError(
            "同じメールアドレスのユーザーが既に登録されています。"
        ) from exc

    user = get_user(normalized)
    if user is None:
        raise UserStoreError("ユーザーの登録後確認に失敗しました。")
    return user


def ensure_user(email: str) -> dict:
    """Cloudflare Accessで認証済みのユーザーを必要なら自動登録する。"""
    normalized = _normalize_email(email)
    if not normalized or "@" not in normalized:
        raise UserStoreError("有効なメールアドレスを指定してください。")

    existing = get_user(normalized)
    if existing is not None:
        return existing

    display_name = normalized.split("@", 1)[0]
    now = _utc_now()

    with _connect() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO users (
                email,
                display_name,
                is_admin,
                is_active,
                created_at,
                updated_at
            ) VALUES (?, ?, 0, 1, ?, ?)
            """,
            (normalized, display_name, now, now),
        )

    user = get_user(normalized)
    if user is None:
        raise UserStoreError("ユーザーの自動登録後確認に失敗しました。")
    return user


def update_user(
    current_email: str,
    *,
    email: str,
    display_name: str,
    is_admin: bool,
) -> dict:
    current_normalized = _normalize_email(current_email)
    new_normalized = _normalize_email(email)

    if not current_normalized:
        raise UserStoreError("更新対象メールアドレスがありません。")
    if not new_normalized or "@" not in new_normalized:
        raise UserStoreError("有効なメールアドレスを指定してください。")

    display_name = str(display_name or "").strip()
    if not display_name:
        display_name = new_normalized.split("@", 1)[0]

    now = _utc_now()

    with _connect() as connection:
        current = connection.execute(
            """
            SELECT email, is_admin
            FROM users
            WHERE email = ? COLLATE NOCASE
            """,
            (current_normalized,),
        ).fetchone()

        if current is None:
            raise UserNotFoundError("更新対象のユーザーが見つかりません。")

        was_admin = bool(current["is_admin"])
        will_be_admin = bool(is_admin)

        if was_admin and not will_be_admin:
            admin_count = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM users
                WHERE is_admin = 1
                """
            ).fetchone()["count"]
            if admin_count <= 1:
                raise LastAdminError(
                    "最後の管理者は管理者解除できません。"
                )

        if new_normalized != current_normalized:
            duplicate = connection.execute(
                """
                SELECT 1
                FROM users
                WHERE email = ? COLLATE NOCASE
                """,
                (new_normalized,),
            ).fetchone()
            if duplicate is not None:
                raise UserAlreadyExistsError(
                    "変更先メールアドレスは既に登録されています。"
                )

        connection.execute(
            """
            UPDATE users
            SET
                email = ?,
                display_name = ?,
                is_admin = ?,
                updated_at = ?
            WHERE email = ? COLLATE NOCASE
            """,
            (
                new_normalized,
                display_name,
                int(bool(is_admin)),
                now,
                current_normalized,
            ),
        )

    user = get_user(new_normalized)
    if user is None:
        raise UserStoreError("ユーザーの更新後確認に失敗しました。")
    return user


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
