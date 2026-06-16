from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from core.paths import data_dir

# The schema ships next to this module (read-only in a packaged build); the
# database itself lives in the writable per-install data directory.
DATABASE_DIR = data_dir("database")
DATABASE_PATH = DATABASE_DIR / "local_learning.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _ensure_column(connection, "user_requests", "parser_source", "TEXT")
        connection.commit()


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    existing_columns = {
        row["name"]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in existing_columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def execute_insert(query: str, parameters: Iterable[Any]) -> int:
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(query, tuple(parameters))
        connection.commit()
        return int(cursor.lastrowid)


def execute_write(query: str, parameters: Iterable[Any] = ()) -> None:
    initialize_database()
    with get_connection() as connection:
        connection.execute(query, tuple(parameters))
        connection.commit()


def fetch_all(query: str, parameters: Iterable[Any] = ()) -> list[dict[str, Any]]:
    initialize_database()
    with get_connection() as connection:
        rows = connection.execute(query, tuple(parameters)).fetchall()
    return [dict(row) for row in rows]
