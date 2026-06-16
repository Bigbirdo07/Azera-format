from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from database.db import execute_insert
from core.privacy_controls import load_privacy_settings


def hash_filename(file_name: str) -> str:
    return hashlib.sha256(Path(file_name).name.encode("utf-8")).hexdigest()


def log_user_request(
    *,
    file_name: str,
    sheet_name: str | None,
    original_request: str,
    generated_command: dict[str, Any],
    parser_confidence: float | None,
    action_type: str | None,
    parser_source: str | None = None,
    success: bool,
    error_message: str | None = None,
) -> int | None:
    privacy = load_privacy_settings()
    if privacy["logging_mode"] == "disabled":
        return None

    if privacy["logging_mode"] == "metadata_only":
        original_request = ""
        generated_command = _metadata_only_command(generated_command)

    return execute_insert(
        """
        INSERT INTO user_requests (
            timestamp,
            filename_hash,
            sheet_name,
            original_request,
            generated_command_json,
            parser_confidence,
            parser_source,
            action_type,
            success,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            hash_filename(file_name),
            sheet_name,
            original_request,
            json.dumps(generated_command, sort_keys=True),
            parser_confidence,
            parser_source,
            action_type,
            1 if success else 0,
            error_message,
        ),
    )


def _metadata_only_command(command: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": command.get("action"),
        "sheet": command.get("sheet"),
        "columns": _extract_columns(command),
    }


def _extract_columns(command: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key in ["column", "group_by", "value_column", "sum_column"]:
        if command.get(key):
            columns.append(str(command[key]))
    for condition in command.get("conditions", []):
        if condition.get("column"):
            columns.append(str(condition["column"]))
    return sorted(set(columns))
