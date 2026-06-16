from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from core.privacy_controls import load_privacy_settings
from database.db import execute_insert


def log_audit_event(
    *,
    username: str | None,
    user_role: str,
    action_type: str | None,
    columns_affected: list[str],
    row_count_affected: int | None,
    success: bool,
    source: str | None,
) -> int | None:
    if load_privacy_settings()["logging_mode"] == "disabled":
        return None
    return execute_insert(
        """
        INSERT INTO audit_log (
            timestamp,
            username,
            user_role,
            action_type,
            columns_affected,
            row_count_affected,
            success,
            source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            username,
            user_role,
            action_type,
            json.dumps(columns_affected),
            row_count_affected,
            1 if success else 0,
            source,
        ),
    )
