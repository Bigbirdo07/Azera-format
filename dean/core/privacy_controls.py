from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.correction_manager import (
    EXPORTS_DIR,
    sync_learning_files,
)
from core.paths import data_dir
from database.db import execute_write, fetch_all


SETTINGS_PATH = data_dir("config") / "privacy_settings.json"
DEFAULT_PRIVACY_SETTINGS = {
    "logging_mode": "metadata_only",
    "block_delete_row_actions": True,
    "warn_row_threshold": 100,
}
LOGGING_MODES = ["disabled", "metadata_only", "command_metadata"]


def load_privacy_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        save_privacy_settings(DEFAULT_PRIVACY_SETTINGS)
        return dict(DEFAULT_PRIVACY_SETTINGS)
    try:
        loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(DEFAULT_PRIVACY_SETTINGS)
    return {**DEFAULT_PRIVACY_SETTINGS, **loaded}


def save_privacy_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean = {**DEFAULT_PRIVACY_SETTINGS, **settings}
    if clean["logging_mode"] not in LOGGING_MODES:
        clean["logging_mode"] = DEFAULT_PRIVACY_SETTINGS["logging_mode"]
    SETTINGS_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")


def clear_all_local_logs() -> None:
    for table in ["audit_log", "user_requests", "corrections"]:
        execute_write(f"DELETE FROM {table}")


def clear_learned_corrections() -> None:
    for table in ["learned_synonyms", "learned_column_mappings", "corrections"]:
        execute_write(f"DELETE FROM {table}")
    sync_learning_files()


def export_privacy_report() -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EXPORTS_DIR / "privacy_report.json"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "privacy_settings": load_privacy_settings(),
        "local_tables": {
            "user_requests": _count("user_requests"),
            "corrections": _count("corrections"),
            "learned_synonyms": _count("learned_synonyms"),
            "learned_column_mappings": _count("learned_column_mappings"),
            "audit_log": _count("audit_log"),
        },
        "cloud_apis": "not configured",
        "telemetry": "not implemented",
        "analytics": "not implemented",
        "remote_logging": "not implemented",
        "student_rows_logged": False,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def logging_enabled() -> bool:
    return load_privacy_settings()["logging_mode"] != "disabled"


def _count(table_name: str) -> int:
    return int(fetch_all(f"SELECT COUNT(*) AS count FROM {table_name}")[0]["count"])
