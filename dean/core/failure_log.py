"""Capture every ask the assistant couldn't answer — clarify, unavailable,
unsupported, or a query/edit that raised — alongside the workbook it was
asked of. Persisted to ``logs/failed_asks.jsonl`` so a daily skim shows the
real-world phrasings the planner needs to learn next.

Design:
- Write-only path during normal operation. Failures here MUST NOT break the
  chat — every public function swallows OSError and writes nothing.
- One JSON line per failure; readers stream the file newest-first.
- Schema is open: the planner can grow new fields and old log lines stay
  parseable.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path("logs/failed_asks.jsonl")
_MAX_COLUMNS_RECORDED = 200  # guard against pathological column counts


def log_failure(
    *,
    user_message: str,
    sheet_name: str,
    columns: list[str] | None,
    intent: str,
    reason: str,
    routing: dict[str, Any] | None = None,
    path: Path | None = None,
) -> None:
    """Append one failure record. Silently no-ops on IO failure."""
    target = Path(path or DEFAULT_PATH)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_message": (user_message or "").strip(),
            "sheet": sheet_name or "",
            "columns": list(columns or [])[:_MAX_COLUMNS_RECORDED],
            "intent": intent or "",
            "reason": reason or "",
        }
        if routing:
            entry["routing"] = {
                "plan_source": routing.get("plan_source"),
                "confidence": routing.get("confidence"),
                "llm_used": routing.get("llm_used"),
                "validation_status": (routing.get("validation") or {}).get("status"),
                "fallback_reason": routing.get("fallback_reason"),
                "band": routing.get("band"),
                "narration": routing.get("narration"),
                "assumption_note": routing.get("assumption_note"),
            }
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False))
            fh.write("\n")
    except OSError:
        # Disk full / permission denied — never break the chat over telemetry.
        return


def read_failures(
    *, limit: int = 100, path: Path | None = None
) -> list[dict[str, Any]]:
    """Return up to ``limit`` failure records, newest first. Empty on IO error
    or missing file."""
    target = Path(path or DEFAULT_PATH)
    if not target.exists():
        return []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    # Walk newest-first by reversing — JSONL is append-only, so the tail is
    # the latest entries.
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
        if len(records) >= limit:
            break
    return records


def clear_failures(*, path: Path | None = None) -> None:
    """Remove the failure log. Useful after a triage pass."""
    target = Path(path or DEFAULT_PATH)
    try:
        if target.exists():
            target.unlink()
    except OSError:
        return
