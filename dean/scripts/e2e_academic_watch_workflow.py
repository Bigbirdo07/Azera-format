"""E2E driver for the Phase N school-roster Academic Watch workflow.

Walks the seven scripted turns from the spec against the synthetic academic
roster (tests/fixtures/academic_roster.xlsx) and asserts that the final state
is what the user would have produced clicking through the live app:

  1. "Show me all teachers that teach Biology."
  2. "Based on all teachers that have Biology, how many of the students have above a 2.00 GPA?"
  3. "Based on this, which students under which professor are not performing well based on GPA?"
  4. "Mark these students under Academic Watch."     (assistant asks to confirm)
  5. "yes"                                            (user confirms)
  6. "Export me a new Excel sheet."

Assertions:
  - A new workbook is written.
  - The original workbook file is unchanged (byte-for-byte).
  - Only Biology students with GPA < 2.0 have Academic Watch == "Yes".
  - The audit log records an academic_watch action.
  - All seven turns made it into the chat history.

Run:
    .venv/bin/python scripts/e2e_academic_watch_workflow.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.confirmed_actions import execute_academic_watch_action  # noqa: E402
from core.excel_loader import load_excel_workbook  # noqa: E402
from core.execution_dispatcher import execute_planned_request  # noqa: E402
from nlp.planner_router import plan_user_request  # noqa: E402


FIXTURE = REPO / "tests" / "fixtures" / "academic_roster.xlsx"


class _FakeUpload:
    def __init__(self, path: Path) -> None:
        self._bytes = path.read_bytes()
        self.name = path.name

    def getvalue(self) -> bytes:
        return self._bytes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"
    original_sha = _sha256(FIXTURE)
    loaded = load_excel_workbook(_FakeUpload(FIXTURE))
    sheets = loaded.sheets
    columns = {name: list(frame.columns) for name, frame in sheets.items()}

    chat_history: list[dict] = []
    audit_dir = Path(tempfile.mkdtemp(prefix="e2e_aw_"))
    audit_path = audit_dir / "audit_log.jsonl"
    output_dir = audit_dir / "outputs"

    state = {"active_filters": []}
    settings = {"strict_privacy_mode": True, "use_local_llm": False}

    def _say_user(text: str) -> None:
        chat_history.append({"role": "user", "content": text})
        print(f"\n> {text}")

    def _say_assistant(text: str) -> None:
        chat_history.append({"role": "assistant", "content": text})
        print(f"  assistant: {text[:160]}")

    def _route(message: str) -> dict:
        _say_user(message)
        routing = plan_user_request(
            user_message=message, sheets=sheets, sheet_columns=columns,
            selected_sheet="Students", conversation_state=state, settings=settings,
        )
        return routing

    # Turn 1 — list distinct Biology teachers.
    r = _route("Show me all teachers that teach Biology.")
    assert r["intent"] == "query"
    plan = r["plan"]
    assert plan["operation"] == "count_unique", plan
    assert plan["value_column"] == "Teacher", plan
    biology_filter = {"column": "Department", "operator": "equals", "value": "Biology"}
    assert biology_filter in plan["filters"], plan["filters"]
    state["active_filters"] = plan["filters"]
    _say_assistant(f"I found the Biology teachers. ({plan['operation']} on Teacher with Department=Biology)")

    # Turn 2 — count students under those teachers with GPA > 2.0.
    r = _route("Based on all teachers that have Biology, how many of the students have above a 2.00 GPA?")
    plan = r["plan"]
    assert plan["operation"] in {"count_rows", "count_unique"}, plan
    assert biology_filter in plan["filters"], plan["filters"]
    assert any(f["column"] == "GPA" and f["operator"] == "greater_than" for f in plan["filters"]), plan["filters"]
    state["active_filters"] = plan["filters"]
    _say_assistant("Keeping the Biology teacher group, I counted students with GPA above 2.00.")

    # Turn 3 — students grouped by professor, GPA < 2.0.
    r = _route("Based on this, which students under which professor are not performing well based on GPA?")
    plan = r["plan"]
    assert plan["group_by"] == "Teacher", plan
    assert biology_filter in plan["filters"], plan["filters"]
    assert any(f["column"] == "GPA" and f["operator"] == "less_than" for f in plan["filters"]), plan["filters"]
    state["active_filters"] = plan["filters"]
    _say_assistant("I interpreted 'not performing well based on GPA' as GPA below 2.00 and grouped by teacher.")

    # Turn 4 — academic watch request → confirmation gate.
    r = _route("Mark these students under Academic Watch.")
    assert r["intent"] == "academic_watch"
    assert r["requires_confirmation"]
    assert r["pending_type"] == "academic_watch"
    confirmation_filters = r["plan"]["filters"]
    assert biology_filter in confirmation_filters
    assert any(f["column"] == "GPA" and f["operator"] == "less_than" for f in confirmation_filters)
    _say_assistant(r["confirmation_reason"])

    # Turn 5 — user confirms; run the action.
    _say_user("yes")
    result = execute_academic_watch_action(
        filters=confirmation_filters,
        sheets=sheets,
        sheet="Students",
        output_dir=output_dir,
        audit_path=audit_path,
        request_summary="Mark these students under Academic Watch.",
    )
    assert result.success, result.message
    assert result.action_type == "academic_watch"
    assert result.rows_affected > 0
    _say_assistant(result.message)
    modified_workbook_path = Path(result.output_file)
    assert modified_workbook_path.exists()

    # Turn 6 — export. With the modified workbook present, that's what we offer.
    _say_user("Export me a new Excel sheet.")
    _say_assistant(f"Export ready: {modified_workbook_path.name}")

    # ----- Final assertions -----
    # 1. Original workbook is byte-for-byte unchanged.
    assert _sha256(FIXTURE) == original_sha, "ORIGINAL WORKBOOK WAS MODIFIED"

    # 2. The modified workbook has Academic Watch = 'Yes' only on the right rows.
    out = pd.read_excel(modified_workbook_path)
    marked = (out["Academic Watch"] == "Yes")
    biology_low = ((out["Department"] == "Biology") & (out["GPA"] < 2.0))
    assert marked.sum() == biology_low.sum(), (marked.sum(), biology_low.sum())
    assert (marked & ~biology_low).sum() == 0, "Academic Watch leaked onto non-matching rows"

    # 3. Row count preserved.
    assert len(out) == len(sheets["Students"])

    # 4. Audit log exists and records the academic_watch action.
    assert audit_path.exists()
    audit_entries = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
    assert any(e.get("action_type") == "academic_watch" for e in audit_entries)

    # 5. Chat history captured all turns.
    assert len(chat_history) == 12, f"expected 12 messages (6 user + 6 assistant), got {len(chat_history)}"

    # Clean up the temp output dir; keep nothing the user has to delete.
    shutil.rmtree(audit_dir, ignore_errors=True)

    print("\n=========================================================")
    print(f"E2E PASS — workflow marked {result.rows_affected} students")
    print("  original workbook unchanged ✓")
    print("  Academic Watch applied to only the matching rows ✓")
    print("  audit log records academic_watch action ✓")
    print("  chat history captured all 6 turns ✓")
    print("=========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
