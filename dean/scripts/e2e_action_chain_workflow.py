"""E2E driver for the Phase Q multi-action chain workflow.

Walks the spec scenario:

  1. Upload the academic synthetic workbook.
  2. Ask: "Show Biology students below 2.0 GPA."
  3. Ask: "Mark these students Academic Watch and export me a new Excel sheet."
  4. Confirm.
  5. Verify the output workbook exists, the original is byte-for-byte
     unchanged, Academic Watch is set only on the matching students,
     the audit log has one action_chain entry, and the chat history
     captured all turns.

Run:
    .venv/bin/python scripts/e2e_action_chain_workflow.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.confirmed_actions import execute_action_chain  # noqa: E402
from core.execution_dispatcher import execute_planned_request  # noqa: E402
from core.session_memory import SessionMemory  # noqa: E402
from nlp.planner_router import plan_user_request  # noqa: E402


FIXTURE = REPO / "tests" / "fixtures" / "academic_roster.xlsx"


@dataclass
class _Loaded:
    sheets: dict
    file_name: str = "academic_roster.xlsx"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"
    original_sha = _sha256(FIXTURE)
    df = pd.read_excel(FIXTURE)
    loaded = _Loaded(sheets={"Students": df})
    cols = {"Students": list(df.columns)}
    memory = SessionMemory()

    chat: list[dict] = []
    work_dir = Path(tempfile.mkdtemp(prefix="e2e_chain_"))
    audit_path = work_dir / "audit_log.jsonl"
    output_dir = work_dir / "outputs"

    def _user(text: str) -> None:
        chat.append({"role": "user", "content": text})
        print(f"\n> {text}")

    def _assistant(text: str) -> None:
        chat.append({"role": "assistant", "content": text})
        print(f"  assistant: {text[:200]}")

    def _ask(message: str) -> dict:
        _user(message)
        routing = plan_user_request(
            user_message=message, sheets=loaded.sheets,
            sheet_columns=cols, selected_sheet="Students",
            conversation_state=asdict(memory), settings={},
        )
        response = execute_planned_request(routing, loaded, settings={},
                                           request_summary=message)
        plan = routing.get("plan") or {}
        if routing.get("intent") == "query" and response.get("success"):
            memory.record_ask(
                request=message, query_plan=plan,
                result_description=response.get("description", "") or "",
                row_count=response.get("row_count"),
                columns_used=response.get("columns") or [],
                sheet=plan.get("sheet", ""),
                summary_table=response.get("result_preview") or [],
                top_group=response.get("top_group"),
            )
        return {"routing": routing, "response": response, "plan": plan}

    # Turn 1 — filter Biology + GPA < 2.0
    turn1 = _ask("Show Biology students below 2.0 GPA.")
    filters = turn1["plan"].get("filters") or []
    assert any(f["column"] == "Department" and f["value"] == "Biology" for f in filters)
    assert any(f["column"] == "GPA" and f["operator"] == "less_than" for f in filters)
    _assistant(turn1["response"]["message"])

    # Turn 2 — chained "mark + export"
    turn2 = _ask("Mark these students Academic Watch and export me a new Excel sheet.")
    routing2 = turn2["routing"]
    assert routing2["intent"] == "action_chain"
    assert routing2["requires_confirmation"]
    assert routing2["pending_type"] == "action_chain"
    _assistant(routing2["confirmation_reason"])

    # Turn 3 — user confirms; run the chain.
    _user("yes")
    result = execute_action_chain(
        actions=routing2["plan"]["actions"],
        filters=routing2["plan"]["filters"],
        sheets=loaded.sheets, sheet="Students",
        output_dir=output_dir, audit_path=audit_path,
        request_summary="Mark these students Academic Watch and export me a new Excel sheet.",
    )
    assert result.success, result.message
    _assistant(result.message)

    # ---- Final assertions ----
    # 1. Output workbook exists.
    output_path = Path(result.output_file)
    assert output_path.exists()

    # 2. Original workbook is byte-for-byte unchanged.
    assert _sha256(FIXTURE) == original_sha, "ORIGINAL WORKBOOK WAS MODIFIED"

    # 3. Academic Watch set on exactly the matching rows.
    out = pd.read_excel(output_path)
    expected = ((out["Department"] == "Biology") & (out["GPA"] < 2.0)).sum()
    marked = (out["Academic Watch"] == "Yes").sum()
    assert marked == expected, (marked, expected)

    # 4. Only ONE workbook produced (edit and export share the same file).
    output_files = list(output_dir.glob("*.xlsx"))
    assert len(output_files) == 1, output_files

    # 5. Audit log has one action_chain entry.
    assert audit_path.exists()
    entries = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["action_type"] == "action_chain"
    assert entry["actions"] == ["academic_watch", "export"]
    assert entry["original_modified"] is False
    assert entry["rows_affected"] == expected

    # 6. Chat history captured all turns.
    expected_messages = 6  # 3 user (Q1, Q2, yes) + 3 assistant
    assert len(chat) == expected_messages, len(chat)

    shutil.rmtree(work_dir, ignore_errors=True)
    print("\n=========================================================")
    print(f"E2E PASS — chain marked {result.rows_affected} students, "
          "single workbook output")
    print("  original unchanged ✓")
    print("  Academic Watch applied only to Biology + GPA<2.0 rows ✓")
    print("  single audit entry of type action_chain ✓")
    print(f"  chat history captured all {len(chat)} turns ✓")
    print("=========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
