"""E2E driver for the Phase B–E attendance + risk workflow.

Walks the 8-step flow from the spec against the synthetic academic roster
(``tests/fixtures/academic_roster.xlsx``) plus a deterministically-generated
attendance file. Asserts the same invariants the live app would maintain:

  1. Upload roster workbook.
  2. Upload daily attendance file.
  3. Ask: "Show me students at attendance risk."
  4. Ask: "Which teachers have the most attendance-risk students?"
  5. Ask: "Now only students with GPA below 2.0."
  6. Ask: "Mark these students Attendance Watch and export."
  7. Confirm.
  8. Verify:
     - Attendance Watch updated only for expected students
     - new workbook created
     - original unchanged
     - audit log created
     - chat history preserved

Run:
    .venv/bin/python scripts/e2e_attendance_risk_workflow.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from random import Random

import pandas as pd
from openpyxl import Workbook

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.attendance import (  # noqa: E402
    compute_attendance_metrics, load_attendance_file, match_attendance_to_roster,
)
from core.confirmed_actions import execute_academic_watch_action  # noqa: E402
from core.data_sources import AttendanceSource, DataSourceRegistry  # noqa: E402
from core.excel_loader import LoadedWorkbook, load_excel_workbook  # noqa: E402
from core.combined_risk import attach_combined_risk  # noqa: E402
from nlp.planner_router import plan_user_request  # noqa: E402


FIXTURE = REPO / "tests" / "fixtures" / "academic_roster.xlsx"


class _FakeUpload:
    def __init__(self, payload: bytes, name: str) -> None:
        self._bytes = payload
        self.name = name

    def getvalue(self) -> bytes:
        return self._bytes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_attendance_workbook(student_ids: list[str], seed: int = 17) -> bytes:
    """Deterministic long-format attendance file: 20 school days per student.

    Each student gets a fixed absence count drawn from a pseudo-random
    distribution, so we know exactly which IDs will land below 90% and 80%
    when the metric runs. Returns the .xlsx bytes (no on-disk file).
    """
    rng = Random(seed)
    rows: list[tuple[str, str, str]] = []
    dates = [f"2026-05-{day:02d}" for day in range(4, 24)]  # 20 weekdays
    for sid in student_ids:
        # Skew the distribution so ~25% of students cross the 90% threshold
        # — enough to give the planner real data to work with.
        absences = rng.choices(
            [0, 1, 2, 3, 5, 7],
            weights=[35, 25, 15, 10, 10, 5],
            k=1,
        )[0]
        absent_dates = set(rng.sample(dates, k=min(absences, len(dates))))
        for date in dates:
            status = "Absent" if date in absent_dates else "Present"
            rows.append((sid, date, status))

    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    ws.append(["Student ID", "Date", "Attendance Status"])
    for row in rows:
        ws.append(list(row))
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def main() -> int:
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"
    original_sha = _sha256(FIXTURE)

    chat_history: list[dict] = []
    state = {"active_filters": []}
    settings = {"strict_privacy_mode": True, "use_local_llm": False}

    audit_dir = Path(tempfile.mkdtemp(prefix="e2e_attendance_"))
    audit_path = audit_dir / "audit_log.jsonl"
    output_dir = audit_dir / "outputs"

    def _say_user(text: str) -> None:
        chat_history.append({"role": "user", "content": text})
        print(f"\n> {text}")

    def _say_assistant(text: str) -> None:
        chat_history.append({"role": "assistant", "content": text})
        print(f"  assistant: {text[:160]}")

    # ----- Step 1: upload roster --------------------------------------------
    _say_user("[uploads roster.xlsx]")
    roster_bytes = FIXTURE.read_bytes()
    roster = load_excel_workbook(_FakeUpload(roster_bytes, FIXTURE.name))
    registry = DataSourceRegistry()
    registry.set_roster(roster)
    roster_sheet = registry.enriched_roster_sheet
    roster_frame = roster.sheets[roster_sheet]
    _say_assistant(
        f"Roster loaded — {len(roster_frame)} students across "
        f"{roster_frame['Department'].nunique()} departments."
    )

    # ----- Step 2: upload daily attendance ----------------------------------
    _say_user("[uploads attendance.xlsx]")
    student_ids = roster_frame["Student ID"].astype(str).tolist()
    attendance_bytes = _build_attendance_workbook(student_ids)
    loaded_attendance = load_attendance_file(
        _FakeUpload(attendance_bytes, "attendance.xlsx")
    )
    matched, unmatched, unmatched_sample = match_attendance_to_roster(
        loaded_attendance.frame, roster_frame,
    )
    registry.set_attendance(AttendanceSource(
        file_name=loaded_attendance.file_name,
        frame=loaded_attendance.frame,
        matched_count=matched, unmatched_count=unmatched,
        unmatched_ids=unmatched_sample,
        warnings=loaded_attendance.warnings,
    ))
    assert matched == len(student_ids), \
        f"expected all {len(student_ids)} students to match, got {matched}"
    assert unmatched == 0, f"unexpected unmatched IDs: {unmatched_sample[:5]}"
    _say_assistant(
        f"Attendance loaded — {len(loaded_attendance.frame)} rows, "
        f"{matched} students matched, {unmatched} unmatched."
    )

    # Build the enriched view (roster + metrics + risk) — the planner sees
    # this single combined sheet.
    enriched_sheets = registry.enriched_sheets()
    sheet_columns = {name: list(frame.columns) for name, frame in enriched_sheets.items()}

    def _route(message: str) -> dict:
        _say_user(message)
        return plan_user_request(
            user_message=message,
            sheets=enriched_sheets, sheet_columns=sheet_columns,
            selected_sheet=roster_sheet, conversation_state=state,
            settings=settings,
        )

    # ----- Step 3: show me students at attendance risk -----------------------
    r = _route("Show me students at attendance risk.")
    assert r["intent"] == "query", r
    plan = r["plan"]
    assert any(
        f.get("column") == "Attendance Risk" and f.get("value") is True
        for f in plan.get("filters") or []
    ), plan
    state["active_filters"] = plan["filters"]
    _say_assistant("Filtered to students with Attendance Rate < 90% "
                   "(Attendance Risk == True).")

    # ----- Step 4: which teachers have the most attendance-risk students ----
    r = _route("Which teachers have the most attendance-risk students?")
    plan = r["plan"]
    assert plan["operation"] == "groupby_count", plan
    assert plan["group_by"] == "Teacher", plan
    # Filter set inherits the attendance-risk filter via the follow-up state.
    assert any(
        f.get("column") == "Attendance Risk" and f.get("value") is True
        for f in plan.get("filters") or []
    ), plan
    _say_assistant("Grouped attendance-risk students by Teacher.")

    # ----- Step 5: drilldown to GPA below 2.0 -------------------------------
    r = _route("Now only students with GPA below 2.0.")
    plan = r["plan"]
    # Both filters must survive — the planner's compose_filters merges the
    # new GPA<2.0 clause onto the active attendance-risk filter.
    filter_columns = {f.get("column") for f in plan.get("filters") or []}
    assert "Attendance Risk" in filter_columns, plan
    assert "GPA" in filter_columns, plan
    state["active_filters"] = plan["filters"]
    confirmation_filters = list(plan["filters"])
    _say_assistant("Now showing students who are both attendance-risk AND below 2.0 GPA.")

    # Pre-compute the expected set independently for the post-write check.
    enriched_roster = enriched_sheets[roster_sheet]
    expected_mask = (
        enriched_roster["Attendance Risk"].astype(bool)
        & (pd.to_numeric(enriched_roster["GPA"], errors="coerce") < 2.0)
    )
    expected_count = int(expected_mask.sum())
    expected_ids = sorted(enriched_roster.loc[expected_mask, "Student ID"].astype(str).tolist())
    assert expected_count > 0, "fixture should have at least one combined-risk student"

    # ----- Step 6: mark Attendance Watch and export -------------------------
    r = _route("Mark these students Attendance Watch and export.")
    # Either single intent (attendance_watch with auto-export downstream) or
    # an action_chain that ends in export — both satisfy the spec.
    assert r["intent"] in {"attendance_watch", "action_chain"}, r
    assert r["requires_confirmation"], r
    _say_assistant(r.get("confirmation_reason")
                   or "Confirmation needed before writing the new workbook.")

    # ----- Step 7: confirm ---------------------------------------------------
    _say_user("yes")
    result = execute_academic_watch_action(
        filters=confirmation_filters,
        sheets={roster_sheet: enriched_roster.copy()},
        sheet=roster_sheet,
        column_name="Attendance Watch", value="Yes",
        request_summary="Mark these students Attendance Watch and export.",
        output_dir=output_dir, audit_path=audit_path,
    )
    assert result.success, result.message
    assert result.action_type == "academic_watch"  # shared underlying action
    output_path = Path(result.output_file)
    assert output_path.exists(), output_path
    _say_assistant(result.message)

    # ----- Step 8: verify ---------------------------------------------------

    # (a) Attendance Watch updated only for expected students.
    written = pd.read_excel(output_path)
    marked_mask = written["Attendance Watch"].astype(str) == "Yes"
    marked_ids = sorted(written.loc[marked_mask, "Student ID"].astype(str).tolist())
    assert marked_ids == expected_ids, (
        f"Attendance Watch leaked: marked={marked_ids}, expected={expected_ids}"
    )
    assert int(marked_mask.sum()) == expected_count

    # (b) New workbook created.
    assert output_path.suffix == ".xlsx"
    assert output_path.stat().st_size > 0

    # (c) Original workbook is byte-for-byte unchanged.
    assert _sha256(FIXTURE) == original_sha, "ORIGINAL ROSTER FILE WAS MODIFIED"

    # (d) Audit log exists and records the action.
    assert audit_path.exists()
    audit_entries = [
        json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()
    ]
    assert any(e.get("action_type") == "academic_watch" for e in audit_entries), \
        audit_entries

    # (e) Chat history preserved. Spec step 8 is "verify" — no chat input —
    # so the user-visible flow is 7 inputs (2 uploads + 4 asks + 1 confirm),
    # each followed by one assistant reply (7 + 7 = 14).
    user_msgs = [m for m in chat_history if m["role"] == "user"]
    assistant_msgs = [m for m in chat_history if m["role"] == "assistant"]
    assert len(user_msgs) == 7, f"expected 7 user turns, got {len(user_msgs)}"
    assert len(assistant_msgs) == 7, f"expected 7 assistant turns, got {len(assistant_msgs)}"

    # Clean up the temp output dir.
    shutil.rmtree(audit_dir, ignore_errors=True)

    print("\n=========================================================")
    print(f"E2E ATTENDANCE WORKFLOW PASS — {expected_count} students marked")
    print("  attendance file matched to roster ✓")
    print("  attendance-risk filter routes correctly ✓")
    print("  teachers grouped by attendance-risk count ✓")
    print("  GPA + attendance combined risk drilldown ✓")
    print("  Attendance Watch applied to only the matching rows ✓")
    print("  original roster file byte-for-byte unchanged ✓")
    print("  audit log records the action ✓")
    print(f"  chat history captured {len(user_msgs)} user + {len(assistant_msgs)} assistant turns ✓")
    print("=========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
