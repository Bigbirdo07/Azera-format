"""Attendance is part of the academic workbook, not a separate required upload.

Covers the spec's 12 tests:
  1. Inline attendance columns are detected on the roster sheet.
  2. "Attendance Rate" maps to canonical attendance_rate.
  3. "Days Absent" maps to canonical days_absent.
  4. "Attendance Watch" maps to canonical attendance_watch.
  5. Attendance-risk query works from inline columns.
  6. GPA + attendance combined query works from one sheet.
  7. Mark Attendance Watch creates the column when missing.
  8. Original workbook remains unchanged.
  9. Workbook with a separate Attendance sheet still works.
 10. Workbook with no attendance still supports GPA/teacher/department.
 11. UI summary reflects "Attendance available" status.
 12. Existing tests stay green (covered by the full pytest run).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook

from core.attendance import detect_workbook_attendance
from core.confirmed_actions import execute_academic_watch_action
from core.data_sources import DataSourceRegistry
from core.excel_loader import LoadedWorkbook, load_excel_workbook
from core.query_engine import run_query
from core.schema import canonical_for
from nlp.planner_router import plan_user_request


# ---- helpers --------------------------------------------------------------


def _loaded(sheets: dict[str, pd.DataFrame], file_name: str = "workbook.xlsx") -> LoadedWorkbook:
    return LoadedWorkbook(
        file_name=file_name, workbook=None, sheets=sheets, warnings=[],
    )


def _roster_inline_attendance() -> pd.DataFrame:
    """Roster + attendance columns on the same sheet (Case 1 from the spec)."""
    return pd.DataFrame({
        "Student ID": [f"S{i:03d}" for i in range(10)],
        "Student Name": list("ABCDEFGHIJ"),
        "Teacher": ["Smith"] * 5 + ["Jones"] * 5,
        "Department": ["Biology"] * 5 + ["Math"] * 5,
        "GPA": [3.5, 1.8, 2.2, 1.5, 3.9, 2.7, 1.7, 2.0, 1.9, 3.1],
        "Academic Standing": ["Good Standing"] * 8 + ["Probation", "Warning"],
        "Attendance Rate": [98.0, 85.0, 92.0, 75.0, 96.0,
                            91.0, 78.0, 88.0, 95.0, 70.0],
        "Days Absent": [2, 6, 4, 10, 1, 3, 9, 5, 2, 12],
    })


def _roster_no_attendance() -> pd.DataFrame:
    """Roster without any attendance columns (Case 3 from the spec)."""
    return pd.DataFrame({
        "Student ID": [f"S{i:03d}" for i in range(5)],
        "Student Name": list("ABCDE"),
        "Teacher": ["Smith"] * 3 + ["Jones"] * 2,
        "Department": ["Biology"] * 3 + ["Math"] * 2,
        "GPA": [3.5, 1.8, 2.2, 1.5, 3.9],
        "Academic Standing": ["Good Standing"] * 5,
    })


def _attendance_sheet_frame() -> pd.DataFrame:
    """Long-format attendance for the same 5 students (Case 2 from the spec)."""
    rows = []
    for sid, absences in [("S000", 1), ("S001", 5), ("S002", 0),
                          ("S003", 3), ("S004", 0)]:
        for i in range(20):
            status = "Absent" if i < absences else "Present"
            rows.append({
                "Student ID": sid,
                "Date": pd.Timestamp(f"2026-05-{i+1:02d}"),
                "Attendance Status": status,
            })
    return pd.DataFrame(rows)


# ---- Test 1: inline attendance columns detected ---------------------------


def test_inline_attendance_columns_detected_on_roster():
    sheets = {"Students": _roster_inline_attendance()}
    detection = detect_workbook_attendance(sheets, "Students")
    assert detection.mode == "inline"
    assert "Attendance Rate" in detection.inline_columns
    assert "Days Absent" in detection.inline_columns
    assert "attendance_rate" in detection.inline_fields
    assert "days_absent" in detection.inline_fields


# ---- Tests 2–4: canonical mappings ----------------------------------------


def test_attendance_rate_maps_to_attendance_rate_canonical():
    assert canonical_for("Attendance Rate") == "attendance_rate"
    # Common variants per the spec.
    for variant in ("Attendance %", "Attendance Percent", "Present Rate",
                    "Attendance"):
        assert canonical_for(variant) == "attendance_rate", variant


def test_days_absent_maps_to_days_absent_canonical():
    assert canonical_for("Days Absent") == "days_absent"
    for variant in ("Absences", "Total Absences", "Absent Days"):
        assert canonical_for(variant) == "days_absent", variant


def test_attendance_watch_maps_to_attendance_watch_canonical():
    assert canonical_for("Attendance Watch") == "attendance_watch"
    for variant in ("Attendance Flag", "Attendance Intervention"):
        assert canonical_for(variant) == "attendance_watch", variant


# ---- Test 5: attendance-risk query works from inline columns ---------------


def test_attendance_risk_query_routes_from_inline_columns():
    """A workbook that already carries Attendance Rate as a roster column
    should answer 'show students with attendance below 90%' immediately —
    no external file required."""
    df = _roster_inline_attendance()
    sheets = {"Students": df}
    sheet_columns = {"Students": list(df.columns)}
    routing = plan_user_request(
        user_message="show students with attendance below 90%",
        sheets=sheets, sheet_columns=sheet_columns,
        selected_sheet="Students", conversation_state=None,
        settings={"llm_enabled": False},
    )
    plan = routing["plan"]
    assert any(
        f.get("column") == "Attendance Rate" and f.get("operator") == "less_than"
        for f in plan.get("filters") or []
    ), plan
    result = run_query(plan, sheets)
    # 5 rows in the fixture have rate < 90.
    assert result.row_count == 5


# ---- Test 6: GPA + attendance combined from the same sheet ----------------


def test_gpa_and_attendance_combined_filter_one_sheet():
    df = _roster_inline_attendance()
    sheets = {"Students": df}
    sheet_columns = {"Students": list(df.columns)}
    routing = plan_user_request(
        user_message="students with GPA below 2.0 and attendance below 90%",
        sheets=sheets, sheet_columns=sheet_columns,
        selected_sheet="Students", conversation_state=None,
        settings={"llm_enabled": False},
    )
    plan = routing["plan"]
    cols = {f.get("column") for f in plan.get("filters") or []}
    assert {"GPA", "Attendance Rate"}.issubset(cols), plan
    result = run_query(plan, sheets)
    # Fixture students that are <2.0 GPA AND <90% attendance:
    # idx 1: 1.8 / 85 ✓, idx 3: 1.5 / 75 ✓, idx 6: 1.7 / 78 ✓ — 3 rows.
    assert result.row_count == 3


# ---- Test 7: Mark Attendance Watch creates the column when missing --------


def test_mark_attendance_watch_creates_column_when_missing(tmp_path):
    df = _roster_inline_attendance()  # NO Attendance Watch column
    assert "Attendance Watch" not in df.columns
    sheets = {"Students": df}
    filters = [{"column": "Attendance Rate", "operator": "less_than", "value": 90}]
    result = execute_academic_watch_action(
        filters=filters, sheets=sheets, sheet="Students",
        column_name="Attendance Watch", value="Yes",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "audit.jsonl",
    )
    assert result.success
    written = pd.read_excel(result.output_file)
    assert "Attendance Watch" in written.columns
    # Only the rows below 90% should be marked.
    marked = (written["Attendance Watch"].astype(str) == "Yes").sum()
    assert marked == 5


# ---- Test 8: original workbook untouched ----------------------------------


def test_original_workbook_unchanged_after_attendance_watch(tmp_path):
    df = _roster_inline_attendance()
    snapshot = df.copy()
    sheets = {"Students": df}
    filters = [{"column": "Attendance Rate", "operator": "less_than", "value": 90}]
    execute_academic_watch_action(
        filters=filters, sheets=sheets, sheet="Students",
        column_name="Attendance Watch", value="Yes",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "audit.jsonl",
    )
    pd.testing.assert_frame_equal(sheets["Students"], snapshot)


# ---- Test 9: separate-Attendance-sheet workbook still works ---------------


def test_workbook_with_sibling_attendance_sheet_is_detected_and_enriched():
    sheets = {
        "Students": _roster_no_attendance(),
        "Attendance": _attendance_sheet_frame(),
    }
    detection = detect_workbook_attendance(sheets, "Students")
    assert detection.mode == "sheet"
    assert detection.attendance_sheet == "Attendance"

    # End-to-end through the registry: enriched_sheets() should now carry
    # Attendance Rate computed from the sibling sheet.
    registry = DataSourceRegistry()
    registry.set_roster(_loaded(sheets, file_name="case2.xlsx"))
    enriched = registry.enriched_sheets()
    roster = enriched["Students"]
    assert "Attendance Rate" in roster.columns
    # S001 had 5 absences out of 20 → 75%; S002 had 0 → 100%.
    by_id = roster.set_index("Student ID")
    assert by_id.loc["S001", "Attendance Rate"] == pytest.approx(75.0, abs=0.1)
    assert by_id.loc["S002", "Attendance Rate"] == 100.0


# ---- Test 10: no-attendance workbook still supports normal workflow -------


def test_no_attendance_workbook_still_supports_normal_workflow():
    df = _roster_no_attendance()
    sheets = {"Students": df}
    sheet_columns = {"Students": list(df.columns)}

    # Detection says 'none'.
    detection = detect_workbook_attendance(sheets, "Students")
    assert detection.mode == "none"
    assert detection.inline_columns == ()

    # Registry attendance_available() is False.
    registry = DataSourceRegistry()
    registry.set_roster(_loaded(sheets, file_name="case3.xlsx"))
    assert registry.attendance_available() is False

    # GPA query still routes.
    routing = plan_user_request(
        user_message="show students with GPA below 2.0",
        sheets=sheets, sheet_columns=sheet_columns,
        selected_sheet="Students", conversation_state=None,
        settings={"llm_enabled": False},
    )
    plan = routing["plan"]
    assert any(
        f.get("column") == "GPA" and f.get("operator") == "less_than"
        for f in plan.get("filters") or []
    ), plan

    # Teacher groupby still works.
    routing = plan_user_request(
        user_message="how many students per teacher",
        sheets=sheets, sheet_columns=sheet_columns,
        selected_sheet="Students", conversation_state=None,
        settings={"llm_enabled": False},
    )
    assert routing["plan"]["operation"] == "groupby_count"
    assert routing["plan"]["group_by"] == "Teacher"


# ---- Test 11: UI summary surfaces "Attendance available" ------------------


def test_summary_reports_attendance_available_for_inline_workbook():
    registry = DataSourceRegistry()
    registry.set_roster(_loaded({"Students": _roster_inline_attendance()},
                                file_name="inline.xlsx"))
    assert registry.attendance_available() is True
    summary = registry.summary()
    wa = summary["workbook_attendance"]
    assert wa["mode"] == "inline"
    assert "Attendance Rate" in wa["inline_columns"]


def test_summary_reports_no_attendance_for_roster_only_workbook():
    registry = DataSourceRegistry()
    registry.set_roster(_loaded({"Students": _roster_no_attendance()},
                                file_name="bare.xlsx"))
    assert registry.attendance_available() is False
    wa = registry.summary()["workbook_attendance"]
    assert wa["mode"] == "none"
    assert wa["inline_columns"] == []


def test_summary_reports_sibling_sheet_attendance():
    registry = DataSourceRegistry()
    sheets = {
        "Students": _roster_no_attendance(),
        "Attendance": _attendance_sheet_frame(),
    }
    registry.set_roster(_loaded(sheets, file_name="case2.xlsx"))
    wa = registry.summary()["workbook_attendance"]
    assert wa["mode"] == "sheet"
    assert wa["attendance_sheet"] == "Attendance"
    assert registry.attendance_available() is True


# ---- Bonus: threshold-on-Days-Absent works without an Attendance Rate ----


def test_days_absent_threshold_filter_when_only_absent_count_present():
    """If a school only carries 'Days Absent' (no rate), the spec asks for
    threshold queries like 'students with more than 5 absences' to work."""
    df = pd.DataFrame({
        "Student ID": ["S1", "S2", "S3", "S4"],
        "Student Name": list("ABCD"),
        "GPA": [3.5, 1.8, 2.2, 1.5],
        "Days Absent": [2, 6, 10, 3],
    })
    sheets = {"Students": df}
    sheet_columns = {"Students": list(df.columns)}
    routing = plan_user_request(
        user_message="which students have more than 5 absences",
        sheets=sheets, sheet_columns=sheet_columns,
        selected_sheet="Students", conversation_state=None,
        settings={"llm_enabled": False},
    )
    plan = routing["plan"]
    assert any(
        f.get("column") == "Days Absent" and f.get("operator") == "greater_than"
        for f in plan.get("filters") or []
    ), plan
    result = run_query(plan, sheets)
    # S2 (6) and S3 (10) satisfy > 5 absences.
    assert result.row_count == 2
