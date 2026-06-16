"""DataSourceRegistry: roster binding, attendance side-load, enriched-view
plumbing, and the protected-schema additions from Phase D."""

from __future__ import annotations

import pandas as pd
import pytest

from core.attendance import compute_attendance_metrics
from core.data_sources import AttendanceSource, DataSourceRegistry
from core.excel_loader import LoadedWorkbook
from core.field_policy import field_status, is_protected, is_safe_editable


def _make_roster() -> LoadedWorkbook:
    return LoadedWorkbook(
        file_name="roster.xlsx",
        workbook=None,  # not used by enriched_sheets / summary
        sheets={
            "Students": pd.DataFrame({
                "Student ID": ["S001", "S002", "S003"],
                "Name": ["Alice", "Bob", "Cara"],
                "Teacher": ["Smith", "Smith", "Lee"],
                "GPA": [3.5, 1.8, 2.1],
            }),
        },
        warnings=[],
    )


def _make_attendance() -> pd.DataFrame:
    return pd.DataFrame({
        "Student ID": ["S001"] * 10 + ["S002"] * 10 + ["S003"] * 10,
        "Date": pd.to_datetime([f"2026-05-{i:02d}" for i in range(1, 11)] * 3),
        "Attendance Status": (["Present"] * 10) + (["Absent"] * 6 + ["Present"] * 4)
                              + (["Present"] * 8 + ["Absent"] * 2),
    })


# ---- registry lifecycle ---------------------------------------------------


def test_set_roster_clears_attendance_when_file_changes():
    registry = DataSourceRegistry()
    registry.set_roster(_make_roster())
    registry.set_attendance(AttendanceSource(
        file_name="old.xlsx", frame=_make_attendance(),
    ))
    # Bind a different roster (different file name) — attendance must clear.
    other = _make_roster()
    object.__setattr__(other, "file_name", "different.xlsx")
    registry.set_roster(other)
    assert registry.attendance is None


def test_set_roster_keeps_attendance_when_same_file_rebound():
    """Streamlit reruns rebind the same roster on every interaction. We
    must NOT lose attendance on a same-file rebind."""
    registry = DataSourceRegistry()
    roster = _make_roster()
    registry.set_roster(roster)
    registry.set_attendance(AttendanceSource(
        file_name="att.xlsx", frame=_make_attendance(),
    ))
    registry.set_roster(roster)  # same file_name
    assert registry.attendance is not None


def test_clear_wipes_everything():
    registry = DataSourceRegistry()
    registry.set_roster(_make_roster())
    registry.set_attendance(AttendanceSource(file_name="a.xlsx", frame=_make_attendance()))
    registry.clear()
    assert registry.roster is None
    assert registry.attendance is None
    assert registry.enriched_roster_sheet == ""


# ---- enriched view --------------------------------------------------------


def test_enriched_sheets_merges_attendance_metrics_onto_roster():
    registry = DataSourceRegistry()
    registry.set_roster(_make_roster())
    registry.set_attendance(AttendanceSource(
        file_name="att.xlsx", frame=_make_attendance(),
    ))
    enriched = registry.enriched_sheets()
    roster = enriched["Students"]
    # Original roster columns survive.
    assert {"Student ID", "Name", "Teacher", "GPA"}.issubset(roster.columns)
    # Attendance metrics are added.
    assert "Attendance Rate" in roster.columns
    assert "Days Absent" in roster.columns
    assert "Attendance Risk" in roster.columns
    # Combined-risk columns are added.
    assert "Risk Level" in roster.columns
    # Long-format attendance is exposed as its own sheet.
    assert "Attendance" in enriched


def test_enriched_sheets_with_no_attendance_still_attaches_risk():
    """A roster-only session should still get a Risk Level column derived
    from whatever signals exist (GPA in this fixture)."""
    registry = DataSourceRegistry()
    registry.set_roster(_make_roster())
    enriched = registry.enriched_sheets()
    roster = enriched["Students"]
    assert "Risk Level" in roster.columns
    assert "GPA Risk" in roster.columns
    assert "Attendance" not in enriched


def test_enriched_columns_match_enriched_sheets():
    registry = DataSourceRegistry()
    registry.set_roster(_make_roster())
    registry.set_attendance(AttendanceSource(
        file_name="att.xlsx", frame=_make_attendance(),
    ))
    cols = registry.enriched_columns()
    sheets = registry.enriched_sheets()
    for name, frame in sheets.items():
        assert cols[name] == list(frame.columns)


def test_summary_reports_loaded_sources():
    registry = DataSourceRegistry()
    registry.set_roster(_make_roster())
    registry.set_attendance(AttendanceSource(
        file_name="att.xlsx", frame=_make_attendance(),
        matched_count=3, unmatched_count=0, unmatched_ids=[],
        warnings=["check this"],
    ))
    s = registry.summary()
    assert s["roster"]["file_name"] == "roster.xlsx"
    assert s["roster"]["rows"] == 3
    assert s["attendance"]["matched"] == 3
    assert "check this" in s["attendance"]["warnings"]
    assert s["assessments"] is None


# ---- Phase D: watch fields + protected schema -----------------------------


def test_attendance_watch_is_safe_editable():
    """The new Attendance Watch column must pass the safe-edit gate so
    execute_academic_watch_action can write it."""
    assert is_safe_editable("Attendance Watch")
    assert is_safe_editable("attendance watch")


def test_computed_attendance_fields_are_protected():
    """Editing computed metrics would silently desynchronise them from the
    uploaded attendance file — they must be protected."""
    for column in ("Attendance Rate", "Days Absent", "Days Present",
                   "Unexcused Absences", "Attendance Risk"):
        assert is_protected(column), f"{column} should be protected"


def test_combined_risk_columns_are_protected():
    for column in ("Risk Level", "Risk Signals", "GPA Risk"):
        assert is_protected(column), f"{column} should be protected"


def test_assessment_score_columns_are_protected():
    """PSAT/SAT scores are uploaded data, not editable."""
    for column in ("Math Score", "Reading/Writing Score",
                   "Total Score", "Benchmark Status"):
        assert is_protected(column), f"{column} should be protected"


def test_existing_safe_fields_remain_safe():
    """Make sure the new additions didn't accidentally demote anything."""
    assert is_safe_editable("Academic Watch")
    assert is_safe_editable("Follow Up Needed")
