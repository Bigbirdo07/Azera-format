"""Phase C: workbook profiling + sensitive-column detection."""

from __future__ import annotations

from core.privacy import (
    classify_sensitivity,
    detect_sensitive_columns,
    get_default_visible_columns,
    is_hidden_by_default,
)
from core.workbook_profiler import profile_workbook


def test_profile_has_students_sheet(sheets, gt):
    profile = profile_workbook("synthetic_students.xlsx", sheets)
    assert "Students" in profile.sheet_names
    students = next(s for s in profile.sheets if s.name == "Students")
    assert students.row_count == len(gt)
    assert "Department" in students.columns and "GPA" in students.columns


def test_profile_includes_workbook_intelligence(sheets):
    profile = profile_workbook("synthetic_students.xlsx", sheets)
    students = next(s for s in profile.sheets if s.name == "Students")
    assert profile.primary_sheet == "Students"
    assert profile.workbook_summary["sheet_count"] >= 1
    assert students.column_count == len(students.columns)
    assert students.canonical_fields["GPA"] == "gpa"
    assert "GPA" in students.numeric_columns
    assert "Student roster questions" in students.answerable_workflows
    assert "Data quality review" in students.answerable_workflows


def test_sensitive_columns_classified(columns):
    detected = detect_sensitive_columns(columns)
    assert detected.get("Email") == "contact"
    assert detected.get("Phone") == "contact"
    assert detected.get("Date of Birth") == "identity_high"
    assert detected.get("Notes") == "notes"
    assert detected.get("Financial Aid Status") == "financial"
    assert detected.get("Conduct Status") == "disciplinary"
    # Non-sensitive analytics columns are not flagged.
    assert "GPA" not in detected
    assert "Department" not in detected


def test_classify_sensitivity_types():
    assert classify_sensitivity("Email") == (True, "contact")
    assert classify_sensitivity("GPA") == (False, "unknown")
    assert classify_sensitivity("Academic Status")[0] is False


def test_default_visible_columns_exclude_sensitive(columns):
    visible = get_default_visible_columns(columns)
    for hidden in ["Email", "Phone", "Date of Birth", "Notes", "Financial Aid Status", "Conduct Status"]:
        assert hidden not in visible
    assert "GPA" in visible and "Department" in visible


def test_is_hidden_by_default():
    assert is_hidden_by_default("Email") is True
    assert is_hidden_by_default("Notes") is True
    assert is_hidden_by_default("GPA") is False
    assert is_hidden_by_default("Name") is False  # identity, shown when needed
