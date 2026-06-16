"""Polish-phase tests: school-office-language capability surface.

Phase 11 (north-star) and Phase 12 (helper) tests rolled together since
they share the same fixtures and helpers.

What's NOT tested here: anything that requires Streamlit rendering
(Streamlit's AppTest harness doesn't easily simulate file uploads, so the
panel layout is sanity-checked via import + unit tests on the helpers it
calls). The chat-greeting flow is also exercised by the existing
``scripts/e2e_attendance_risk_workflow.py`` E2E.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from core.workbook_capabilities import (
    CATEGORY_ORDER,
    Capability,
    detect_capabilities,
    group_detected_fields,
    missing_field_messages,
    readiness_checks,
    readiness_issues,
    upload_assistant_message,
)
from core.institution_context import InstitutionMode, Role, role_prompt_snippets, workflow_templates
from nlp.dynamic_suggestions import build_dynamic_suggestions


# ---- shared fixtures ------------------------------------------------------


_FULL_COVERAGE_COLUMNS = [
    "Student ID", "Student Name", "Teacher", "Department", "Major",
    "GPA", "Academic Standing", "Attendance Rate", "Days Absent",
    "Days Tardy", "Unexcused Absences", "Academic Watch", "Attendance Watch",
]

_ROSTER_ONLY_COLUMNS = [
    "Student ID", "Student Name", "Teacher", "Department", "Major",
    "GPA", "Academic Standing",
]

_BARE_GPA_COLUMNS = ["Student ID", "Student Name", "GPA"]


def _full_coverage_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Student ID": ["S001", "S002", "S003", "S004"],
        "Student Name": ["Alice", "Bob", "Cara", "Dan"],
        "Teacher": ["Smith", "Smith", "Jones", "Lee"],
        "Department": ["Biology", "Biology", "Math", "Math"],
        "Major": ["Bio", "Bio", "Math", "Math"],
        "GPA": [3.5, 1.8, 2.2, 1.5],
        "Academic Standing": ["Good Standing", "Probation", "Good Standing", "Warning"],
        "Attendance Rate": [98.0, 85.0, 92.0, 75.0],
        "Days Absent": [2, 6, 4, 10],
        "Days Tardy": [0, 1, 0, 2],
        "Unexcused Absences": [0, 3, 1, 5],
        "Academic Watch": ["", "", "", ""],
        "Attendance Watch": ["", "", "", ""],
    })


# ============================================================================
# Phase 12 — helper tests
# ============================================================================


# 12-1: Field grouping → Roster bucket correct.
def test_group_detected_fields_roster_bucket():
    grouped = group_detected_fields(_FULL_COVERAGE_COLUMNS)
    roster = set(grouped["Roster"])
    # Student Name is the Student label; Teacher/Department/Major all land
    # in Roster too.
    assert {"Teacher", "Department", "Student", "Major", "Student ID"} <= roster


# 12-2: Field grouping → Performance bucket correct.
def test_group_detected_fields_performance_bucket():
    grouped = group_detected_fields(_FULL_COVERAGE_COLUMNS)
    assert grouped["Performance"] == ["GPA", "Academic Standing"]


# 12-3: Field grouping → Attendance bucket correct.
def test_group_detected_fields_attendance_bucket():
    grouped = group_detected_fields(_FULL_COVERAGE_COLUMNS)
    attendance = set(grouped["Attendance"])
    assert {"Attendance Rate", "Days Absent", "Days Tardy",
            "Unexcused Absences"} <= attendance


# 12-4: Field grouping → Actions bucket correct.
def test_group_detected_fields_actions_bucket():
    grouped = group_detected_fields(_FULL_COVERAGE_COLUMNS)
    assert set(grouped["Actions"]) == {"Academic Watch", "Attendance Watch"}


def test_group_detected_fields_export_always_present():
    """Export is always available — the action layer can always write a
    new workbook from a roster."""
    grouped = group_detected_fields(_BARE_GPA_COLUMNS)
    assert grouped["Export"] == ["Export updated workbook", "Export filtered list"]


# 12-5: Capabilities computed from detected fields are correct.
def test_capabilities_for_full_coverage_are_all_available():
    caps = detect_capabilities(_FULL_COVERAGE_COLUMNS)
    by_key = {c.key: c for c in caps}
    for key in ("teacher_department", "gpa_performance", "major_grouping",
                "academic_standing", "attendance_risk",
                "academic_watch_updates", "attendance_watch_updates",
                "export"):
        assert by_key[key].available, key


def test_capabilities_for_roster_only_drops_attendance():
    caps = detect_capabilities(_ROSTER_ONLY_COLUMNS)
    by_key = {c.key: c for c in caps}
    assert by_key["gpa_performance"].available
    assert by_key["academic_standing"].available
    assert not by_key["attendance_risk"].available
    # Watch capabilities are creatable — always available.
    assert by_key["academic_watch_updates"].available
    assert by_key["attendance_watch_updates"].available


# 12-6: Missing-attendance message appears when attendance is absent.
def test_missing_attendance_message_present_when_absent():
    messages = missing_field_messages(_ROSTER_ONLY_COLUMNS)
    assert any("Attendance not detected" in m for m in messages)


# 12-7: Attendance capability appears when attendance fields exist.
def test_attendance_capability_appears_when_fields_exist():
    caps = detect_capabilities(_FULL_COVERAGE_COLUMNS)
    attendance_cap = next(c for c in caps if c.key == "attendance_risk")
    assert attendance_cap.available
    assert attendance_cap.title == "Attendance-risk review"


def test_attendance_capability_unlocked_by_sibling_sheet_flag():
    """A workbook with a sibling Attendance sheet has no inline attendance
    columns but should still unlock the attendance capability via the
    explicit `attendance_available=True` flag."""
    caps = detect_capabilities(
        _ROSTER_ONLY_COLUMNS, attendance_available=True,
    )
    attendance_cap = next(c for c in caps if c.key == "attendance_risk")
    assert attendance_cap.available


# 12-8: Watch fields treated as creatable when missing.
def test_academic_watch_creatable_message_when_missing():
    messages = missing_field_messages(_ROSTER_ONLY_COLUMNS)
    assert any(
        "Academic Watch" in m and "create that column in the exported workbook" in m
        for m in messages
    )


def test_attendance_watch_creatable_message_when_missing():
    messages = missing_field_messages(_ROSTER_ONLY_COLUMNS)
    assert any(
        "Attendance Watch" in m and "create that column in the exported workbook" in m
        for m in messages
    )


def test_watch_capabilities_always_available_regardless_of_columns():
    """Per the spec: 'Academic Watch / Attendance Watch updates' is
    always a supported capability — the column gets created on export
    when absent."""
    caps = detect_capabilities(_BARE_GPA_COLUMNS)
    by_key = {c.key: c for c in caps}
    assert by_key["academic_watch_updates"].available
    assert by_key["attendance_watch_updates"].available
    # And the note must say the column will be created.
    assert "exported workbook" in by_key["academic_watch_updates"].note
    assert "exported workbook" in by_key["attendance_watch_updates"].note


# 12-9: Suggested questions only include workflows supported by detected fields.
def test_suggested_questions_for_no_attendance_workbook_omit_attendance():
    df = _full_coverage_frame().drop(
        columns=["Attendance Rate", "Days Absent", "Days Tardy",
                 "Unexcused Absences"]
    )
    suggestions = build_dynamic_suggestions(df, list(df.columns))
    texts = [q.text.lower() for q in suggestions]
    assert not any("attendance below" in t for t in texts)
    assert not any("attendance watch" in t for t in texts)


# 12-10: Developer-only terms are not present in capability/grouping labels.
_FORBIDDEN_DEVELOPER_TERMS = (
    "canonical_for", "dispatcher", "validator object",
    "planner source", "routing result", "raw JSON",
    "execution object", "llm fallback",
)


def test_capability_labels_use_school_office_language():
    caps = detect_capabilities(_FULL_COVERAGE_COLUMNS)
    blob = " ".join(c.title + " " + c.note for c in caps).lower()
    for term in _FORBIDDEN_DEVELOPER_TERMS:
        assert term.lower() not in blob, term


def test_missing_messages_use_school_office_language():
    messages = missing_field_messages(_ROSTER_ONLY_COLUMNS)
    blob = " ".join(messages).lower()
    for term in _FORBIDDEN_DEVELOPER_TERMS:
        assert term.lower() not in blob, term


# ============================================================================
# Phase 11 — north-star tests
# ============================================================================


# 11-1: Full coverage shows correct capabilities.
def test_full_coverage_workbook_shows_correct_capabilities():
    caps = detect_capabilities(_FULL_COVERAGE_COLUMNS)
    titles = {c.title for c in caps if c.available}
    expected = {
        "Teacher/instructor and department questions",
        "GPA performance review",
        "Major-based grouping",
        "Academic standing review",
        "Attendance-risk review",
        "Academic Watch updates",
        "Attendance Watch updates",
        "Export updated workbook",
    }
    assert expected <= titles


def test_institution_mode_changes_labels_but_not_backend_plans():
    generic = detect_capabilities(_FULL_COVERAGE_COLUMNS, mode=InstitutionMode.GENERIC)
    college = detect_capabilities(_FULL_COVERAGE_COLUMNS, mode=InstitutionMode.COLLEGE)
    assert {c.key for c in generic} == {c.key for c in college}
    assert any(c.title.startswith("Professor") for c in college)


def test_k12_mode_uses_teacher_grade_attendance_language():
    caps = detect_capabilities(_FULL_COVERAGE_COLUMNS, mode=InstitutionMode.PK12)
    titles = " ".join(c.title for c in caps)
    assert "Teacher" in titles
    assert "Attendance" in titles


def test_pk12_parent_guardian_contact_field_is_detected_when_present():
    grouped = group_detected_fields(["Student ID", "Student Name", "Parent/Guardian Contact"])
    assert "Parent/guardian contact" in grouped["Roster"]


def test_college_mode_uses_professor_advisor_retention_language():
    caps = detect_capabilities(_FULL_COVERAGE_COLUMNS, mode=InstitutionMode.COLLEGE)
    titles = " ".join(c.title for c in caps)
    assert "Professor" in titles
    assert "Retention" in titles or "Advisor" in titles


def test_role_setting_changes_suggested_workflow_snippets():
    assert "department summaries" in role_prompt_snippets(Role.ADMIN, InstitutionMode.GENERIC)
    assert "student risk lists" in role_prompt_snippets(Role.COUNSELOR, InstitutionMode.GENERIC)
    assert "students under my classes" in role_prompt_snippets(Role.TEACHER, InstitutionMode.GENERIC)
    assert "missing IDs" in role_prompt_snippets(Role.REGISTRAR, InstitutionMode.GENERIC)


def test_readiness_panel_flags_missing_and_duplicate_ids():
    checks = readiness_checks(["Student Name", "GPA"])
    labels = dict(checks)
    assert labels["Student ID"] == "issue found"
    assert labels["GPA"] == "found"

    df = pd.DataFrame({"Student ID": ["A", "A", "B"], "Student Name": ["X", "Y", "Z"]})
    assert "duplicate Student IDs" in readiness_issues(df)


# 11-2: Attendance available message when attendance columns exist.
def test_upload_greeting_mentions_attendance_when_attendance_columns_exist():
    msg = upload_assistant_message(
        _FULL_COVERAGE_COLUMNS,
        attendance_available=True,
    ).lower()
    assert "attendance" in msg
    assert "review" in msg or "risk" in msg


def test_upload_greeting_omits_attendance_when_missing():
    msg = upload_assistant_message(_ROSTER_ONLY_COLUMNS).lower()
    assert "i do not see attendance" in msg
    # And the still-available workflows should be named.
    assert "gpa" in msg
    assert "academic watch" in msg


# 11-3: No-attendance workbook still supports GPA + standing workflows.
def test_no_attendance_workbook_still_offers_gpa_and_standing_capabilities():
    caps = detect_capabilities(_ROSTER_ONLY_COLUMNS)
    by_key = {c.key: c for c in caps}
    assert by_key["gpa_performance"].available
    assert by_key["academic_standing"].available
    assert by_key["teacher_department"].available
    # And missing-field messages explain attendance is the only gap.
    messages = missing_field_messages(_ROSTER_ONLY_COLUMNS)
    assert any("Attendance not detected" in m for m in messages)


# 11-4: Suggested questions change based on detected fields.
def test_suggested_questions_change_with_detected_fields():
    full_df = _full_coverage_frame()
    full_texts = [q.text for q in build_dynamic_suggestions(full_df, list(full_df.columns))]
    bare_df = full_df.drop(columns=[
        "Attendance Rate", "Days Absent", "Days Tardy", "Unexcused Absences",
    ])
    bare_texts = [q.text for q in build_dynamic_suggestions(bare_df, list(bare_df.columns))]
    # Full coverage surfaces attendance prompts; the bare workbook doesn't.
    assert any("attendance below 90%" in t for t in full_texts)
    assert not any("attendance below 90%" in t for t in bare_texts)
    # Both share the basic GPA top-N prompt — proves it's not a coincidence.
    assert any("Top 10 students by GPA" == t for t in full_texts)
    assert any("Top 10 students by GPA" == t for t in bare_texts)


# 11-5 + 11-6: Watch-missing messages mention "create on export".
def test_academic_watch_missing_message_says_create_on_export():
    messages = missing_field_messages(_ROSTER_ONLY_COLUMNS)
    target = next(m for m in messages if "Academic Watch field not found" in m)
    assert "create that column in the exported workbook" in target


def test_attendance_watch_missing_message_says_create_on_export():
    messages = missing_field_messages(_ROSTER_ONLY_COLUMNS)
    target = next(m for m in messages if "Attendance Watch field not found" in m)
    assert "create that column in the exported workbook" in target


# 11-7: Raw schema / debug terms not visible in normal-mode output.
def test_upload_greeting_uses_school_office_language():
    msg = upload_assistant_message(
        _FULL_COVERAGE_COLUMNS, attendance_available=True,
    ).lower()
    for term in _FORBIDDEN_DEVELOPER_TERMS:
        assert term.lower() not in msg, term
    # Also, no internal canonical names should leak.
    for canonical in ("academic_status", "attendance_rate", "days_absent"):
        assert canonical not in msg


# 11-8: Upload assistant message is generated per workbook (text varies on
# detected fields, so two distinct workbooks get distinct greetings).
def test_upload_message_adapts_per_workbook():
    full = upload_assistant_message(
        _FULL_COVERAGE_COLUMNS, attendance_available=True,
    )
    bare = upload_assistant_message(_ROSTER_ONLY_COLUMNS)
    # Different shape → different text.
    assert full != bare
    # The "once per upload" wiring is enforced in app.py
    # (`_maybe_post_upload_greeting`) — covered by the integration test
    # below that imports app without rendering.
    import app  # noqa: F401
    assert hasattr(app, "_maybe_post_upload_greeting")


# 11-9: North-star workflow suggestions are present for full coverage.
def test_north_star_suggestions_appear_for_full_coverage_workbook():
    df = _full_coverage_frame()
    texts = [q.text for q in build_dynamic_suggestions(df, list(df.columns))]
    # The four spec-listed example prompts (or close paraphrases) should
    # all be present.
    assert any("Show me Biology students" == t for t in texts), texts
    assert any("Top 10 students by GPA" == t for t in texts), texts
    assert any("attendance below 90%" in t for t in texts), texts
    assert any("Mark these students Academic Watch and export" == t for t in texts), texts


def test_north_star_includes_attendance_watch_export_when_attendance_present():
    df = _full_coverage_frame()
    texts = [q.text for q in build_dynamic_suggestions(df, list(df.columns))]
    assert any("Mark these students Attendance Watch and export" == t for t in texts), texts


# 11-10: Suite-green check — sanity import that the panel orchestration
# still works. The full pytest run is the authoritative "existing tests
# stay green" gate.
def test_app_imports_after_polish_phase():
    import app
    # The new helpers are reachable.
    assert hasattr(app, "_render_capability_summary")
    assert hasattr(app, "_maybe_post_upload_greeting")
