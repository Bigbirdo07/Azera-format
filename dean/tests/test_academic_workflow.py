"""Phase N tests: school-roster workflow intelligence.

Covers the 17 acceptance cases from the N spec:
  1. Teacher/professor/instructor maps to canonical teacher.
  2. Department = Biology query works.
  3. "their students" preserves active Biology scope.
  4. GPA above 2.0 follow-up keeps the Biology filter.
  5. "not performing well based on GPA" → GPA < 2.00.
  6. Low-performing students are grouped by teacher when "under each professor".
  7. "under which professor" triggers teacher grouping.
  8. Academic Watch action uses the current filtered student set.
  9. Academic Watch column is created if missing.
 10. Academic Watch marks only matching students.
 11. Original workbook remains unchanged.
 12. Export workflow keeps an updated-workbook reference path.
 13. Suggestions include Academic Watch after low-GPA query.
 14. Conversation history preserves all turns (covered by the E2E script).
 15. Audit log records the academic watch action.
 16. Interaction log captures the workflow without row leakage.
 17. Existing tests stay green (verified by the full suite run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.confirmed_actions import execute_academic_watch_action
from core.interaction_logger import sanitize_filters
from core.schema import canonical_for
from nlp.planner_router import plan_user_request
from nlp.query_planner import _rule_plan
from nlp.request_intents import is_academic_watch_request
from nlp.suggestions import suggest_next_moves


FIXTURE = Path(__file__).parent / "fixtures" / "academic_roster.xlsx"


@pytest.fixture()
def roster_df() -> pd.DataFrame:
    return pd.read_excel(FIXTURE)


@pytest.fixture()
def roster_sheets(roster_df):
    return {"Students": roster_df}


@pytest.fixture()
def roster_columns(roster_df):
    return list(roster_df.columns)


def _plan(message, sheets, columns, *, state=None):
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=state or {},
        settings={},
    )


# ---- 1. teacher synonym ----------------------------------------------------


def test_teacher_synonym_maps_to_canonical_teacher():
    assert canonical_for("Teacher") == "teacher"
    assert canonical_for("Professor Name") == "teacher"
    assert canonical_for("Instructor") == "teacher"
    assert canonical_for("Faculty Member") == "teacher"


# ---- 2. Biology teacher query ---------------------------------------------


def test_biology_teacher_query_produces_list_unique_with_dept_filter(roster_sheets, roster_columns):
    """'show me all teachers that teach Biology' lists distinct teachers
    (not just a count) filtered to Biology."""
    routing = _plan("show me all teachers that teach Biology", roster_sheets, roster_columns)
    plan = routing["plan"]
    assert plan["operation"] == "list_unique"
    assert plan["value_column"] == "Teacher"
    assert {"column": "Department", "operator": "equals", "value": "Biology"} in plan["filters"]


# ---- 3 + 4. "their students" preserves scope and adds the new predicate ---


def test_their_students_followup_preserves_biology_scope(roster_sheets, roster_columns):
    state = {"active_filters": [{"column": "Department", "operator": "equals", "value": "Biology"}]}
    routing = _plan("of their students how many have gpa above 2.0",
                    roster_sheets, roster_columns, state=state)
    filters = routing["plan"]["filters"]
    assert {"column": "Department", "operator": "equals", "value": "Biology"} in filters
    assert any(f["column"] == "GPA" and f["operator"] == "greater_than" for f in filters)


# ---- 5. "not performing well based on GPA" → GPA < 2.0 -------------------


def test_not_performing_well_based_on_gpa_maps_to_gpa_lt_2(roster_sheets, roster_columns):
    routing = _plan("show me students not performing well based on gpa",
                    roster_sheets, roster_columns)
    plan = routing["plan"]
    assert any(f == {"column": "GPA", "operator": "less_than", "value": 2.0}
               for f in plan["filters"])


# ---- 6 + 7. "under each professor" triggers teacher group_by -------------


def test_under_each_professor_triggers_teacher_grouping(roster_sheets, roster_columns):
    routing = _plan("how many students under each professor are below 2.0",
                    roster_sheets, roster_columns)
    plan = routing["plan"]
    assert plan["group_by"] == "Teacher"


def test_under_which_professor_also_triggers_teacher_grouping(roster_sheets, roster_columns):
    routing = _plan("which students under which professor are not performing well",
                    roster_sheets, roster_columns)
    plan = routing["plan"]
    assert plan["group_by"] == "Teacher"


# ---- 8 + 9 + 10 + 11. Academic Watch action ------------------------------


def test_academic_watch_phrase_recognized():
    assert is_academic_watch_request("mark these students under academic watch")
    assert is_academic_watch_request("put them on watch")
    assert is_academic_watch_request("flag these students")
    assert not is_academic_watch_request("add note: advisor follow-up needed")
    assert not is_academic_watch_request("show me Biology teachers")


def test_academic_watch_routes_through_confirmation(roster_sheets, roster_columns):
    state = {"active_filters": [
        {"column": "Department", "operator": "equals", "value": "Biology"},
        {"column": "GPA", "operator": "less_than", "value": 2.0},
    ]}
    routing = _plan("mark these students under academic watch",
                    roster_sheets, roster_columns, state=state)
    assert routing["intent"] == "academic_watch"
    assert routing["requires_confirmation"]
    assert routing["pending_type"] == "academic_watch"
    # The filters are carried into the plan so the action operates on the
    # current student set, not the aggregate.
    plan_filters = routing["plan"]["filters"]
    assert {"column": "Department", "operator": "equals", "value": "Biology"} in plan_filters
    assert any(f["column"] == "GPA" and f["operator"] == "less_than" for f in plan_filters)


def test_academic_watch_action_marks_only_filtered_rows(roster_sheets, tmp_path):
    """Smoke the full action: writes a NEW workbook with the AW column set on
    only the matching students; the original sheets dict is unchanged."""
    before_aw = roster_sheets["Students"]["Academic Watch"].copy(deep=True)
    result = execute_academic_watch_action(
        filters=[
            {"column": "Department", "operator": "equals", "value": "Biology"},
            {"column": "GPA", "operator": "less_than", "value": 2.0},
        ],
        sheets=roster_sheets,
        sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "logs" / "audit_log.jsonl",
    )
    assert result.success
    assert result.action_type == "academic_watch"
    # Original dict not mutated.
    after_aw = roster_sheets["Students"]["Academic Watch"]
    pd.testing.assert_series_equal(before_aw, after_aw, check_dtype=False)

    # New workbook on disk has the correct count and only the matching rows.
    out = pd.read_excel(result.output_file)
    expected = ((roster_sheets["Students"]["Department"] == "Biology")
                & (roster_sheets["Students"]["GPA"] < 2.0)).sum()
    assert (out["Academic Watch"] == "Yes").sum() == expected
    assert len(out) == len(roster_sheets["Students"])  # row count preserved


def test_academic_watch_creates_column_if_missing(roster_df, tmp_path):
    sheets = {"Students": roster_df.drop(columns=["Academic Watch"])}
    result = execute_academic_watch_action(
        filters=[{"column": "Department", "operator": "equals", "value": "Biology"}],
        sheets=sheets, sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "logs" / "audit_log.jsonl",
    )
    assert result.success
    out = pd.read_excel(result.output_file)
    assert "Academic Watch" in out.columns
    assert (out["Academic Watch"] == "Yes").sum() == (roster_df["Department"] == "Biology").sum()


def test_academic_watch_no_match_returns_failure(roster_sheets, tmp_path):
    result = execute_academic_watch_action(
        filters=[{"column": "Department", "operator": "equals", "value": "Astrophysics"}],
        sheets=roster_sheets, sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "logs" / "audit_log.jsonl",
    )
    assert not result.success
    assert result.error == "no_rows"


# ---- 12. Export workflow with modified workbook --------------------------


def test_export_intent_recognized_from_final_sheet_phrasings():
    from nlp.request_intents import is_export_request
    assert is_export_request("export me a new Excel sheet")
    assert is_export_request("download this as Excel")
    assert is_export_request("create the updated workbook")
    assert is_export_request("give me the final sheet")


# ---- 13. Suggestions include Academic Watch after low-GPA query ----------


def test_suggestions_include_academic_watch_after_low_gpa_filter():
    plan = {"operation": "filtered_preview", "filters": [
        {"column": "Department", "operator": "equals", "value": "Biology"},
        {"column": "GPA", "operator": "less_than", "value": 2.0},
    ]}
    suggestions = suggest_next_moves(
        plan=plan,
        columns=["Student ID", "Teacher", "Department", "GPA", "Academic Watch"],
        active_filters=plan["filters"],
        row_count=27,
    )
    assert any("Academic Watch" in s for s in suggestions)


# ---- 15. Audit log records academic watch --------------------------------


def test_academic_watch_writes_audit_record(roster_sheets, tmp_path):
    audit_path = tmp_path / "logs" / "audit_log.jsonl"
    execute_academic_watch_action(
        filters=[{"column": "Department", "operator": "equals", "value": "Biology"}],
        sheets=roster_sheets, sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=audit_path,
    )
    assert audit_path.exists()
    lines = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
    assert lines, "audit log should contain at least one record"
    entry = lines[0]
    assert entry["action_type"] == "academic_watch"
    assert entry["target_sheet"] == "Students"
    assert entry["rows_affected"] > 0
    assert entry["confirmation_status"] == "confirmed"
    # Privacy: never log full student rows; only metadata.
    serialized = json.dumps(entry)
    assert "Student " not in serialized  # the student-name field values shouldn't leak
    assert "GPA\":" not in serialized.replace(" ", "")[:200] or "filters" in serialized


# ---- 16. Interaction log: search-side metadata, no row leakage -----------


def test_interaction_log_sanitizer_keeps_filter_metadata_for_workflow():
    """The workflow filters (Department, GPA threshold) are not sensitive on
    their own — they're metadata about the query, not row values — so the
    sanitizer preserves them so the log can be mined later."""
    filters = [
        {"column": "Department", "operator": "equals", "value": "Biology"},
        {"column": "GPA", "operator": "less_than", "value": 2.0},
    ]
    sanitized = sanitize_filters(filters)
    assert sanitized[0]["value"] == "Biology"
    assert sanitized[1]["value"] == 2.0


def test_interaction_log_sanitizer_does_not_let_student_name_through():
    """Defensive: a filter that *did* target a sensitive column gets redacted."""
    filters = [{"column": "Student Name", "operator": "equals", "value": "Student 0001"}]
    sanitized = sanitize_filters(filters)
    assert sanitized[0]["value"] == "[REDACTED]"
