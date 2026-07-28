"""Pivot routing: explicit "pivot" phrasing already worked; this locks in the
implicit two-dimension cross-tab trigger added to widen coverage, and confirms
it never hijacks ordinary single-group "by X" questions or nonsense pairings
like "sort by GPA and name"."""

from __future__ import annotations

import pandas as pd
import pytest

from nlp.planner_router import plan_user_request


def _sheets():
    frame = pd.DataFrame(
        {
            "Student ID": ["S1", "S2", "S3", "S4", "S5"],
            "Name": ["A", "B", "C", "D", "E"],
            "Year": ["Senior", "Junior", "Freshman", "Senior", "Sophomore"],
            "Standing": ["Good Standing", "Bad Standing", "Good Standing", "Bad Standing", "Good Standing"],
            "Advisor": ["Dr. A", "Dr. A", "Dr. B", "Dr. B", "Dr. C"],
            "GPA": [3.4, 2.1, 3.0, 1.8, 3.7],
            "Attendance Rate": [94.0, 88.0, 97.0, 82.0, 91.0],
        }
    )
    return {"Students": frame}, list(frame.columns)


def _route(message):
    sheets, columns = _sheets()
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=None,
        settings={"llm_enabled": False},
        llm_call=None,
    )


@pytest.mark.parametrize("message", [
    "average GPA by advisor and year",
    "break down attendance rate by advisor and standing",
    "show a crosstab of standing by advisor",
    "cross tab of year by advisor",
])
def test_explicit_and_near_explicit_pivot_language_routes_to_pivot(message):
    routing = _route(message)
    assert routing["plan"]["operation"] == "pivot_table_summary"


def test_implicit_two_dimension_breakdown_without_pivot_word_routes_to_pivot():
    routing = _route("average GPA by advisor and standing")
    assert routing["plan"]["operation"] == "pivot_table_summary"
    assert routing["plan"]["pivot_rows"] == "Advisor"
    assert routing["plan"]["pivot_columns"] == "Standing"


def test_single_dimension_by_question_does_not_hijack_pivot():
    routing = _route("what is the average GPA by advisor")
    assert routing["plan"]["operation"] != "pivot_table_summary"


def test_sort_by_gpa_and_name_does_not_become_a_pivot():
    routing = _route("sort students by GPA and name")
    assert routing["plan"]["operation"] != "pivot_table_summary"


def _skyward_style_sheets():
    # Distinct from _sheets() above on purpose: a real "Grade" column
    # (9-12) alongside a longer, non-bare GPA column name -- the shape
    # that exposed the crash below (the college fixture's bare "GPA"
    # column always resolves via exact match, masking the bug).
    frame = pd.DataFrame(
        {
            "Student ID": ["S1", "S2", "S3", "S4", "S5"],
            "Grade": [9, 10, 11, 12, 9],
            "Current Cumulative GPA": [3.4, 2.1, 3.0, 1.8, 3.7],
            "Advisor": ["Dr. A", "Dr. A", "Dr. B", "Dr. B", "Dr. C"],
            "Risk Level": ["Low Risk", "High Risk", "Low Risk", "Moderate Risk", "Low Risk"],
        }
    )
    return {"Students": frame}, list(frame.columns)


def _route_skyward(message):
    sheets, columns = _skyward_style_sheets()
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=None,
        settings={"llm_enabled": False},
        llm_call=None,
    )


def test_pivot_by_grade_resolves_gpa_value_column_not_grade_itself():
    # Regression: "gpa" resolved to the "Grade" column instead of "Current
    # Cumulative GPA" -- the "grade point average" alias for gpa contains
    # "grade" as a whole word, and a real "Grade" column matched it via
    # loose substring-of-alias-phrase matching in _find_column_by_names.
    # That set pivot_rows == value_column == "Grade", which crashed pandas
    # ("Grouper for 'Grade' not 1-dimensional") the moment the pivot
    # actually executed -- run_query() below is the real execution path,
    # not just routing, so this catches the crash, not just the plan.
    from core.query_engine import run_query

    routing = _route_skyward("pivot table of average gpa by grade and risk level")
    plan = routing["plan"]
    assert plan["operation"] == "pivot_table_summary"
    assert plan["pivot_rows"] == "Grade"
    assert plan["pivot_columns"] == "Risk Level"
    assert plan["value_column"] == "Current Cumulative GPA"
    sheets, _ = _skyward_style_sheets()
    result = run_query(plan, sheets)  # must not raise
    assert result.row_count > 0
