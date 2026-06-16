"""Dynamic suggestions: shape per workbook + every emitted suggestion must
route through the planner without falling into clarify."""

from __future__ import annotations

import pandas as pd
import pytest

from nlp.dynamic_suggestions import build_dynamic_suggestions
from nlp.planner_router import plan_user_request


def _plan(message: str, sheets: dict, columns_by_sheet: dict) -> dict:
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns=columns_by_sheet,
        selected_sheet="Students",
        conversation_state=None,
        settings={"llm_enabled": False},
    )


def _expect_no_clarify(question_text: str, sheets: dict, columns_by_sheet: dict) -> None:
    routing = _plan(question_text, sheets, columns_by_sheet)
    intent = routing["intent"]
    assert intent != "clarify", (
        f"Dynamic suggestion fell into clarify: {question_text!r} -> {routing}"
    )
    assert intent != "unavailable", (
        f"Dynamic suggestion was unavailable: {question_text!r}"
    )


# ---- shape tests ----------------------------------------------------------


def test_college_fixture_emits_core_intents(sheets, columns):
    suggestions = build_dynamic_suggestions(sheets["Students"], columns)
    text = " | ".join(q.text for q in suggestions)
    assert any("Top 10 students by GPA" in q.text for q in suggestions)
    assert any("Bottom 10" in q.text for q in suggestions)
    assert any("Show me freshmen" in q.text for q in suggestions), text
    assert any("List every Advisor" in q.text or "List every Department" in q.text
               for q in suggestions), text


def test_k12_fixture_uses_grade_phrasing():
    df = pd.DataFrame({
        "Student ID": [f"S{i}" for i in range(8)],
        "Name": list("ABCDEFGH"),
        "Grade": ["K", "1", "5", "5", "7", "10", "12", "K"],
        "Teacher": ["Smith"] * 4 + ["Jones"] * 4,
        "GPA": [3.5, 2.8, 3.9, 2.1, 1.7, 3.2, 4.0, 3.0],
    })
    suggestions = build_dynamic_suggestions(df, list(df.columns))
    text = " | ".join(q.text for q in suggestions)
    # Kindergarten wins over numeric grades when both exist.
    assert any("kindergarten" in q.text.lower() for q in suggestions), text
    assert any("Top 10 students by GPA" in q.text for q in suggestions)
    assert any("List every Teacher" in q.text for q in suggestions)


def test_k12_numbered_grades_no_kindergarten():
    """If the workbook has no K, fall back to the smallest numbered grade."""
    df = pd.DataFrame({
        "Name": list("ABCD"),
        "Grade": ["3", "5", "5", "7"],
        "GPA": [3.0, 2.5, 3.5, 4.0],
    })
    suggestions = build_dynamic_suggestions(df, list(df.columns))
    text = " | ".join(q.text for q in suggestions)
    assert any("3rd graders" in q.text for q in suggestions), text


def test_minimal_workbook_still_yields_useful_questions():
    df = pd.DataFrame({"Name": ["A", "B", "C"], "GPA": [3.5, 2.0, 4.0]})
    suggestions = build_dynamic_suggestions(df, list(df.columns))
    text = " | ".join(q.text for q in suggestions)
    assert any("Top 10 students by GPA" in q.text for q in suggestions), text
    assert any("Show me just" in q.text for q in suggestions), text


def test_skips_listing_high_cardinality_name_columns(sheets, columns):
    """`List every Name` would be a wall of 600 rows — skip it."""
    suggestions = build_dynamic_suggestions(sheets["Students"], columns)
    assert not any(q.text == "List every Name" for q in suggestions)


def test_dedupes_sample_value_across_columns():
    """Department and Major can both contain 'Accounting' — only suggest it once."""
    df = pd.DataFrame({
        "Name": list("ABCD"),
        "Department": ["Accounting", "Accounting", "Biology", "Biology"],
        "Major": ["Accounting", "Biology", "Accounting", "Biology"],
        "GPA": [3.0, 3.5, 2.0, 4.0],
    })
    suggestions = build_dynamic_suggestions(df, list(df.columns))
    texts = [q.text for q in suggestions]
    accounting_filters = [t for t in texts if t == "Show me Accounting students"]
    assert len(accounting_filters) <= 1, texts


# ---- routing tests --------------------------------------------------------


def test_every_college_suggestion_routes_without_clarify(sheets, columns):
    suggestions = build_dynamic_suggestions(sheets["Students"], columns)
    assert suggestions
    columns_by_sheet = {"Students": columns}
    for question in suggestions:
        _expect_no_clarify(question.text, sheets, columns_by_sheet)


def test_every_k12_suggestion_routes_without_clarify():
    df = pd.DataFrame({
        "Student ID": [f"S{i}" for i in range(6)],
        "Name": list("ABCDEF"),
        "Grade": ["K", "1", "5", "5", "7", "12"],
        "Teacher": ["Smith", "Smith", "Jones", "Jones", "Lee", "Lee"],
        "GPA": [3.5, 2.8, 3.9, 2.1, 1.7, 4.0],
    })
    sheets = {"Students": df}
    columns_by_sheet = {"Students": list(df.columns)}
    suggestions = build_dynamic_suggestions(df, list(df.columns))
    assert suggestions
    for question in suggestions:
        _expect_no_clarify(question.text, sheets, columns_by_sheet)
