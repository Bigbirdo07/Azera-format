"""Figure-generating questions a dean / counselor would actually ask.

Guards the chart-intent detection in ui/figures_panel against the regressions
found while exercising 15 real questions: grouping by an identity column
(Student ID / First Name) instead of the named category, and missing the
"average X by Y" and "each X" phrasings.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import FakeUpload, FIXTURE
from core.excel_loader import load_excel_workbook
from ui.figures_panel import (
    compute_chart,
    detect_chart_intent,
    is_chart_request,
    suggested_figure_questions,
)


@pytest.fixture(scope="module")
def students() -> pd.DataFrame:
    return load_excel_workbook(FakeUpload(FIXTURE)).sheets["Students"]


@pytest.fixture(scope="module")
def columns(students) -> list[str]:
    return list(students.columns)


# (question, expected_chart_type, expected_field, expected_metric)
FIGURE_CASES = [
    ("Show me a bar chart of students by department", "bar", "Department", "count"),
    ("Pie chart of academic status", "pie", "Academic Status", "count"),
    ("Show the GPA distribution", "histogram", "GPA", "count"),
    ("Average GPA by major", "bar", "Major", "average"),
    ("Bar chart of students by year", "bar", "Year", "count"),
    ("How many students does each advisor have? show a chart", "bar", "Advisor", "count"),
    ("Visualize the number of students in each major", "bar", "Major", "count"),
    ("Average GPA by department", "bar", "Department", "average"),
    ("Pie chart of graduation status", "pie", "Graduation Status", "count"),
    ("Chart of students by financial aid status", "bar", "Financial Aid Status", "count"),
    ("Show a bar graph of conduct status", "bar", "Conduct Status", "count"),
    ("Histogram of credits completed", "histogram", "Credits Completed", "count"),
    ("Average credits completed by year", "bar", "Year", "average"),
    ("Show me a chart of average GPA by advisor", "bar", "Advisor", "average"),
    ("Plot how many students are in good vs bad academic standing", "bar", "Academic Status", "count"),
]


@pytest.mark.parametrize("question,chart_type,field,metric", FIGURE_CASES)
def test_figure_question_produces_correct_chart(question, chart_type, field, metric, columns, students):
    assert is_chart_request(question), f"not detected as a chart request: {question!r}"
    intent = detect_chart_intent(question, columns)
    assert intent is not None, f"no chart intent for: {question!r}"
    assert intent.chart_type == chart_type, f"{question!r}: {intent.chart_type} != {chart_type}"
    assert intent.field == field, f"{question!r}: grouped by {intent.field!r}, expected {field!r}"
    assert intent.metric == metric, f"{question!r}: metric {intent.metric!r} != {metric!r}"
    # The chart must actually compute a non-empty, sensibly-sized summary
    # (the old bugs produced 600-row "one bar per student" charts).
    summary = compute_chart(intent, students)
    assert summary is not None and not summary.empty
    assert len(summary) <= 30, f"{question!r}: {len(summary)} groups looks like an identity-column chart"


def test_average_by_phrasing_is_a_figure_without_the_word_chart():
    # "average X by Y" should be treated as a figure request on its own.
    assert is_chart_request("average GPA by major")
    assert is_chart_request("mean credits completed per year")
    # A plain question without a breakdown should not become a chart.
    assert not is_chart_request("what is the average GPA")


def test_chart_never_groups_by_identity_column(columns, students):
    # Even a vague request must not group by Student ID / names.
    for question in ("chart of students", "show a chart of the students by name"):
        intent = detect_chart_intent(question, columns)
        if intent is not None:
            assert intent.field not in {"Student ID", "First Name", "Last Name", "Name"}


# --- data-aware figure suggestions -------------------------------------------


def test_every_suggested_chip_renders_a_real_chart(columns, students):
    suggestions = suggested_figure_questions(columns)
    assert suggestions, "expected figure suggestions for the standard workbook"
    for label, query in suggestions:
        intent = detect_chart_intent(query, columns)
        assert intent is not None, f"chip {label!r} -> {query!r} produced no intent"
        assert intent.field in columns
        summary = compute_chart(intent, students)
        assert summary is not None and not summary.empty
        assert len(summary) <= 30


def test_suggestions_are_data_aware():
    # No GPA column -> no "Average GPA" chips, but count charts still offered.
    no_gpa = suggested_figure_questions(["Student ID", "Department", "Year"])
    labels = [label for label, _ in no_gpa]
    assert labels  # still suggests count charts
    assert not any("Average GPA" in label for label in labels)

    # A bare ID + one category -> exactly the one count chart, nothing invented.
    minimal = suggested_figure_questions(["Student ID", "Major"])
    assert minimal == [("Students by major", "bar chart of students by major")]


def test_average_chip_only_when_numeric_column_resolves():
    # "Average GPA by major" must not appear when there's no GPA to average.
    labels = [label for label, _ in suggested_figure_questions(["Student ID", "Major"])]
    assert not any("Average" in label for label in labels)
