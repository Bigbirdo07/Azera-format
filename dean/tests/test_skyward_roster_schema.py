"""Validates Dean's schema/sensitivity/query layers against a workbook
shaped like a REAL Skyward export (scripts/make_skyward_workbook.py, built
from knowledge/skyward_field_map.json) -- as opposed to the hand-invented
mock rosters used everywhere else. Fast and deterministic: no LLM calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.data_sources import DataSourceRegistry
from core.excel_loader import load_excel_workbook
from core.execution_dispatcher import execute_planned_request
from core.privacy import classify_sensitivity
from core.schema import canonical_for
from nlp.planner_router import plan_user_request
from scripts.make_skyward_workbook import DEFAULT_OUT, write_workbook

FIXTURE = DEFAULT_OUT


def _ensure_fixture() -> None:
    if not FIXTURE.exists():
        write_workbook(FIXTURE)


_ensure_fixture()


class FakeUpload:
    def __init__(self, path: Path):
        self._bytes = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self) -> bytes:
        return self._bytes


@pytest.fixture()
def loaded():
    return load_excel_workbook(FakeUpload(FIXTURE))


@pytest.fixture()
def sheets(loaded):
    reg = DataSourceRegistry()
    reg.set_roster(loaded)
    return reg.enriched_sheets() or dict(loaded.sheets)


def test_workbook_loads_without_error(loaded):
    assert "Students" in loaded.sheets
    assert len(loaded.sheets["Students"]) == 250


@pytest.mark.parametrize(
    "header,expected_concept",
    [
        ("Student ID", "student_id"),
        ("Name", "full_name"),
        ("Grad Year", "grad_year"),
        ("Birth Date", "date_of_birth"),
        ("Phone", "phone"),
        ("Email", "email"),
        ("Home Address", "home_address"),
        ("Current Cumulative GPA", "cumulative_gpa"),
        ("Advisor", "advisor"),
        ("Guardian Phone", "parent_guardian_contact"),
        ("Guardian Email", "parent_guardian_contact"),
        ("Excused Absences", "excused_absences"),
        ("Unexcused Absences", "unexcused_absences"),
        ("Tardies", "days_tardy"),
        ("Attendance Rate", "attendance_rate"),
        ("Discipline Information", "conduct_status"),
        ("SAT Math", "sat_math"),
        ("SAT EBRW", "sat_ebrw"),
        ("SAT Total", "sat_total"),
        ("PSAT Math", "psat_math"),
        ("PSAT Reading/Writing", "psat_reading_writing"),
        ("PSAT Total", "psat_total"),
        ("Entry Date", "entry_date"),
        ("Withdrawal Date", "withdrawal_date"),
        ("Emergency Contact", "emergency_contact"),
    ],
)
def test_mapped_fields_resolve_to_expected_concept(header, expected_concept):
    assert canonical_for(header) == expected_concept


def test_grade_column_has_no_canonical_concept_but_still_answers_questions():
    # Known gap: canonical_for("Grade") returns None (no direct concept
    # mapping for grade-level), unlike "Year" in the other mock rosters.
    # The query planner still resolves it via literal column-name matching,
    # so this documents current behavior rather than asserting an ideal.
    assert canonical_for("Grade") is None


def test_guardian_name_and_student_name_share_a_concept():
    # Known ambiguity: "Guardian Name" resolves to the same "full_name"
    # concept as the student's own "Name" column. Not exercised by any
    # question in this test file, but worth knowing before building a
    # feature that resolves "the student's name" by concept alone rather
    # than by explicit column.
    assert canonical_for("Guardian Name") == canonical_for("Name") == "full_name"


@pytest.mark.parametrize(
    "column,expected_sensitive,expected_type",
    [
        ("Home Address", True, "contact"),
        ("Guardian Phone", True, "contact"),
        ("Guardian Email", True, "contact"),
        ("Emergency Contact", True, "contact"),
        ("Discipline Information", True, "disciplinary"),
        ("Student ID", True, "identity"),
    ],
)
def test_sensitive_columns_are_flagged(column, expected_sensitive, expected_type):
    sensitive, sensitivity_type = classify_sensitivity(column)
    assert sensitive == expected_sensitive
    assert sensitivity_type == expected_type


def _ask(sheets, loaded, question: str) -> str:
    sheet_columns = {n: list(f.columns) for n, f in sheets.items()}
    selected_sheet = next(iter(sheets))
    routing = plan_user_request(
        user_message=question, sheets=sheets, sheet_columns=sheet_columns,
        selected_sheet=selected_sheet, conversation_state={}, settings={"llm_enabled": False},
    )
    response = execute_planned_request(routing, loaded, {"llm_enabled": False}, request_summary=question)
    return response.get("message", "")


def test_grade_level_question_answers_correctly(sheets, loaded):
    df = sheets["Students"]
    expected = int((df["Grade"] == 9).sum())
    message = _ask(sheets, loaded, "how many students are in grade 9")
    assert str(expected) in message


def test_gpa_threshold_question_answers_correctly(sheets, loaded):
    df = sheets["Students"]
    expected = int((df["Current Cumulative GPA"] >= 3.5).sum())
    message = _ask(sheets, loaded, "how many students have a current cumulative gpa of 3.5 or higher")
    assert str(expected) in message


def test_advisor_count_question_answers_correctly(sheets, loaded):
    df = sheets["Students"]
    advisor = df["Advisor"].iloc[0]
    expected = int((df["Advisor"] == advisor).sum())
    message = _ask(sheets, loaded, f"how many students does {advisor} advise")
    assert str(expected) in message


def test_home_address_presence_question_answers_correctly(sheets, loaded):
    df = sheets["Students"]
    expected = len(df)  # every row has a Home Address in this generator
    message = _ask(sheets, loaded, "how many students have a home address on file")
    assert str(expected) in message
