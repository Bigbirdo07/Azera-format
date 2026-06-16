"""Phase O tests: interpretation transparency + dean-roster ambiguity.

The four spec acceptance cases plus unit tests for the cross-concept fallback
helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from core.execution_dispatcher import execute_planned_request
from nlp.ambiguity import detect_ambiguity
from nlp.planner_router import plan_user_request
from nlp.request_intents import is_academic_watch_request
from nlp.synonym_mapper import (
    load_json,
    match_column_for_concept_with_fallback,
)


# ---- fixtures --------------------------------------------------------------


@dataclass
class _Loaded:
    sheets: dict
    file_name: str = "fixture.xlsx"


@pytest.fixture()
def dean_roster() -> pd.DataFrame:
    """Synthetic dean-office roster with Advisor + Discipline (no Teacher / Department).
    Five advisors, ten students per advisor with mixed GPAs around the 2.5
    cutoff so the two ambiguity readings give different counts."""
    advisors = ["Dr. Alpha", "Dr. Bravo", "Dr. Cathy", "Dr. Delta", "Dr. Echo"]
    disciplines = ["Education", "Biology", "Education", "Math", "Math"]
    rows = []
    sid = 0
    for advisor, discipline in zip(advisors, disciplines):
        # 4 students below 2.5, 6 above — so each advisor "contains" a student
        # below 2.5, but only some advisors have an aggregate avg < 2.5.
        for gpa in [1.8, 2.1, 2.3, 2.4, 2.6, 2.9, 3.1, 3.3, 3.6, 3.8]:
            sid += 1
            rows.append({
                "Student ID": f"S{sid:03d}",
                "Advisor": advisor,
                "Discipline": discipline,
                "GPA": gpa,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def loaded_dean(dean_roster):
    return _Loaded(sheets={"Students": dean_roster})


@pytest.fixture()
def synonyms():
    return load_json("synonyms.json")


def _route(message, loaded):
    cols = list(next(iter(loaded.sheets.values())).columns)
    return plan_user_request(
        user_message=message,
        sheets=loaded.sheets,
        sheet_columns={"Students": cols},
        selected_sheet="Students",
        conversation_state={},
        settings={},
    )


# ---- O.1: cross-concept fallback (unit) -----------------------------------


def test_teacher_falls_back_to_advisor_when_no_teacher_column(synonyms):
    cols = ["Student ID", "Advisor", "Discipline", "GPA"]
    column, score, fallback_from = match_column_for_concept_with_fallback(
        "teacher", cols, synonyms,
    )
    assert column == "Advisor"
    assert score >= 0.55
    assert fallback_from == "teacher"


def test_professor_falls_back_to_advisor(synonyms):
    cols = ["Student ID", "Advisor", "Discipline", "GPA"]
    column, _, fallback_from = match_column_for_concept_with_fallback(
        "professor", cols, synonyms,
    )
    assert column == "Advisor"
    assert fallback_from == "professor"


def test_teacher_does_not_fall_back_when_teacher_column_present(synonyms):
    cols = ["Student ID", "Teacher", "Advisor", "GPA"]
    column, _, fallback_from = match_column_for_concept_with_fallback(
        "teacher", cols, synonyms,
    )
    assert column == "Teacher"
    assert fallback_from is None


def test_department_falls_back_to_discipline(synonyms):
    cols = ["Student ID", "Advisor", "Discipline", "GPA"]
    column, _, fallback_from = match_column_for_concept_with_fallback(
        "department", cols, synonyms,
    )
    assert column == "Discipline"
    assert fallback_from == "department"


def test_unrelated_concept_returns_no_match(synonyms):
    cols = ["Student ID", "Advisor", "Discipline", "GPA"]
    column, _, fallback_from = match_column_for_concept_with_fallback(
        "housing_status", cols, synonyms,
    )
    assert column is None
    assert fallback_from is None


# ---- O.2: detector returns the right two interpretations ------------------


def test_container_aggregate_ambiguity_detected(synonyms):
    cols = ["Student ID", "Advisor", "Discipline", "GPA"]
    res = detect_ambiguity(
        "how many teachers contain students with less than an average of 2.5",
        sheet="Students", columns=cols, synonyms=synonyms,
    )
    assert res is not None
    assert res.kind == "container_vs_aggregate"
    assert res.primary_plan["operation"] == "count_unique"
    assert res.primary_plan["value_column"] == "Advisor"
    assert {"column": "GPA", "operator": "less_than", "value": 2.5} \
        in res.primary_plan["filters"]
    assert res.alternative_spec["kind"] == "aggregate_then_count_groups"
    assert res.alternative_spec["group_column"] == "Advisor"
    assert ("teacher", "Advisor") in res.column_mapping


def test_container_aggregate_no_match_without_container_verb(synonyms):
    cols = ["Student ID", "Advisor", "GPA"]
    # 'show me students below 2.5' is a flat filter — not ambiguous.
    assert detect_ambiguity("show me students below 2.5",
                            sheet="Students", columns=cols, synonyms=synonyms) is None


# ---- O.3: performing-well ambiguity ---------------------------------------


def test_performing_well_ambiguity_detected(synonyms):
    cols = ["Student ID", "Advisor", "Discipline", "GPA"]
    res = detect_ambiguity(
        "which department has the best average of students performing well",
        sheet="Students", columns=cols, synonyms=synonyms,
    )
    assert res is not None
    assert res.kind == "performing_well"
    assert res.primary_plan["operation"] == "groupby_average"
    assert res.primary_plan["group_by"] == "Discipline"
    assert res.alternative_spec["kind"] == "rate_above_cutoff_by_group"
    assert ("department", "Discipline") in res.column_mapping


# ---- O.2 end-to-end through router + dispatcher ---------------------------


def test_teachers_contain_students_routes_with_both_results(loaded_dean):
    routing = _route(
        "how many teachers contain students with less than an average of 2.5",
        loaded_dean,
    )
    assert routing["band"] == "medium"
    assert routing["ambiguity"]["kind"] == "container_vs_aggregate"
    assert routing["alternatives"]
    assert routing["plan"]["value_column"] == "Advisor"

    response = execute_planned_request(routing, loaded_dean, settings={},
                                       request_summary="...")
    message = response["message"]
    assert "Advisor" in message
    assert "teacher as Advisor" in message
    assert "Alternative interpretation" in message
    # The primary (container) count is 5 — every advisor has at least one
    # student below 2.5. The aggregate count is 0 — every advisor's mean is
    # above 2.5 by construction. The dispatcher must surface both numbers.
    assert "5" in message
    assert "0" in message


def test_professors_have_students_under_threshold_maps_to_advisor(loaded_dean):
    routing = _route("how many professors have students under 2.5", loaded_dean)
    assert routing["ambiguity"]["kind"] == "container_vs_aggregate"
    assert routing["plan"]["value_column"] == "Advisor"
    response = execute_planned_request(routing, loaded_dean, settings={},
                                       request_summary="...")
    assert "professor as Advisor" in response["message"]


# ---- O.3 end-to-end -------------------------------------------------------


def test_department_performance_ranks_both_readings(loaded_dean):
    routing = _route(
        "which department has the best average of students performing well",
        loaded_dean,
    )
    assert routing["ambiguity"]["kind"] == "performing_well"
    assert routing["plan"]["operation"] == "groupby_average"
    assert routing["plan"]["group_by"] == "Discipline"
    response = execute_planned_request(routing, loaded_dean, settings={},
                                       request_summary="...")
    message = response["message"]
    assert "department as Discipline" in message
    assert "Alternative interpretation" in message
    # Mention the share-above-cutoff alt format.
    assert "share of students" in message and "≥ 2.5" in message


# ---- O.4: Academic Watch action still routes through confirmation --------


def test_academic_watch_intent_still_recognized():
    """Phase O changes must not weaken the existing confirmed-action flow."""
    assert is_academic_watch_request("mark students under 2.5 as Academic Watch")
    assert is_academic_watch_request("flag these students")
    assert not is_academic_watch_request("add note: advisor follow-up needed")


def test_academic_watch_request_routes_through_confirmation(loaded_dean):
    state_like = {"active_filters": [{"column": "GPA", "operator": "less_than", "value": 2.5}]}
    cols = list(next(iter(loaded_dean.sheets.values())).columns)
    routing = plan_user_request(
        user_message="mark these students as Academic Watch",
        sheets=loaded_dean.sheets,
        sheet_columns={"Students": cols},
        selected_sheet="Students",
        conversation_state=state_like,
        settings={},
    )
    assert routing["intent"] == "academic_watch"
    assert routing["requires_confirmation"]


# ---- Interpretation line is always present when ambiguity fires ----------


def test_interpretation_line_lists_concept_to_column_mappings(loaded_dean):
    routing = _route("how many teachers have students under 2.5", loaded_dean)
    assumption = routing.get("assumption_note", "")
    # Must mention both mappings: the entity (teacher→Advisor) and gpa.
    assert "teacher as Advisor" in assumption
    assert "gpa as GPA" in assumption
