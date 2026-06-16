"""Phase P tests: result drilldown + follow-up context + action chaining.

The five spec acceptance cases plus unit tests for the new memory fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import pytest

from core.execution_dispatcher import execute_planned_request
from core.session_memory import SessionMemory
from nlp.drilldown import resolve_drilldown
from nlp.planner_router import plan_user_request


# ---- fixtures --------------------------------------------------------------


@dataclass
class _Loaded:
    sheets: dict
    file_name: str = "fixture.xlsx"


@pytest.fixture()
def roster() -> pd.DataFrame:
    """Dean-style roster: Advisor (no Teacher), Discipline (no Department),
    GPA, plus a Major to break-down on. Designed so 'teachers contain
    students under 2.5' has rows on every advisor."""
    advisors = ["Dr. Alpha", "Dr. Bravo", "Dr. Cathy", "Dr. Delta", "Dr. Echo"]
    rows = []
    sid = 0
    for advisor in advisors:
        for gpa, major in zip([1.5, 1.8, 2.1, 2.6, 3.0, 3.4],
                              ["Biology", "Chemistry", "Math", "Biology", "Chemistry", "Math"]):
            sid += 1
            rows.append({
                "Student ID": f"S{sid:03d}",
                "Advisor": advisor,
                "Discipline": "Sciences",
                "Major": major,
                "GPA": gpa,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def loaded(roster):
    return _Loaded(sheets={"Students": roster})


@pytest.fixture()
def memory():
    return SessionMemory()


def _step(memory: SessionMemory, loaded, message: str) -> dict:
    """Plan + execute one turn, record into memory like the UI does."""
    cols = list(next(iter(loaded.sheets.values())).columns)
    routing = plan_user_request(
        user_message=message,
        sheets=loaded.sheets,
        sheet_columns={"Students": cols},
        selected_sheet="Students",
        conversation_state=asdict(memory),
        settings={},
    )
    response = execute_planned_request(routing, loaded, settings={},
                                       request_summary=message)
    plan = routing.get("plan") or {}
    if routing.get("intent") == "query" and response.get("success"):
        memory.record_ask(
            request=message, query_plan=plan,
            result_description=response.get("description", "") or "",
            row_count=response.get("row_count"),
            columns_used=response.get("columns") or [],
            sheet=plan.get("sheet", ""),
            summary_table=response.get("result_preview") or [],
            top_group=response.get("top_group"),
        )
    return {"routing": routing, "response": response, "plan": plan}


# ---- P.1: memory records result shape -------------------------------------


def test_memory_records_group_level_after_count_unique(memory, loaded):
    _step(memory, loaded, "how many teachers have students under 2.5")
    assert memory.last_result_type == "aggregate"
    assert {"column": "GPA", "operator": "less_than", "value": 2.5} in memory.last_row_filter
    assert memory.last_referent_label


def test_memory_records_winner_after_top_n_groupby(memory, loaded):
    """After 'best avg department' the winner column + value should be stored."""
    _step(memory, loaded, "which department has the best average of students performing well")
    # The ambiguity detector emits a groupby_average with limit=1, so the
    # winner should be captured from the result table.
    assert memory.last_group_winner_column == "Discipline"
    assert memory.last_group_winner_value


# ---- P.2 + P.3: drilldown plans -------------------------------------------


def test_row_drilldown_after_group_count(memory, loaded):
    """'how many advisors have students under 2.5' → 'which students are those'
    must produce a filtered_preview using the prior row filter."""
    _step(memory, loaded, "how many advisors have students under 2.5")
    second = _step(memory, loaded, "which students are those")
    assert second["routing"].get("drilldown_kind") == "row_drilldown"
    assert second["plan"]["operation"] == "filtered_preview"
    assert {"column": "GPA", "operator": "less_than", "value": 2.5} \
        in second["plan"]["filters"]
    assert "previous filter" in second["response"]["message"].lower()


def test_generic_listing_after_count_unique_lists_distinct_values(memory, loaded):
    """'how many majors are there' -> 'what are they' should list the majors,
    not ask for clarification or require a row-level filter."""
    _step(memory, loaded, "how many majors are there")
    second = _step(memory, loaded, "what are they")
    assert second["routing"].get("drilldown_kind") == "distinct_listing"
    assert second["plan"]["operation"] == "groupby_count"
    assert second["plan"]["group_by"] == "Major"
    assert second["plan"]["limit"] is None
    assert {row["Major"] for row in second["response"]["result_preview"]} == {
        "Biology", "Chemistry", "Math",
    }


def test_winner_drilldown_after_top_department(memory, loaded):
    """'which department has the best avg' → 'show me students in that
    department' must add {winner_col equals winner_value} as a filter."""
    _step(memory, loaded, "which department has the best average of students performing well")
    second = _step(memory, loaded, "show me students in that department")
    assert second["routing"].get("drilldown_kind") == "winner_drilldown"
    filters = second["plan"]["filters"]
    assert any(f.get("column") == "Discipline" and f.get("operator") == "equals"
               for f in filters)


def test_show_students_in_top_group_after_group_count(memory, loaded):
    _step(memory, loaded, "how many students in each department")
    top_value = memory.last_group_winner_value
    second = _step(memory, loaded, "Show the students in the top group")
    assert second["routing"].get("drilldown_kind") == "winner_drilldown"
    assert second["plan"]["operation"] == "filtered_preview"
    assert {"column": "Discipline", "operator": "equals", "value": top_value} \
        in second["plan"]["filters"]


def test_named_top_group_drilldown_keeps_prior_filter(memory, loaded):
    """After grouping a filtered set, naming the visible top group should add
    that group filter instead of returning the whole prior filtered set."""
    _step(memory, loaded, "show me students above 2.0 gpa")
    _step(memory, loaded, "group them by advisor")
    top_advisor = memory.last_group_winner_value
    second = _step(memory, loaded, f"show me the students that are {top_advisor}")
    assert second["routing"].get("drilldown_kind") == "winner_drilldown"
    filters = second["plan"]["filters"]
    assert {"column": "GPA", "operator": "greater_than", "value": 2.0} in filters
    assert {"column": "Advisor", "operator": "equals", "value": top_advisor} in filters
    assert second["response"]["row_count"] == int(
        ((loaded.sheets["Students"]["GPA"] > 2.0)
         & (loaded.sheets["Students"]["Advisor"] == top_advisor)).sum()
    )


def test_breakdown_carries_prior_filter_and_resolves_column(memory, loaded):
    """'mention low GPA students' → 'break it down by advisor' must apply
    the prior GPA filter AND group by Advisor (resolved from the bare noun)."""
    _step(memory, loaded, "show me students below 2.5")
    second = _step(memory, loaded, "break it down by advisor")
    assert second["routing"].get("drilldown_kind") == "breakdown"
    assert second["plan"]["operation"] == "groupby_count"
    assert second["plan"]["group_by"] == "Advisor"
    assert any(f.get("column") == "GPA" for f in second["plan"]["filters"])


def test_breakdown_resolves_teacher_to_advisor(memory, loaded):
    """Cross-concept fallback flows into the breakdown too: 'by teacher' on
    an advisor-only roster picks Advisor."""
    _step(memory, loaded, "show me students below 2.5")
    second = _step(memory, loaded, "break it down by teacher")
    assert second["plan"]["group_by"] == "Advisor"


# ---- P.5: clarification when no context exists ----------------------------


def test_drilldown_with_no_prior_result_returns_none(loaded):
    """Cold start (no prior result) — drilldown resolver MUST decline."""
    fresh = SessionMemory()
    plan = resolve_drilldown("show me those", fresh,
                             columns=["Advisor", "GPA"])
    assert plan is None


def test_drilldown_after_filtered_preview_does_not_fire():
    """A row-level result is already row-level; 'show me those' shouldn't
    re-route to drilldown — the normal follow-up resolver handles it."""
    memory = SessionMemory(
        last_operation="filtered_preview",
        last_row_filter=[{"column": "GPA", "operator": "less_than", "value": 2.5}],
        last_sheet="Students",
    )
    # Despite the prior result, drilldown should decline because last_op is
    # already filtered_preview — there's nothing to drill INTO.
    plan = resolve_drilldown("show me the students", memory,
                             columns=["Advisor", "GPA"])
    assert plan is None


# ---- Safe action chaining (P.5 spec test 2) -------------------------------


def test_academic_watch_chaining_uses_prior_filter(memory, loaded):
    """'how many professors have students under 2.5' → 'mark them academic
    watch' must reach the academic_watch confirmation gate carrying the
    prior GPA filter."""
    _step(memory, loaded, "how many professors have students under 2.5")
    cols = list(next(iter(loaded.sheets.values())).columns)
    routing = plan_user_request(
        user_message="mark them academic watch",
        sheets=loaded.sheets,
        sheet_columns={"Students": cols},
        selected_sheet="Students",
        conversation_state=asdict(memory),
        settings={},
    )
    assert routing["intent"] == "academic_watch"
    assert routing["requires_confirmation"]
    filters = routing["plan"]["filters"]
    assert any(f.get("column") == "GPA" and f.get("operator") == "less_than"
               for f in filters)


# ---- Interpretation transparency continues to fire ------------------------


def test_drilldown_response_includes_context_reminder(memory, loaded):
    _step(memory, loaded, "how many advisors have students under 2.5")
    second = _step(memory, loaded, "which students are those")
    msg = second["response"]["message"]
    assert "previous filter" in msg.lower()
    assert "gpa" in msg.lower()
