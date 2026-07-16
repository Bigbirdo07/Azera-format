"""L.12 – L.15 vague-term resolution and corrected execution."""

from __future__ import annotations

import pytest

from core.interaction_logger import extract_corrected_request, is_correction_message
from nlp.planner_router import plan_user_request
from nlp.vague_terms import resolve_vague_term


COLUMNS_FULL = ["Department", "Major", "Year", "GPA", "Academic Status", "Advisor"]
COLUMNS_NO_RISK = ["Department", "Major", "Year"]
COLUMNS_GPA_ONLY = ["Department", "Major", "Year", "GPA"]


# ---- resolver unit tests ---------------------------------------------------


def test_struggling_builds_status_filter_when_both_columns_exist():
    res = resolve_vague_term(message="show me struggling students",
                             sheet="Students",
                             columns=COLUMNS_FULL,
                             categorical_values={"Academic Status": ["Good Standing", "Warning",
                                                                     "Probation", "At Risk"]})
    assert res is not None
    assert res.has_plan
    assert res.query["operation"] == "filtered_preview"
    assert res.query["filters"]
    status_filter = res.query["filters"][0]
    assert status_filter["column"] == "Academic Status"
    assert status_filter["operator"] == "in"
    assert set(status_filter["value"]) == {"Warning", "Probation", "At Risk"}
    assert "Academic Status" in res.assumption
    assert res.alternatives


def test_struggling_falls_back_to_gpa_when_only_gpa_exists():
    res = resolve_vague_term(message="who is struggling",
                             sheet="Students",
                             columns=COLUMNS_GPA_ONLY,
                             categorical_values={})
    assert res is not None and res.has_plan
    assert res.query["filters"][0]["column"] == "GPA"
    assert res.query["filters"][0]["operator"] == "less_than"
    assert res.query["filters"][0]["value"] == 2.5


def test_struggling_clarifies_when_no_supporting_columns():
    res = resolve_vague_term(message="who is struggling",
                             sheet="Students",
                             columns=COLUMNS_NO_RISK,
                             categorical_values={})
    assert res is not None
    assert res.has_plan is False
    assert res.clarification


def test_overloaded_advisors_groups_by_advisor_count():
    res = resolve_vague_term(message="show me overloaded advisors",
                             sheet="Students",
                             columns=COLUMNS_FULL,
                             categorical_values={})
    assert res is not None and res.has_plan
    assert res.query["operation"] == "groupby_count"
    assert res.query["group_by"] == "Advisor"
    assert res.query["sort"]["direction"] == "descending"


def test_no_advisor_uses_is_blank():
    res = resolve_vague_term(message="students with no advisor",
                             sheet="Students",
                             columns=COLUMNS_FULL,
                             categorical_values={})
    assert res is not None and res.has_plan
    assert res.query["filters"][0]["column"] == "Advisor"
    assert res.query["filters"][0]["operator"] == "is_blank"


def test_specific_query_returns_none():
    """Specific phrasings should NOT be resolved by the vague-term layer."""
    res = resolve_vague_term(message="show me Accounting students",
                             sheet="Students",
                             columns=COLUMNS_FULL,
                             categorical_values={})
    assert res is None


# ---- end-to-end through the planner ----------------------------------------


def _route(message, sheets, columns, *, state=None):
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=state,
        settings={"llm_enabled": False},
    )


def test_struggling_returns_concrete_filter_via_router(sheets, columns):
    routing = _route("show me struggling students", sheets, columns)
    assert routing["intent"] == "query"
    assert routing["band"] == "medium"
    # The bug we're fixing: must NOT be a bare filtered_preview with empty filters.
    plan = routing["plan"]
    assert plan["filters"], "vague-term plan must carry filters, not return whole sheet"
    assert routing["assumption_note"]
    assert routing["alternatives"]


def test_struggling_clarifies_when_workbook_lacks_supporting_columns(sheets):
    bare_columns = ["Department", "Year"]
    routing = _route("show me struggling students", sheets, bare_columns)
    assert routing["intent"] == "clarify"
    # Confirmation reason holds the resolver's clarification text.
    assert "definition" in (routing["confirmation_reason"] or "").lower()


def test_overloaded_advisors_via_router(sheets, columns):
    routing = _route("show me overloaded advisors", sheets, columns)
    assert routing["intent"] == "query"
    assert routing["plan"]["operation"] == "groupby_count"
    assert routing["plan"]["group_by"] == "Advisor"


def test_no_advisor_via_router(sheets, columns):
    routing = _route("show me students with no advisor", sheets, columns)
    assert routing["intent"] == "query"
    # The rule planner may already match this as `is_missing`; the vague-term
    # resolver uses `is_blank`. Either is acceptable — both mean "no value".
    operator = routing["plan"]["filters"][0]["operator"]
    assert operator in {"is_blank", "is_missing"}
    assert routing["plan"]["filters"][0]["column"] == "Advisor"


# ---- specific-phrase queries are NOT hijacked by the vague layer -----------


def test_specific_phrase_with_filter_still_wins(sheets, columns):
    """A query the rule planner can fully ground should keep its plan."""
    routing = _route("show me Accounting students", sheets, columns)
    assert routing["band"] == "high"
    # Accounting filter must be present, not a vague-term fallback.
    cols_used = [f["column"] for f in routing["plan"]["filters"]]
    assert "Department" in cols_used


# ---- correction extraction (L.15) ------------------------------------------


def test_extract_strips_correction_prefix():
    assert extract_corrected_request("no, I mean students on probation") == "students on probation"
    assert extract_corrected_request("Actually, show me seniors only") == "show me seniors only"
    assert extract_corrected_request("I meant GPA below 2.0") == "GPA below 2.0"


def test_extract_passes_through_normal_message():
    assert extract_corrected_request("show me Accounting students") == "show me Accounting students"


def test_correction_message_executes_corrected_query(sheets, columns):
    """The extracted message must yield a usable plan, not an average_column trip."""
    routing = _route("students on probation", sheets, columns)
    assert routing["intent"] == "query"
    # Either a direct value match landed it as a high-confidence filter, or
    # the vague resolver picked it up at medium. Either way, it must NOT be a
    # bare filtered_preview with empty filters and it must NOT be average_column.
    plan = routing["plan"]
    assert plan["operation"] != "average_column"
    assert plan["filters"], "corrected plan must carry filters"


# ---- L.14 safeguard: never broad whole-sheet on vague-risk -----------------


def test_vague_risk_never_returns_bare_filtered_preview(sheets, columns):
    for message in [
        "show me struggling students",
        "students at risk",
        "who needs advisor attention",
        "show me concerning students",
        "underperforming students",
        "falling behind students",
    ]:
        routing = _route(message, sheets, columns)
        if routing["intent"] != "query":
            continue  # clarify is also acceptable per L.14
        plan = routing["plan"]
        # The plan must be BOUNDED, not a whole-sheet dump. Bounded means any of:
        # a filter, a grouping, a sort+limit (e.g. the intervention summary's
        # top-50 by signal), or a specialized summary operation that isn't a bare
        # preview/count.
        bounded = (
            plan["filters"]
            or plan.get("group_by")
            or (plan.get("sort") and plan.get("limit"))
            or plan.get("operation") not in {"filtered_preview", "count_rows"}
        )
        assert bounded, (
            f"{message!r} produced a bare whole-sheet plan: {plan}"
        )
