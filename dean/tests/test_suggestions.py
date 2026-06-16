"""L.3: grounded next-move suggestions after each answer."""

from __future__ import annotations

from nlp.planner_router import plan_user_request
from nlp.suggestions import suggest_next_moves


def _route(message, sheets, columns, *, state=None):
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=state,
        settings={"llm_enabled": False},
    )


# ---- suggestions unit tests -------------------------------------------------


def test_filtered_preview_suggests_export_and_note(columns):
    plan = {
        "operation": "filtered_preview",
        "filters": [{"column": "Department", "operator": "equals", "value": "Accounting"}],
    }
    suggestions = suggest_next_moves(plan=plan, columns=columns, active_filters=plan["filters"], row_count=10)
    assert any("export" in s.lower() for s in suggestions)
    assert any("note" in s.lower() for s in suggestions)


def test_empty_result_suggests_broadening():
    plan = {"operation": "filtered_preview", "filters": []}
    suggestions = suggest_next_moves(plan=plan, columns=["Department"], row_count=0)
    assert any("clear" in s.lower() or "broader" in s.lower() for s in suggestions)


def test_groupby_suggests_chart_and_sort(columns):
    plan = {"operation": "groupby_count", "group_by": "Department"}
    suggestions = suggest_next_moves(plan=plan, columns=columns, active_filters=[], row_count=5)
    assert any("chart" in s.lower() for s in suggestions)
    assert any("sort" in s.lower() for s in suggestions)


def test_grouping_column_excludes_already_filtered(columns):
    """If Department is filtered, the suggestion should pick another grouping column."""
    plan = {
        "operation": "filtered_preview",
        "filters": [{"column": "Department", "operator": "equals", "value": "Biology"}],
    }
    suggestions = suggest_next_moves(plan=plan, columns=columns, active_filters=plan["filters"], row_count=5)
    group_suggestions = [s for s in suggestions if s.lower().startswith("group these by")]
    # Either we found a non-Department column, or we found nothing (acceptable).
    for suggestion in group_suggestions:
        assert "department" not in suggestion.lower()


def test_suggestions_capped_at_three():
    plan = {"operation": "filtered_preview", "filters": []}
    suggestions = suggest_next_moves(
        plan=plan,
        columns=["Department", "Advisor", "Major", "Year", "Academic Status"],
        active_filters=[],
        row_count=5,
    )
    assert len(suggestions) <= 3


# ---- planner_router attaches suggestions -----------------------------------


def test_routing_attaches_suggestions(sheets, columns):
    routing = _route("Show me Accounting students", sheets, columns)
    assert routing["suggestions"]
    assert len(routing["suggestions"]) <= 3


def test_suggestions_only_reference_existing_columns(sheets, columns):
    routing = _route("Show me Accounting students", sheets, columns)
    for suggestion in routing["suggestions"]:
        # Every column-referencing suggestion must use a real workbook column.
        if suggestion.lower().startswith(("group these by", "break this down by", "now group these by")):
            referenced = suggestion.split("by", 1)[1].strip()
            assert referenced in columns, f"Suggestion references missing column: {referenced}"
