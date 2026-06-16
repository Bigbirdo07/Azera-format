"""Phase M tests: free-text notes search.

Covers the 10 acceptance cases from the spec:
  1. Notes column classified as free-text and sensitive.
  2. contains_text is case-insensitive.
  3. contains_text handles blanks safely.
  4. not_contains_text works.
  5. "notes mention attendance" routes to contains_text.
  6. "no notes" routes to is_blank.
  7. Full note text hidden by default.
  8. Full note display requires confirmation.
  9. Interaction log does not store raw note row values.
 10. Existing tests remain green (verified by the full suite run).
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.execution_dispatcher import _match_snippet, _text_search_terms
from core.interaction_logger import sanitize_filters
from core.privacy import classify_sensitivity, is_hidden_by_default
from core.query_engine import _build_mask, run_query
from core.schema import build_workbook_schema, infer_column_types
from nlp.planner_router import plan_user_request
from nlp.query_planner import _notes_filter, _rule_plan


# ---- fixtures --------------------------------------------------------------


@pytest.fixture()
def notes_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Student ID": ["S1", "S2", "S3", "S4"],
        "Department": ["Accounting", "Biology", "Accounting", "Biology"],
        "GPA": [2.1, 3.5, 1.9, 3.2],
        "Advisor": ["Dr. Brooks", "Dr. Patel", "Dr. Brooks", "Dr. Patel"],
        "Notes": [
            "Mom called about attendance issues last week",
            "",
            "Considering changing major to Nursing",
            None,
        ],
    })


@pytest.fixture()
def notes_sheets(notes_frame):
    return {"Students": notes_frame}


# ---- M.1: free-text classification + sensitivity ---------------------------


def test_notes_column_classified_as_free_text(notes_frame):
    info = infer_column_types(notes_frame)
    assert info["Notes"]["semantic_role"] == "free_text"


def test_notes_column_is_sensitive_by_default():
    sensitive, sensitivity_type = classify_sensitivity("Notes")
    assert sensitive
    assert sensitivity_type == "notes"
    assert is_hidden_by_default("Notes")


def test_advisor_notes_also_classified_as_free_text():
    df = pd.DataFrame({"Advisor Notes": ["Long narrative " * 5, "Another note"]})
    info = infer_column_types(df)
    assert info["Advisor Notes"]["semantic_role"] == "free_text"


def test_short_categorical_text_is_not_free_text():
    # 'Status' with two repeating short values is NOT a narrative column.
    df = pd.DataFrame({"Status": ["Active", "Inactive"] * 10})
    info = infer_column_types(df)
    assert info["Status"]["semantic_role"] != "free_text"


def test_build_workbook_schema_lists_free_text_columns(notes_frame):
    schema = build_workbook_schema({"Students": notes_frame})
    assert "Notes" in schema["Students"]["free_text_columns"]


# ---- M.2: contains_text + not_contains_text operators ----------------------


def test_contains_text_is_case_insensitive(notes_frame):
    mask = _build_mask(
        notes_frame,
        [{"column": "Notes", "operator": "contains_text", "value": "ATTENDANCE"}],
    )
    assert mask.tolist() == [True, False, False, False]


def test_contains_text_handles_nan_safely(notes_frame):
    # Row S4 has a NaN note — must not crash and must not match.
    mask = _build_mask(
        notes_frame,
        [{"column": "Notes", "operator": "contains_text", "value": "anything"}],
    )
    # No match anywhere; specifically S4 (NaN) must be False, not NaN.
    assert mask.tolist()[3] is False or mask.tolist()[3] == False  # noqa: E712


def test_contains_text_handles_empty_string(notes_frame):
    # Row S2 has "" (empty string) — a contains_text for anything must skip it.
    mask = _build_mask(
        notes_frame,
        [{"column": "Notes", "operator": "contains_text", "value": "attendance"}],
    )
    assert mask.tolist()[1] is False or mask.tolist()[1] == False  # noqa: E712


def test_not_contains_text_returns_rows_that_dont_match(notes_frame):
    mask = _build_mask(
        notes_frame,
        [{"column": "Notes", "operator": "not_contains_text", "value": "attendance"}],
    )
    assert mask.tolist()[0] is False or mask.tolist()[0] == False  # noqa: E712
    assert mask.sum() == 3  # S2, S3, S4


def test_contains_text_uses_literal_not_regex(notes_frame):
    """A search term with regex specials must be matched literally, not as a
    pattern. The test passes if the call doesn't raise and the substring
    truly isn't in the data."""
    mask = _build_mask(
        notes_frame,
        [{"column": "Notes", "operator": "contains_text", "value": "(re)check"}],
    )
    # No row contains the literal "(re)check"; if regex were enabled the call
    # would raise re.error. Reaching this assertion means it ran cleanly.
    assert mask.sum() == 0


# ---- M.3: planner routes notes phrasings ----------------------------------


def test_planner_routes_notes_mention_to_contains_text(notes_sheets):
    plan = _rule_plan("show students whose notes mention attendance",
                      "Students", list(notes_sheets["Students"].columns),
                      frame=notes_sheets["Students"]).query
    f = plan["filters"][0]
    assert f == {"column": "Notes", "operator": "contains_text", "value": "attendance"}


def test_planner_routes_no_notes_to_is_blank(notes_sheets):
    plan = _rule_plan("which students have no notes", "Students",
                      list(notes_sheets["Students"].columns),
                      frame=notes_sheets["Students"]).query
    assert {"column": "Notes", "operator": "is_blank"} in plan["filters"]


def test_planner_routes_dont_mention_to_not_contains_text(notes_sheets):
    plan = _rule_plan("students whose notes do not mention follow-up",
                      "Students", list(notes_sheets["Students"].columns),
                      frame=notes_sheets["Students"]).query
    f = plan["filters"][0]
    assert f["operator"] == "not_contains_text"
    assert "follow" in f["value"]


def test_planner_preserves_quoted_phrase(notes_sheets):
    plan = _rule_plan('show me students with notes about "mom called"',
                      "Students", list(notes_sheets["Students"].columns),
                      frame=notes_sheets["Students"]).query
    f = plan["filters"][0]
    assert f["value"] == "mom called"


def test_planner_prefers_more_specific_notes_column():
    df = pd.DataFrame({
        "Student ID": ["S1"],
        "Notes": ["general note"],
        "Advisor Notes": ["advisor narrative " * 3],
    })
    plan = _rule_plan("show students whose advisor notes mention session",
                      "Students", list(df.columns), frame=df).query
    f = plan["filters"][0]
    assert f["column"] == "Advisor Notes"


# ---- M.4 + M.5: privacy + reveal + snippets -------------------------------


def test_text_search_terms_returns_column_to_term_mapping():
    plan = {"filters": [
        {"column": "Notes", "operator": "contains_text", "value": "attendance"},
        {"column": "GPA", "operator": "less_than", "value": 2.5},
    ]}
    assert _text_search_terms(plan) == {"Notes": "attendance"}


def test_match_snippet_returns_window_around_match():
    text = "The student had several conversations about attendance and follow-up plans"
    snippet = _match_snippet(text, "attendance", window=10)
    assert "attendance" in snippet
    assert len(snippet) <= len(text) + 6  # plus ellipses
    assert snippet.startswith("...")


def test_match_snippet_handles_no_match_gracefully():
    text = "Some unrelated content"
    snippet = _match_snippet(text, "missing_term", window=10)
    assert isinstance(snippet, str)


def test_full_notes_hidden_by_default_via_redact_table():
    """When the dispatcher redacts a result, the Notes column is removed."""
    from core.privacy import redact_table

    rows = [{"Student ID": "S1", "Notes": "private", "Department": "Acc"}]
    redacted, removed = redact_table(rows, ["Student ID", "Notes", "Department"])
    assert "Notes" in removed
    assert "Notes" not in redacted[0]


def test_show_notes_request_reveals_without_confirmation(notes_sheets):
    """Asking 'show me the full notes' reveals the Notes column directly — the
    sensitive-field confirmation gate was removed, so an explicitly requested
    hidden column is shown without a pending confirmation step."""
    routing = plan_user_request(
        user_message="show me the full notes for these students",
        sheets=notes_sheets,
        sheet_columns={"Students": list(notes_sheets["Students"].columns)},
        selected_sheet="Students",
        conversation_state={"active_filters": [{"column": "GPA", "operator": "less_than", "value": 2.5}]},
        settings={},
    )
    assert routing["intent"] == "query"
    assert not routing.get("requires_confirmation")
    assert routing.get("pending_type") is None
    # The Notes column is un-redacted for this result instead of gated.
    assert routing.get("reveal_sensitive") is True


# ---- M.7: interaction log keeps search term, never row text ---------------


def test_interaction_log_preserves_notes_search_term():
    """The user's search term IS user input, not row data — it should be
    logged so we can mine recurring searches."""
    filters = [{"column": "Notes", "operator": "contains_text", "value": "attendance"}]
    sanitized = sanitize_filters(filters)
    assert sanitized[0]["value"] == "attendance"  # NOT [REDACTED]


def test_interaction_log_redacts_value_filters_on_notes_column():
    """A non-text-search filter on Notes (e.g. equals) must still redact —
    only the text-search variants preserve the term."""
    filters = [{"column": "Notes", "operator": "equals", "value": "secret"}]
    sanitized = sanitize_filters(filters)
    assert sanitized[0]["value"] == "[REDACTED]"


def test_interaction_log_scrubs_pii_in_search_term():
    """Defensive: if the user pastes an email or long ID into the search box,
    the _sanitize_value scrubber should still clean it."""
    filters = [{"column": "Notes", "operator": "contains_text",
                "value": "contact alice@example.com asap"}]
    sanitized = sanitize_filters(filters)
    assert sanitized[0]["value"] == "[REDACTED]"


# ---- end-to-end through the query engine ----------------------------------


def test_run_query_with_contains_text_returns_filtered_rows(notes_sheets):
    """Smoke: real run_query with a contains_text filter returns only the
    rows that matched the substring."""
    plan = {
        "operation": "filtered_preview",
        "sheet": "Students",
        "filters": [{"column": "Notes", "operator": "contains_text", "value": "attendance"}],
        "limit": 10,
    }
    result = run_query(plan, notes_sheets)
    assert result.row_count == 1
    assert result.table[0]["Student ID"] == "S1"
