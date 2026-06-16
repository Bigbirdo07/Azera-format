"""L.1 + L.2: always-on conversational mode and plain-English plan narration."""

from __future__ import annotations

import json

import pytest

from core import execution_dispatcher
from core.execution_dispatcher import execute_planned_request
from core.llm_config import from_app_settings
from nlp.narration import narrate_plan
from nlp.planner_router import plan_user_request


# ---- narration unit tests ---------------------------------------------------


def test_narration_handles_simple_filter():
    plan = {
        "operation": "filtered_preview",
        "filters": [
            {"column": "Department", "operator": "equals", "value": "Accounting"},
            {"column": "GPA", "operator": "less_than", "value": 2.5},
        ],
    }
    text = narrate_plan(plan=plan)
    assert text.startswith("I understood this as")
    assert "Department" in text and "Accounting" in text
    assert "GPA" in text and "2.5" in text


def test_narration_handles_followup_replacement():
    prior = [{"column": "Department", "operator": "equals", "value": "Accounting"}]
    new = [{"column": "Department", "operator": "equals", "value": "Biology"}]
    plan = {"operation": "filtered_preview", "filters": new}
    text = narrate_plan(plan=plan, context_action="followup",
                        prior_filters=prior, new_filters=new, additive=False)
    assert "replaced" in text.lower()
    assert "Biology" in text


def test_narration_handles_groupby_followup():
    prior = [{"column": "Department", "operator": "equals", "value": "Accounting"}]
    plan = {"operation": "groupby_count", "filters": prior, "group_by": "Advisor"}
    text = narrate_plan(plan=plan, context_action="followup",
                        prior_filters=prior, new_filters=[], additive=False)
    assert "Advisor" in text


def test_narration_includes_sort_and_limit():
    plan = {
        "operation": "filtered_preview",
        "filters": [{"column": "GPA", "operator": "less_than", "value": 3.0}],
        "sort": {"column": "GPA", "direction": "ascending"},
        "limit": 5,
    }
    text = narrate_plan(plan=plan)
    assert "sorted" in text.lower()
    assert "5" in text


# ---- planner_router emits narration ----------------------------------------


def _route(message, sheets, columns, *, state=None):
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=state,
        settings={"llm_enabled": False},
    )


def test_routing_includes_narration_on_rules_path(sheets, columns):
    routing = _route("Show me Accounting students", sheets, columns)
    assert routing["plan_source"] == "rules"
    assert routing["narration"].startswith("I understood this as")
    assert "Accounting" in routing["narration"]


def test_routing_narration_describes_followup(sheets, columns):
    state = {
        "active_filters": [{"column": "Department", "operator": "equals", "value": "Accounting"}],
        "sheet": "Students",
    }
    routing = _route("what about Biology", sheets, columns, state=state)
    assert "Biology" in routing["narration"]
    assert "replaced" in routing["narration"].lower()


# ---- llm_config strict-privacy gating --------------------------------------


def test_strict_privacy_forces_conversation_llm_off():
    config = from_app_settings({
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "strict_privacy_mode": True,
    })
    assert config["llm_enabled"] is False
    assert config["conversation_llm_enabled"] is False


def test_conversation_llm_requires_llm_enabled():
    # Even if conversation toggle is on, the umbrella LLM must be enabled.
    config = from_app_settings({
        "use_local_llm": False,
        "conversation_llm_enabled": True,
        "strict_privacy_mode": False,
    })
    assert config["conversation_llm_enabled"] is False


def test_conversation_llm_on_when_both_set():
    config = from_app_settings({
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "strict_privacy_mode": False,
    })
    assert config["llm_enabled"] is True
    assert config["conversation_llm_enabled"] is True


def test_full_row_access_requires_conversation_llm_and_non_strict():
    config = from_app_settings({
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "local_llm_full_row_access": True,
        "strict_privacy_mode": False,
    })
    assert config["local_llm_full_row_access"] is True

    strict_config = from_app_settings({
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "local_llm_full_row_access": True,
        "strict_privacy_mode": True,
    })
    assert strict_config["local_llm_full_row_access"] is False


# ---- dispatcher: narrator gets a name-safe row sample ----------------------


class _Loaded:
    def __init__(self, sheets):
        self.sheets = sheets
        self.file_name = "synthetic_students.xlsx"


def test_dispatcher_conversation_payload_carries_name_safe_rows(sheets, columns, monkeypatch):
    captured: dict = {}

    def fake_converse(**kwargs):
        captured.update(kwargs)
        return "Conversational reply.", None

    # Patch the import inside _run_query.
    import nlp.local_model as lm
    monkeypatch.setattr(lm, "converse_about_result_with_model", fake_converse)

    routing = _route("Show me Accounting students", sheets, columns)
    settings = {
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "strict_privacy_mode": False,
    }
    response = execute_planned_request(routing, _Loaded(sheets), settings,
                                       request_summary="Show me Accounting students")

    # The conversational narrator was called.
    assert response["conversation_llm_used"] is True
    assert response["message"] == "Conversational reply."

    # The verified result still only carries summary fields, not row dicts.
    verified = captured["verified_result"]
    assert set(verified.keys()) <= {"operation", "value", "row_count", "description", "columns"}
    # And the active_context only carries plan-level metadata (no rows).
    active = captured["active_context"]
    assert "filters" in active

    # NEW: the narrator now receives a bounded, name-safe sample of result rows
    # so it can name specific students. It must carry student names...
    row_sample = captured["row_sample"]
    assert isinstance(row_sample, list) and row_sample
    assert any(
        any("name" in key.lower() for key in row)
        for row in row_sample
    ), "expected name columns in the row sample"

    # ...but must still exclude IDs and hidden-by-default contact fields.
    payload_json = json.dumps(captured, default=str)
    for forbidden in ("@", "555-", "Student ID", "ssn"):
        assert forbidden.lower() not in payload_json.lower()
    for row in row_sample:
        assert "Student ID" not in row
        assert "Email" not in row
        assert "Phone" not in row


def test_dispatcher_full_local_row_access_passes_unredacted_rows(sheets, columns, monkeypatch):
    captured: dict = {}

    def fake_converse(**kwargs):
        captured.update(kwargs)
        return "Conversational reply.", None

    import nlp.local_model as lm
    monkeypatch.setattr(lm, "converse_about_result_with_model", fake_converse)

    routing = _route("Show me Accounting students", sheets, columns)
    settings = {
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "local_llm_full_row_access": True,
        "strict_privacy_mode": False,
    }
    response = execute_planned_request(routing, _Loaded(sheets), settings,
                                       request_summary="Show me Accounting students")

    assert response["conversation_llm_used"] is True
    assert captured["row_sample_policy"] == "full_local_rows"
    assert captured["hidden_sensitive_fields"] == []
    row_sample = captured["row_sample"]
    assert isinstance(row_sample, list) and row_sample
    assert {"Student ID", "Email", "Phone"} <= set(row_sample[0])
    assert any("@" in str(row.get("Email", "")) for row in row_sample)
    assert all(row.get("Department") == "Accounting" for row in row_sample)


def test_dispatcher_full_local_row_access_samples_matching_rows_for_count(sheets, columns, monkeypatch):
    captured: dict = {}

    def fake_converse(**kwargs):
        captured.update(kwargs)
        return "Conversational reply.", None

    import nlp.local_model as lm
    monkeypatch.setattr(lm, "converse_about_result_with_model", fake_converse)

    routing = _route("how many seniors are there", sheets, columns)
    settings = {
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "local_llm_full_row_access": True,
        "strict_privacy_mode": False,
    }
    response = execute_planned_request(routing, _Loaded(sheets), settings,
                                       request_summary="how many seniors are there")

    assert response["conversation_llm_used"] is True
    assert response["operation"] == "count_rows"
    row_sample = captured["row_sample"]
    assert isinstance(row_sample, list) and row_sample
    assert {"Student ID", "Email", "Phone"} <= set(row_sample[0])
    assert all(row.get("Year") == "Senior" for row in row_sample)


def test_dispatcher_falls_back_to_deterministic_message_when_llm_disabled(sheets, columns):
    routing = _route("Show me Accounting students", sheets, columns)
    response = execute_planned_request(routing, _Loaded(sheets), settings={
        "use_local_llm": False,
        "conversation_llm_enabled": False,
        "strict_privacy_mode": False,
    }, request_summary="Show me Accounting students")
    assert response["conversation_llm_used"] is False
    assert response["narration"].startswith("I understood this as")
    # The deterministic message leads with narration.
    assert response["message"].startswith("I understood this as")


def test_dispatcher_strict_privacy_skips_conversation_llm(sheets, columns, monkeypatch):
    called = {"n": 0}

    def fake_converse(**kwargs):
        called["n"] += 1
        return "Should not be called.", None

    import nlp.local_model as lm
    monkeypatch.setattr(lm, "converse_about_result_with_model", fake_converse)

    routing = _route("Show me Accounting students", sheets, columns)
    response = execute_planned_request(routing, _Loaded(sheets), settings={
        "use_local_llm": True,
        "conversation_llm_enabled": True,
        "strict_privacy_mode": True,
    }, request_summary="Show me Accounting students")

    assert called["n"] == 0
    assert response["conversation_llm_used"] is False


# ---- narrator rejects meta/JSON descriptions -------------------------------


def test_narrator_rejects_meta_json_description(monkeypatch):
    """A small model that describes the prompt JSON instead of answering must be
    dropped (returns None) so the dispatcher uses the deterministic phrasing."""
    import nlp.local_model as lm

    meta = ("The provided JSON data appears to be a response from an API. "
            "data: An array of objects, each representing a student. "
            "active_context: metadata about the current context.")
    monkeypatch.setattr(lm, "_call_ollama", lambda *a, **k: (meta, None))
    reply, err = lm.converse_about_result_with_model(
        user_question="Top 10 students by GPA",
        understood_plan="Top 10 by GPA",
        verified_result={"operation": "filtered_preview", "row_count": 10},
        model_name="test-model",
        row_sample=[{"Name": "Ada", "GPA": 3.98}],
    )
    assert reply is None
    assert err and "meta" in err.lower()


def test_narrator_keeps_real_answer(monkeypatch):
    import nlp.local_model as lm

    answer = ("Here are the top students by GPA: Ada Lovelace (3.98) and "
              "Alan Turing (3.95), among the 10 shown on the right.")
    monkeypatch.setattr(lm, "_call_ollama", lambda *a, **k: (answer, None))
    reply, err = lm.converse_about_result_with_model(
        user_question="Top 10 students by GPA",
        understood_plan="Top 10 by GPA",
        verified_result={"operation": "filtered_preview", "row_count": 10},
        model_name="test-model",
        row_sample=[{"Name": "Ada", "GPA": 3.98}],
    )
    assert reply == answer
    assert err is None


def test_narrator_rejects_count_that_conflicts_with_verified_result(monkeypatch):
    import nlp.local_model as lm

    answer = (
        "I filtered to Health Administration and found 11 students. "
        "The filtered list is shown on the right."
    )
    monkeypatch.setattr(lm, "_call_ollama", lambda *a, **k: (answer, None))
    reply, err = lm.converse_about_result_with_model(
        user_question="show me all students in Health Administration",
        understood_plan="Major = Health Administration",
        verified_result={"operation": "filtered_preview", "row_count": 25},
        model_name="test-model",
        row_sample=[{"Name": "Hannah Thompson", "Major": "Health Administration"}],
    )
    assert reply is None
    assert err and "verified workbook counts" in err


def test_narrator_rejects_sample_size_as_answer_count(monkeypatch):
    import nlp.local_model as lm

    answer = "I found 40 seniors in the sheet."
    monkeypatch.setattr(lm, "_call_ollama", lambda *a, **k: (answer, None))
    reply, err = lm.converse_about_result_with_model(
        user_question="how many seniors are there",
        understood_plan="Year = Senior",
        verified_result={"operation": "count_rows", "row_count": 84, "value": 84},
        model_name="test-model",
        row_sample=[{"Name": f"Student {i}", "Year": "Senior"} for i in range(40)],
    )
    assert reply is None
    assert err and "verified workbook counts" in err


def test_narrator_rejects_unverified_group_count(monkeypatch):
    import nlp.local_model as lm

    answer = (
        "Here are the top Advisors by Count: Dr. Nadia Pierce (10), "
        "Prof. Omar Sloan (10), and Dr. Victor Ford (5). "
        "There are 3 more Advisors with lower counts."
    )
    monkeypatch.setattr(lm, "_call_ollama", lambda *a, **k: (answer, None))
    reply, err = lm.converse_about_result_with_model(
        user_question="Group these by Advisor",
        understood_plan="Using Major = Health Administration, group by Advisor",
        verified_result={"operation": "groupby_count", "row_count": 25},
        model_name="test-model",
        row_sample=[
            {"Advisor": "Dr. Nadia Pierce", "Count": 10},
            {"Advisor": "Prof. Omar Sloan", "Count": 10},
            {"Advisor": "Dr. Victor Ford", "Count": 5},
        ],
    )
    assert reply is None
    assert err and "verified workbook counts" in err


def test_narrator_rejects_hidden_field_claim_when_none_hidden(monkeypatch):
    import nlp.local_model as lm

    answer = (
        "Here are the seniors: there are 84 of them. "
        "Those fields that were hidden by default stayed hidden."
    )
    monkeypatch.setattr(lm, "_call_ollama", lambda *a, **k: (answer, None))
    reply, err = lm.converse_about_result_with_model(
        user_question="how many seniors are there",
        understood_plan="Year = Senior",
        verified_result={"operation": "count_rows", "row_count": 84, "value": 84},
        model_name="test-model",
        hidden_sensitive_fields=[],
        row_sample=[],
    )
    assert reply is None
    assert err and "hidden fields" in err


# ---- L.8: strict privacy preserves sanitized logging ------------------------


def test_strict_privacy_still_allows_interaction_logging(tmp_path):
    """Per L.8 (Option A): strict privacy disables LLM calls but interaction
    logging remains because the log is sanitized at write time."""
    from core.interaction_logger import log_interaction, read_records

    log_path = tmp_path / "log.jsonl"
    routing = {
        "plan_source": "rules", "intent": "query", "confidence": 0.9, "band": "high",
        "validation": {"status": "passed", "errors": []},
        "plan": {"operation": "filtered_preview", "sheet": "Students", "filters": []},
    }
    entry_id = log_interaction(
        user_message="Show me Accounting students",
        routing=routing,
        response={"response_type": "answer", "row_count": 5, "columns": []},
        session_id="s1",
        sheet_columns={"Students": ["Department"]},
        enabled=True,  # logging stays on under strict privacy
        path=log_path,
    )
    assert entry_id
    assert len(read_records(log_path)) == 1


def test_logging_can_be_disabled_independently(tmp_path):
    from core.interaction_logger import log_interaction, read_records

    log_path = tmp_path / "log.jsonl"
    entry_id = log_interaction(
        user_message="anything",
        routing={"plan_source": "rules", "intent": "query", "confidence": 0.9,
                 "validation": {"status": "passed", "errors": []},
                 "plan": {"operation": "filtered_preview"}},
        response=None,
        session_id="s1",
        enabled=False,
        path=log_path,
    )
    assert entry_id is None
    assert read_records(log_path) == []


# ---- L.10 closing: low-confidence path still clarifies ----------------------


def test_low_confidence_message_clarifies(sheets, columns):
    """A message that matches no rule and no LLM produces a clarification."""
    routing = _route("flarble glarble whatever", sheets, columns)
    assert routing["intent"] == "clarify"
    assert routing["band"] in {"low", "high"}  # band may not apply to non-query intents
