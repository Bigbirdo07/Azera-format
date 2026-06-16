"""L.5 + L.6 + L.7: sanitized interaction log, correction capture, rule mining."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.interaction_logger import (
    DEFAULT_LOG_PATH,
    build_record,
    is_correction_message,
    log_interaction,
    read_records,
    sanitize_filters,
    sanitize_user_message,
    sanitize_validated_plan,
    workbook_schema_hash,
)
from scripts.analyze_interaction_logs import render_report, summarize


# ---- sanitization unit tests -----------------------------------------------


def test_email_in_message_is_redacted():
    sanitized, redactions = sanitize_user_message("contact maria at maria@example.edu please")
    assert "@example.edu" not in sanitized
    assert "[REDACTED:email address]" in sanitized
    assert redactions


def test_phone_in_message_is_redacted():
    sanitized, redactions = sanitize_user_message("call 555-123-4567 about this student")
    assert "555-123-4567" not in sanitized
    assert redactions


def test_long_id_number_in_message_is_redacted():
    sanitized, redactions = sanitize_user_message("show student 1234567890")
    assert "1234567890" not in sanitized
    assert redactions


def test_named_person_pattern_is_redacted():
    sanitized, redactions = sanitize_user_message("the student named Jane Doe needs follow up")
    assert "Jane Doe" not in sanitized
    assert any("name" in reason.lower() for reason in redactions)


def test_specific_query_has_no_redactions():
    sanitized, redactions = sanitize_user_message("Show me Accounting students below 2.5 GPA")
    assert sanitized == "Show me Accounting students below 2.5 GPA"
    assert redactions == []


def test_sanitize_filters_redacts_sensitive_columns():
    filters = [
        {"column": "Email", "operator": "equals", "value": "maria@example.edu"},
        {"column": "Department", "operator": "equals", "value": "Accounting"},
        {"column": "GPA", "operator": "less_than", "value": 2.5},
    ]
    safe = sanitize_filters(filters)
    email_filter = next(f for f in safe if f["column"] == "Email")
    dept_filter = next(f for f in safe if f["column"] == "Department")
    gpa_filter = next(f for f in safe if f["column"] == "GPA")
    assert email_filter["value"] == "[REDACTED]"
    assert dept_filter["value"] == "Accounting"  # non-sensitive category preserved
    assert gpa_filter["value"] == 2.5            # numeric preserved


def test_sanitize_validated_plan_strips_extras():
    plan = {
        "operation": "filtered_preview",
        "sheet": "Students",
        "filters": [{"column": "Department", "operator": "equals", "value": "Accounting"}],
        "group_by": "Year",
        "sort": {"column": "GPA", "direction": "descending"},
        "limit": 10,
        "plain_english_question": "should be dropped",  # not in sanitized output
    }
    safe = sanitize_validated_plan(plan)
    assert safe["operation"] == "filtered_preview"
    assert "plain_english_question" not in safe
    assert safe["filters"][0]["value"] == "Accounting"


def test_workbook_schema_hash_is_stable():
    a = workbook_schema_hash({"Students": ["GPA", "Department"]})
    b = workbook_schema_hash({"Students": ["Department", "GPA"]})
    assert a == b
    c = workbook_schema_hash({"Students": ["GPA", "Department", "Email"]})
    assert a != c


# ---- record construction ---------------------------------------------------


def test_build_record_excludes_row_data():
    routing = {
        "plan_source": "rules",
        "intent": "query",
        "confidence": 0.7,
        "band": "medium",
        "assumption_note": "I interpreted this as: filter to Accounting students.",
        "alternatives": ["Use GPA below 2.0"],
        "suggestions": ["Group these by Advisor"],
        "validation": {"status": "passed", "errors": []},
        "plan": {
            "operation": "filtered_preview",
            "sheet": "Students",
            "filters": [{"column": "Department", "operator": "equals", "value": "Accounting"}],
        },
    }
    response = {
        "response_type": "answer",
        "row_count": 14,
        "columns": ["Department", "GPA", "Advisor"],
        # Deliberately include row-shaped data to confirm it does NOT leak.
        "result_preview": [{"Department": "Accounting", "GPA": 2.1, "Advisor": "Smith"}],
        "removed": ["Email"],
    }
    record = build_record(user_message="Show me Accounting students",
                          routing=routing, response=response,
                          session_id="session-1", workbook_schema_hash_value="abc")
    # Result shape only carries counts, not rows.
    assert record["result_shape"] == {"rows": 14, "columns": 3}
    # Serialized record must not contain any row-content tokens.
    blob = json.dumps(record, default=str)
    assert "Smith" not in blob
    assert "result_preview" not in blob


# ---- correction detection (L.6) --------------------------------------------


def test_is_correction_detects_explicit_correction():
    assert is_correction_message("no, I mean students on probation")
    assert is_correction_message("Actually, I want only seniors")
    assert is_correction_message("I meant GPA below 2.0")
    assert is_correction_message("let me clarify")


def test_is_correction_ignores_normal_queries():
    assert not is_correction_message("Show me Accounting students")
    assert not is_correction_message("Group them by advisor")
    assert not is_correction_message("How many students are there")


# ---- end-to-end log write + read -------------------------------------------


def test_log_interaction_writes_sanitized_record(tmp_path):
    log_path = tmp_path / "log.jsonl"
    routing = {
        "plan_source": "rules", "intent": "query", "confidence": 0.9, "band": "high",
        "validation": {"status": "passed", "errors": []},
        "plan": {"operation": "filtered_preview", "sheet": "Students",
                 "filters": [{"column": "Email", "operator": "equals", "value": "maria@example.edu"}]},
    }
    response = {"response_type": "answer", "row_count": 1, "columns": ["Department"],
                "result_preview": [{"Department": "Biology"}], "removed": []}
    entry_id = log_interaction(
        user_message="lookup maria@example.edu",
        routing=routing, response=response,
        session_id="s1", sheet_columns={"Students": ["Email", "Department"]},
        path=log_path,
    )
    assert entry_id

    records = read_records(log_path)
    assert len(records) == 1
    record = records[0]
    # Email in message is redacted.
    assert "maria@example.edu" not in record["user_message"]
    # Email in filter value is redacted.
    assert record["validated_plan"]["filters"][0]["value"] == "[REDACTED]"
    # Row content does not leak.
    blob = json.dumps(record)
    assert "Biology" not in blob
    # PII redactions flag the record as unsafe for mining.
    assert record["safe_for_rule_mining"] is False


def test_correction_link_recorded(tmp_path):
    log_path = tmp_path / "log.jsonl"
    first = log_interaction(
        user_message="who is struggling",
        routing={"plan_source": "rules", "intent": "query", "confidence": 0.65,
                 "band": "medium",
                 "assumption_note": "I interpreted this as: GPA below 2.5",
                 "validation": {"status": "passed", "errors": []},
                 "plan": {"operation": "filtered_preview", "sheet": "Students",
                          "filters": [{"column": "GPA", "operator": "less_than", "value": 2.5}]}},
        response={"response_type": "answer", "row_count": 32, "columns": ["GPA"]},
        session_id="s1", path=log_path,
    )
    second = log_interaction(
        user_message="no I mean students on probation",
        routing={"plan_source": "rules", "intent": "query", "confidence": 0.9,
                 "band": "high",
                 "validation": {"status": "passed", "errors": []},
                 "plan": {"operation": "filtered_preview", "sheet": "Students",
                          "filters": [{"column": "Academic Status", "operator": "equals", "value": "Probation"}]}},
        response={"response_type": "answer", "row_count": 7, "columns": ["Academic Status"]},
        session_id="s1",
        corrects_entry_id=first,
        correction_message="no I mean students on probation",
        path=log_path,
    )
    assert second
    records = read_records(log_path)
    assert len(records) == 2
    correction = records[1]
    assert correction["user_corrected"] is True
    assert correction["corrects_entry_id"] == first


# ---- rule-mining script (L.7) ----------------------------------------------


def _make_records(*entries: dict) -> list[dict]:
    return list(entries)


def test_summary_counts_intents_and_bands():
    records = _make_records(
        {"intent": "query", "plan_source": "rules", "band": "high",
         "normalized_message": "show accounting students",
         "validated_plan": {"operation": "filtered_preview", "filters": []},
         "safe_for_rule_mining": True},
        {"intent": "clarify", "plan_source": "clarification", "band": "low",
         "normalized_message": "fix it", "safe_for_rule_mining": True},
        {"intent": "query", "plan_source": "llm", "band": "medium",
         "normalized_message": "who is struggling",
         "assumption_used": "GPA below 2.5",
         "validated_plan": {"operation": "filtered_preview", "filters": []},
         "safe_for_rule_mining": True},
    )
    summary = summarize(records)
    assert summary["total"] == 3
    assert summary["by_intent"]["query"] == 2
    assert summary["by_band"]["medium"] == 1
    assert summary["repeated_assumptions"]["GPA below 2.5"] == 1


def test_rule_candidates_require_repeats():
    records = _make_records(
        {"intent": "query", "plan_source": "rules", "band": "medium",
         "normalized_message": "who is struggling",
         "validated_plan": {"operation": "filtered_preview", "filters": [{"column": "GPA"}]},
         "safe_for_rule_mining": True},
        {"intent": "query", "plan_source": "rules", "band": "medium",
         "normalized_message": "who is struggling",
         "validated_plan": {"operation": "filtered_preview", "filters": [{"column": "GPA"}]},
         "safe_for_rule_mining": True},
        # A one-off doesn't become a rule candidate.
        {"intent": "query", "plan_source": "rules", "band": "medium",
         "normalized_message": "one off question",
         "validated_plan": {"operation": "count_rows", "filters": []},
         "safe_for_rule_mining": True},
    )
    summary = summarize(records)
    messages = {message for message, _ in summary["rule_candidates"]}
    assert "who is struggling" in messages
    assert "one off question" not in messages


def test_render_report_includes_sections(tmp_path):
    summary = summarize([
        {"intent": "query", "plan_source": "rules", "band": "medium",
         "normalized_message": "who is struggling",
         "assumption_used": "GPA below 2.5",
         "validated_plan": {"operation": "filtered_preview",
                            "filters": [{"column": "GPA"}]},
         "safe_for_rule_mining": True},
    ])
    report = render_report(summary)
    assert "# Interaction Learning Report" in report
    assert "## Candidate deterministic rules" in report
    assert "## Corrections" in report
