"""Phase G: planner router, JSON repair, and LLM-plan safety (LLM is mocked)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from nlp.planner_router import OllamaUnavailable, clean_json_text, plan_user_request
from scripts.eval_planner import load_cases, run_eval


def _route(message, sheets, sheet_columns, *, enabled=False, llm_call=None, state=None, settings=None):
    merged_settings = {"llm_enabled": enabled}
    if settings:
        merged_settings.update(settings)
    return plan_user_request(
        user_message=message,
        sheets=sheets,
        sheet_columns=sheet_columns,
        selected_sheet="Students",
        conversation_state=state,
        settings=merged_settings,
        llm_call=llm_call,
    )


def _mock(plan: dict):
    return lambda prompt: json.dumps(plan)


def _mock_dean_sheets():
    frame = pd.DataFrame(
        {
            "Student ID": ["S1", "S2", "S3", "S4", "S5"],
            "Name": ["A", "B", "C", "D", "E"],
            "Year": ["Senior", "Junior", "Freshman", "Senior", "Sophomore"],
            "Discipline": ["Engineering", "Business", "Education", "Engineering", "Arts and Sciences"],
            "Standing": ["Good Standing", "Bad Standing", "Good Standing", "Bad Standing", "Good Standing"],
            "Location": ["Main Campus", "Online", "Main Campus", "Health Campus", "North Campus"],
            "Advisor": ["Dr. A", "Dr. A", "Dr. B", "Dr. B", "Dr. C"],
            "Major": ["Computer Engineering", "Data Analytics", "Education", "Computer Engineering", "Data Analytics"],
            "Second Major": ["Spanish", None, "Chemistry", "", "Mathematics"],
            "GPA": [3.4, 2.1, 3.0, 1.8, 3.7],
            "Attendance Rate": [94.0, 88.0, 97.0, 82.0, 91.0],
            "Days Absent": [2, 9, 1, 12, 4],
            "SAT Total": [1240, 980, 1120, 1010, 1300],
            "PSAT Total": [1160, 900, 1080, 940, 1210],
            "Attendance Category": [
                "Great Attendance",
                "Needs Attendance Support",
                "Great Attendance",
                "Needs Attendance Support",
                "Great Attendance",
            ],
        }
    )
    return {"Students": frame}, list(frame.columns)


def _mock_nadia_sheets():
    frame = pd.DataFrame(
        {
            "Student ID": ["S1", "S2", "S3", "S4"],
            "Name": ["A", "B", "C", "D"],
            "Year": ["Senior", "Junior", "Senior", "Freshman"],
            "Standing": ["Good Standing", "Bad Standing", "Good Standing", "Bad Standing"],
            "Advisor": ["Dr. Nadia Pierce", "Dr. Nadia Pierce", "Dr. Victor Ford", "Dr. Nadia Pierce"],
            "Major": ["Health Administration", "Nursing", "Marketing", "Health Administration"],
            "Location": ["North Campus", "Main Campus", "Online", "North Campus"],
            "GPA": [3.2, 2.4, 3.7, 2.8],
            "Risk Reason": ["", "GPA below threshold", "", "Bad standing"],
        }
    )
    return {"Students": frame}, list(frame.columns)


def _mock_comparison_sheets():
    frame = pd.DataFrame(
        {
            "Student ID": ["S1", "S2", "S3", "S4"],
            "Name": ["A", "B", "C", "D"],
            "Discipline": ["Nursing", "Nursing", "Business", "Education"],
            "Major": ["Nursing", "Health Administration", "Marketing", "Nursing"],
            "Advisor": ["Dr. Nadia Pierce", "Dr. Nadia Pierce", "Prof. Omar Sloan", "Prof. Omar Sloan"],
            "Standing": ["Good Standing", "Bad Standing", "Good Standing", "Good Standing"],
            "GPA": [3.2, 2.4, 3.7, 2.9],
            "Attendance Rate": [0.95, 0.84, 0.98, 0.9],
        }
    )
    return {"Students": frame}, list(frame.columns)


# 1-4: routing -----------------------------------------------------------------


def test_rules_used_when_confident(sheets, columns):
    r = _route("Show me Accounting students", sheets, {"Students": columns})
    assert r["plan_source"] == "rules" and r["llm_used"] is False


def test_llm_used_when_rules_low_and_enabled(sheets, columns):
    plan = {"intent": "query", "operation": "filter",
            "filters": [{"column": "GPA", "operator": "less_than", "value": 2.5}], "confidence": 0.6}
    # Message is a genuinely open-ended read (not a specialized intent like
    # advisor-attention/intervention, which the rules engine now answers itself).
    r = _route("which students have an unusual profile", sheets, {"Students": columns},
               enabled=True, llm_call=_mock(plan))
    assert r["plan_source"] == "llm" and r["llm_used"] is True


def test_llm_is_primary_when_enabled_even_for_confident_rules(sheets, columns):
    # LLM-first: once the local model is enabled it plans every (non-ranking)
    # question, even ones the rules engine would answer confidently. There is no
    # confidence gate and no force toggle.
    plan = {
        "intent": "query", "operation": "count",
        "filters": [{"column": "Department", "operator": "equals", "value": "Accounting"}],
        "confidence": 0.8,
    }
    r = _route(
        "Show me Accounting students",
        sheets,
        {"Students": columns},
        state={},
        enabled=True,
        llm_call=_mock(plan),
    )
    assert r["plan_source"] == "llm" and r["llm_used"] is True


def test_llm_repairs_missing_value_filter_for_distinct_count(sheets, columns):
    # "how many majors" is not a ranking query, so the model plans it; the
    # repair pass fixes the malformed missing-value filter into count_unique.
    bad_llm_plan = {
        "intent": "query", "operation": "filter",
        "filters": [{"column": "Major", "operator": "equals"}],
        "confidence": 0.6,
    }
    r = _route(
        "how many majors are there?",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    assert r["plan_source"] == "llm" and r["llm_used"] is True
    assert r["validation"]["status"] == "passed"
    assert r["plan"]["operation"] == "count_unique"
    assert r["plan"]["value_column"] == "Major"
    assert r["plan"]["filters"] == []


def test_llm_repairs_attendance_average_by_group():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "aggregate",
        "group_by": "Advisor",
        "filters": [],
        "confidence": 0.68,
    }
    r = _route(
        "average Attendance Rate by advisor",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    p = r["plan"]
    assert r["plan_source"] == "llm"
    assert r["validation"]["status"] == "passed"
    assert p["operation"] == "groupby_average"
    assert p["group_by"] == "Advisor"
    assert p["value_column"] == "Attendance Rate"


def test_llm_repairs_assessment_average_rank_by_group():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "aggregate",
        "group_by": "Major",
        "filters": [],
        "sort": {"column": "Count", "direction": "desc"},
        "confidence": 0.7,
    }
    r = _route(
        "which major has the highest average SAT Total",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    p = r["plan"]
    assert r["plan_source"] == "llm"
    assert r["validation"]["status"] == "passed"
    assert p["operation"] == "groupby_average"
    assert p["group_by"] == "Major"
    assert p["value_column"] == "SAT Total"
    assert p["sort"] == {"column": "SAT Total", "direction": "desc"}
    assert p["limit"] == 1


def test_llm_repairs_missing_metric_filter_for_average_by_group():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "filter",
        "group_by": "Year",
        "filters": [{"column": "Days Absent", "operator": "equals"}],
        "confidence": 0.62,
    }
    r = _route(
        "average Days Absent by year",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    p = r["plan"]
    assert r["plan_source"] == "llm"
    assert r["validation"]["status"] == "passed"
    assert p["operation"] == "groupby_average"
    assert p["group_by"] == "Year"
    assert p["value_column"] == "Days Absent"
    assert p["filters"] == []


def test_llm_repairs_missing_attendance_support_filter():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "aggregate",
        "group_by": "Advisor",
        "filters": [],
        "confidence": 0.7,
    }
    r = _route(
        "which advisors have the most students needing attendance support",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    p = r["plan"]
    assert p["operation"] == "groupby_count"
    assert p["group_by"] == "Advisor"
    assert {"column": "Attendance Category", "operator": "equals", "value": "Needs Attendance Support"} in p["filters"]


def test_llm_repairs_attendance_calendar_days_to_attendance_rate():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [
            {"column": "SAT Total", "operator": "less_than", "value": 1000},
            {"column": "Attendance Calendar Days", "operator": "less_than", "value": 90},
        ],
        "confidence": 0.7,
    }
    # Add the misleading column the live model selected so validation can pass
    # before the repair corrects the semantic target.
    sheets["Students"]["Attendance Calendar Days"] = [90, 90, 90, 90, 90]
    columns = list(sheets["Students"].columns)
    r = _route(
        "show students with SAT Total below 1000 and attendance below 90%",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    p = r["plan"]
    assert {"column": "Attendance Rate", "operator": "less_than", "value": 0.9} in p["filters"]
    assert not any(f.get("column") == "Attendance Calendar Days" for f in p["filters"])


def test_rules_attendance_below_keeps_standing_filter_and_rate_filter():
    sheets, columns = _mock_dean_sheets()
    r = _route(
        "how many students are in good standing but attendance below 90%",
        sheets,
        {"Students": columns},
    )
    p = r["plan"]
    assert p["operation"] == "count_rows"
    assert {"column": "Standing", "operator": "equals", "value": "Good Standing"} in p["filters"]
    assert any(f.get("column") == "Attendance Rate" and f.get("operator") == "less_than" for f in p["filters"])


def test_represented_majors_among_bad_standing_lists_unique_values():
    sheets, columns = _mock_dean_sheets()
    r = _route(
        "what majors are represented among bad standing students",
        sheets,
        {"Students": columns},
    )
    p = r["plan"]
    assert p["operation"] == "list_unique"
    assert p["value_column"] == "Major"
    assert {"column": "Standing", "operator": "equals", "value": "Bad Standing"} in p["filters"]


def test_llm_repairs_show_nursing_majors_to_filtered_preview():
    sheets, columns = _mock_dean_sheets()
    sheets["Students"].loc[len(sheets["Students"])] = {
        "Student ID": "S6",
        "Name": "F",
        "Year": "Junior",
        "Discipline": "Health Sciences",
        "Standing": "Bad Standing",
        "Location": "Main Campus",
        "Advisor": "Dr. D",
        "Major": "Nursing",
        "Second Major": "",
        "GPA": 2.2,
        "Attendance Rate": 89.0,
        "Days Absent": 8,
        "SAT Total": 990,
        "PSAT Total": 910,
        "Attendance Category": "Needs Attendance Support",
    }
    columns = list(sheets["Students"].columns)
    bad_llm_plan = {
        "intent": "query",
        "operation": "count_unique",
        "value_column": "Major",
        "filters": [
            {"column": "Major", "operator": "equals", "value": "Nursing"},
            {"column": "Standing", "operator": "equals", "value": "Bad Standing"},
        ],
        "confidence": 0.7,
    }
    r = _route(
        "show nursing majors with bad standing",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    p = r["plan"]
    assert p["operation"] == "filtered_preview"
    assert p["value_column"] == ""


def test_llm_repairs_value_on_wrong_categorical_column(sheets, columns):
    bad_llm_plan = {
        "intent": "query",
        "operation": "count",
        "filters": [{"column": "Academic Status", "operator": "equals", "value": "Senior"}],
        "confidence": 0.6,
    }
    r = _route(
        "how many seniors are there",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    assert r["plan_source"] == "llm"
    assert r["llm_used"] is True
    assert r.get("fallback_reason") is None
    assert r["validation"]["status"] == "passed"
    assert r["plan"]["operation"] == "count_rows"
    assert r["plan"]["filters"] == [
        {"column": "Year", "operator": "equals", "value": "Senior"}
    ]


def test_valid_but_weaker_llm_plan_is_semantically_repaired_by_rules(sheets, columns):
    bad_llm_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Year", "operator": "equals", "value": "Senior"}],
        "confidence": 0.8,
    }
    r = _route(
        "how many seniors are there",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    assert r["plan_source"] == "llm"
    assert r["llm_used"] is True
    assert r.get("semantic_repaired") is True
    assert r["plan"]["operation"] == "count_rows"
    assert r["plan"]["filters"] == [
        {"column": "Year", "operator": "equals", "value": "Senior"}
    ]


def test_valid_llm_plan_missing_filter_is_semantically_repaired():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Standing", "operator": "equals", "value": "Bad Standing"}],
        "confidence": 0.8,
    }
    r = _route(
        "show freshmen with bad standing",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    assert r["plan_source"] == "llm"
    assert r.get("semantic_repaired") is True
    assert {"column": "Year", "operator": "equals", "value": "Freshman"} in r["plan"]["filters"]
    assert {"column": "Standing", "operator": "equals", "value": "Bad Standing"} in r["plan"]["filters"]


def test_summary_about_partial_advisor_name_routes_to_cohort_summary():
    sheets, columns = _mock_nadia_sheets()
    r = _route(
        "what summary can you say about nadia students?",
        sheets,
        {"Students": columns},
    )
    assert r["plan_source"] == "rules"
    assert r["plan"]["operation"] == "cohort_summary"
    assert r["plan"]["filters"] == [
        {"column": "Advisor", "operator": "equals", "value": "Dr. Nadia Pierce"}
    ]


def test_summary_word_does_not_trigger_risk_reason_text_search():
    sheets, columns = _mock_nadia_sheets()
    r = _route(
        "what summary can you say about nadia students?",
        sheets,
        {"Students": columns},
    )
    assert not any(
        f.get("column") == "Risk Reason" and f.get("operator") == "contains_text"
        for f in r["plan"]["filters"]
    )


def test_compare_two_advisors_routes_to_cohort_comparison():
    sheets, columns = _mock_comparison_sheets()
    r = _route(
        "compare Dr. Nadia Pierce and Prof. Omar Sloan's students",
        sheets,
        {"Students": columns},
    )
    assert r["intent"] == "query"
    assert r["plan"]["operation"] == "cohort_comparison"
    assert r["plan"]["group_by"] == "Advisor"
    assert r["plan"]["filters"] == [
        {
            "column": "Advisor",
            "operator": "in",
            "value": ["Dr. Nadia Pierce", "Prof. Omar Sloan"],
        }
    ]


def test_compare_two_named_advisors_excludes_shared_first_name():
    # Regression: "compare Victor Ford and Marta Stone" must not pull in
    # "Prof. Victor Chen" just because it shares the first name "Victor".
    frame = pd.DataFrame(
        {
            "Student ID": ["S1", "S2", "S3"],
            "Name": ["A", "B", "C"],
            "Advisor": ["Dr. Victor Ford", "Dr. Mara Stone", "Prof. Victor Chen"],
            "GPA": [3.2, 2.4, 3.7],
        }
    )
    sheets = {"Students": frame}
    r = _route(
        "compare Victor Ford and Marta Stone",
        sheets,
        {"Students": list(frame.columns)},
    )
    assert r["plan"]["operation"] == "cohort_comparison"
    assert r["plan"]["filters"] == [
        {
            "column": "Advisor",
            "operator": "in",
            "value": ["Dr. Victor Ford", "Dr. Mara Stone"],
        }
    ]


def test_compare_two_category_values_routes_to_cohort_comparison():
    # Caught live: "compare Good Standing vs Bad Standing students" was
    # falling through to a plain count_rows filtered to just "Bad Standing"
    # -- half the comparison silently dropped. _detect_cohort_comparison only
    # ever matched two ADVISOR names; this generalizes it to any column whose
    # values match both phrases.
    sheets, columns = _mock_comparison_sheets()
    r = _route("Compare Good Standing vs Bad Standing students", sheets, {"Students": columns})
    assert r["plan"]["operation"] == "cohort_comparison"
    assert r["plan"]["group_by"] == "Standing"
    assert r["plan"]["filters"] == [
        {"column": "Standing", "operator": "in", "value": ["Good Standing", "Bad Standing"]}
    ]


def test_missing_sat_scores_resolves_via_concept_not_literal_header():
    # Caught live: "missing SAT scores" declined ("local model disabled")
    # because the blank-value filter detector only matched the literal
    # column header text ("SAT Total"), not the generic phrase a user
    # actually types. Falls back to concept/synonym resolution.
    sheets, columns = _mock_dean_sheets()
    r = _route("Which students are missing SAT scores?", sheets, {"Students": columns})
    assert r["intent"] == "query"
    assert r["plan"]["filters"] == [{"column": "SAT Total", "operator": "is_missing"}]


def test_ambiguous_shared_value_asks_for_clarification_options():
    sheets, columns = _mock_comparison_sheets()
    r = _route("show Nursing students", sheets, {"Students": columns})
    assert r["intent"] == "clarify"
    assert "Nursing" in r["confirmation_reason"]
    assert "Show students where Discipline is Nursing" in r["clarify_options"]
    assert "Show students majoring in Nursing" in r["clarify_options"]

    grouped = _route("group Nursing students by year", sheets, {"Students": columns})
    assert grouped["intent"] == "clarify"
    assert "Group students where Discipline is Nursing by year" in grouped["clarify_options"]
    assert "Group students majoring in Nursing by year" in grouped["clarify_options"]

    located = _route("show Nursing students on Main Campus", sheets, {"Students": columns})
    assert located["intent"] == "clarify"
    assert "Show students where Discipline is Nursing on Main Campus" in located["clarify_options"]
    assert "Show students majoring in Nursing on Main Campus" in located["clarify_options"]


def test_groupby_repair_drops_accidental_filter_on_group_column():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "aggregate",
        "group_by": "Year",
        "filters": [{"column": "Year", "operator": "equals", "value": "Freshman"}],
        "confidence": 0.8,
    }
    r = _route(
        "count students by year",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    assert r["plan_source"] == "llm"
    assert r["plan"]["operation"] == "groupby_count"
    assert r["plan"]["group_by"] == "Year"
    assert r["plan"]["filters"] == []


def test_average_question_repair_keeps_filters_and_uses_average_column():
    sheets, columns = _mock_dean_sheets()
    bad_llm_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Year", "operator": "equals", "value": "Freshman"}],
        "confidence": 0.8,
    }
    r = _route(
        "average GPA for freshmen",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    assert r["plan_source"] == "llm"
    assert r["plan"]["operation"] == "average_column"
    assert r["plan"]["value_column"] == "GPA"
    assert {"column": "Year", "operator": "equals", "value": "Freshman"} in r["plan"]["filters"]


def test_validation_repaired_but_wrong_plan_is_semantically_repaired():
    sheets, columns = _mock_dean_sheets()
    calls = {"n": 0}
    bad_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [
            {"column": "Year", "operator": "equals", "value": None},
            {"column": "Discipline", "operator": "equals", "value": "Engineering"},
        ],
        "confidence": 0.0,
    }
    repaired_but_wrong = {
        "intent": "query",
        "operation": "filter",
        "filters": [
            {"column": "Year", "operator": "equals", "value": "Junior"},
            {"column": "Discipline", "operator": "equals", "value": "Engineering"},
        ],
        "confidence": 0.8,
    }

    def llm(prompt):
        calls["n"] += 1
        return json.dumps(bad_plan if calls["n"] == 1 else repaired_but_wrong)

    r = _route(
        "show Main Campus juniors",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=llm,
    )
    assert r.get("semantic_repaired") is True
    assert r.get("validation_repaired") is True
    assert {"column": "Year", "operator": "equals", "value": "Junior"} in r["plan"]["filters"]
    assert {"column": "Location", "operator": "equals", "value": "Main Campus"} in r["plan"]["filters"]
    assert not any(f.get("column") == "Discipline" for f in r["plan"]["filters"])


def test_ranking_performance_query_resolved_by_rules_guardrail(sheets, columns):
    # "highest well performing major" is a ranking query: the rules guardrail
    # resolves it deterministically (groupby average, sort desc, limit 1) and
    # the model's (here deliberately malformed) plan is ignored. This is the
    # guardrail that stops "top 10 by GPA" from being mangled into a filter.
    bad_llm_plan = {
        "intent": "query", "operation": "filter",
        "filters": [{"column": "Major", "operator": "equals"}],
        "confidence": 0.6,
    }
    r = _route(
        "what is the highest well performing major",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    assert r["plan_source"] == "rules" and r["llm_used"] is False
    assert r["plan"]["operation"] == "groupby_average"
    assert r["plan"]["group_by"] == "Major"
    assert r["plan"]["value_column"] == "GPA"
    assert r["plan"]["sort"] == {"column": "GPA", "direction": "desc"}
    assert r["plan"]["limit"] == 1


def test_llm_not_used_when_disabled(sheets, columns):
    r = _route("which students have an unusual profile", sheets, {"Students": columns}, enabled=False)
    assert r["plan_source"] == "clarification" and r["llm_used"] is False


# Ranking guardrail — the "Top 10 by GPA" regression -------------------------


def test_top_n_by_gpa_rules_only(sheets, columns):
    r = _route("Top 10 students by GPA", sheets, {"Students": columns}, enabled=False)
    plan = r["plan"]
    assert plan["operation"] == "filtered_preview"
    assert plan["sort"] == {"column": "GPA", "direction": "desc"}
    assert plan["limit"] == 10
    assert plan["filters"] == []


def test_top_n_by_gpa_guardrail_overrides_bad_llm(sheets, columns):
    # The exact failure that was reported: with the model enabled, a weak model
    # mangles "Top 10 by GPA" into a GPA>=3.0 filter with the default limit 50.
    # The ranking guardrail must override it back to sort desc + limit 10.
    bad_llm_plan = {
        "intent": "query", "operation": "filter",
        "filters": [{"column": "GPA", "operator": "greater_than_or_equal", "value": 3.0}],
        "limit": 50, "confidence": 0.7,
    }
    r = _route("Top 10 students by GPA", sheets, {"Students": columns},
               enabled=True, llm_call=_mock(bad_llm_plan))
    assert r["plan_source"] == "rules" and r["llm_used"] is False
    plan = r["plan"]
    assert plan["sort"] == {"column": "GPA", "direction": "desc"}
    assert plan["limit"] == 10
    assert plan["filters"] == []


def test_bottom_n_ranking_guardrail(sheets, columns):
    # "lowest" is also a ranking cue; a bad model plan must not win.
    bad_llm_plan = {"intent": "query", "operation": "filter",
                    "filters": [{"column": "GPA", "operator": "less_than", "value": 2.0}],
                    "limit": 50, "confidence": 0.7}
    r = _route("bottom 5 students by GPA", sheets, {"Students": columns},
               enabled=True, llm_call=_mock(bad_llm_plan))
    assert r["plan_source"] == "rules" and r["llm_used"] is False
    assert r["plan"]["sort"] == {"column": "GPA", "direction": "asc"}
    assert r["plan"]["limit"] == 5


def test_ollama_unavailable_is_graceful(sheets, columns):
    def boom(prompt):
        raise OllamaUnavailable("connection refused")

    r = _route("which students have an unusual profile", sheets, {"Students": columns}, enabled=True, llm_call=boom)
    assert r["plan_source"] == "clarification"
    assert r["fallback_reason"] and "unavailable" in r["fallback_reason"]


# 5-7: JSON repair -------------------------------------------------------------


def test_valid_llm_plan_passes(sheets, columns):
    plan = {"intent": "query", "operation": "count", "filters": [], "confidence": 0.7}
    r = _route("weird stuff", sheets, {"Students": columns}, enabled=True, llm_call=_mock(plan))
    assert r["plan_source"] == "llm" and r["validation"]["status"] == "passed"


def test_markdown_wrapped_json_is_cleaned():
    raw = "```json\n{\"intent\": \"query\", \"operation\": \"count\"}\n```"
    assert clean_json_text(raw) == '{"intent": "query", "operation": "count"}'


def test_invalid_json_is_repaired(sheets, columns):
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Sure, here is the plan you asked for."  # prose
        return json.dumps({"intent": "query", "operation": "count", "filters": [], "confidence": 0.6})

    r = _route("vague", sheets, {"Students": columns}, enabled=True, llm_call=flaky)
    assert r["plan_source"] == "llm" and calls["n"] == 2


def test_invalid_llm_plan_gets_validation_repair_before_rules_fallback(sheets, columns):
    calls = {"n": 0}
    bad_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Program", "operator": "equals", "value": "Accounting"}],
        "confidence": 0.7,
    }
    repaired_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Department", "operator": "equals", "value": "Accounting"}],
        "confidence": 0.9,
    }

    def repairable(prompt):
        calls["n"] += 1
        return json.dumps(bad_plan if calls["n"] == 1 else repaired_plan)

    r = _route(
        "Show me Accounting students",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=repairable,
    )
    assert calls["n"] == 2
    assert r["plan_source"] == "llm"
    assert r["llm_used"] is True
    assert r.get("validation_repaired") is True
    assert r["plan"]["filters"] == [
        {"column": "Department", "operator": "equals", "value": "Accounting"}
    ]


def test_validation_fallback_carries_llm_validation_errors(sheets, columns):
    bad_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Academic Status", "operator": "equals", "value": "Alien"}],
        "confidence": 0.7,
    }

    r = _route(
        "how many alien students are there",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_plan),
    )
    assert r["plan_source"] == "rules"
    assert r["fallback_reason"] == "LLM plan failed validation; used rules plan"
    assert r.get("llm_validation_errors")
    assert "Alien" in r["llm_validation_errors"][0]


def test_unrecoverable_json_fails_safely(sheets, columns):
    r = _route("vague", sheets, {"Students": columns}, enabled=True, llm_call=lambda p: "no json at all")
    assert r["plan_source"] == "clarification"


def test_prose_response_is_rejected(sheets, columns):
    r = _route("vague", sheets, {"Students": columns}, enabled=True,
               llm_call=lambda p: "I think you should look at GPA.")
    assert r["plan_source"] == "clarification"


# 8-11: safety validation ------------------------------------------------------


def test_nonexistent_column_rejected(sheets, columns):
    plan = {"intent": "query", "operation": "filter",
            "filters": [{"column": "Housing", "operator": "equals", "value": "Dorm"}], "confidence": 0.6}
    r = _route("housing issues", sheets, {"Students": columns}, enabled=True, llm_call=_mock(plan))
    assert r["plan_source"] == "clarification" and r["validation"]["status"] == "failed"


def test_protected_field_update_rejected(sheets, columns):
    r = _route("change their GPA to 4.0", sheets, {"Students": columns})
    assert r["intent"] == "field_update" and r["validation"]["status"] == "failed"


def test_sensitive_export_requires_confirmation(sheets, columns):
    r = _route("export this list with emails", sheets, {"Students": columns})
    assert r["intent"] == "export" and r["requires_confirmation"] is True


def test_action_intents_cannot_bypass_confirmation(sheets, columns):
    for message in ["export this list", "add note: follow up", "set Follow Up Needed to Yes"]:
        r = _route(message, sheets, {"Students": columns})
        assert r["requires_confirmation"] is True
        assert r["plan_source"] == "rules"  # router never executes


def test_excessive_limit_rejected(sheets, columns):
    plan = {"intent": "query", "operation": "filter", "filters": [], "limit": 100000, "confidence": 0.6}
    r = _route("dump everything", sheets, {"Students": columns}, enabled=True, llm_call=_mock(plan))
    assert r["plan_source"] == "clarification" and r["validation"]["status"] == "failed"


# 12-13: explanation + debug ---------------------------------------------------


def test_explanation_prompt_contains_only_summary():
    from nlp.model_prompt import build_explain_prompt

    verified = {"operation": "count_rows", "value": 177, "row_count": 177, "description": "177 students match."}
    prompt = build_explain_prompt(user_question="how many?", verified_result=verified)
    assert "177" in prompt
    assert "@" not in prompt and "Student " not in prompt  # no raw rows


def test_debug_state_shows_plan_source_and_llm_usage():
    from core.schema import build_debug_state

    schema = {"Students": {"columns": ["GPA"], "canonical_map": {}, "column_types": {}, "sensitive": {}, "default_visible": ["GPA"]}}
    state = build_debug_state({}, schema, "Students",
                              routing={"plan_source": "local_llm", "llm_used": True, "validation_status": "passed"})
    assert state["routing"]["plan_source"] == "local_llm"
    assert state["routing"]["llm_used"] is True


# 14-15: eval harness ----------------------------------------------------------


def test_eval_cases_load_and_validate():
    cases = load_cases()
    assert len(cases) >= 25
    for case in cases:
        assert "prompt" in case and "expected_intent" in case


def test_eval_harness_rules_only_passes():
    rows = run_eval(llm_enabled=False)
    failures = [r for r in rows if not r["pass"]]
    assert not failures, failures


# 16: superlative performance queries (lowest/worst/best) ----------------------


def _performance_plan(message, sheets, columns):
    r = _route(message, sheets, {"Students": columns})
    return r.get("plan") or {}


# Advisor performance questions now route to the purpose-built
# `advisor_outcome_summary` (ranked by a composite Outcome Score, top 20). The
# meaningful contract these tests guard is unchanged: grouped by Advisor, and the
# sort DIRECTION matches the superlative (lowest -> asc, best -> desc).


def test_lowest_performing_advisor_sorts_ascending_with_limit(sheets, columns):
    plan = _performance_plan("what advisor has the lowest performing students", sheets, columns)
    assert plan["operation"] == "advisor_outcome_summary"
    assert plan["group_by"] == "Advisor"
    assert plan["sort"] == {"column": "Outcome Score", "direction": "asc"}
    assert plan["limit"] == 20


def test_best_performing_advisor_sorts_descending_with_limit(sheets, columns):
    plan = _performance_plan("who is the best performing advisor", sheets, columns)
    assert plan["operation"] == "advisor_outcome_summary"
    assert plan["group_by"] == "Advisor"
    assert plan["sort"] == {"column": "Outcome Score", "direction": "desc"}
    assert plan["limit"] == 20


def test_teacher_best_gpa_record_falls_back_to_advisor(sheets, columns):
    plan = _performance_plan("what teacher has the best gpa record", sheets, columns)
    assert plan["operation"] == "advisor_outcome_summary"
    assert plan["group_by"] == "Advisor"
    assert plan["sort"] == {"column": "Outcome Score", "direction": "desc"}


def test_professor_best_gpa_falls_back_to_advisor(sheets, columns):
    plan = _performance_plan("what professor has the best gpa", sheets, columns)
    assert plan["operation"] == "advisor_outcome_summary"
    assert plan["group_by"] == "Advisor"
    assert plan["sort"] == {"column": "Outcome Score", "direction": "desc"}


def test_class_best_gpa_uses_year_when_no_course_column(sheets, columns):
    plan = _performance_plan("what class has the best gpa", sheets, columns)
    assert plan["operation"] == "groupby_average"
    assert plan["group_by"] == "Year"
    assert plan["value_column"] == "GPA"
    assert plan["sort"] == {"column": "GPA", "direction": "desc"}
    assert plan["limit"] == 1


def test_possessive_advisors_students_groups_by_advisor_not_student_id(sheets, columns):
    # Caught live: normalize_text turns "advisor's" into "advisor s"
    # (apostrophe -> space), leaving a stray "s" token that displaced
    # "advisor" under from_end truncation and silently grouped by Student ID
    # instead -- "S10046 has the lowest GPA" instead of an advisor name.
    plan = _performance_plan("which advisor's students have the lowest average gpa", sheets, columns)
    assert plan["group_by"] == "Advisor"


def test_unresolved_by_phrase_declines_instead_of_silently_dropping_it(sheets, columns):
    # Caught live: fixing "color" as a hard-edit trigger left this question
    # falling through to a plain full-roster listing ("300 matching rows")
    # with no indication the "favorite color" grouping was never understood.
    # It should decline the same way "housing status" already does.
    r = _route("show me students by their favorite color", sheets, {"Students": columns})
    assert r["intent"] == "clarify"
    assert "favorite color" in r["confirmation_reason"]


def test_by_phrase_that_resolves_still_works_normally(sheets, columns):
    for message in ("show me students by advisor", "list students by year", "top 10 students by GPA"):
        r = _route(message, sheets, {"Students": columns})
        assert r["intent"] != "clarify", message


def test_students_verb_phrase_does_not_false_positive_as_unavailable_field(sheets, columns):
    # Widening the unavailable-field guard to filtered_preview exposed a
    # latent bug: "which students HAVE an unusual profile" was captured by
    # _unavailable_field's "student(s) X" regex as the field name "have an",
    # short-circuiting straight to a high-confidence clarification instead of
    # ever reaching the LLM fallback path this question was meant to exercise.
    for message in (
        "which students have an unusual profile",
        "which students are in Accounting",
        "which students need extra support",
    ):
        r = _route(message, sheets, {"Students": columns})
        if r["intent"] == "clarify":
            assert "have an" not in r["confirmation_reason"], message


def test_worst_gpa_with_category_groups_by_category_not_gpa(sheets, columns):
    """Regression: 'which major has the worst gpa' must group by Major, not GPA.
    The GPA column is the ranking value, never a grouping candidate."""
    plan = _performance_plan("which major has the worst gpa", sheets, columns)
    assert plan["group_by"] == "Major"
    assert plan["value_column"] == "GPA"
    assert plan["sort"]["direction"] == "asc"


def test_highest_well_performing_major_groups_by_major(sheets, columns):
    plan = _performance_plan("what is the highest well performing major", sheets, columns)
    assert plan["operation"] == "groupby_average"
    assert plan["group_by"] == "Major"
    assert plan["value_column"] == "GPA"
    assert plan["sort"] == {"column": "GPA", "direction": "desc"}
    assert plan["limit"] == 1


def test_lower_gpa_per_average_phrasing(sheets, columns):
    """The user's actual follow-up phrasing — verbose but unambiguous."""
    plan = _performance_plan(
        "what advisor has students that has on average a lower gpa per average?",
        sheets, columns,
    )
    assert plan["operation"] == "groupby_average"
    assert plan["group_by"] == "Advisor"
    assert plan["sort"]["direction"] == "asc"
    assert plan["limit"] == 1


def test_highest_gpa_with_department_sorts_descending(sheets, columns):
    plan = _performance_plan("which department has the highest gpa", sheets, columns)
    assert plan["group_by"] == "Department"
    assert plan["sort"]["direction"] == "desc"
    assert plan["limit"] == 1


def test_how_many_students_in_each_department_groups_by_department(sheets, columns):
    plan = _performance_plan("how many students in each department", sheets, columns)
    assert plan["operation"] == "groupby_count"
    assert plan["group_by"] == "Department"


def test_struggling_students_phrasing_picks_ascending(sheets, columns):
    """'struggling students' should be treated as a low-GPA superlative."""
    plan = _performance_plan("which advisor has the most struggling students", sheets, columns)
    assert plan["group_by"] == "Advisor"
    assert plan["sort"]["direction"] == "asc"


# 17: semantic-coherence checks in validate_llm_plan --------------------------


from nlp.planner_router import validate_llm_plan


def _validate(query, sheets, sheet="Students"):
    plan = {"intent": "query"}
    return validate_llm_plan(plan, query, sheets, sheet)


def _query(filters, **extra):
    base = {"sheet": "Students", "operation": "filtered_preview",
            "filters": filters, "group_by": "", "value_column": "", "limit": 10}
    base.update(extra)
    return base


def test_validator_rejects_contains_with_none_value(sheets):
    """The 'Advisor contains None' LLM mistake we saw on llama3.2:3b."""
    result = _validate(_query([{"column": "Advisor", "operator": "contains", "value": None}]), sheets)
    assert not result["ok"]
    assert any("missing a value" in e for e in result["errors"])


def test_validator_rejects_equals_with_empty_string(sheets):
    result = _validate(_query([{"column": "Department", "operator": "equals", "value": "  "}]), sheets)
    assert not result["ok"]


def test_validator_rejects_in_with_non_list(sheets):
    result = _validate(_query([{"column": "Department", "operator": "in", "value": "Accounting"}]), sheets)
    assert not result["ok"]
    assert any("list of values" in e for e in result["errors"])


def test_validator_accepts_in_with_list(sheets):
    result = _validate(
        _query([{"column": "Department", "operator": "in", "value": ["Accounting", "Biology"]}]),
        sheets,
    )
    assert result["ok"], result["errors"]


def test_validator_rejects_between_with_wrong_arity(sheets):
    result = _validate(_query([{"column": "GPA", "operator": "between", "value": [2.0]}]), sheets)
    assert not result["ok"]
    assert any("exactly 2" in e for e in result["errors"])


def test_validator_rejects_between_with_non_numeric_bounds(sheets):
    result = _validate(_query([{"column": "GPA", "operator": "between", "value": ["low", "high"]}]), sheets)
    assert not result["ok"]


def test_validator_rejects_string_op_on_numeric_column(sheets):
    result = _validate(_query([{"column": "GPA", "operator": "contains", "value": "2"}]), sheets)
    assert not result["ok"]
    assert any("doesn't apply to numeric" in e for e in result["errors"])


def test_validator_rejects_numeric_op_with_non_numeric_value(sheets):
    result = _validate(_query([{"column": "GPA", "operator": "less_than", "value": "low"}]), sheets)
    assert not result["ok"]
    assert any("numeric value" in e for e in result["errors"])


def test_validator_rejects_invented_enum_value(sheets):
    """The Academic Status = 'incomplete' mistake from the smoke. 'incomplete'
    isn't in the column's actual domain — must be rejected."""
    result = _validate(
        _query([{"column": "Academic Status", "operator": "equals", "value": "incomplete"}]),
        sheets,
    )
    assert not result["ok"]
    assert any("not a known value" in e for e in result["errors"])
    # The error message should also show the actual known values for review.
    msg = "\n".join(result["errors"])
    assert "Probation" in msg or "Good Standing" in msg or "At Risk" in msg or "Warning" in msg


def test_validator_accepts_known_enum_value(sheets):
    result = _validate(
        _query([{"column": "Academic Status", "operator": "equals", "value": "At Risk"}]),
        sheets,
    )
    assert result["ok"], result["errors"]


def test_validator_enum_check_is_case_insensitive(sheets):
    """'at risk' lowercased should still match 'At Risk' in the domain."""
    result = _validate(
        _query([{"column": "Academic Status", "operator": "equals", "value": "at risk"}]),
        sheets,
    )
    assert result["ok"], result["errors"]


def test_validator_passes_is_missing_without_value(sheets):
    """is_missing / is_not_missing legitimately take no value."""
    result = _validate(_query([{"column": "Notes", "operator": "is_missing"}]), sheets)
    assert result["ok"], result["errors"]


def test_validator_accepts_values_field_synonym(sheets):
    """llama3.2:3b sometimes emits 'values' instead of 'value' — accept it."""
    result = _validate(
        _query([{"column": "Department", "operator": "in",
                 "values": ["Accounting", "Biology"]}]),
        sheets,
    )
    assert result["ok"], result["errors"]


# --- column projection -------------------------------------------------------


def test_rule_parses_just_x_and_y_into_select_columns(sheets, columns):
    r = _route("show me just student name and gpa", sheets, {"Students": columns})
    plan = r["plan"]
    assert plan["operation"] == "filtered_preview"
    assert plan["select_columns"] == ["Name", "GPA"]


def test_rule_parses_columns_of_phrase(sheets, columns):
    # The phrase "students" resolves to "Student ID" (the literal-substring
    # column match beats the synonym path). That's an acceptable student
    # column for the projection; the important thing is the planner emits
    # a select_columns list with the right shape.
    r = _route("show me columns of just students and gpa", sheets, {"Students": columns})
    plan = r["plan"]
    assert plan["operation"] == "filtered_preview"
    assert plan["select_columns"] == ["Student ID", "GPA"]


def test_rule_parses_only_show_phrase(sheets, columns):
    r = _route("only show advisor and department", sheets, {"Students": columns})
    plan = r["plan"]
    assert plan["operation"] == "filtered_preview"
    assert plan["select_columns"] == ["Advisor", "Department"]


def test_rule_does_not_project_on_plain_filter_request(sheets, columns):
    """A 'show students with GPA<2.0' query should NOT pick up a projection."""
    r = _route("show me students with gpa below 2.0", sheets, {"Students": columns})
    plan = r["plan"]
    assert plan.get("select_columns") in (None, [])


def test_projection_carries_through_to_engine(sheets, columns):
    """End-to-end: planner emits select_columns, engine narrows the preview."""
    from core.query_engine import run_query

    r = _route("just student name and gpa", sheets, {"Students": columns})
    plan = r["plan"]
    result = run_query(plan, sheets)
    assert result.table
    assert all(set(row.keys()) <= {"Name", "GPA"} for row in result.table)


# --- top-N / bottom-N -------------------------------------------------------


def test_top_n_by_column(sheets, columns):
    r = _route("top 10 students by GPA", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "filtered_preview"
    assert p["sort"] == {"column": "GPA", "direction": "desc"}
    assert p["limit"] == 10
    assert not p["group_by"]


def test_bottom_n_by_column(sheets, columns):
    r = _route("bottom 5 students by GPA", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "filtered_preview"
    assert p["sort"] == {"column": "GPA", "direction": "asc"}
    assert p["limit"] == 5


def test_n_lowest_column_students(sheets, columns):
    r = _route("10 lowest GPA students", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "filtered_preview"
    assert p["sort"] == {"column": "GPA", "direction": "asc"}
    assert p["limit"] == 10


def test_top_n_does_not_clobber_average(sheets, columns):
    """An 'average X' question is not a top-N row preview."""
    r = _route("average GPA", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "average_column"
    assert p["value_column"] == "GPA"


# --- list_unique parsing ----------------------------------------------------


def test_list_all_x_returns_list_unique(sheets, columns):
    r = _route("list all departments", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "list_unique"
    assert p["value_column"] == "Department"


def test_what_x_do_we_have_returns_list_unique(sheets, columns):
    r = _route("what advisors do we have", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "list_unique"
    assert p["value_column"] == "Advisor"


def test_what_majors_are_listed_returns_list_unique(sheets, columns):
    r = _route("what majors are listed", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "list_unique"
    assert p["value_column"] == "Major"


def test_roster_list_fields_return_list_unique():
    sheets, columns = _mock_dean_sheets()
    sheets["Students"]["Attendance Category"] = [
        "Great Attendance",
        "Good Attendance",
        "Needs Attendance Support",
        "Good Attendance",
        "Great Attendance",
    ]
    columns = list(sheets["Students"].columns)

    checks = {
        "what years are listed": "Year",
        "what locations are listed": "Location",
        "what attendance categories are listed": "Attendance Category",
    }
    for question, expected_column in checks.items():
        r = _route(question, sheets, {"Students": columns})
        assert r["plan"]["operation"] == "list_unique"
        assert r["plan"]["value_column"] == expected_column


def test_how_many_students_are_there_stays_row_count():
    sheets, columns = _mock_dean_sheets()
    r = _route("how many students are there", sheets, {"Students": columns})
    assert r["plan"]["operation"] == "count_rows"
    assert r["plan"]["value_column"] == ""
    assert r["intent"] == "query"


def test_data_quality_summary_routes_to_query():
    sheets, columns = _mock_dean_sheets()
    r = _route("give me a data quality summary", sheets, {"Students": columns})
    assert r["plan"]["operation"] == "data_quality_summary"
    assert r["intent"] == "query"


def test_explicit_majoring_prefers_major_over_discipline():
    sheets, columns = _mock_dean_sheets()
    sheets["Students"].loc[0, "Discipline"] = "Nursing"
    sheets["Students"].loc[0, "Major"] = "Nursing"
    columns = list(sheets["Students"].columns)
    r = _route("show students majoring in Nursing", sheets, {"Students": columns})
    assert {"column": "Major", "operator": "equals", "value": "Nursing"} in r["plan"]["filters"]


def test_assessment_and_attendance_metrics_do_not_fall_back_to_gpa():
    sheets, columns = _mock_dean_sheets()
    sheets["Students"]["SAT Total"] = [1200, 1100, 1300, 900, 1000]
    sheets["Students"]["Attendance Rate"] = [0.95, 0.88, 0.99, 0.91, 0.84]
    columns = list(sheets["Students"].columns)

    sat = _route("which major has the highest average SAT Total", sheets, {"Students": columns})
    assert sat["plan"]["operation"] == "groupby_average"
    assert sat["plan"]["group_by"] == "Major"
    assert sat["plan"]["value_column"] == "SAT Total"

    attendance = _route("show students with Attendance Rate below 90%", sheets, {"Students": columns})
    assert attendance["plan"]["filters"] == [
        {"column": "Attendance Rate", "operator": "less_than", "value": 0.9}
    ]


def test_llm_missing_major_filter_repairs_to_list_unique(sheets, columns):
    bad_llm_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Major", "operator": "equals"}],
        "confidence": 0.7,
    }
    r = _route(
        "what majors are listed",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(bad_llm_plan),
    )
    p = r["plan"]
    assert r["validation"]["status"] == "passed"
    assert p["operation"] == "list_unique"
    assert p["value_column"] == "Major"
    assert p["filters"] == []


def test_how_many_x_stays_count_unique(sheets, columns):
    """'how many' is a count, never a list."""
    r = _route("how many departments are there", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "count_unique"
    assert p["value_column"] == "Department"


def test_computer_engineering_is_not_misread_as_put_edit_cue():
    sheets, columns = _mock_dean_sheets()
    r = _route("how many Computer Engineering students are there", sheets, {"Students": columns})
    p = r["plan"]
    assert r["intent"] == "query"
    assert p["operation"] == "count_rows"
    assert {"column": "Major", "operator": "equals", "value": "Computer Engineering"} in p["filters"]


def test_show_students_majoring_in_value_stays_row_preview():
    sheets, columns = _mock_dean_sheets()
    r = _route("show students majoring in Data Analytics", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "filtered_preview"
    assert p.get("value_column") in ("", None)
    assert {"column": "Major", "operator": "equals", "value": "Data Analytics"} in p["filters"]


def test_inline_condition_on_watch_action_targets_matching_students_not_all():
    # Caught live: "mark these as watch" is designed to act on the active
    # filter context from a PRIOR turn. A natural single-sentence phrasing
    # with its own inline condition had no prior turn to draw from, so the
    # filter silently dropped and the action defaulted to ALL rows -- the
    # confirmation text said so, but a user who didn't read carefully could
    # confirm marking every student instead of the ~46 GPA-below-2.0 ones.
    sheets, columns = _mock_dean_sheets()
    r = _route("Mark students with GPA below 2.0 as Academic Watch", sheets, {"Students": columns})
    assert r["intent"] == "academic_watch"
    assert r["plan"]["filters"] == [{"column": "GPA", "operator": "less_than", "value": 2.0}]
    assert "ALL rows" not in r["confirmation_reason"]
    assert "GPA" in r["confirmation_reason"]


def test_watch_action_with_no_referent_still_falls_back_to_all_rows():
    # "these" with no prior filter turn genuinely has nothing to parse --
    # ALL rows is the honest answer here, not a regression.
    sheets, columns = _mock_dean_sheets()
    r = _route("Mark these students as Academic Watch", sheets, {"Students": columns})
    assert r["intent"] == "academic_watch"
    assert r["plan"]["filters"] == []
    assert "ALL rows" in r["confirmation_reason"]


def test_reading_missing_academic_watch_does_not_route_to_edit_action():
    sheets, columns = _mock_dean_sheets()
    r = _route("how many students are on Academic Watch", sheets, {"Students": columns})
    assert r["intent"] == "clarify"
    assert r["plan_source"] == "clarification"
    assert "does not include Academic Watch" in r["confirmation_reason"]


def test_missing_academic_watch_clarification_short_circuits_llm():
    sheets, columns = _mock_dean_sheets()
    llm_plan = {
        "intent": "query",
        "operation": "filter",
        "filters": [{"column": "Discipline", "operator": "equals", "value": "Education"}],
        "confidence": 0.9,
    }
    r = _route(
        "show Education students on Academic Watch",
        sheets,
        {"Students": columns},
        enabled=True,
        llm_call=_mock(llm_plan),
    )
    assert r["intent"] == "clarify"
    assert r["llm_used"] is False
    assert "does not include Academic Watch" in r["confirmation_reason"]


def test_average_gpa_by_advisor_uses_gpa_not_standing():
    sheets, columns = _mock_dean_sheets()
    r = _route("which advisor has the lowest average GPA", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "groupby_average"
    assert p["group_by"] == "Advisor"
    assert p["value_column"] == "GPA"
    assert not any(f.get("column") == "Standing" for f in p["filters"])


def test_worst_average_gpa_by_major_groups_by_major():
    sheets, columns = _mock_dean_sheets()
    r = _route("which major has the worst average GPA", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "groupby_average"
    assert p["group_by"] == "Major"
    assert p["value_column"] == "GPA"
    assert p["sort"] == {"column": "GPA", "direction": "asc"}
    assert p["limit"] == 1


def test_top_majors_by_student_count_is_grouped_count():
    sheets, columns = _mock_dean_sheets()
    r = _route("show the top 5 majors by student count", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "groupby_count"
    assert p["group_by"] == "Major"
    assert p["sort"] == {"column": "Count", "direction": "desc"}
    assert p["limit"] == 5


def test_students_with_second_major_maps_to_not_missing():
    sheets, columns = _mock_dean_sheets()
    r = _route("how many students have a second major", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "count_rows"
    assert {"column": "Second Major", "operator": "is_not_missing"} in p["filters"]


# --- grade / year filtering -------------------------------------------------


def test_freshmen_plural_resolves_to_year_filter(sheets, columns):
    r = _route("show me freshmen", sheets, {"Students": columns})
    p = r["plan"]
    assert any(
        f.get("column") == "Year" and f.get("value") == "Freshman"
        for f in p.get("filters") or []
    )


def test_seniors_plural_resolves_to_year_filter(sheets, columns):
    r = _route("show me seniors", sheets, {"Students": columns})
    p = r["plan"]
    assert any(
        f.get("column") == "Year" and f.get("value") == "Senior"
        for f in p.get("filters") or []
    )


def test_k12_grade_number_filter():
    """K-12 'show me 5th graders' / 'grade 5' / '5th grade' against a Grade
    column with values K..12."""
    import pandas as pd
    df = pd.DataFrame({
        "Student ID": ["S1", "S2", "S3", "S4"],
        "Name": ["A", "B", "C", "D"],
        "Grade": ["K", "5", "5", "12"],
    })
    sheets = {"Students": df}
    columns = list(df.columns)
    for msg in ("show me 5th graders", "show students in 5th grade", "show students in grade 5"):
        r = _route(msg, sheets, {"Students": columns})
        p = r["plan"]
        assert any(
            f.get("column") == "Grade" and f.get("value") == "5"
            for f in p.get("filters") or []
        ), msg


def test_k12_kindergarten_filter():
    import pandas as pd
    df = pd.DataFrame({
        "Student ID": ["S1", "S2"],
        "Name": ["A", "B"],
        "Grade": ["K", "5"],
    })
    r = _route("list kindergarten students", {"Students": df}, {"Students": list(df.columns)})
    p = r["plan"]
    assert any(
        f.get("column") == "Grade" and f.get("value") == "K"
        for f in p.get("filters") or []
    )


# --- "list X by Y" = sort, not group ----------------------------------------


def test_list_students_by_year_is_sort_not_group(sheets, columns):
    r = _route("list students by year", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "filtered_preview"
    assert p["sort"] == {"column": "Year", "direction": "asc"}
    assert not p["group_by"]


def test_count_by_x_stays_groupby(sheets, columns):
    r = _route("count students by department", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "groupby_count"
    assert p["group_by"] == "Department"


def test_average_by_x_stays_groupby_average(sheets, columns):
    r = _route("average GPA by department", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "groupby_average"
    assert p["group_by"] == "Department"


# --- range / between filters ------------------------------------------------


def test_between_x_and_y_parses_range(sheets, columns):
    r = _route("students with GPA between 2.0 and 2.5", sheets, {"Students": columns})
    p = r["plan"]
    assert any(
        f.get("column") == "GPA" and f.get("operator") == "between"
        and f.get("value") == [2.0, 2.5]
        for f in p.get("filters") or []
    ), p


def test_from_x_to_y_parses_range(sheets, columns):
    r = _route("students with credits from 30 to 60", sheets, {"Students": columns})
    p = r["plan"]
    assert any(
        f.get("column") == "Credits Completed" and f.get("operator") == "between"
        and f.get("value") == [30.0, 60.0]
        for f in p.get("filters") or []
    ), p


def test_bare_x_to_y_with_column_hint(sheets, columns):
    r = _route("show students with credits 30 to 60", sheets, {"Students": columns})
    p = r["plan"]
    assert any(
        f.get("operator") == "between" and f.get("value") == [30.0, 60.0]
        for f in p.get("filters") or []
    ), p


# --- percent_rows -----------------------------------------------------------


def test_what_percent_routes_to_percent_rows(sheets, columns):
    r = _route("what percent of students are on probation", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "percent_rows"
    assert any(
        f.get("column") == "Academic Status" and f.get("value") == "Probation"
        for f in p.get("filters") or []
    )


def test_share_of_routes_to_percent_rows(sheets, columns):
    r = _route("what fraction of students have GPA below 2.0", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "percent_rows"
    assert any(
        f.get("column") == "GPA" and f.get("operator") == "less_than"
        for f in p.get("filters") or []
    )


def test_how_many_stays_count_not_percent(sheets, columns):
    """'how many' is a count, never a percent."""
    r = _route("how many students are on probation", sheets, {"Students": columns})
    p = r["plan"]
    assert p["operation"] == "count_rows"


# --- OR filter mode --------------------------------------------------------


def test_or_between_two_filters_sets_any_mode(sheets, columns):
    r = _route("students on probation or with GPA below 2.0", sheets, {"Students": columns})
    p = r["plan"]
    assert p["filter_mode"] == "any"
    assert len(p["filters"]) >= 2


def test_and_keeps_all_mode(sheets, columns):
    r = _route("students on probation and below 2.0", sheets, {"Students": columns})
    p = r["plan"]
    assert p["filter_mode"] == "all"


def test_single_filter_stays_all_even_with_or_word(sheets, columns):
    """'or' in a message with only one resolvable filter shouldn't flip mode."""
    r = _route("show me students or freshmen", sheets, {"Students": columns})
    p = r["plan"]
    # Either no second filter resolved (mode stays 'all'), or both resolved
    # — but if only one resolved we must NOT flip to 'any'.
    if len(p.get("filters") or []) < 2:
        assert p.get("filter_mode", "all") == "all"


def test_or_carries_through_to_engine_result(sheets, columns):
    """End-to-end: OR routing must hit the engine and produce a strict
    superset of the AND result."""
    from core.query_engine import run_query

    r_or = _route("students on probation or with GPA below 2.0",
                  sheets, {"Students": columns})
    r_and = _route("students on probation and below 2.0",
                   sheets, {"Students": columns})

    or_result = run_query(r_or["plan"], sheets)
    and_result = run_query(r_and["plan"], sheets)
    assert or_result.value >= and_result.value
