"""Phase O: dean-roster ambiguity detection.

Recognizes two classes of school-roster questions where the most natural
English phrasing supports more than one defensible interpretation:

  1. Container vs. aggregate
     "how many advisors contain students with GPA below 2.5"
        A (container, default): count distinct Advisors where any student row
          has GPA < 2.5
        B (aggregate, alt):     count Advisors whose AVERAGE student GPA < 2.5

  2. "Performing well" department ranking
     "which department has the best average of students performing well"
        A (avg-GPA, default):   department with the highest mean GPA
        B (rate-above-cutoff, alt): department with the highest share of
          students at or above the GPA cutoff (default 2.5)

When detected, the planner builds the primary plan and stashes the alternative
description on the routing so the dispatcher can ALSO compute the alternative
in pandas and surface both results in one response. The user can click the
alternative chip to re-route the follow-up through the alternative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nlp.synonym_mapper import (
    load_json,
    match_column_for_concept_with_fallback,
    normalize_text,
)


# Entity tokens we accept as the group axis for container/aggregate questions.
_ENTITY_CONCEPTS = ("teacher", "professor", "instructor", "advisor",
                    "department", "discipline", "major")
# Phrases that imply "this group has student rows with the condition".
_CONTAINER_VERBS = ("have students", "has students", "contain students",
                    "contains students", "containing students", "with students",
                    "who have students", "who has students", "that have students",
                    "that contain students", "whose students")
# Phrases that imply "this group's aggregate metric meets the condition".
_AGGREGATE_HINTS = ("whose average", "with an average", "averaging",
                    "with average", "average gpa")
# Phrases that mean "the average of N" appears inside the user wording — that
# nudges interpretation B but doesn't pick it (the spec's example uses
# "less than an average of 2.5" and still treats container as primary).
_AVERAGE_HINTS = ("an average of", "average of", "average gpa")
# Performance-ambiguity phrases that imply a "best department by …" question.
_BEST_AVERAGE_PHRASES = (
    "best average", "highest average", "top average",
    "best performance", "top performance", "highest performance",
    "strongest students", "best-performing", "best performing department",
    "top department", "best department",
)
# "Performing well" suggests an "above cutoff" reading is also reasonable.
_PERFORMING_WELL_PHRASES = (
    "performing well", "doing well", "performing strongly", "do well",
    "above standard", "above average", "above cutoff",
)
# Default numeric threshold when the user said "performing well" without one.
_DEFAULT_GPA_CUTOFF = 2.5

# Numeric comparison words → operator key.
_COMPARISON_WORDS = (
    (("less than", "below", "under", "lower than", "fewer than"), "less_than"),
    (("at most", "no more than", "or less", "or below"), "less_or_equal"),
    (("greater than", "above", "over", "more than", "higher than"), "greater_than"),
    (("at least", "no less than", "or more", "or above", "or higher"), "greater_or_equal"),
)


# ---- public dataclasses ----------------------------------------------------


@dataclass
class AmbiguityResolution:
    """The container-vs-aggregate or performance ambiguity result.

    The planner uses ``primary_plan`` as the operation to actually plan; the
    dispatcher runs ``alternative_spec`` separately in pandas and joins both
    results into the response message.
    """
    kind: str                       # "container_vs_aggregate" | "performing_well"
    primary_plan: dict[str, Any]    # the validated plan the rule planner emits
    alternative_spec: dict[str, Any]  # pandas-runnable description of the alt
    assumption_note: str            # "I interpreted this as advisors ..."
    alternative_phrase: str         # human label for the alt chip
    column_mapping: list[tuple[str, str]]  # [("teacher", "Advisor"), ...]


# ---- entry point -----------------------------------------------------------


def detect_ambiguity(
    user_request: str,
    *,
    sheet: str,
    columns: list[str],
    synonyms: dict[str, list[str]] | None = None,
) -> AmbiguityResolution | None:
    """Return an AmbiguityResolution if the request is one of the supported
    school-roster ambiguity patterns. Otherwise return None and let the
    normal planner handle the request."""
    if synonyms is None:
        synonyms = load_json("synonyms.json")
    text = normalize_text(user_request)

    container = _detect_container_vs_aggregate(text, sheet, columns, synonyms)
    if container is not None:
        return container

    performance = _detect_performing_well(text, sheet, columns, synonyms)
    if performance is not None:
        return performance

    return None


# ---- container / aggregate -------------------------------------------------


def _detect_container_vs_aggregate(
    text: str, sheet: str, columns: list[str], synonyms: dict[str, list[str]]
) -> AmbiguityResolution | None:
    # Need: an entity word AND a "have/contain students" phrase AND a numeric
    # GPA comparison. Anything less ambiguous than that we let the regular
    # planner handle.
    entity_concept = _first_entity_concept(text)
    if entity_concept is None:
        return None
    if not any(verb in text for verb in _CONTAINER_VERBS):
        return None

    threshold_match = _extract_numeric_comparison(text)
    if threshold_match is None:
        return None
    operator, threshold = threshold_match

    group_column, _, fallback_from = match_column_for_concept_with_fallback(
        entity_concept, columns, synonyms,
    )
    if not group_column:
        return None

    # GPA is what "students with less than 2.5" almost always means on a
    # roster; we don't try to map an arbitrary metric here.
    gpa_column, _, _ = match_column_for_concept_with_fallback("gpa", columns, synonyms)
    if not gpa_column:
        return None

    # Build the primary (container) plan: count_unique(group_column) WHERE
    # the row-level GPA condition matches. The dispatcher will run this
    # through the existing query engine.
    primary_plan = {
        "request_type": "ask_question",
        "operation": "count_unique",
        "sheet": sheet,
        "filters": [{"column": gpa_column, "operator": operator, "value": threshold}],
        "group_by": "",
        "value_column": group_column,
        "sort": None,
        "limit": None,
        "plain_english_question": text,
        "confidence": 0.85,
    }

    # The alternative is computed inline by the dispatcher because no single
    # existing operation expresses "filter groups by their aggregate."
    alternative_spec = {
        "kind": "aggregate_then_count_groups",
        "group_column": group_column,
        "metric_column": gpa_column,
        "aggregate": "mean",
        "operator": operator,
        "threshold": threshold,
    }

    column_mapping: list[tuple[str, str]] = []
    if fallback_from and fallback_from != group_column.lower():
        column_mapping.append((fallback_from, group_column))
    column_mapping.append(("gpa", gpa_column))

    op_word = _operator_to_phrase(operator)
    primary_label = (
        f"{group_column}s with at least one student where "
        f"{gpa_column} {op_word} {threshold}"
    )
    alt_label = (
        f"{group_column}s whose average student {gpa_column} "
        f"{op_word} {threshold}"
    )
    assumption = f"I interpreted this as {primary_label}."

    return AmbiguityResolution(
        kind="container_vs_aggregate",
        primary_plan=primary_plan,
        alternative_spec=alternative_spec,
        assumption_note=assumption,
        alternative_phrase=alt_label,
        column_mapping=column_mapping,
    )


# ---- "performing well" department ranking ---------------------------------


def _detect_performing_well(
    text: str, sheet: str, columns: list[str], synonyms: dict[str, list[str]]
) -> AmbiguityResolution | None:
    # Both signals must be present: "which/what department has the …" framing
    # AND a "performing well" / "best average" cue.
    has_dept_question = (
        ("department" in text or "discipline" in text or "major" in text)
        and ("which" in text or "what" in text or "best" in text)
    )
    if not has_dept_question:
        return None
    if _mentions_non_gpa_metric(text):
        return None
    best_avg_hit = any(phrase in text for phrase in _BEST_AVERAGE_PHRASES)
    performing_well_hit = any(phrase in text for phrase in _PERFORMING_WELL_PHRASES)
    if not (best_avg_hit or performing_well_hit):
        return None

    group_column, _, fallback_from_dept = match_column_for_concept_with_fallback(
        "department", columns, synonyms,
    )
    if not group_column:
        return None
    gpa_column, _, _ = match_column_for_concept_with_fallback("gpa", columns, synonyms)
    if not gpa_column:
        return None

    # Primary: top group by average GPA descending, limit 1.
    primary_plan = {
        "request_type": "ask_question",
        "operation": "groupby_average",
        "sheet": sheet,
        "filters": [],
        "group_by": group_column,
        "value_column": gpa_column,
        "sort": {"column": gpa_column, "direction": "desc"},
        "limit": 1,
        "plain_english_question": text,
        "confidence": 0.85,
    }

    # Alternative: rate of students with GPA >= cutoff, by group, top 1.
    threshold = _extract_threshold(text) or _DEFAULT_GPA_CUTOFF
    alternative_spec = {
        "kind": "rate_above_cutoff_by_group",
        "group_column": group_column,
        "metric_column": gpa_column,
        "operator": "greater_or_equal",
        "threshold": threshold,
    }

    column_mapping: list[tuple[str, str]] = []
    if fallback_from_dept and fallback_from_dept != group_column.lower():
        column_mapping.append((fallback_from_dept, group_column))
    column_mapping.append(("gpa", gpa_column))

    assumption = (
        f"I interpreted department as {group_column} and 'performing well' "
        f"as the {group_column} with the highest average {gpa_column}."
    )
    alt_phrase = (
        f"Compare by share of students with {gpa_column} ≥ {threshold} instead"
    )

    return AmbiguityResolution(
        kind="performing_well",
        primary_plan=primary_plan,
        alternative_spec=alternative_spec,
        assumption_note=assumption,
        alternative_phrase=alt_phrase,
        column_mapping=column_mapping,
    )


def _mentions_non_gpa_metric(text: str) -> bool:
    return any(
        token in text
        for token in (
            "sat", "psat", "attendance", "absent", "absence", "days present",
            "days absent", "english", "math score", "total score",
        )
    )


# ---- helpers ---------------------------------------------------------------


def _first_entity_concept(text: str) -> str | None:
    for concept in _ENTITY_CONCEPTS:
        if f" {concept}" in f" {text}" or text.startswith(concept) or f"{concept}s" in text:
            return concept
    return None


def _extract_numeric_comparison(text: str) -> tuple[str, float] | None:
    """Find a (operator, threshold) pair from common comparison phrasings."""
    for phrases, operator in _COMPARISON_WORDS:
        for phrase in phrases:
            # Allow "an average of" / "average of" between the comparison word
            # and the number, since that's the wording that creates the
            # ambiguity (e.g. "less than an average of 2.5").
            pattern = rf"{re.escape(phrase)}\s+(?:an?\s+)?(?:average\s+(?:of\s+)?)?(\d+(?:\.\d+)?)"
            match = re.search(pattern, text)
            if match:
                try:
                    return operator, float(match.group(1))
                except ValueError:
                    continue
    return None


def _extract_threshold(text: str) -> float | None:
    """Find a bare numeric GPA threshold in the text (e.g. '2.5', '2.0')."""
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        try:
            value = float(match.group(1))
            if 0 <= value <= 4.5:
                return value
        except ValueError:
            pass
    return None


def _operator_to_phrase(operator: str) -> str:
    return {
        "less_than": "<",
        "less_or_equal": "≤",
        "greater_than": ">",
        "greater_or_equal": "≥",
    }.get(operator, operator)
