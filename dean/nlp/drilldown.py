"""Phase P: result drilldown + follow-up context resolution.

Recognizes two follow-up patterns that the regular planner doesn't handle:

  Pattern A — row drilldown after a group-level result.
    User: "How many teachers have students under 2.5?"
    A:    "7 advisors have at least one student below 2.5 GPA."
    User: "Which students are those?" / "list them" / "show me the students"
      → switch to filtered_preview using the row-level filter (GPA < 2.5).

  Pattern B — winner drilldown after a top-N group result.
    User: "Which department has the best average GPA?"
    A:    "Education has the highest GPA (3.04)."
    User: "Show me students in that department"
      → add {Department equals Education} as a filter on a fresh
        filtered_preview.

Both patterns produce a complete plan dict the dispatcher can run directly,
plus a context_reminder string the narrator prepends to the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nlp.synonym_mapper import (
    load_json,
    load_synonyms_with_learned,
    match_column_for_concept_with_fallback,
    normalize_text,
)


# Row drilldown — phrases that explicitly want the underlying STUDENT rows.
# These always route to filtered_preview with the prior row filter (when one
# exists), even if the previous turn was count_unique on a class column.
_ROW_DRILLDOWN_STUDENT_PHRASES = (
    "which students are those", "which students are these",
    "which students are they",
    "list the students", "list those students", "list these students",
    "show me the students", "show me those students", "show me these students",
    "show those students", "show these students", "show the students",
    "who are those students", "who are these students",
)
# Generic listing phrases — meaning depends on the prior result type:
#  - after count_unique on column X → list the distinct X values (groupby_count).
#  - after groupby_* / count_rows → show the underlying matching rows.
_GENERIC_LISTING_PHRASES = (
    "which ones", "list them", "list those", "list these",
    "show me them", "show me those", "show me these",
    "who are they", "which are they", "which are those",
    "what are they", "what are those", "what are these", "what are the",
    "name them", "name those", "name these", "name the ones",
    "show me the list", "show the list", "give me the list",
)
_ROW_DRILLDOWN_PHRASES = _ROW_DRILLDOWN_STUDENT_PHRASES + _GENERIC_LISTING_PHRASES
# Drilldown that wants to keep the row filter AND add a group_by axis.
_BREAKDOWN_PHRASES = (
    "break it down by", "break that down by", "break this down by",
    "break them down by", "broken down by",
    "split it by", "split that by", "split them by",
    "grouped by",
)
# "in that <X>" / "for that <X>" / "from that <X>" — winner drilldown.
# We don't require a specific entity word; the noun after "that" should
# match the column the winner came from, but the detector accepts any noun
# and validates against memory.last_group_winner_column.
_WINNER_REFERENT_PATTERNS = (
    re.compile(r"\b(?:in|for|from|within)\s+(?:that|this|the)\s+([a-z]+(?:\s+[a-z]+)?)"),
    re.compile(r"\b(?:that|this)\s+([a-z]+(?:\s+[a-z]+)?)(?:'s)?\b"),
)


@dataclass
class DrilldownPlan:
    """A complete plan dict + a context reminder string."""
    plan: dict[str, Any]
    context_reminder: str
    kind: str  # "row_drilldown" | "winner_drilldown" | "breakdown"


def resolve_drilldown(
    user_message: str, memory, *, columns: list[str] | None = None,
) -> DrilldownPlan | None:
    """Map a follow-up message + prior result context to a drilldown plan.

    Returns None when the message doesn't look like a drilldown OR memory has
    no usable prior result for the pattern that matched. ``columns`` is the
    list of column names in the active sheet; used to resolve breakdown
    target nouns ("by advisor" → "Advisor").
    """
    if not user_message:
        return None
    text = normalize_text(user_message)
    if not text:
        return None

    def _g(key: str, default=None):
        if isinstance(memory, dict):
            return memory.get(key, default)
        return getattr(memory, key, default)

    sheet = _g("active_sheet", "") or _g("last_sheet", "")
    last_op = _g("last_operation", "") or ""
    row_filter = list(_g("last_row_filter", []) or [])
    referent = _g("last_referent_label", "") or ""
    winner_column = _g("last_group_winner_column", "") or ""
    winner_value = _g("last_group_winner_value", "") or ""
    summary_table = list(_g("last_summary_table", []) or [])

    # Pattern A — row / distinct-value drilldown
    if _matches_phrase(text, _ROW_DRILLDOWN_PHRASES):
        # Must follow a group-level or aggregate result to be meaningful.
        if last_op not in {"count_unique", "groupby_count", "groupby_sum",
                           "groupby_average", "count_rows"}:
            return None

        # Special case: prior turn was "how many <X> are there"
        # (count_unique on column X) AND the user used a generic listing
        # phrase ("what are they" / "list them"), NOT one that explicitly
        # named "students". Emit a groupby_count by that column so the
        # user sees each distinct value with its row count.
        student_specific = _matches_phrase(text, _ROW_DRILLDOWN_STUDENT_PHRASES)
        if last_op == "count_unique" and not student_specific:
            last_plan = _g("last_query_plan", {}) or {}
            target_column = last_plan.get("value_column", "") or ""
            if target_column:
                plan = _filtered_preview_plan(sheet, row_filter)
                plan["operation"] = "groupby_count"
                plan["group_by"] = target_column
                plan["value_column"] = ""
                plan["limit"] = None
                reminder = (
                    f"Listing the distinct {target_column} values "
                    f"from the previous count."
                )
                if row_filter:
                    reminder += f" Carrying the previous filter: {_describe_filters(row_filter)}."
                return DrilldownPlan(
                    plan=plan, context_reminder=reminder,
                    kind="distinct_listing",
                )

        # Standard row drilldown — requires a row-level filter to act on.
        plan = _filtered_preview_plan(sheet, row_filter)
        referenced_group_value = _referenced_group_value(
            text, winner_column, winner_value, summary_table,
        )
        if not referenced_group_value and _references_top_group(text):
            referenced_group_value = str(winner_value or "")
        if (
            winner_column
            and referenced_group_value
            and last_op in {"groupby_count", "groupby_sum", "groupby_average"}
        ):
            plan["filters"] = [
                f for f in plan["filters"] if f.get("column") != winner_column
            ]
            plan["filters"].append({
                "column": winner_column, "operator": "equals", "value": referenced_group_value,
            })
            reminder = (
                f"I'm using the previous filter: {_describe_filters(row_filter)}. "
                f"Also filtering to {winner_column} = {referenced_group_value}."
            )
            return DrilldownPlan(
                plan=plan,
                context_reminder=reminder,
                kind="winner_drilldown",
            )
        if not row_filter:
            return None
        return DrilldownPlan(
            plan=plan,
            context_reminder=f"I'm using the previous filter: {_describe_filters(row_filter)}.",
            kind="row_drilldown",
        )

    # Pattern B — "break it down by X"
    breakdown_target = _extract_breakdown_target(text)
    if breakdown_target:
        group_column = _resolve_to_column(breakdown_target, columns or [])
        if not group_column:
            return None
        plan = _filtered_preview_plan(sheet, row_filter)
        plan["operation"] = "groupby_count"
        plan["group_by"] = group_column
        plan["value_column"] = ""
        plan["limit"] = None
        return DrilldownPlan(
            plan=plan,
            context_reminder=f"I'm using the previous filter: {_describe_filters(row_filter)}." if row_filter else "",
            kind="breakdown",
        )

    # Pattern B — winner drilldown ("in that department")
    if winner_column and winner_value and _references_winner(text, winner_column):
        # If the user also typed a row-level predicate ("students not performing
        # well in that department"), keep the prior row filter as a starting
        # point AND let the regular planner add the new predicate on top — for
        # the bare "show me students in that department" case, we just emit a
        # filtered_preview with the winner filter.
        plan = _filtered_preview_plan(sheet, row_filter)
        # Append the winner filter (replacing any prior condition on the same column).
        plan["filters"] = [f for f in plan["filters"] if f.get("column") != winner_column]
        plan["filters"].append({
            "column": winner_column, "operator": "equals", "value": winner_value,
        })
        reminder = f"I'm using the previous {winner_column.lower()} winner: {winner_value}."
        if row_filter:
            reminder += f" Carrying the previous filter: {_describe_filters(row_filter)}."
        return DrilldownPlan(plan=plan, context_reminder=reminder,
                              kind="winner_drilldown")

    return None


# ---- helpers ---------------------------------------------------------------


def _matches_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _extract_breakdown_target(text: str) -> str | None:
    for cue in _BREAKDOWN_PHRASES:
        if cue in text:
            tail = text.split(cue, 1)[1].strip()
            # Take 1–3 words as the candidate group concept.
            tokens = tail.split()
            if not tokens:
                return None
            candidate = " ".join(tokens[:2]).rstrip(".,;:!?")
            return candidate or None
    return None


def _references_winner(text: str, winner_column: str) -> bool:
    """True if the user's wording targets the winner of a prior group result.

    Accepts "in that <col>", "for that <col>", "from that <col>", plus the
    bare "that <col>". The noun is matched against the winner column either
    directly (substring) OR via the cross-concept aliases — so "in that
    department" resolves to a Discipline winner because department↔discipline
    are aliased in synonym_mapper.CONCEPT_ALIASES.
    """
    from nlp.synonym_mapper import CONCEPT_ALIASES

    target = winner_column.lower()
    aliases: set[str] = {target}
    for concept, alias_chain in CONCEPT_ALIASES.items():
        chain = {concept} | set(alias_chain)
        if any(target == name or name in target for name in chain):
            aliases.update(chain)
    for pattern in _WINNER_REFERENT_PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group(1).strip()
            if not phrase:
                continue
            if any(alias in phrase or phrase in alias for alias in aliases):
                return True
    return False


def _references_winner_value(text: str, winner_value: str) -> bool:
    phrase = normalize_text(str(winner_value))
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _references_top_group(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "top group", "the top group", "this top group", "that top group",
            "winning group", "highest group", "largest group",
        )
    )


def _referenced_group_value(
    text: str,
    winner_column: str,
    winner_value: str,
    summary_table: list[dict[str, Any]],
) -> str:
    if winner_value and _references_winner_value(text, winner_value):
        return str(winner_value)
    if not winner_column:
        return ""
    for row in summary_table:
        if not isinstance(row, dict):
            continue
        value = row.get(winner_column)
        if value is not None and _references_winner_value(text, str(value)):
            return str(value)
    return ""


def _resolve_to_column(target_phrase: str, columns: list[str]) -> str | None:
    """Resolve a breakdown noun ('advisor', 'department') to a real column.

    Uses synonyms + the cross-concept fallback so 'advisor' lands on Advisor
    and 'department' lands on Discipline when only the latter exists.
    """
    if not target_phrase or not columns:
        return None
    # Direct case-insensitive column-name match.
    needle = normalize_text(target_phrase)
    for column in columns:
        if normalize_text(column) == needle:
            return column
    # Synonym / fallback resolution.
    synonyms = load_synonyms_with_learned()
    column, _score, _fallback_from = match_column_for_concept_with_fallback(
        needle.replace(" ", "_"), columns, synonyms,
    )
    if column:
        return column
    # Try the bare word too (some single-token concepts like 'advisor').
    column, _, _ = match_column_for_concept_with_fallback(
        needle.split()[0] if needle else "", columns, synonyms,
    )
    return column


def _filtered_preview_plan(sheet: str, filters: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "request_type": "ask_question",
        "operation": "filtered_preview",
        "sheet": sheet,
        "filters": [dict(f) for f in filters],
        "group_by": "",
        "value_column": "",
        "sort": None,
        "sort_by": "",
        "limit": 50,
        "plain_english_question": "",
        "confidence": 0.85,
    }


def _describe_filters(filters: list[dict[str, Any]]) -> str:
    if not filters:
        return "no filter"
    bits = []
    op_phrase = {
        "less_than": "<", "less_or_equal": "≤",
        "greater_than": ">", "greater_or_equal": "≥",
        "equals": "=", "not_equals": "≠",
    }
    for f in filters:
        column = f.get("column", "")
        op = f.get("operator", "")
        value = f.get("value")
        phrase = op_phrase.get(op, op)
        if op in {"is_missing", "is_not_missing", "is_blank", "is_not_blank"}:
            bits.append(f"{column} {phrase}")
        elif isinstance(value, list):
            bits.append(f"{column} {phrase} {value}")
        else:
            bits.append(f"{column} {phrase} {value}")
    return " AND ".join(bits)
