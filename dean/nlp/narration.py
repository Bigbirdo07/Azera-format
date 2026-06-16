"""Plain-English narration of validated plans.

Produces a short "I understood this as..." sentence for any planner result.
Deterministic and dependency-free so it works regardless of LLM availability
or privacy mode. When the conversational LLM is enabled, the narrator is fed
the same text as context (never raw rows) and may polish the wording.
"""

from __future__ import annotations

from typing import Any

from nlp.conversation import FOLLOWUP, describe_filters


_OPERATION_PHRASE = {
    "filtered_preview": "list the matching rows",
    "count_rows": "count the matching rows",
    "count_unique": "count distinct values",
    "average_column": "compute the average",
    "sum_column": "compute the total",
    "min_column": "find the minimum",
    "max_column": "find the maximum",
    "groupby_count": "group the rows and count each group",
    "groupby_sum": "group the rows and total each group",
    "groupby_average": "group the rows and average each group",
    "missing_summary": "summarize missing values",
    "duplicate_check": "check for duplicate rows",
    "data_quality_summary": "summarize data quality",
}

_OPERATOR_PHRASE = {
    "equals": "=",
    "not_equals": "≠",
    "contains": "contains",
    "not_contains": "does not contain",
    "contains_any": "contains any of",
    "starts_with": "starts with",
    "ends_with": "ends with",
    "greater_than": ">",
    "greater_or_equal": "≥",
    "less_than": "<",
    "less_or_equal": "≤",
    "between": "between",
    "in": "in",
    "not_in": "not in",
    "is_blank": "is blank",
    "is_not_blank": "is not blank",
    "is_missing": "is missing",
    "is_not_missing": "is present",
}


def narrate_plan(
    *,
    plan: dict[str, Any],
    context_action: str = "fresh",
    prior_filters: list[dict[str, Any]] | None = None,
    new_filters: list[dict[str, Any]] | None = None,
    additive: bool = False,
) -> str:
    """Return one short sentence describing the plan."""
    if context_action == FOLLOWUP:
        change = _describe_followup_change(prior_filters or [], new_filters or [], plan, additive)
        if change:
            return change

    operation = plan.get("operation") or "filtered_preview"
    filters = plan.get("filters") or []
    group_by = plan.get("group_by") or ""
    value_column = plan.get("value_column") or ""
    sort = plan.get("sort") or None
    limit = plan.get("limit")

    head = _operation_head(operation, group_by=group_by, value_column=value_column)
    pieces: list[str] = [f"I understood this as: {head}"]

    if filters:
        pieces.append("where " + _describe_conditions(filters))

    if group_by and "group" not in head:
        pieces.append(f"by {group_by}")

    if sort and sort.get("column"):
        direction = (sort.get("direction") or "descending").lower()
        pieces.append(f"sorted by {sort['column']} {direction}")

    if isinstance(limit, int) and limit > 0:
        pieces.append(f"limited to {limit}")

    sentence = " ".join(part for part in pieces if part).strip()
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _operation_head(operation: str, *, group_by: str, value_column: str) -> str:
    base = _OPERATION_PHRASE.get(operation, _OPERATION_PHRASE["filtered_preview"])
    if operation in {"average_column", "sum_column", "min_column", "max_column"} and value_column:
        base = base + f" of {value_column}"
    if operation in {"groupby_count", "groupby_sum", "groupby_average"} and group_by:
        base = base.replace("group the rows", f"group the rows by {group_by}")
        if operation == "groupby_sum" and value_column:
            base = base + f" of {value_column}"
        if operation == "groupby_average" and value_column:
            base = base + f" of {value_column}"
    return base


def _describe_conditions(conditions: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for condition in conditions:
        column = condition.get("column", "?")
        operator = condition.get("operator", "equals")
        phrase = _OPERATOR_PHRASE.get(operator, operator.replace("_", " "))
        value = condition.get("value")
        if operator in {"is_blank", "is_not_blank", "is_missing", "is_not_missing"}:
            parts.append(f"{column} {phrase}")
        elif isinstance(value, list):
            parts.append(f"{column} {phrase} [{', '.join(map(str, value))}]")
        else:
            parts.append(f"{column} {phrase} {value}")
    return " and ".join(parts)


def _describe_followup_change(
    prior: list[dict[str, Any]],
    new: list[dict[str, Any]],
    plan: dict[str, Any],
    additive: bool,
) -> str:
    prior_columns = {f.get("column") for f in prior}
    replaced = [f for f in new if f.get("column") in prior_columns]
    added = [f for f in new if f.get("column") not in prior_columns]
    group_by = plan.get("group_by") or ""

    parts: list[str] = []
    if replaced and not additive:
        names = ", ".join(dict.fromkeys(f["column"] for f in replaced if f.get("column")))
        parts.append(f"I replaced the {names} filter with {_describe_conditions(replaced)}")
    elif replaced and additive:
        parts.append("I added " + _describe_conditions(replaced) + " to the existing selection")
    if added:
        lead = " and added " if parts else "I kept the current filters and added "
        parts.append(lead + _describe_conditions(added))
    if group_by:
        lead = " and grouped " if parts else "I kept the current filters and grouped "
        parts.append(lead + f"by {group_by}")

    if not parts:
        return ""
    sentence = "".join(parts).strip()
    if not sentence.endswith("."):
        sentence += "."
    return sentence
