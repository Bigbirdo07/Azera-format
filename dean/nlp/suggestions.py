"""Grounded next-move suggestions for the assistant.

Produces 2–3 short, safe follow-up prompts the user can click after each
answer. Suggestions are derived from the validated plan, the result shape,
the available columns, and the active filters — never from row data.

Phrasing uses the assistant's existing follow-up cues ("now", "only", "what
about", "group these by ...") so the planner can route the click as a normal
follow-up.

When the conversational LLM is enabled, this list is also passed to the
narrator as `allowed_next_actions` so the model can mention them naturally
without inventing options the app cannot honor.
"""

from __future__ import annotations

from typing import Any

from core.privacy import is_hidden_by_default


# Columns that are useful for grouping in the dean-office workbook.
_PREFERRED_GROUPING_COLUMNS = (
    "Advisor",
    "Department",
    "Major",
    "Program",
    "Year",
    "Academic Status",
    "Status",
    "College",
)

_MAX_SUGGESTIONS = 3


def suggest_next_moves(
    *,
    plan: dict[str, Any],
    columns: list[str],
    active_filters: list[dict[str, Any]] | None = None,
    removed_fields: list[str] | None = None,
    row_count: int | None = None,
) -> list[str]:
    """Return up to 3 grounded next-move prompts.

    The suggestions are plain English using the assistant's follow-up cues so
    each one routes back through the normal planner when the user clicks it.
    """
    operation = plan.get("operation") or "filtered_preview"
    active = active_filters or []
    filtered_columns = {f.get("column") for f in active}

    suggestions: list[str] = []

    if isinstance(row_count, int) and row_count == 0:
        suggestions.append("Clear the filters and show all students")
        suggestions.append("Try a broader filter")
        return _dedupe_and_cap(suggestions)

    # Notes-search follow-ups (M.6). When the user just ran a free-text search
    # on a notes column, the most useful next moves are advisor/status grouping
    # and export — exactly the priority outreach workflow.
    if _is_text_search(active):
        if _column_exists(columns, "Advisor"):
            suggestions.append("Group these by advisor")
        if _column_exists(columns, "Academic Status"):
            suggestions.append("Group these by academic status")
        suggestions.append("Export this list")
        suggestions.append("Add a follow-up note to these students")
        return _dedupe_and_cap(suggestions)

    if operation == "filtered_preview":
        # Academic workflow chips: when the active selection is the low-GPA set
        # under a teacher/professor workbook, prioritize the Academic Watch
        # action and teacher grouping.
        if _has_gpa_below_filter(active) and _has_teacher_column(columns):
            if "teacher" not in {str(c).lower() for c in filtered_columns}:
                suggestions.append("Group by teacher")
            suggestions.append("Mark these students Academic Watch")
            suggestions.append("Export this list")
            return _dedupe_and_cap(suggestions)
        group_column = _pick_grouping_column(columns, filtered_columns, plan.get("group_by"))
        if group_column:
            suggestions.append(f"Group these by {group_column}")
        suggestions.append("Export this filtered list")
        suggestions.append("Add a follow-up note to these students")

    elif operation == "count_rows":
        # Same low-GPA → Academic Watch shortcut for the "how many" count form.
        if _has_gpa_below_filter(active) and _has_teacher_column(columns):
            suggestions.append("Group by teacher")
            suggestions.append("Mark these students Academic Watch")
        group_column = _pick_grouping_column(columns, filtered_columns, plan.get("group_by"))
        if group_column:
            suggestions.append(f"Now group these by {group_column}")
        suggestions.append("Show me the matching students")

    elif operation in {"groupby_count", "groupby_sum", "groupby_average"}:
        suggestions.append("Sort highest to lowest")
        suggestions.append("Show the students in the top group")
        suggestions.append("Make a bar chart of this")

    elif operation in {"average_column", "sum_column", "min_column", "max_column"}:
        group_column = _pick_grouping_column(columns, filtered_columns, plan.get("group_by"))
        if group_column:
            suggestions.append(f"Break this down by {group_column}")
        suggestions.append("Show me the matching students")

    elif operation == "cohort_summary":
        if _column_exists(columns, "GPA"):
            suggestions.append("Show average GPA for these students")
        if _column_exists(columns, "Standing"):
            suggestions.append("Break these down by Standing")
        if _column_exists(columns, "Attendance Rate"):
            suggestions.append("Show attendance for these students")
        suggestions.append("Show me the matching students")

    elif operation == "cohort_comparison":
        suggestions.append("Make a bar chart of this")
        suggestions.append("Show me the matching students")
        suggestions.append("Export this comparison")

    elif operation in {"missing_summary", "duplicate_check", "data_quality_summary"}:
        suggestions.append("Show me the affected students")
        suggestions.append("Export a data-quality report")

    if removed_fields:
        # We deliberately do not suggest showing hidden sensitive fields here;
        # users can ask explicitly and that path requires confirmation.
        pass

    return _dedupe_and_cap(suggestions)


_TEXT_SEARCH_OPS = {"contains_text", "not_contains_text"}


def _is_text_search(active_filters: list[dict[str, Any]]) -> bool:
    return any(f.get("operator") in _TEXT_SEARCH_OPS for f in active_filters)


def _column_exists(columns: list[str], name: str) -> bool:
    target = name.lower()
    return any(str(c).lower() == target for c in columns)


def _has_gpa_below_filter(active_filters: list[dict[str, Any]]) -> bool:
    """True if a filter narrows to low-GPA students (the typical pre-Watch state)."""
    for f in active_filters:
        column = str(f.get("column", "")).lower()
        operator = f.get("operator")
        if "gpa" not in column:
            continue
        if operator in {"less_than", "less_or_equal"}:
            try:
                if float(f.get("value", 0)) <= 2.5:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _has_teacher_column(columns: list[str]) -> bool:
    return any("teacher" in str(c).lower() or "professor" in str(c).lower()
               or "instructor" in str(c).lower() for c in columns)


def _pick_grouping_column(
    columns: list[str],
    already_filtered: set[str | None],
    current_group: str | None,
) -> str | None:
    """Pick the first preferred grouping column that exists and is not in use."""
    lower_existing = {str(c).lower(): str(c) for c in columns if c}
    used = {str(c).lower() for c in already_filtered if c}
    if current_group:
        used.add(str(current_group).lower())
    for candidate in _PREFERRED_GROUPING_COLUMNS:
        actual = lower_existing.get(candidate.lower())
        if actual and actual.lower() not in used and not is_hidden_by_default(actual):
            return actual
    return None


def _dedupe_and_cap(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(item.strip())
        if len(output) >= _MAX_SUGGESTIONS:
            break
    return output
