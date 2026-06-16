"""Conversation context manager for multi-turn follow-ups.

Turns a sequence of messages into a composable set of active filters:

    "Show Accounting students"   -> Department = Accounting
    "now only below 2.5"         -> + GPA < 2.5            (compose)
    "now only seniors"           -> + Year = Senior        (compose)
    "what about Biology"         -> Department = Biology    (replace same column)
    "clear that"                 -> drop filters
    "start over"                 -> reset everything

It also detects categorical *value* filters by matching a column's distinct
values against the user's words locally (e.g. "Accounting" -> Department,
"seniors" -> Year). Distinct values are workbook metadata used on-device only;
no rows are sent anywhere.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from nlp.synonym_mapper import normalize_text


RESET = "reset"
CLEAR = "clear"
FOLLOWUP = "followup"
FRESH = "fresh"

_RESET_CUES = (
    "start over",
    "start again",
    "reset",
    "new search",
    "clear everything",
    "forget that",
    "forget everything",
    "brand new",
)
_CLEAR_CUES = (
    "clear that",
    "clear the filter",
    "clear filter",
    "clear filters",
    "remove the filter",
    "remove filter",
    "remove filters",
    "drop the filter",
    "no filters",
    "show all",
    "show everyone",
    "show everybody",
    "all students",
    "everyone",
)
# Words that signal "modify the current selection" rather than start fresh.
_FOLLOWUP_CUES = (
    "now ",
    "only ",
    "just ",
    "also ",
    "and ",
    "then ",
    "narrow",
    "within",
    "filter to",
    "filter down",
    "what about",
    "how about",
    "of those",
    "of these",
    "of them",
    "of their",
    "from those",
    "from these",
    "among them",
    "among those",
    "their ",
    "theirs",
    "them",
    "those",
    "these",
    "that group",
    "from that group",
    "in that department",
    "in that group",
    "drill",
    "further",
    "based on this",
    "based on these",
    "based on that",
    "based on the above",
    "based on all",
    "under each",
    "under those",
    "under these",
    "under which",
    "under their",
)

# Columns we never scan for value matches (handled elsewhere or sensitive).
_MAX_CATEGORICAL_CARDINALITY = 60


def classify_context_action(user_request: str) -> str:
    text = normalize_text(user_request)
    if any(cue in text for cue in _RESET_CUES):
        return RESET
    if any(cue in text for cue in _CLEAR_CUES):
        return CLEAR
    if _looks_like_followup(text) or is_additive(user_request):
        return FOLLOWUP
    return FRESH


def _looks_like_followup(text: str) -> bool:
    padded = f" {text} "
    return any(text.startswith(cue) or f" {cue}" in padded for cue in _FOLLOWUP_CUES)


# Verbs that mean "change/create something in the workbook" even inside a
# follow-up. ("sort"/"group" alone are treated as view refinements, not edits.)
_HARD_EDIT_CUES = (
    "highlight",
    "color",
    "colour",
    "move ",
    "put ",
    "send ",
    "copy ",
    "create",
    "make a",
    "make an",
    "build",
    "generate",
    "format",
    "reformat",
    "freeze",
    "autofit",
    "export",
    "save as",
    "download",
    "conditional format",
    "remove duplicate",
    "dedupe",
    "chart",
    "graph",
    "plot",
    "figure",
    "visualization",
    "visualisation",
    "report",
    "dashboard",
    "new sheet",
    "new tab",
    "another sheet",
    "another tab",
    "add a column",
    "add column",
    "flag",
)


def has_hard_edit_cue(user_request: str) -> bool:
    text = normalize_text(user_request)
    for cue in _HARD_EDIT_CUES:
        phrase = normalize_text(cue)
        if not phrase:
            continue
        pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"(?!\w)"
        if re.search(pattern, text):
            return True
    return False


_ADDITIVE_CUES = ("also", "include", " too", "as well", "add ", "plus ", "along with")
# Standalone-word cues that would otherwise match inside other words
# (e.g. "or" inside "professor"). Checked as whole tokens.
_ADDITIVE_WORD_CUES = ("or",)


def is_additive(user_request: str) -> bool:
    text = f" {normalize_text(user_request)} "
    if any(cue in text for cue in _ADDITIVE_CUES):
        return True
    tokens = set(text.split())
    return any(word in tokens for word in _ADDITIVE_WORD_CUES)


def compose_filters(
    active: list[dict[str, Any]],
    new: list[dict[str, Any]],
    action: str,
    additive: bool = False,
) -> list[dict[str, Any]]:
    """Combine the new message's filters with the active ones per the action.

    FOLLOWUP: a new filter on a column already present replaces that column's
    condition, unless `additive` ("include Biology too") — then the values are
    merged into an `in` filter so both are kept.
    """
    if action in (RESET, CLEAR, FRESH):
        return list(new)

    merged: list[dict[str, Any]] = [dict(f) for f in active]
    for condition in new:
        column = condition.get("column")
        matched = next((f for f in merged if f.get("column") == column), None)
        if matched is None:
            merged.append(dict(condition))
            continue
        if additive:
            values = _as_value_list(matched) + _as_value_list(condition)
            deduped = list(dict.fromkeys(values))
            matched["operator"] = "in"
            matched["value"] = deduped
        else:
            matched.update({k: condition[k] for k in condition})
            for key in list(matched):
                if key not in condition:
                    matched.pop(key, None)
    return merged


def _as_value_list(condition: dict[str, Any]) -> list[Any]:
    value = condition.get("value")
    if isinstance(value, list):
        return list(value)
    return [value] if value is not None else []


def detect_value_filters(
    user_request: str,
    frame: pd.DataFrame,
    skip_columns: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Detect equals/in filters by matching column distinct values to the text.

    Matches whole words/phrases, case-insensitive, with simple plural handling.
    Each value is attributed to only one column (first match wins) to avoid the
    same word filtering two columns (e.g. 'Accounting' as both Dept and Major).
    """
    skip = {normalize_text(c) for c in (skip_columns or set())}
    text = normalize_text(user_request)
    singular_text = _singularize(text)
    filters: list[dict[str, Any]] = []
    used_values: set[str] = set()
    used_columns: set[str] = set()

    for column in frame.columns:
        if normalize_text(column) in skip or column in used_columns:
            continue
        series = frame[column]
        is_text = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        if not is_text or pd.api.types.is_numeric_dtype(series):
            continue
        uniques = series.dropna().astype(str).unique()
        if len(uniques) == 0 or len(uniques) > _MAX_CATEGORICAL_CARDINALITY:
            continue
        for value in uniques:
            normalized_value = normalize_text(value)
            if not normalized_value or len(normalized_value) < 3 or normalized_value in used_values:
                continue
            singular_value = _singularize(normalized_value)
            if (
                _phrase_in(normalized_value, text)
                or _phrase_in(singular_value, text)
                or _phrase_in(normalized_value, singular_text)
                or _phrase_in(singular_value, singular_text)
            ):
                filters.append({"column": column, "operator": "equals", "value": value})
                used_values.add(normalized_value)
                used_columns.add(column)
                break

    return filters


def describe_filters(active: list[dict[str, Any]]) -> str:
    """Human-readable summary of the active filters for the UI context line."""
    if not active:
        return "none"
    parts = []
    for condition in active:
        column = condition.get("column", "?")
        operator = str(condition.get("operator", "")).replace("_", " ")
        value = condition.get("value")
        if condition.get("operator") in {"is_missing", "is_not_missing"}:
            parts.append(f"{column} {operator}")
        elif isinstance(value, list):
            parts.append(f"{column} in [{', '.join(map(str, value))}]")
        else:
            parts.append(f"{column} {operator} {value}")
    return "; ".join(parts)


_CONFIRM_YES = ("yes", "confirm", "confirmed", "go ahead", "do it", "proceed", "export it", "show them", "show it", "yep", "sure", "ok", "okay")
_CONFIRM_NO = ("no", "cancel", "never mind", "nevermind", "stop", "abort", "do not", "dont", "keep them hidden", "keep hidden", "no thanks")
_CONFIRM_LEAD = {"yes", "no", "confirm", "cancel", "ok", "okay", "yep", "sure", "stop", "proceed", "abort", "nevermind"}


def classify_confirmation(user_request: str) -> str:
    """Return 'yes', 'no', or 'unclear' for a pending-action response.

    Only short, essentially-bare confirmations count. A full query that merely
    contains "no" (e.g. "show students with no advisor") is 'unclear', so it
    cannot accidentally cancel or approve a pending action.
    """
    text = normalize_text(user_request)
    words = text.split()
    if not words:
        return "unclear"
    # Long messages are treated as new requests, not confirmations.
    if len(words) > 4 and words[0] not in _CONFIRM_LEAD:
        return "unclear"

    def _matches(cues: tuple[str, ...]) -> bool:
        return any(text == cue or text.startswith(cue + " ") for cue in cues) or words[0] in cues

    if _matches(_CONFIRM_NO) or words[0] == "no":
        return "no"
    if _matches(_CONFIRM_YES):
        return "yes"
    return "unclear"


def is_bare_context_command(user_request: str) -> bool:
    """True when the message is only a reset/clear command with no other query."""
    text = normalize_text(user_request)
    for cue in _RESET_CUES + _CLEAR_CUES:
        text = text.replace(cue, " ")
    leftover = [
        word
        for word in text.split()
        if word not in {"the", "that", "filter", "filters", "please", "and", "all", "everything", "context", "now", "ok", "okay"}
    ]
    return len(leftover) == 0


def _phrase_in(phrase: str, text: str) -> bool:
    if not phrase:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def _singularize(text: str) -> str:
    # Irregular plurals: -men → -man (freshmen, women, etc.) — runs first so the
    # generic -s/-ies rules below don't accidentally strip the m/n.
    text = re.sub(r"\b([a-z]+)men\b", r"\1man", text)
    text = re.sub(r"\b([a-z]+)ies\b", r"\1y", text)
    return re.sub(r"\b([a-z]{3,}?)s\b", r"\1", text)
