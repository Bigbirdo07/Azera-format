"""Deterministic detection of action-style requests (export / note / field update).

Shared by the chat layer and the planner router so both agree on what counts as
an export, a note edit, or a field update.
"""

from __future__ import annotations

import re

from nlp.synonym_mapper import normalize_text


_EXPORT_CUES = ("export", "download this", "download the", "save this list", "save the list",
                "create a file", "save as excel", "give me the final", "final sheet",
                "create the updated workbook", "updated workbook")
_NOTE_CUES = ("add a note", "add note", "add notes", "note ", "flag these", "flag them for", "mark these for", "mark them for")
# School-roster "mark these students under Academic Watch" / "flag these students" /
# "put them on watch" / "mark as follow up needed" / "set Academic Watch to Yes".
_ACADEMIC_WATCH_CUES = (
    "academic watch", "watch list", "watchlist", "put them on watch",
    "put these on watch", "put those on watch", "put on watch",
    "follow up needed", "follow-up needed",
    "mark as follow up", "mark as follow-up",
    "mark them as flagged", "flag these students", "flag those students",
    "flag them", "flag these", "intervention needed",
    "mark these students", "mark those students",
)
# Attendance Watch is its own concept (separate column, separate workflow) —
# detect explicit attendance-watch phrasings so the action writes to the
# Attendance Watch column instead of the Academic Watch column.
_ATTENDANCE_WATCH_CUES = (
    "attendance watch", "attendance flag", "attendance intervention",
    "mark as attendance watch", "put on attendance watch",
    "put them on attendance watch", "flag for attendance",
)
_READ_ONLY_CUES = (
    "show", "list", "how many", "count", "which", "who", "what", "average",
    "avg", "mean", "group", "sort", "top", "bottom", "find",
)
_WATCH_ACTION_VERBS = (
    "mark", "set", "put", "flag", "add", "create", "make",
)


def is_export_request(request: str) -> bool:
    text = normalize_text(request)
    return any(cue in text for cue in _EXPORT_CUES)


def is_note_request(request: str) -> bool:
    text = normalize_text(request)
    # Academic Watch verbs ("mark these students under academic watch",
    # "flag them", "put them on watch") are NOT generic note adds — they go
    # through is_academic_watch_request and the dedicated action below.
    if is_academic_watch_request(request):
        return False
    return any(cue in text for cue in _NOTE_CUES)


_EXPLICIT_NOTE_PREFIXES = ("add a note", "add note", "add notes", "note:", "comment:")


def is_academic_watch_request(request: str) -> bool:
    """True if the message asks to set the Academic Watch / Follow Up Needed
    flag on the currently selected students.

    Explicit note prefixes ("add note: ...", "note: ...") win — those are
    note-edit requests regardless of the note content.
    Attendance-watch phrasings (handled by ``is_attendance_watch_request``)
    are excluded so they don't also trip the Academic Watch path.
    """
    text = normalize_text(request)
    if any(prefix in text for prefix in _EXPLICIT_NOTE_PREFIXES):
        return False
    if is_attendance_watch_request(request):
        return False
    if any(cue in text for cue in _READ_ONLY_CUES) and not any(
        re.search(rf"(?<!\w){verb}(?!\w)", text) for verb in _WATCH_ACTION_VERBS
    ):
        return False
    return any(cue in text for cue in _ACADEMIC_WATCH_CUES)


def is_attendance_watch_request(request: str) -> bool:
    """True if the message asks to mark the Attendance Watch flag.

    Separate from Academic Watch so the action writes to the right column;
    the underlying execution code path is shared (just a different
    ``column_name``).
    """
    text = normalize_text(request)
    if any(prefix in text for prefix in _EXPLICIT_NOTE_PREFIXES):
        return False
    if any(cue in text for cue in _READ_ONLY_CUES) and not any(
        re.search(rf"(?<!\w){verb}(?!\w)", text) for verb in _WATCH_ACTION_VERBS
    ):
        return False
    return any(cue in text for cue in _ATTENDANCE_WATCH_CUES)


def parse_note(request: str) -> str:
    match = re.search(r"(?:add\s+a?\s*notes?|notes?|comment)\s*[:\-]\s*(.+)$", request, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?:flag|mark)\s+(?:them|these|those)?\s*(?:students?\s+)?for\s+(.+)$", request, re.IGNORECASE)
    if match:
        return f"For {match.group(1).strip()}"
    return ""


def parse_field_update(request: str) -> tuple[str, str] | None:
    match = re.search(
        r"\b(?:set|change|update)\s+(?:their\s+|the\s+|these\s+students?\s+|those\s+students?\s+)?(.+?)\s+to\s+(.+)$",
        request,
        re.IGNORECASE,
    )
    if not match:
        return None
    field_name = match.group(1).strip().strip(",")
    value = match.group(2).strip().rstrip(".")
    if not field_name or not value:
        return None
    return field_name, value


# Chat-driven risk-threshold updates ("change the GPA risk threshold to
# 2.5"). Ordered specific-before-generic throughout: "severe attendance
# risk" before the bare "attendance risk" entry, and every "psat ..." cue
# before the corresponding "sat ..." cue (real bug caught by this file's own
# tests: "psat math benchmark" literally contains "sat math benchmark" as a
# substring, so checking "sat math benchmark" first misresolved "set the
# psat math benchmark to 480" to sat_math_benchmark_threshold instead of
# psat_math_benchmark_threshold) -- same class of bug as this session's
# _NUMERIC_CONCEPTS ordering fix.
_RISK_SETTING_FIELD_CUES: tuple[tuple[str, str], ...] = (
    ("severe attendance risk", "severe_attendance_risk_threshold"),
    ("severe attendance threshold", "severe_attendance_risk_threshold"),
    ("attendance risk", "attendance_risk_threshold"),
    ("attendance threshold", "attendance_risk_threshold"),
    ("gpa risk", "gpa_risk_threshold"),
    ("gpa threshold", "gpa_risk_threshold"),
    ("unexcused absence concern", "unexcused_absence_concern"),
    ("unexcused absences threshold", "unexcused_absence_concern"),
    ("unexcused absence threshold", "unexcused_absence_concern"),
    ("tardy concern", "tardy_concern"),
    ("tardies threshold", "tardy_concern"),
    ("tardy threshold", "tardy_concern"),
    ("high risk signal count", "high_risk_signal_count"),
    ("high risk signal threshold", "high_risk_signal_count"),
    ("moderate risk signal count", "moderate_risk_signal_count"),
    ("moderate risk signal threshold", "moderate_risk_signal_count"),
    ("psat math benchmark", "psat_math_benchmark_threshold"),
    ("psat reading writing benchmark", "psat_reading_writing_benchmark_threshold"),
    ("psat reading benchmark", "psat_reading_writing_benchmark_threshold"),
    ("sat math benchmark", "sat_math_benchmark_threshold"),
    ("sat ebrw benchmark", "sat_ebrw_benchmark_threshold"),
    ("sat reading benchmark", "sat_ebrw_benchmark_threshold"),
)
_RISK_SETTING_VERBS = ("set", "change", "update", "lower", "raise", "increase", "decrease", "make")
_TRAILING_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")


def is_risk_setting_update_request(request: str) -> bool:
    text = normalize_text(request)
    if not any(re.search(rf"(?<!\w){verb}(?!\w)", text) for verb in _RISK_SETTING_VERBS):
        return False
    return any(cue in text for cue, _field in _RISK_SETTING_FIELD_CUES)


def parse_risk_setting_update(request: str) -> tuple[str, float] | None:
    """Parse "change the GPA risk threshold to 2.5" into a (field, value)
    pair validated against RiskSettings' real field names. Returns None
    when no known threshold cue or no number resolves -- the caller then
    falls through to existing behavior (e.g. parse_field_update) untouched.
    """
    if not is_risk_setting_update_request(request):
        return None
    text = normalize_text(request)
    field = None
    for cue, mapped_field in _RISK_SETTING_FIELD_CUES:
        if cue in text:
            field = mapped_field
            break
    if not field:
        return None
    # "to <value>" is the common shape ("change X to 2.5"); fall back to the
    # last bare number in the message ("make the gpa threshold 2.5").
    to_match = re.search(r"\bto\s+(-?\d+(?:\.\d+)?)\b", text)
    if to_match:
        return field, float(to_match.group(1))
    numbers = _TRAILING_NUMBER_RE.findall(text)
    if numbers:
        return field, float(numbers[-1])
    return None
