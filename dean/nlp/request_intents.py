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
