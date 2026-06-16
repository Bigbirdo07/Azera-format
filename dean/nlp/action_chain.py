"""Phase Q: parse single-message action chains.

The school-office workflow case is:

    "Mark these students Academic Watch and export me a new Excel sheet"
    "Add note: advisor follow-up needed and export"
    "Set Follow Up Needed to Yes and export"
    "Flag them for follow-up and download this as Excel"

The parser only handles **edit-then-export** chains for now (per the spec).
Anything more elaborate falls back to single-intent routing.

The function returns ``None`` when the message is a single action so the
existing `_classify_action_intent` plumbing stays in charge for those cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nlp.request_intents import (
    _ACADEMIC_WATCH_CUES,
    _ATTENDANCE_WATCH_CUES,
    is_academic_watch_request,
    is_attendance_watch_request,
    is_export_request,
    is_note_request,
    parse_field_update,
    parse_note,
)
from nlp.synonym_mapper import normalize_text


# Joining words that link the edit phrase to the export phrase. We accept
# bare "and"/"then" plus a few longer forms.
# Joiners between the edit phrase and the export phrase. We deliberately
# avoid " and export " / " then export " here so the export verb survives
# into the right-side phrase for is_export_request().
_CHAIN_JOINS = (
    " and then ", " then ", " and after that ", " after that ",
    " and also ", " also ", " and ", " plus ",
)


@dataclass
class ChainedAction:
    """One step in a parsed action chain."""
    type: str                       # "academic_watch" | "note_edit" | "field_update" | "export"
    payload: dict[str, Any]         # per-type details
    raw_phrase: str = ""            # the substring of the user message that produced this


def parse_action_chain(message: str) -> list[ChainedAction] | None:
    """Return the chain when the message combines an edit-style verb with an
    export. Returns None for single-action messages.

    Recognized edit verbs (each routes to the action it triggers):
      - mark these students academic watch / flag them / put them on watch / set academic watch
      - add a note: ... / note: ...
      - set <field> to <value>

    Followed by any export phrase:
      - export / export me / export the/this list / download this as excel /
        give me the final / create the updated workbook
    """
    if not message:
        return None
    text = normalize_text(message)
    if not text:
        return None
    if not _looks_chained(text):
        return None

    edit_phrase, export_phrase = _split_at_join(text, message)
    if not edit_phrase or not export_phrase:
        return None
    # The right-hand side must actually be an export verb on its own.
    if not is_export_request(export_phrase):
        return None

    edit_action = _classify_edit_step(message, edit_phrase)
    if edit_action is None:
        return None

    return [
        edit_action,
        ChainedAction(type="export", payload={"target": "updated_workbook"},
                      raw_phrase=export_phrase.strip()),
    ]


# ---- helpers ---------------------------------------------------------------


def _looks_chained(text: str) -> bool:
    """Cheap pre-check: at least one chain join AND at least one export cue
    AND at least one edit cue (academic_watch / note / field_update verb)."""
    padded = f" {text} "
    if not any(join in padded for join in _CHAIN_JOINS):
        return False
    if not _has_export_cue(padded):
        return False
    if not _has_edit_cue(padded):
        return False
    return True


_EXPORT_TOKEN_CUES = (
    "export", "download", "excel sheet", "save this list", "save the list",
    "create a file", "final sheet", "updated workbook",
)


def _has_export_cue(padded_text: str) -> bool:
    return any(cue in padded_text for cue in _EXPORT_TOKEN_CUES)


def _has_edit_cue(padded_text: str) -> bool:
    if any(f" {cue} " in padded_text or padded_text.startswith(f"{cue} ")
           or padded_text.endswith(f" {cue}")
           for cue in _ACADEMIC_WATCH_CUES + _ATTENDANCE_WATCH_CUES):
        return True
    if "add a note" in padded_text or "add note" in padded_text or "note:" in padded_text:
        return True
    if re.search(r"\b(?:set|change|update)\s+.+?\s+to\s+", padded_text):
        return True
    return False


def _split_at_join(text: str, original_message: str) -> tuple[str, str]:
    """Return (edit_phrase, export_phrase) split at the chain joiner.

    Returns the splits from the *original* message (case-preserved) so that
    downstream parsers like parse_note / parse_field_update still see proper
    quoting and casing. Splits at the LAST join that has an export verb to
    the right — that way "mark academic watch and follow up and export"
    finds the right cleavage.
    """
    lowered = original_message.lower()
    padded = f" {lowered} "
    best_index = -1
    for join in sorted(_CHAIN_JOINS, key=len, reverse=True):
        # Look from the right so the LAST join wins; that maximizes the
        # edit-side phrase and minimizes the export side.
        position = padded.rfind(join)
        if position < 0:
            continue
        right_side = padded[position + len(join):].strip()
        if _has_export_cue(" " + right_side + " "):
            # Reconstruct in original (case-preserved) message.
            real_position = position - 1  # padding offset
            if real_position > best_index:
                best_index = real_position
                best_join = join
    if best_index < 0:
        return "", ""
    left = original_message[:best_index].strip()
    right = original_message[best_index + len(best_join):].strip()
    return left, right


def _classify_edit_step(original_message: str, edit_phrase: str) -> ChainedAction | None:
    """Pick the right edit-action shape from the edit-side phrase."""
    if is_attendance_watch_request(edit_phrase):
        return ChainedAction(
            type="attendance_watch",
            payload={"value": "Yes", "column_hint": "Attendance Watch"},
            raw_phrase=edit_phrase,
        )
    if is_academic_watch_request(edit_phrase):
        # Honor follow-up-needed wording → set that column instead.
        column = "Academic Watch"
        text_low = edit_phrase.lower()
        if "follow up" in text_low or "follow-up" in text_low or "followup" in text_low:
            column = "Follow Up Needed"
        return ChainedAction(
            type="academic_watch",
            payload={"value": "Yes", "column_hint": column},
            raw_phrase=edit_phrase,
        )

    if is_note_request(edit_phrase):
        return ChainedAction(
            type="note_edit",
            payload={"note": parse_note(edit_phrase)},
            raw_phrase=edit_phrase,
        )

    field_update = parse_field_update(edit_phrase)
    if field_update is not None:
        field, value = field_update
        return ChainedAction(
            type="field_update",
            payload={"field": field, "value": value},
            raw_phrase=edit_phrase,
        )
    return None
