"""Follow-up reference resolver.

Lets the assistant understand short follow-ups that lean on the previous turn:
"highlight them", "move those students", "make a chart of that". It maps the
reference to the last result set held in core.session_memory. If there is no
previous result to point at, it asks the user to clarify instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.session_memory import SessionMemory


# Whole-word reference tokens.
_REFERENCE_WORDS = (
    "them",
    "those",
    "these",
    "that",
    "ones",
    "they",
)
# Multi-word reference phrases.
_REFERENCE_PHRASES = (
    "those students",
    "these students",
    "same group",
    "same students",
    "the same",
    "previous result",
    "the ones",
    "the ones from before",
    "from before",
    "that group",
    "that result",
)


@dataclass
class FollowupResolution:
    is_followup: bool
    resolved: bool
    needs_clarification: bool = False
    clarification_question: str = ""
    filters: list[dict[str, Any]] = field(default_factory=list)
    sheet: str = ""
    description: str = ""
    referent_note: str = ""


_QUESTION_WORDS = {
    "how", "which", "who", "whose", "what", "where", "when", "why",
    "are", "is", "do", "does", "can",
}
# Words that carry no standalone meaning in a bare reference command.
_FILLER_WORDS = {
    "the", "a", "an", "to", "of", "out", "from", "in", "on", "for", "with",
    "those", "these", "that", "them", "they", "ones", "one", "same", "group",
    "groups", "students", "student", "rows", "row", "people", "previous",
    "result", "results", "before", "now", "please", "and", "then", "all",
}


def has_reference(user_request: str) -> bool:
    text = _normalize(user_request)
    if any(phrase in text for phrase in _REFERENCE_PHRASES):
        return True
    tokens = set(re.findall(r"[a-z]+", text))
    return any(word in tokens for word in _REFERENCE_WORDS)


def is_bare_reference(user_request: str) -> bool:
    """True only when the message leans entirely on a prior result (e.g.
    "move them", "highlight those"). A self-contained question or richer command
    is NOT a bare reference, even if it happens to contain "those"/"that"."""
    words = re.findall(r"[a-z]+", _normalize(user_request))
    if any(word in _QUESTION_WORDS for word in words):
        return False
    content = [word for word in words if word not in _FILLER_WORDS]
    return len(content) <= 2


def resolve_followup(user_request: str, memory: SessionMemory) -> FollowupResolution:
    if not has_reference(user_request):
        return FollowupResolution(is_followup=False, resolved=False)

    # We can only act on "them"/"those" if the previous turn produced an actual
    # row selection (filters). A whole-sheet summary (e.g. "which columns have
    # missing values") gives nothing concrete to highlight or move.
    if not memory.last_filters:
        return FollowupResolution(
            is_followup=True,
            resolved=False,
            needs_clarification=True,
            clarification_question="Which students or rows do you mean? I don't have a specific selection from before to act on.",
        )

    description = memory.last_result_description or memory.last_request or "the previous result"
    note = f"Refers to the previous result: {description}."
    return FollowupResolution(
        is_followup=True,
        resolved=True,
        filters=list(memory.last_filters),
        sheet=memory.last_sheet,
        description=description,
        referent_note=note,
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()
