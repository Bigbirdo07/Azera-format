"""Durable glossary for the free-form analyst.

The analyst answers better when it shares the user's vocabulary: "struggling"
means GPA < 2.5 here, "advisor" is the column called "Counselor", and so on.
Rather than stand up a parallel store, this rides the existing
``core.correction_manager`` learning tables (already persisted to SQLite, synced
to JSON, and carried in the export/import learning pack):

- a glossary *definition* ("struggling = GPA < 2.5") is stored as a learned
  synonym whose ``mapped_concept`` holds the definition text and whose source is
  ``GLOSSARY_SOURCE``;
- a column *alias* ("advisor" -> column "Counselor") reuses learned column
  mappings as-is.

All reads are best-effort: if the database is unavailable (offline/tests) the
glossary is simply empty and the analyst runs without it.
"""

from __future__ import annotations

import re
from typing import Any

GLOSSARY_SOURCE = "user_glossary"

# Explicit teach forms — phrased so plainly they never hijack a real question,
# so they skip the question-word guard below.
_EXPLICIT_TEACH_RES = (
    re.compile(r"^\s*define\s+(?P<phrase>.+?)\s+as\s+(?P<definition>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*when\s+i\s+say\s+(?P<phrase>.+?)[, ]+\s*(?:i\s+mean|use|treat\s+it\s+as|it\s+means)\s+(?P<definition>.+?)\s*$", re.IGNORECASE),
)
# Bare "X means Y" / "X = Y" — convenient but ambiguous, so it is gated by the
# question-word guard to avoid catching statements/questions.
_BARE_TEACH_RE = re.compile(
    r"^\s*(?:the\s+term\s+)?(?P<phrase>.+?)\s+(?:means|=)\s+(?P<definition>.+?)\s*$",
    re.IGNORECASE,
)
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(how|what|which|who|when|where|why|show|list|count|find|is|are|do|does)\b",
    re.IGNORECASE,
)


def _valid_pair(phrase: str, definition: str) -> tuple[str, str] | None:
    phrase = (phrase or "").strip().strip("\"'")
    definition = (definition or "").strip()
    # A term is a few words at most; a definition must be non-trivial.
    if phrase and definition and len(phrase.split()) <= 5 and len(definition) >= 2:
        return phrase, definition
    return None


def parse_teach(message: str) -> tuple[str, str] | None:
    """Detect a glossary-teaching command. Returns (phrase, definition) or None."""
    text = (message or "").strip()
    if not text or text.endswith("?"):
        return None
    for pattern in _EXPLICIT_TEACH_RES:
        match = pattern.search(text)
        if match:
            pair = _valid_pair(match.group("phrase"), match.group("definition"))
            if pair:
                return pair
    if _QUESTION_LEAD_RE.match(text):
        return None
    match = _BARE_TEACH_RE.search(text)
    if match:
        return _valid_pair(match.group("phrase"), match.group("definition"))
    return None


def teach_term(phrase: str, definition: str) -> None:
    """Persist a glossary definition via the learning store (best-effort)."""
    try:
        from core.correction_manager import add_learned_synonym, sync_learning_files

        add_learned_synonym(phrase=phrase, mapped_concept=definition, source=GLOSSARY_SOURCE)
        sync_learning_files()
    except Exception:
        pass  # offline / DB unavailable — teaching is best-effort


def _safe_rows(getter) -> list[dict[str, Any]]:
    try:
        return getter() or []
    except Exception:
        return []


def get_glossary_terms() -> list[tuple[str, str]]:
    from core.correction_manager import get_learned_synonyms

    return [
        (row["phrase"], row["mapped_concept"])
        for row in _safe_rows(get_learned_synonyms)
        if row.get("phrase") and row.get("mapped_concept")
    ]


def get_column_aliases() -> list[tuple[str, str]]:
    from core.correction_manager import get_learned_column_mappings

    return [
        (row["raw_column_name"], row["standard_concept"])
        for row in _safe_rows(get_learned_column_mappings)
        if row.get("raw_column_name") and row.get("standard_concept")
    ]


def build_glossary_block(max_terms: int = 20) -> str:
    """Render the glossary as a prompt block, or "" when nothing is learned."""
    terms = get_glossary_terms()[:max_terms]
    aliases = get_column_aliases()[:max_terms]
    if not terms and not aliases:
        return ""
    lines = ["Glossary — apply these definitions whenever the user's wording appears:"]
    for phrase, definition in terms:
        lines.append(f'- "{phrase}" means: {definition}')
    for raw, concept in aliases:
        lines.append(f'- when the user says "{concept}", use the column "{raw}"')
    return "\n".join(lines) + "\n"
