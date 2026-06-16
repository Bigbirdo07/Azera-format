from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"


def load_json(file_name: str) -> dict[str, Any]:
    path = KNOWLEDGE_DIR / file_name
    if not path.exists():
        return {} if file_name.endswith(".json") else {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json_list(file_name: str) -> list[dict[str, Any]]:
    path = KNOWLEDGE_DIR / file_name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def normalize_text(value: str) -> str:
    text = str(value).lower()
    # Preserve decimal points between digits so "2.5" survives normalization.
    text = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)
    text = re.sub(r"[^a-z0-9\x00]+", " ", text)
    text = text.replace("\x00", ".")
    return re.sub(r"\s+", " ", text).strip()


def concept_from_text(text: str, synonyms: dict[str, list[str]]) -> tuple[str | None, float]:
    normalized_text = normalize_text(text)
    best_concept: str | None = None
    best_score = 0.0

    for concept, phrases in synonyms.items():
        for phrase in phrases:
            normalized_phrase = normalize_text(phrase)
            if not normalized_phrase:
                continue
            if normalized_phrase in normalized_text:
                score = min(0.95, 0.55 + (len(normalized_phrase) / max(len(normalized_text), 1)))
                if score > best_score:
                    best_concept = concept
                    best_score = score

    return best_concept, best_score


# Concept equivalence classes — when a user-named concept has no column on
# this workbook, we try the other concepts in the same class as a graceful
# fallback. This is what lets "teacher" / "professor" land on an Advisor
# column when the roster doesn't have a dedicated Teacher field.
#
# Membership is bidirectional within a class: any concept in the class can
# stand in for any other. We never overwrite a direct match — fallback only
# fires when the primary concept produced no result.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    # Faculty / advising entity. "teacher" / "professor" / "instructor" all
    # mean the same person on most school rosters; "advisor" is the same
    # concept under a different organizational label.
    "teacher":    ("advisor",),
    "professor":  ("advisor", "teacher"),
    "instructor": ("advisor", "teacher"),
    "faculty":    ("advisor", "teacher"),
    "advisor":    ("teacher",),
    # Discipline / department / school of study. "department" → "discipline"
    # is the dean-office convention; "school" is occasionally the same.
    "department": ("discipline", "school"),
    "discipline": ("department",),
    # When a workbook has no course/class section column, "class" is usually
    # the student's class year in dean-roster usage.
    "course":     ("year",),
}


def match_column_for_concept(
    concept: str,
    columns: list[str],
    synonyms: dict[str, list[str]],
) -> tuple[str | None, float]:
    """Resolve a concept token to a real column. Direct synonyms first;
    aliases from CONCEPT_ALIASES are NOT considered here. Use
    ``match_column_for_concept_with_fallback`` when you want the fallback
    behavior plus an audit trail."""
    if not concept:
        return None, 0.0

    concept_terms = [concept.replace("_", " "), *synonyms.get(concept, [])]
    return match_column_by_terms(concept_terms, columns)


def match_column_for_concept_with_fallback(
    concept: str,
    columns: list[str],
    synonyms: dict[str, list[str]],
    *,
    min_score: float = 0.55,
) -> tuple[str | None, float, str | None]:
    """Resolve a concept with cross-concept fallback.

    Returns ``(column, score, fallback_from_concept)``. ``fallback_from`` is
    None when the direct concept matched, or the original concept name when
    a member of CONCEPT_ALIASES[concept] resolved instead. The caller can
    use this signal to emit an assumption_note like
    "I interpreted teacher as Advisor because no Teacher column was found."
    """
    column, score = match_column_for_concept(concept, columns, synonyms)
    if column and score >= min_score:
        return column, score, None

    for alias in CONCEPT_ALIASES.get(concept, ()):
        alt_column, alt_score = match_column_for_concept(alias, columns, synonyms)
        if alt_column and alt_score >= min_score:
            return alt_column, alt_score, concept

    return column, score, None


def match_column_by_terms(terms: list[str], columns: list[str]) -> tuple[str | None, float]:
    best_column: str | None = None
    best_score = 0.0
    normalized_columns = {column: normalize_text(column) for column in columns}

    for column, normalized_column in normalized_columns.items():
        for term in terms:
            normalized_term = normalize_text(term)
            if not normalized_term:
                continue
            if normalized_column == normalized_term:
                score = 1.0
            elif normalized_term in normalized_column or normalized_column in normalized_term:
                score = 0.88
            else:
                term_tokens = set(normalized_term.split())
                column_tokens = set(normalized_column.split())
                overlap = len(term_tokens & column_tokens)
                score = overlap / max(len(term_tokens | column_tokens), 1)

            if score > best_score:
                best_column = column
                best_score = score

    if best_score < 0.35:
        return None, best_score
    return best_column, best_score
