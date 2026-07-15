"""Question library — loader, workbook-aware filter, and lookup utilities.

The library lives in ``knowledge/question_library.json`` and groups questions
by category. Each question declares ``requires_columns`` as a list of concept
names (looked up via ``knowledge/synonyms.json``) or literal column names.
A question is *askable* on a given workbook only if every required concept
resolves to a real column in that workbook.

The chat UI uses this in two places:
  - Render the suggested-question chips (filtered to what's askable today).
  - When a user clicks a chip, stash its ``id`` so the post-answer suggestion
    list can be replaced by the entry's curated ``follow_ups`` instead of the
    dynamic suggester.

This module is read-only and has no Streamlit dependency, so it's easy to
unit-test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from nlp.synonym_mapper import (
    KNOWLEDGE_DIR,
    load_json,
    load_synonyms_with_learned,
    match_column_by_terms,
    match_column_for_concept,
    normalize_text,
)


LIBRARY_PATH = KNOWLEDGE_DIR / "question_library.json"
_CONCEPT_MATCH_THRESHOLD = 0.55


# ---- data shapes -----------------------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    requires_columns: tuple[str, ...]
    follow_ups: tuple[str, ...]


@dataclass(frozen=True)
class Category:
    id: str
    title: str
    blurb: str
    questions: tuple[Question, ...]


@dataclass
class AskableCategory:
    """A category filtered to questions whose required columns are present."""
    id: str
    title: str
    blurb: str
    questions: list[Question] = field(default_factory=list)


# ---- loading ---------------------------------------------------------------


@lru_cache(maxsize=1)
def _raw_library(path_str: str) -> dict[str, Any]:
    """Cached raw library load — keyed by path so tests with a tmp path work."""
    path = Path(path_str)
    if not path.exists():
        return {"version": 1, "categories": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_library(path: Path | None = None) -> list[Category]:
    """Return the library as immutable Category objects."""
    raw = _raw_library(str(path or LIBRARY_PATH))
    categories: list[Category] = []
    for category in raw.get("categories", []):
        questions = tuple(
            Question(
                id=str(entry.get("id", "")),
                text=str(entry.get("text", "")).strip(),
                requires_columns=tuple(str(c) for c in entry.get("requires_columns", [])),
                follow_ups=tuple(str(f) for f in entry.get("follow_ups", [])),
            )
            for entry in category.get("questions", [])
            if entry.get("text") and entry.get("id")
        )
        categories.append(Category(
            id=str(category.get("id", "")),
            title=str(category.get("title", "")),
            blurb=str(category.get("blurb", "")),
            questions=questions,
        ))
    return categories


def clear_cache() -> None:
    """Reset the LRU cache (used by tests to swap fixtures)."""
    _raw_library.cache_clear()


# ---- column resolution -----------------------------------------------------


def _resolve_required(
    required: str,
    columns: list[str],
    synonyms: dict[str, list[str]] | None,
) -> bool:
    """Decide whether ``required`` (a concept token or literal column name) is
    present in the workbook's columns."""
    if not required:
        return True
    required = required.strip()
    # Direct case-insensitive column-name match.
    normalized_required = normalize_text(required)
    for column in columns:
        if normalize_text(column) == normalized_required:
            return True
    # Synonyms-based concept resolution.
    if synonyms is not None:
        column, score = match_column_for_concept(required, columns, synonyms)
        if column and score >= _CONCEPT_MATCH_THRESHOLD:
            return True
    # Last-ditch fuzzy match on the required token alone.
    column, score = match_column_by_terms([required], columns)
    return bool(column and score >= _CONCEPT_MATCH_THRESHOLD)


def askable_categories(
    columns: list[str],
    *,
    synonyms: dict[str, list[str]] | None = None,
    library: list[Category] | None = None,
) -> list[AskableCategory]:
    """Return the library trimmed to questions answerable on ``columns``.

    Categories with no surviving questions are dropped entirely.
    """
    if synonyms is None:
        synonyms = load_synonyms_with_learned()
    cats = library if library is not None else load_library()
    askable: list[AskableCategory] = []
    for category in cats:
        keep = [
            question for question in category.questions
            if all(_resolve_required(req, columns, synonyms) for req in question.requires_columns)
        ]
        if keep:
            askable.append(AskableCategory(
                id=category.id, title=category.title, blurb=category.blurb,
                questions=keep,
            ))
    return askable


# ---- lookups for follow-up routing ----------------------------------------


def lookup_by_id(
    template_id: str,
    *,
    library: list[Category] | None = None,
) -> Question | None:
    if not template_id:
        return None
    for category in (library if library is not None else load_library()):
        for question in category.questions:
            if question.id == template_id:
                return question
    return None


def lookup_for_message(
    user_message: str,
    *,
    library: list[Category] | None = None,
) -> Question | None:
    """Best-effort match of a free-typed message to a library entry.

    Uses normalized exact match first, then a "all-tokens-of-template-present"
    check. Anything fuzzier is left to the dynamic suggester so we don't
    accidentally route an unrelated question through curated follow-ups.
    """
    if not user_message:
        return None
    target = normalize_text(user_message)
    if not target:
        return None
    target_tokens = set(target.split())
    cats = library if library is not None else load_library()
    # Pass 1: exact normalized match.
    for category in cats:
        for question in category.questions:
            if normalize_text(question.text) == target:
                return question
    # Pass 2: all template tokens present in the message and short enough that
    # this isn't a long unrelated sentence.
    if len(target_tokens) > 12:
        return None
    for category in cats:
        for question in category.questions:
            template_tokens = set(normalize_text(question.text).split())
            if template_tokens and template_tokens.issubset(target_tokens):
                return question
    return None
