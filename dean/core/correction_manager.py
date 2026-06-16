from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.paths import data_dir
from database.db import execute_insert, execute_write, fetch_all, initialize_database


KNOWLEDGE_DIR = data_dir("knowledge")
EXPORTS_DIR = data_dir("exports")
LEARNED_SYNONYMS_PATH = KNOWLEDGE_DIR / "learned_synonyms.json"
LEARNED_COLUMN_MAPPINGS_PATH = KNOWLEDGE_DIR / "learned_column_mappings.json"


def save_correction(
    *,
    request_id: int | None,
    original_request: str,
    incorrect_command: dict[str, Any],
    corrected_command: dict[str, Any],
    correction_type: str,
    better_phrase: str | None = None,
    mapped_concept: str | None = None,
    raw_column_name: str | None = None,
) -> int:
    correction_id = execute_insert(
        """
        INSERT INTO corrections (
            request_id,
            original_request,
            incorrect_command_json,
            corrected_command_json,
            correction_type,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            original_request,
            json.dumps(incorrect_command, sort_keys=True),
            json.dumps(corrected_command, sort_keys=True),
            correction_type,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    if better_phrase and mapped_concept:
        add_learned_synonym(
            phrase=better_phrase,
            mapped_concept=mapped_concept,
            source=f"correction:{correction_id}",
        )

    if raw_column_name and mapped_concept:
        add_learned_column_mapping(
            raw_column_name=raw_column_name,
            standard_concept=mapped_concept,
            confidence=0.95,
            source=f"correction:{correction_id}",
        )

    sync_learning_files()
    return correction_id


def add_learned_synonym(phrase: str, mapped_concept: str, source: str) -> None:
    execute_write(
        """
        INSERT OR IGNORE INTO learned_synonyms (phrase, mapped_concept, source, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            phrase.strip(),
            mapped_concept.strip(),
            source,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def add_learned_column_mapping(
    raw_column_name: str,
    standard_concept: str,
    confidence: float,
    source: str,
) -> None:
    execute_write(
        """
        INSERT OR IGNORE INTO learned_column_mappings (
            raw_column_name,
            standard_concept,
            confidence,
            source,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            raw_column_name.strip(),
            standard_concept.strip(),
            confidence,
            source,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def sync_learning_files() -> None:
    initialize_database()
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    LEARNED_SYNONYMS_PATH.write_text(
        json.dumps(get_learned_synonyms(), indent=2),
        encoding="utf-8",
    )
    LEARNED_COLUMN_MAPPINGS_PATH.write_text(
        json.dumps(get_learned_column_mappings(), indent=2),
        encoding="utf-8",
    )


def get_learned_synonyms() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT phrase, mapped_concept, source, created_at
        FROM learned_synonyms
        ORDER BY created_at DESC, id DESC
        """
    )


def get_learned_column_mappings() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT raw_column_name, standard_concept, confidence, source, created_at
        FROM learned_column_mappings
        ORDER BY created_at DESC, id DESC
        """
    )


def get_correction_examples() -> list[dict[str, Any]]:
    return fetch_all(
        """
        SELECT
            original_request,
            incorrect_command_json,
            corrected_command_json,
            correction_type,
            created_at
        FROM corrections
        ORDER BY created_at DESC, id DESC
        """
    )


def export_learning_pack() -> Path:
    initialize_database()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORTS_DIR / "local_learning_pack.json"
    learning_pack = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "learned_synonyms": get_learned_synonyms(),
        "learned_column_mappings": get_learned_column_mappings(),
        "correction_examples": get_correction_examples(),
    }
    export_path.write_text(json.dumps(learning_pack, indent=2), encoding="utf-8")
    return export_path


def import_learning_pack(pack: dict[str, Any]) -> None:
    for item in pack.get("learned_synonyms", []):
        if item.get("phrase") and item.get("mapped_concept"):
            add_learned_synonym(
                phrase=item["phrase"],
                mapped_concept=item["mapped_concept"],
                source=item.get("source", "import"),
            )

    for item in pack.get("learned_column_mappings", []):
        if item.get("raw_column_name") and item.get("standard_concept"):
            add_learned_column_mapping(
                raw_column_name=item["raw_column_name"],
                standard_concept=item["standard_concept"],
                confidence=float(item.get("confidence", 0.8)),
                source=item.get("source", "import"),
            )

    for item in pack.get("correction_examples", []):
        if item.get("incorrect_command_json") and item.get("corrected_command_json"):
            execute_insert(
                """
                INSERT INTO corrections (
                    request_id,
                    original_request,
                    incorrect_command_json,
                    corrected_command_json,
                    correction_type,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    None,
                    item.get("original_request", ""),
                    item.get("incorrect_command_json", "{}"),
                    item.get("corrected_command_json", "{}"),
                    item.get("correction_type", "import"),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    sync_learning_files()
