"""Unit tests for scripts.promote_learned_patterns.

Uses a tiny synthetic JSONL fixture so the assertions are deterministic and
do not depend on whatever the project log happens to contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.promote_learned_patterns import (
    build_report,
    mine_corrections,
    mine_llm_cache_candidates,
    mine_synonym_gap_candidates,
    read_log,
    render_json,
    render_markdown,
)


# ---- fixture ---------------------------------------------------------------


def _row(**overrides):
    """Default record shape — overrides win."""
    base = {
        "id": "x",
        "session_id": "s1",
        "user_message": "",
        "normalized_message": "",
        "plan_source": "rules",
        "intent": "query",
        "operation": "filtered_preview",
        "validated_plan": {"operation": "filtered_preview", "filters": []},
        "validation_status": "passed",
        "llm_used": False,
        "user_corrected": False,
        "corrects_entry_id": None,
        "correction_message": None,
        "safe_for_rule_mining": True,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def log_path(tmp_path) -> Path:
    """Build a small synthetic log with one row per category."""
    path = tmp_path / "interaction_learning.jsonl"
    records = [
        # 4× the same LLM-handled phrase with the same plan -> cache candidate.
        *[_row(id=f"l{i}", session_id=f"s{i}", normalized_message="who needs attention",
               plan_source="llm", llm_used=True,
               validated_plan={"operation": "count_rows", "filters": []})
          for i in range(4)],
        # 4× a clarify cluster about housing -> synonym gap.
        *[_row(id=f"c{i}", session_id=f"sc{i}", normalized_message="filter by housing",
               intent="clarify", plan_source="clarification", llm_used=False,
               validated_plan=None)
          for i in range(4)],
        # 2× a clarify cluster too small to make the cut (min_count=3).
        *[_row(id=f"n{i}", session_id=f"sn{i}", normalized_message="something rare",
               intent="unsupported", plan_source="clarification", llm_used=False)
          for i in range(2)],
        # Confirmation noise — must be ignored by the synonym miner.
        _row(id="y1", normalized_message="yes", intent="clarify"),
        _row(id="y2", normalized_message="no", intent="clarify"),
        # A correction pair: original then a corrective follow-up.
        _row(id="orig", normalized_message="who needs help", llm_used=True,
             validated_plan={"operation": "count_rows", "filters": []}),
        _row(id="fix", normalized_message="no i mean show their balance",
             user_corrected=True, corrects_entry_id="orig",
             correction_message="show their balance",
             validated_plan={"operation": "filtered_preview",
                             "filters": [{"column": "Balance", "operator": "greater_than", "value": 0}]}),
    ]
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return path


# ---- mining ---------------------------------------------------------------


def test_llm_cache_candidates_finds_repeated_phrase(log_path):
    rows = read_log(log_path)
    candidates = mine_llm_cache_candidates(rows, min_count=3)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.phrase == "who needs attention"
    assert c.count == 4
    assert c.canonical_plan["operation"] == "count_rows"
    assert c.plan_variance == 1
    assert c.sessions == 4


def test_llm_cache_candidates_respects_min_count(log_path):
    rows = read_log(log_path)
    # min_count high enough to exclude the 4-row cluster.
    assert mine_llm_cache_candidates(rows, min_count=10) == []


def test_synonym_gap_candidates_picks_up_clarify_clusters(log_path):
    rows = read_log(log_path)
    candidates = mine_synonym_gap_candidates(rows, min_count=3)
    phrases = [c.phrase for c in candidates]
    assert "filter by housing" in phrases
    assert "yes" not in phrases  # confirmation noise filtered out
    assert "no" not in phrases
    housing = next(c for c in candidates if c.phrase == "filter by housing")
    assert "housing" in housing.noun_tokens


def test_synonym_gap_suggests_concept_when_token_matches_known(log_path):
    rows = read_log(log_path)
    known = {"housing_status": ["housing", "dorm", "residence"]}
    candidates = mine_synonym_gap_candidates(rows, min_count=3, known_concepts=known)
    housing = next(c for c in candidates if c.phrase == "filter by housing")
    assert housing.suggested_concept == "housing_status"
    assert housing.unmatched_tokens == []


def test_synonym_gap_no_concept_match_when_token_unknown(log_path):
    rows = read_log(log_path)
    candidates = mine_synonym_gap_candidates(rows, min_count=3, known_concepts={})
    housing = next(c for c in candidates if c.phrase == "filter by housing")
    assert housing.suggested_concept is None
    assert "housing" in housing.unmatched_tokens


def test_synonym_gap_flags_unmatched_noun_even_when_suggestion_fires(tmp_path):
    """When the phrase contains a known single-token match ('status') AND an
    unknown noun ('housing'), the suggester surfaces the partial match as a
    hint, but unmatched_tokens still flags the real gap. The auto-apply path
    (tested separately) refuses to write when unmatched_tokens is non-empty."""
    path = tmp_path / "log.jsonl"
    records = [
        _row(id=f"m{i}", session_id=f"s{i}",
             normalized_message="what is their housing status",
             intent="clarify", plan_source="clarification")
        for i in range(4)
    ]
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rows = read_log(path)
    known = {"enrollment_status": ["status", "enrolled"]}
    candidates = mine_synonym_gap_candidates(rows, min_count=3, known_concepts=known)
    mixed = next(c for c in candidates if c.phrase == "what is their housing status")
    assert mixed.suggested_concept == "enrollment_status"  # partial-match hint
    assert "housing" in mixed.unmatched_tokens  # the real gap is still flagged


def test_synonym_gap_prefers_longest_match_when_bigram_is_known(tmp_path):
    """When a longer concept phrase ('housing status') is in concept_lookup,
    pick it over the single-token alternative ('status' under enrollment_status)."""
    path = tmp_path / "log.jsonl"
    records = [
        _row(id=f"m{i}", session_id=f"s{i}",
             normalized_message="what is their housing status",
             intent="clarify", plan_source="clarification")
        for i in range(4)
    ]
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rows = read_log(path)
    known = {
        "housing_status": ["housing", "housing status", "dorm"],
        "enrollment_status": ["status", "enrolled"],
    }
    candidates = mine_synonym_gap_candidates(rows, min_count=3, known_concepts=known)
    match = next(c for c in candidates if c.phrase == "what is their housing status")
    assert match.suggested_concept == "housing_status"


def test_synonym_gap_word_boundary_prevents_partial_match(tmp_path):
    """A key like 'id' must not match inside 'housing'."""
    path = tmp_path / "log.jsonl"
    records = [
        _row(id=f"m{i}", session_id=f"s{i}",
             normalized_message="filter by housing", intent="clarify")
        for i in range(4)
    ]
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rows = read_log(path)
    known = {"student_id": ["id"]}
    candidates = mine_synonym_gap_candidates(rows, min_count=3, known_concepts=known)
    housing = next(c for c in candidates if c.phrase == "filter by housing")
    assert housing.suggested_concept is None  # 'id' substring inside 'housing' must not match


def test_corrections_resolve_original_phrase(log_path):
    rows = read_log(log_path)
    corrections = mine_corrections(rows)
    assert len(corrections) == 1
    ex = corrections[0]
    assert ex.original_phrase == "who needs help"
    assert ex.correction_message == "show their balance"
    assert ex.incorrect_plan["operation"] == "count_rows"
    assert ex.corrected_plan["operation"] == "filtered_preview"


def test_corrections_dedupe_replayed_scenarios(tmp_path):
    """The same correction repeated across e2e replays collapses into one row."""
    path = tmp_path / "log.jsonl"
    records = []
    for i in range(4):
        records.append(_row(id=f"orig{i}", normalized_message="show me struggling students",
                            llm_used=True,
                            validated_plan={"operation": "filtered_preview", "filters": [{"a": 1}]}))
        records.append(_row(id=f"fix{i}", normalized_message="now use gpa below 2 5 instead",
                            user_corrected=True, corrects_entry_id=f"orig{i}",
                            correction_message="Now use GPA below 2.5 instead",
                            validated_plan={"operation": "filtered_preview",
                                            "filters": [{"a": 1}, {"b": 2}]}))
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    rows = read_log(path)
    corrections = mine_corrections(rows)
    assert len(corrections) == 1  # 4 replays collapse to one


# ---- rendering -------------------------------------------------------------


def test_build_report_combines_all_categories(log_path):
    rows = read_log(log_path)
    report = build_report(rows, log_path=log_path, min_count=3, known_concepts={"balance": ["balance"]})
    assert report.record_count == len(rows)
    assert len(report.llm_cache) == 1
    assert len(report.synonym_gaps) >= 1
    assert len(report.corrections) == 1


def test_render_markdown_includes_each_section(log_path):
    rows = read_log(log_path)
    report = build_report(rows, log_path=log_path, min_count=3, known_concepts={})
    md = render_markdown(report)
    assert "LLM-cache candidates" in md
    assert "Synonym-gap candidates" in md
    assert "User corrections" in md
    assert "who needs attention" in md
    assert "filter by housing" in md


def test_render_json_is_parseable(log_path):
    rows = read_log(log_path)
    report = build_report(rows, log_path=log_path, min_count=3, known_concepts={})
    payload = json.loads(render_json(report))
    assert payload["record_count"] == len(rows)
    assert isinstance(payload["llm_cache_candidates"], list)
    assert isinstance(payload["synonym_gap_candidates"], list)
    assert isinstance(payload["corrections"], list)


# ---- read_log -------------------------------------------------------------


def test_read_log_missing_file_returns_empty(tmp_path):
    assert read_log(tmp_path / "nope.jsonl") == []


def test_read_log_skips_invalid_json(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"id":"a"}\nnot-json\n{"id":"b"}\n', encoding="utf-8")
    rows = read_log(path)
    assert [r["id"] for r in rows] == ["a", "b"]
