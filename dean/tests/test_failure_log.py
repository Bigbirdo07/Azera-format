"""failure_log unit + planner-integration tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from core.failure_log import clear_failures, log_failure, read_failures


# ---- unit tests -----------------------------------------------------------


def test_log_and_read_roundtrip(tmp_path):
    path = tmp_path / "failed_asks.jsonl"
    log_failure(
        user_message="show me hovercrafts",
        sheet_name="Roster",
        columns=["Name", "GPA"],
        intent="clarify",
        reason="vague term without supporting columns",
        routing={"plan_source": "rules", "confidence": 0.4, "llm_used": False},
        path=path,
    )
    records = read_failures(path=path)
    assert len(records) == 1
    entry = records[0]
    assert entry["user_message"] == "show me hovercrafts"
    assert entry["sheet"] == "Roster"
    assert entry["columns"] == ["Name", "GPA"]
    assert entry["intent"] == "clarify"
    assert entry["reason"] == "vague term without supporting columns"
    assert entry["routing"]["plan_source"] == "rules"
    assert entry["routing"]["confidence"] == 0.4


def test_read_returns_newest_first(tmp_path):
    path = tmp_path / "f.jsonl"
    for i in range(5):
        log_failure(user_message=f"q{i}", sheet_name="S", columns=[],
                    intent="clarify", reason=str(i), path=path)
    records = read_failures(path=path)
    assert [r["reason"] for r in records] == ["4", "3", "2", "1", "0"]


def test_read_honors_limit(tmp_path):
    path = tmp_path / "f.jsonl"
    for i in range(20):
        log_failure(user_message=f"q{i}", sheet_name="S", columns=[],
                    intent="clarify", reason=str(i), path=path)
    records = read_failures(limit=5, path=path)
    assert len(records) == 5
    assert records[0]["reason"] == "19"
    assert records[-1]["reason"] == "15"


def test_clear_removes_log(tmp_path):
    path = tmp_path / "f.jsonl"
    log_failure(user_message="q", sheet_name="S", columns=[],
                intent="clarify", reason="r", path=path)
    assert read_failures(path=path)
    clear_failures(path=path)
    assert read_failures(path=path) == []


def test_read_missing_file_is_empty(tmp_path):
    assert read_failures(path=tmp_path / "nope.jsonl") == []


def test_log_failure_swallows_io_error(tmp_path):
    """Pointing at an unwritable path must NOT raise — chat would break."""
    bad = tmp_path / "nonexistent" / "deep" / "path"
    # Create the parent as a file so mkdir(exist_ok=True) and open() both fail.
    (tmp_path / "nonexistent").write_text("not a dir")
    log_failure(
        user_message="q", sheet_name="S", columns=[],
        intent="clarify", reason="r",
        path=bad / "f.jsonl",
    )
    # Nothing to assert — success is "no exception."


def test_columns_field_is_capped(tmp_path):
    from core.failure_log import _MAX_COLUMNS_RECORDED

    path = tmp_path / "f.jsonl"
    many = [f"col_{i}" for i in range(_MAX_COLUMNS_RECORDED + 50)]
    log_failure(user_message="q", sheet_name="S", columns=many,
                intent="clarify", reason="r", path=path)
    [entry] = read_failures(path=path)
    assert len(entry["columns"]) == _MAX_COLUMNS_RECORDED


def test_corrupt_lines_are_skipped(tmp_path):
    path = tmp_path / "f.jsonl"
    log_failure(user_message="ok1", sheet_name="S", columns=[],
                intent="clarify", reason="r", path=path)
    with path.open("a") as fh:
        fh.write("this is not json\n")
    log_failure(user_message="ok2", sheet_name="S", columns=[],
                intent="clarify", reason="r", path=path)
    records = read_failures(path=path)
    messages = [r["user_message"] for r in records]
    assert messages == ["ok2", "ok1"]


# ---- planner integration --------------------------------------------------


def test_clarify_routing_would_be_captured():
    """A genuinely vague request should route to clarify — that's the shape
    the chat_panel hook will capture."""
    from nlp.planner_router import plan_user_request

    df = pd.DataFrame({"Name": ["A", "B"], "GPA": [3.0, 2.0]})
    routing = plan_user_request(
        user_message="hovercrafts",
        sheets={"Students": df},
        sheet_columns={"Students": ["Name", "GPA"]},
        selected_sheet="Students",
        conversation_state=None,
        settings={"llm_enabled": False},
    )
    assert routing["intent"] in {"clarify", "unavailable", "unsupported"}
