"""Phase Q tests: multi-action command chaining.

The 11 spec acceptance cases.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import pytest

from core.confirmed_actions import execute_action_chain
from core.execution_dispatcher import execute_planned_request
from core.session_memory import SessionMemory
from nlp.action_chain import parse_action_chain
from nlp.planner_router import plan_user_request


FIXTURE = Path(__file__).parent / "fixtures" / "academic_roster.xlsx"


@dataclass
class _Loaded:
    sheets: dict
    file_name: str = "academic_roster.xlsx"


@pytest.fixture()
def roster() -> pd.DataFrame:
    return pd.read_excel(FIXTURE)


@pytest.fixture()
def loaded(roster):
    return _Loaded(sheets={"Students": roster})


@pytest.fixture()
def memory():
    return SessionMemory()


def _route(message, loaded, state=None):
    cols = list(next(iter(loaded.sheets.values())).columns)
    return plan_user_request(
        user_message=message,
        sheets=loaded.sheets,
        sheet_columns={"Students": cols},
        selected_sheet="Students",
        conversation_state=state or {},
        settings={},
    )


# ---- Q.1: chain parsing ----------------------------------------------------


def test_chain_parses_academic_watch_plus_export():
    chain = parse_action_chain(
        "mark these students academic watch and export me a new Excel sheet"
    )
    assert chain is not None
    assert [s.type for s in chain] == ["academic_watch", "export"]


def test_chain_parses_note_plus_export():
    chain = parse_action_chain("add note: advisor follow-up needed and export")
    assert chain is not None
    assert chain[0].type == "note_edit"
    assert chain[0].payload["note"]
    assert chain[1].type == "export"


def test_chain_parses_field_update_plus_export():
    chain = parse_action_chain("set Follow Up Needed to Yes and export")
    assert chain is not None
    # Reuses the academic_watch step since "follow up needed" triggers the
    # academic-watch path with the Follow Up Needed column.
    assert chain[0].type in {"academic_watch", "field_update"}
    assert chain[1].type == "export"


def test_chain_returns_none_for_single_action():
    assert parse_action_chain("mark these students academic watch") is None
    assert parse_action_chain("export this list") is None


# ---- Q.2: confirmation gate -----------------------------------------------


def test_chain_routes_through_confirmation(loaded):
    routing = _route(
        "mark these students Academic Watch and export me a new Excel sheet",
        loaded,
        state={"active_filters": [{"column": "GPA", "operator": "less_than", "value": 2.0}]},
    )
    assert routing["intent"] == "action_chain"
    assert routing["requires_confirmation"]
    assert routing["pending_type"] == "action_chain"
    assert "Confirmation needed" in routing["confirmation_reason"]


def test_chain_confirmation_lists_each_step(loaded):
    routing = _route(
        "mark these students Academic Watch and export me a new Excel sheet",
        loaded,
        state={"active_filters": [{"column": "GPA", "operator": "less_than", "value": 2.0}]},
    )
    reason = routing["confirmation_reason"]
    assert "Set Academic Watch" in reason
    assert "Save a new workbook" in reason
    assert "Export" in reason


# ---- Q.3: execution -------------------------------------------------------


def test_chain_execution_marks_only_target_rows(loaded, tmp_path):
    original_sha = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    result = execute_action_chain(
        actions=[
            {"type": "academic_watch", "value": "Yes", "column_hint": "Academic Watch"},
            {"type": "export", "target": "updated_workbook"},
        ],
        filters=[
            {"column": "Department", "operator": "equals", "value": "Biology"},
            {"column": "GPA", "operator": "less_than", "value": 2.0},
        ],
        sheets=loaded.sheets,
        sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "logs" / "audit_log.jsonl",
    )
    assert result.success
    out = pd.read_excel(result.output_file)
    expected = ((loaded.sheets["Students"]["Department"] == "Biology")
                & (loaded.sheets["Students"]["GPA"] < 2.0)).sum()
    marked = (out["Academic Watch"] == "Yes").sum()
    assert marked == expected
    # Original file is byte-for-byte unchanged.
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == original_sha


def test_chain_produces_single_output_workbook(loaded, tmp_path):
    """The edit step writes one workbook; the export step points at that
    same file. There should be exactly one .xlsx in the outputs directory."""
    output_dir = tmp_path / "outputs"
    execute_action_chain(
        actions=[
            {"type": "academic_watch", "value": "Yes", "column_hint": "Academic Watch"},
            {"type": "export", "target": "updated_workbook"},
        ],
        filters=[{"column": "Department", "operator": "equals", "value": "Biology"}],
        sheets=loaded.sheets,
        sheet="Students",
        output_dir=output_dir,
        audit_path=tmp_path / "logs" / "audit_log.jsonl",
    )
    files = list(output_dir.glob("*.xlsx"))
    assert len(files) == 1


def test_chain_writes_single_audit_entry(loaded, tmp_path):
    audit_path = tmp_path / "logs" / "audit_log.jsonl"
    execute_action_chain(
        actions=[
            {"type": "academic_watch", "value": "Yes", "column_hint": "Academic Watch"},
            {"type": "export", "target": "updated_workbook"},
        ],
        filters=[{"column": "Department", "operator": "equals", "value": "Biology"}],
        sheets=loaded.sheets,
        sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=audit_path,
    )
    lines = [json.loads(l) for l in audit_path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["action_type"] == "action_chain"
    assert entry["actions"] == ["academic_watch", "export"]
    assert entry["original_modified"] is False
    assert entry["confirmation_status"] == "confirmed"


def test_cancelled_chain_produces_no_output(loaded, tmp_path):
    """No execute_* call means no output, no audit entry. This test asserts
    that the dispatcher/router contract is: nothing happens until execute is
    called. (Equivalent to the UI 'No, cancel' path.)"""
    output_dir = tmp_path / "outputs"
    audit_path = tmp_path / "logs" / "audit_log.jsonl"
    _route("mark these students Academic Watch and export me a new Excel sheet",
           loaded,
           state={"active_filters": [{"column": "GPA", "operator": "less_than", "value": 2.0}]})
    # Without calling execute_action_chain, nothing should have been written.
    assert not output_dir.exists() or not list(output_dir.glob("*.xlsx"))
    assert not audit_path.exists()


def test_note_plus_export_chain_runs(loaded, tmp_path):
    result = execute_action_chain(
        actions=[
            {"type": "note_edit", "note": "Advisor follow-up needed"},
            {"type": "export", "target": "updated_workbook"},
        ],
        filters=[{"column": "Department", "operator": "equals", "value": "Biology"}],
        sheets=loaded.sheets,
        sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "logs" / "audit_log.jsonl",
    )
    assert result.success
    assert result.output_file
    assert result.rows_affected > 0


def test_field_update_plus_export_chain_runs(loaded, tmp_path):
    result = execute_action_chain(
        actions=[
            {"type": "field_update", "field": "Follow Up Needed", "value": "Yes"},
            {"type": "export", "target": "updated_workbook"},
        ],
        filters=[{"column": "Department", "operator": "equals", "value": "Biology"}],
        sheets=loaded.sheets,
        sheet="Students",
        output_dir=tmp_path / "outputs",
        audit_path=tmp_path / "logs" / "audit_log.jsonl",
    )
    assert result.success
    out = pd.read_excel(result.output_file)
    biology_count = (loaded.sheets["Students"]["Department"] == "Biology").sum()
    assert (out["Follow Up Needed"] == "Yes").sum() == biology_count


def test_chain_uses_prior_drilldown_filter(memory, loaded):
    """The chain should pick up filters from a prior drilldown turn so
    'show me low-GPA Biology students' → 'mark them and export' works."""
    cols = list(next(iter(loaded.sheets.values())).columns)

    # Turn 1 — filter Biology + GPA < 2.0
    routing1 = plan_user_request(
        user_message="show me Biology students below 2.0 GPA",
        sheets=loaded.sheets,
        sheet_columns={"Students": cols},
        selected_sheet="Students",
        conversation_state=asdict(memory),
        settings={},
    )
    resp1 = execute_planned_request(routing1, loaded, settings={},
                                    request_summary="...")
    plan1 = routing1["plan"]
    memory.record_ask(
        request="show me Biology students below 2.0 GPA",
        query_plan=plan1, result_description=resp1.get("description", "") or "",
        row_count=resp1.get("row_count"),
        columns_used=resp1.get("columns") or [],
        sheet=plan1.get("sheet", ""),
        summary_table=resp1.get("result_preview") or [],
        top_group=resp1.get("top_group"),
    )

    # Turn 2 — chained command using "these students"
    routing2 = plan_user_request(
        user_message="mark these students Academic Watch and export me a new Excel sheet",
        sheets=loaded.sheets,
        sheet_columns={"Students": cols},
        selected_sheet="Students",
        conversation_state=asdict(memory),
        settings={},
    )
    assert routing2["intent"] == "action_chain"
    chain_filters = routing2["plan"]["filters"]
    assert {"column": "Department", "operator": "equals", "value": "Biology"} \
        in chain_filters
    assert any(f["column"] == "GPA" and f["operator"] == "less_than"
               for f in chain_filters)
