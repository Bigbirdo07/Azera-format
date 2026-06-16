"""Phase F: confirmed local-only actions, output safety, and audit trail."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from core import confirmed_actions as ca
from core.confirmed_actions import (
    append_note,
    execute_add_note_action,
    execute_export_action,
    execute_update_field_action,
)


@pytest.fixture()
def io_dirs(tmp_path):
    return tmp_path / "outputs", tmp_path / "logs" / "audit_log.jsonl"


@pytest.fixture()
def roster():
    df = pd.DataFrame({
        "Student ID": ["S1", "S2", "S3", "S4"],
        "Email": ["a@x.edu", "b@x.edu", "c@x.edu", "d@x.edu"],
        "Department": ["Accounting", "Biology", "Accounting", "Biology"],
        "GPA": [2.1, 3.5, 1.9, 3.2],
        "Notes": ["old", "", "", ""],
    })
    return {"Students": df}


ACC = [{"column": "Department", "operator": "equals", "value": "Accounting"}]


def test_append_note():
    assert append_note("", "x") == "x"
    assert append_note(None, "x") == "x"
    assert append_note("a", "b") == "a; b"


def test_export_creates_file_with_matching_rows(roster, io_dirs):
    out, audit = io_dirs
    result = execute_export_action(filters=ACC, sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    assert result.success and result.rows_affected == 2
    assert Path(result.output_file).exists()
    exported = pd.read_excel(result.output_file)
    assert set(exported["Student ID"]) == {"S1", "S3"}


def test_add_note_appends_and_creates_only_on_matches(roster, io_dirs):
    out, audit = io_dirs
    result = execute_add_note_action(filters=ACC, note="Follow up", sheets=roster, sheet="Students",
                                     output_dir=out, audit_path=audit)
    assert result.success and result.rows_affected == 2
    saved = pd.read_excel(result.output_file).fillna("")
    notes = dict(zip(saved["Student ID"], saved["Notes"]))
    assert notes["S1"] == "old; Follow up"   # appended to existing
    assert notes["S3"] == "Follow up"        # added to matching empty
    assert notes["S2"] == "" and notes["S4"] == ""  # non-matching untouched


def test_add_note_creates_notes_column_if_missing(io_dirs):
    out, audit = io_dirs
    sheets = {"Students": pd.DataFrame({"Student ID": ["S1", "S2"], "Department": ["Accounting", "Biology"]})}
    result = execute_add_note_action(filters=ACC, note="Hi", sheets=sheets, sheet="Students",
                                     output_dir=out, audit_path=audit)
    saved = pd.read_excel(result.output_file)
    assert "Notes" in saved.columns


def test_original_sheets_unchanged(roster, io_dirs):
    out, audit = io_dirs
    before = roster["Students"].copy()
    execute_add_note_action(filters=ACC, note="x", sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    pd.testing.assert_frame_equal(roster["Students"], before)


def test_update_safe_field_creates_workbook(roster, io_dirs):
    out, audit = io_dirs
    result = execute_update_field_action(filters=ACC, field_name="Follow Up Needed", value="Yes",
                                         sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    assert result.success and result.rows_affected == 2
    saved = pd.read_excel(result.output_file).fillna("")
    flags = dict(zip(saved["Student ID"], saved["Follow Up Needed"]))
    assert flags["S1"] == "Yes" and flags["S3"] == "Yes" and flags["S2"] == ""


def test_protected_field_blocked(roster, io_dirs):
    out, audit = io_dirs
    result = execute_update_field_action(filters=ACC, field_name="GPA", value=4.0,
                                         sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    assert not result.success and result.error == "protected_field"
    assert result.output_file is None
    assert not (out.exists() and list(out.glob("*.xlsx")))  # nothing written


def test_unknown_field_blocked(roster, io_dirs):
    out, audit = io_dirs
    result = execute_update_field_action(filters=ACC, field_name="Department", value="X",
                                         sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    assert not result.success and result.error == "not_editable"


def test_audit_records_export_and_note(roster, io_dirs):
    out, audit = io_dirs
    execute_export_action(filters=ACC, sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    execute_add_note_action(filters=ACC, note="x", sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    actions = [r["action_type"] for r in records]
    assert "export" in actions and "add_note" in actions
    note_rec = next(r for r in records if r["action_type"] == "add_note")
    assert note_rec["rows_affected"] == 2
    assert note_rec["columns_changed"] == ["Notes"]
    assert note_rec["confirmation_status"] == "confirmed"


def test_audit_has_no_raw_sensitive_values(roster, io_dirs):
    out, audit = io_dirs
    execute_export_action(filters=ACC, sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    blob = audit.read_text()
    # No raw email/row values leak into the audit log.
    assert "a@x.edu" not in blob and "c@x.edu" not in blob
    # The export flagged Email as a sensitive field involved (name only).
    record = json.loads(blob.splitlines()[0])
    assert "Email" in record["sensitive_fields_involved"]


def test_output_filename_is_sanitized(roster, io_dirs):
    out, audit = io_dirs
    result = execute_export_action(filters=ACC, sheets=roster, sheet="Students", output_dir=out, audit_path=audit)
    name = Path(result.output_file).name
    assert re.fullmatch(r"student_export_\d{8}_\d{6}(_\d+)?\.xlsx", name)
    assert "@" not in name and "accounting" not in name.lower()
