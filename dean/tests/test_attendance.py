"""Attendance ingestion, matching, metrics, and risk threshold."""

from __future__ import annotations

from datetime import datetime, timedelta
from io import BytesIO

import pandas as pd
import pytest
from openpyxl import Workbook

from core.attendance import (
    ATTENDANCE_RISK_THRESHOLD,
    SEVERE_ATTENDANCE_RISK_THRESHOLD,
    canonicalise_attendance_frame,
    compute_attendance_metrics,
    load_attendance_file,
    match_attendance_to_roster,
)


# ---- helpers ---------------------------------------------------------------


class _FakeUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


def _make_attendance_xlsx(rows: list[dict]) -> _FakeUpload:
    wb = Workbook()
    ws = wb.active
    ws.title = "Attendance"
    if rows:
        headers = list(rows[0].keys())
        ws.append(headers)
        for row in rows:
            ws.append([row[h] for h in headers])
    buf = BytesIO()
    wb.save(buf)
    return _FakeUpload("attendance.xlsx", buf.getvalue())


def _roster_with_ids(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "Student ID": ids,
        "Name": [f"Student {sid}" for sid in ids],
        "GPA": [3.0] * len(ids),
    })


# ---- Test 1: attendance file detected --------------------------------------


def test_attendance_file_is_detected_and_canonicalised():
    upload = _make_attendance_xlsx([
        {"StudentID": "S001", "Date": "2026-06-01", "Status": "Present"},
        {"StudentID": "S002", "Date": "2026-06-01", "Status": "Absent"},
    ])
    loaded = load_attendance_file(upload)
    assert loaded.file_name == "attendance.xlsx"
    assert not loaded.frame.empty
    # Column names are normalised regardless of source spelling.
    assert "Student ID" in loaded.frame.columns
    assert "Attendance Status" in loaded.frame.columns
    # Status values are canonicalised.
    assert set(loaded.frame["Attendance Status"]) == {"Present", "Absent"}


def test_attendance_loader_warns_on_missing_required_columns():
    upload = _make_attendance_xlsx([
        {"Name": "Alice", "When": "2026-06-01"},  # no Student ID, no real status
    ])
    loaded = load_attendance_file(upload)
    joined = " ".join(loaded.warnings).lower()
    assert "student id" in joined


# ---- Test 2: matching ------------------------------------------------------


def test_attendance_matches_to_roster_by_student_id():
    attendance = pd.DataFrame({
        "Student ID": ["S001", "S002", "S003"],
        "Date": pd.to_datetime(["2026-06-01"] * 3),
        "Attendance Status": ["Present", "Absent", "Present"],
    })
    roster = _roster_with_ids(["S001", "S002", "S004"])
    matched, unmatched, unmatched_ids = match_attendance_to_roster(attendance, roster)
    assert matched == 2  # S001, S002
    assert unmatched == 1  # S003 not in roster
    assert unmatched_ids == ["S003"]


# ---- Test 3: unmatched warning ---------------------------------------------


def test_unmatched_attendance_rows_can_be_surfaced():
    attendance = pd.DataFrame({
        "Student ID": ["S999", "S888"],
        "Date": pd.to_datetime(["2026-06-01", "2026-06-01"]),
        "Attendance Status": ["Absent", "Absent"],
    })
    roster = _roster_with_ids(["S001"])
    matched, unmatched, unmatched_ids = match_attendance_to_roster(attendance, roster)
    assert matched == 0
    assert unmatched == 2
    assert sorted(unmatched_ids) == ["S888", "S999"]


# ---- Test 4: Attendance Rate calculated ------------------------------------


def test_attendance_rate_matches_expected_share():
    attendance = pd.DataFrame({
        "Student ID": ["S001"] * 10,
        "Date": pd.to_datetime([f"2026-05-{i:02d}" for i in range(1, 11)]),
        "Attendance Status": ["Present"] * 8 + ["Absent"] * 2,
    })
    metrics = compute_attendance_metrics(attendance)
    row = metrics.loc[metrics["Student ID"] == "S001"].iloc[0]
    assert row["Days Present"] == 8
    assert row["Days Absent"] == 2
    assert row["Attendance Rate"] == 80.0


def test_tardies_count_in_attendance_rate_but_as_their_own_day_type():
    attendance = pd.DataFrame({
        "Student ID": ["S001"] * 4,
        "Date": pd.to_datetime(["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"]),
        "Attendance Status": ["Present", "Tardy", "Tardy", "Absent"],
    })
    metrics = compute_attendance_metrics(attendance)
    row = metrics.iloc[0]
    assert row["Days Present"] == 1
    assert row["Days Tardy"] == 2
    assert row["Days Absent"] == 1
    # Present + Tardy + Excused count as "attended" for rate purposes.
    assert row["Attendance Rate"] == 75.0


# ---- Test 5: Attendance Risk threshold -------------------------------------


def test_attendance_risk_flips_at_90_percent():
    attendance = pd.DataFrame({
        "Student ID": ["LOW", "EDGE", "HIGH", "SEVERE"],
        "Date": pd.to_datetime(["2026-05-01"] * 4),
        "Attendance Status": ["Absent", "Absent", "Present", "Absent"],
    })
    # Build a 10-row history per student so 1 absence = 90%, exactly the
    # threshold (must NOT flag), 2 absences = 80% (flag), all-absent = 0%.
    full = []
    for sid, absences in (("LOW", 1), ("EDGE", 0), ("HIGH", 0), ("SEVERE", 10)):
        for i in range(10):
            full.append({
                "Student ID": sid,
                "Date": pd.Timestamp(f"2026-05-{i+1:02d}"),
                "Attendance Status": "Absent" if i < absences else "Present",
            })
    frame = pd.DataFrame(full)
    metrics = compute_attendance_metrics(frame).set_index("Student ID")
    # 90% rate = NOT at risk (threshold is strict <).
    assert metrics.loc["LOW", "Attendance Rate"] == 90.0
    assert metrics.loc["LOW", "Attendance Risk"] is False or \
        metrics.loc["LOW", "Attendance Risk"] == False  # noqa: E712
    # 100% rate = not at risk.
    assert metrics.loc["HIGH", "Attendance Risk"] == False  # noqa: E712
    assert metrics.loc["EDGE", "Attendance Risk"] == False  # noqa: E712
    # 0% rate = at risk + severe.
    assert metrics.loc["SEVERE", "Attendance Risk"] == True  # noqa: E712
    assert metrics.loc["SEVERE", "Severe Attendance Risk"] == True  # noqa: E712


def test_recent_absences_window_limits_to_last_14_days():
    today = datetime(2026, 6, 1)
    rows = [
        {"Student ID": "S1", "Date": pd.Timestamp(today),
         "Attendance Status": "Absent"},  # today, counts
        {"Student ID": "S1", "Date": pd.Timestamp(today - timedelta(days=10)),
         "Attendance Status": "Absent"},  # within 14 days
        {"Student ID": "S1", "Date": pd.Timestamp(today - timedelta(days=20)),
         "Attendance Status": "Absent"},  # older than 14 days, excluded
        {"Student ID": "S1", "Date": pd.Timestamp(today),
         "Attendance Status": "Present"},
    ]
    metrics = compute_attendance_metrics(pd.DataFrame(rows), today=today)
    row = metrics.iloc[0]
    assert row["Recent Absences"] == 2


# ---- Test for canonicalisation edge cases ----------------------------------


def test_status_canonicalisation_handles_synonyms():
    frame = pd.DataFrame({
        "Student ID": ["S1", "S2", "S3", "S4"],
        "Date": ["2026-06-01"] * 4,
        "Attendance Status": ["P", "absent", "Late", "EX"],
    })
    cleaned, _ = canonicalise_attendance_frame(frame)
    assert list(cleaned["Attendance Status"]) == ["Present", "Absent", "Tardy", "Excused"]


def test_attendance_loader_constants_match_thresholds():
    # Lock the contract — Phase B specifies < 90% and < 80%.
    assert ATTENDANCE_RISK_THRESHOLD == 90.0
    assert SEVERE_ATTENDANCE_RISK_THRESHOLD == 80.0
