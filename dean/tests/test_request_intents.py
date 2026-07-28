"""Tests for nlp/request_intents.py's chat-driven risk-threshold parsing.

Covers the parse_risk_setting_update / is_risk_setting_update_request pair
added to let a dean type "change the GPA risk threshold to 2.5" instead of
using the manual settings-panel form.
"""

from __future__ import annotations

import pytest

from nlp.request_intents import (
    is_risk_setting_update_request,
    parse_field_update,
    parse_risk_setting_update,
)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("change the GPA risk threshold to 2.5", ("gpa_risk_threshold", 2.5)),
        ("change the gpa threshold to 2.5", ("gpa_risk_threshold", 2.5)),
        ("set the attendance risk threshold to 88", ("attendance_risk_threshold", 88.0)),
        ("update the attendance threshold to 92", ("attendance_risk_threshold", 92.0)),
        ("set the unexcused absence concern to 4", ("unexcused_absence_concern", 4.0)),
        ("change the tardy concern to 6", ("tardy_concern", 6.0)),
        ("set the high risk signal count to 3", ("high_risk_signal_count", 3.0)),
        ("set the moderate risk signal count to 1", ("moderate_risk_signal_count", 1.0)),
        ("set the sat math benchmark to 550", ("sat_math_benchmark_threshold", 550.0)),
        ("set the sat ebrw benchmark to 480", ("sat_ebrw_benchmark_threshold", 480.0)),
        ("set the psat math benchmark to 480", ("psat_math_benchmark_threshold", 480.0)),
        ("set the psat reading writing benchmark to 460",
         ("psat_reading_writing_benchmark_threshold", 460.0)),
        # "to <value>" fallback: bare trailing number with no "to".
        ("make the gpa threshold 3.0", ("gpa_risk_threshold", 3.0)),
    ],
)
def test_parse_risk_setting_update_resolves_known_fields(message, expected):
    assert parse_risk_setting_update(message) == expected


def test_severe_attendance_does_not_collide_with_bare_attendance():
    # Regression-shaped: "severe attendance risk" must resolve to the
    # severe_* field, not get swallowed by the more generic "attendance
    # risk" cue -- same ordering class of bug as this session's
    # _NUMERIC_CONCEPTS fix for unexcused/excused absences.
    assert parse_risk_setting_update("change the severe attendance risk to 75") == (
        "severe_attendance_risk_threshold", 75.0,
    )
    assert parse_risk_setting_update("change the severe attendance threshold to 70") == (
        "severe_attendance_risk_threshold", 70.0,
    )
    assert parse_risk_setting_update("change the attendance risk threshold to 88") == (
        "attendance_risk_threshold", 88.0,
    )


@pytest.mark.parametrize(
    "message",
    [
        "change their advisor to Dr. Smith",
        "set the note to follow up next week",
        "how many students are at risk",
        "change the GPA risk threshold",  # no number at all
    ],
)
def test_parse_risk_setting_update_returns_none_for_non_matches(message):
    assert parse_risk_setting_update(message) is None


def test_plain_field_update_is_not_swallowed_by_risk_setting_detector():
    # The two detectors must stay mutually exclusive on ordinary workbook
    # field-update phrasing -- a real column name, not a threshold cue.
    assert parse_risk_setting_update("change their advisor to Dr. Smith") is None
    assert parse_field_update("change their advisor to Dr. Smith") == ("advisor", "Dr. Smith")


def test_is_risk_setting_update_request_requires_both_verb_and_cue():
    assert is_risk_setting_update_request("change the GPA risk threshold to 2.5")
    assert not is_risk_setting_update_request("the GPA risk threshold is 2.5")  # no action verb
    assert not is_risk_setting_update_request("change their advisor to Dr. Smith")  # no threshold cue
