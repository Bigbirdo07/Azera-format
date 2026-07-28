"""Tests for core/risk_settings.py's chat-driven threshold-update helper.

apply_risk_setting_update is the execution half of the "change the GPA risk
threshold to 2.5" feature; nlp/request_intents.py and nlp/planner_router.py
produce the (field, value) command, this applies it.
"""

from __future__ import annotations

from core.risk_settings import (
    RiskSettings,
    apply_risk_setting_update,
    load_risk_settings,
)


def test_apply_risk_setting_update_returns_old_and_new():
    session_state: dict = {}
    old, new = apply_risk_setting_update(session_state, "gpa_risk_threshold", 2.5)
    assert old.gpa_risk_threshold == 2.0  # dataclass default, untouched
    assert new.gpa_risk_threshold == 2.5


def test_apply_risk_setting_update_persists_to_session_state():
    session_state: dict = {}
    apply_risk_setting_update(session_state, "attendance_risk_threshold", 88.0)
    reloaded = load_risk_settings(session_state)
    assert reloaded.attendance_risk_threshold == 88.0


def test_apply_risk_setting_update_does_not_mutate_the_old_instance():
    # RiskSettings is frozen -- the only correct way to "change" one field is
    # a fresh instance via dataclasses.replace. This guards against a future
    # regression that tries to mutate in place.
    session_state: dict = {"risk_settings": RiskSettings().to_dict()}
    old_before = load_risk_settings(session_state)
    old, new = apply_risk_setting_update(session_state, "tardy_concern", 8)
    assert old is not new
    assert old.tardy_concern == old_before.tardy_concern == 5
    assert new.tardy_concern == 8


def test_apply_risk_setting_update_casts_count_fields_to_int():
    session_state: dict = {}
    _, new = apply_risk_setting_update(session_state, "unexcused_absence_concern", 4.0)
    assert new.unexcused_absence_concern == 4
    assert isinstance(new.unexcused_absence_concern, int)


def test_apply_risk_setting_update_keeps_other_fields_unchanged():
    session_state: dict = {}
    _, new = apply_risk_setting_update(session_state, "gpa_risk_threshold", 2.5)
    defaults = RiskSettings()
    assert new.attendance_risk_threshold == defaults.attendance_risk_threshold
    assert new.tardy_concern == defaults.tardy_concern
