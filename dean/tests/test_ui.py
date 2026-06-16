"""Phase I: simplified dean-friendly UI — structure and behavior."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.conftest import FIXTURE, FakeUpload
from ui.chat_panel import _rules_fallback_note

REPO_ROOT = Path(__file__).resolve().parents[1]


def _texts(at):
    """All visible markdown/title/caption/info text, lowercased."""
    parts = []
    for collection in ("markdown", "title", "caption", "info", "subheader"):
        for el in getattr(at, collection, []):
            parts.append(str(getattr(el, "value", "")))
    return " ".join(parts).lower()


def test_app_loads_with_no_workbook():
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.session_state["current_user"] = {"username": "t", "role": "Editor"}
    at.run()
    assert not at.exception
    text = _texts(at)
    assert "dean assistant" in text
    assert "no workbook loaded" in text


def test_workbook_summary_appears(chat):
    text = _texts(chat.at)
    assert "synthetic_students.xlsx" in text
    assert "students" in text and "rows" in text
    assert "workbook loaded" in text


def test_llm_fallback_note_hidden_from_normal_chat():
    note = _rules_fallback_note(
        {
            "plan_source": "rules",
            "fallback_reason": "LLM plan failed validation; used rules plan",
        },
        {"use_local_llm": True, "strict_privacy_mode": False},
    )
    assert note is None


def test_filter_is_applied_and_recorded(chat):
    # The "Current view" card was removed from the UI; filtering is now
    # reflected in conversation memory (and downstream results), not a chip card.
    chat.send("Show me Accounting students")
    filters = chat.memory().get("active_filters") or []
    assert filters, "filter should be recorded in conversation memory"
    joined = str(filters).lower()
    assert "accounting" in joined or "department" in joined


def test_clear_filters_button(chat):
    chat.send("Show me Accounting students")
    assert chat.memory()["active_filters"]
    chat.send("clear that")
    assert not chat.memory()["active_filters"]


def test_start_over_button(chat):
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    chat.send("start over")
    mem = chat.memory()
    assert not mem["active_filters"] and not mem.get("pending_action")


def test_sensitive_request_shows_confirmation_card(chat):
    chat.send("show me all student emails and GPAs")
    assert chat.get("assistant_mode") == "ask_question"
    assert not chat.memory().get("pending_action")


def test_confirm_export_creates_output(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("yes, export")
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_cancel_clears_confirmation(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("no")
    assert not chat.memory().get("pending_action")
    assert not chat.get("latest_output_file")


def test_debug_hidden_by_default(chat):
    # The debug toggle exists and is off; the debug panel caption is not shown.
    assert chat.get("show_debug_toggle") in (None, False)
    assert "no raw sensitive rows" not in _texts(chat.at)


def test_debug_visible_when_enabled(chat):
    chat.at.session_state["show_debug_toggle"] = True
    chat.at.run()
    assert "no raw sensitive rows" in _texts(chat.at)


def test_llm_settings_hidden_for_non_admin(chat):
    # Editor (non-admin) sees no LLM toggle in the sidebar.
    labels = [getattr(c, "label", "") for c in chat.at.checkbox]
    assert not any("local llm" in str(lbl).lower() for lbl in labels)
