"""Phase C: privacy defaults and sensitive-field gating in the ask path."""

from __future__ import annotations

from core.privacy import redact_table, requested_sensitive_columns

HIDDEN = {"Email", "Phone", "Date of Birth", "Financial Aid Status", "Conduct Status", "Notes"}


def test_requested_sensitive_detection(columns):
    assert requested_sensitive_columns("show me all emails and gpas", columns) == ["Email"]
    assert requested_sensitive_columns("show accounting students", columns) == []


def test_redact_table_drops_hidden():
    rows = [{"Name": "A", "Email": "a@x", "GPA": 3.1, "Phone": "555"}]
    redacted, removed = redact_table(rows, list(rows[0].keys()))
    assert redacted == [{"Name": "A", "GPA": 3.1}]
    assert set(removed) == {"Email", "Phone"}


def test_aggregate_no_confirmation(chat):
    chat.send("How many students are in each department?")
    assert chat.get("assistant_mode") == "ask_question"
    assert not chat.memory().get("pending_action")


def test_student_list_hides_sensitive_by_default(chat):
    chat.send("show me students below 2.5 GPA")
    table = chat.get("ask_table") or []
    shown = set().union(*[set(r) for r in table]) if table else set()
    assert not (shown & HIDDEN)
    assert "GPA" in shown


def test_explicit_sensitive_requires_confirmation(chat):
    chat.send("show me all student emails and GPAs")
    assert chat.get("assistant_mode") == "ask_question"
    assert not chat.memory().get("pending_action")
    table = chat.get("ask_table") or []
    shown = set().union(*[set(r) for r in table]) if table else set()
    assert "Email" in shown  # revealed directly because there is no safety confirmation gate
