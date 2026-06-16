"""Phase C: pending-action confirm/cancel flows (export, edit).

Note: the sensitive-field "show_sensitive" confirmation gate was removed —
explicitly requested hidden columns are now revealed directly (see
test_sensitive_request_reveals_directly)."""

from __future__ import annotations


def _shown_columns(chat):
    table = chat.get("ask_table") or []
    return set().union(*[set(r) for r in table]) if table else set()


def test_sensitive_request_reveals_directly(chat):
    # The sensitive-field confirmation gate was removed: explicitly asking for
    # Email shows it directly, with no pending confirmation step.
    chat.send("show me all student emails and GPAs")
    assert not chat.memory().get("pending_action")
    assert chat.get("assistant_mode") == "ask_question"
    assert "Email" in _shown_columns(chat)


def test_export_with_sensitive_requires_confirmation(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    pending = chat.memory().get("pending_action") or {}
    assert pending.get("type") == "export"
    assert "Email" in pending.get("reason", "")
    # Nothing exported before confirmation.
    assert not chat.get("latest_output_file")


def test_export_confirm_creates_file(chat, tmp_path):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("yes, export")
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_export_cancel_no_file(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("no, cancel")
    assert not chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_note_edit_requires_confirmation(chat):
    chat.send("add a note to these students")
    pending = chat.memory().get("pending_action") or {}
    assert pending.get("type") == "note_edit"
    # No edit executed (no export file, still awaiting confirmation).
    assert not chat.get("latest_output_file")


# --- E.6 safer confirmation handling ----------------------------------------


def test_yes_with_no_pending_does_nothing(chat):
    chat.send("yes")
    assert not chat.memory().get("pending_action")
    assert not chat.get("latest_output_file")


def test_cancel_with_no_pending_does_nothing(chat):
    chat.send("no")
    assert not chat.memory().get("pending_action")
    assert not chat.get("latest_output_file")


def test_pending_export_expires_on_unrelated_query(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    assert (chat.memory().get("pending_action") or {}).get("type") == "export"
    chat.send("how many students are in each department")
    assert not chat.memory().get("pending_action")  # expired
    assert not chat.get("latest_output_file")  # nothing exported
    assert chat.get("assistant_mode") == "ask_question"  # the new query ran


def test_pending_note_edit_expires_on_unrelated_query(chat):
    chat.send("add a note to these students")
    assert (chat.memory().get("pending_action") or {}).get("type") == "note_edit"
    chat.send("how many students are in each department")
    assert not chat.memory().get("pending_action")
    assert chat.get("assistant_mode") == "ask_question"


def test_query_with_no_does_not_cancel_pending_wrongly(chat):
    # A full query that merely contains the word "no" must not be read as cancel.
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("show me students with no advisor")
    # Pending expired (unrelated query) and the query was processed, not a bare cancel.
    assert not chat.memory().get("pending_action")
    assert chat.get("assistant_mode") == "ask_question"


def test_confirming_export_does_not_run_note_edit(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("yes, export")
    # Export ran (file created); no note-edit side effect.
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


# --- Phase F: confirmed-action execution through the chat -------------------


def test_add_note_confirmed_creates_workbook(chat):
    chat.send("show me Accounting students")
    chat.send("add note: Advisor follow-up needed")
    assert (chat.memory().get("pending_action") or {}).get("type") == "note_edit"
    chat.send("yes, do it")
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_protected_gpa_update_refused_no_pending(chat):
    chat.send("show me Accounting students")
    chat.send("change their GPA to 4.0")
    assert not chat.memory().get("pending_action")  # never gated
    assert not chat.get("latest_output_file")  # nothing written


def test_safe_field_update_confirmed(chat):
    chat.send("show me Accounting students")
    chat.send("set Follow Up Needed to Yes")
    assert (chat.memory().get("pending_action") or {}).get("type") == "field_update"
    chat.send("yes, do it")
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_pending_clears_after_failed_action(chat):
    # An empty-note edit is gated, then fails on execution; pending must clear.
    chat.send("show me Accounting students")
    chat.send("add a note to these students")  # no note text
    assert (chat.memory().get("pending_action") or {}).get("type") == "note_edit"
    chat.send("yes, do it")
    assert not chat.memory().get("pending_action")  # cleared after failure
    assert not chat.get("latest_output_file")  # nothing written
