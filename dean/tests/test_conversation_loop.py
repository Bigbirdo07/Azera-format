"""Phase J — persistent chat conversation UX.

These tests pin down the user-visible behavior that makes follow-up questions
feel like a single thread instead of one-shot prompts.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from tests.conftest import Chat, FIXTURE, FakeUpload, REPO_ROOT


def _last_user(messages: list[dict]) -> dict | None:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg
    return None


def _last_assistant(messages: list[dict]) -> dict | None:
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            return msg
    return None


def _assistants(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["role"] == "assistant"]


def _users(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m["role"] == "user"]


def _cols(filters: list[dict]) -> set[str]:
    return {f["column"] for f in filters or []}


# J.1 + J.2 — history persists chronologically -------------------------------


def test_first_user_and_assistant_appear_in_history(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    messages = chat.get("chat_messages") or []
    assert _users(messages)[-1]["content"] == "Show me Accounting students"
    assert _assistants(messages), "expected an assistant reply"
    assert chat.get("assistant_mode") == "ask_question"


def test_second_message_appends_does_not_replace(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    first = list(chat.get("chat_messages") or [])
    chat.send("now only below 2.5 GPA")
    second = list(chat.get("chat_messages") or [])
    assert len(second) > len(first), "follow-up must append to history"
    # The first user prompt is still present.
    contents = [m["content"] for m in second if m["role"] == "user"]
    assert "Show me Accounting students" in contents
    assert "now only below 2.5 GPA" in contents


def test_history_survives_three_turns(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    chat.send("now only seniors")
    user_messages = [m["content"] for m in (chat.get("chat_messages") or []) if m["role"] == "user"]
    assert user_messages == [
        "Show me Accounting students",
        "now only below 2.5 GPA",
        "now only seniors",
    ]


# J.4 — follow-up planner uses active context --------------------------------


def test_followup_accumulates_filters_via_context(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    filters = chat.memory()["active_filters"]
    assert _cols(filters) == {"Department", "GPA"}


# J.5 — no export until explicitly requested ---------------------------------


def test_filter_does_not_create_export_file(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    # No download attachment should have been auto-attached.
    messages = chat.get("chat_messages") or []
    attachments = [m.get("attachment", {}) for m in messages if m["role"] == "assistant"]
    assert not any(a.get("type") == "download" for a in attachments)
    # latest_output_file should still be unset.
    assert not chat.get("latest_output_file")


# J.6 — confirmation card lives in chat history ------------------------------


def test_confirmation_appears_as_attachment(chat: Chat) -> None:
    # Use the export gate (a pending action that still exists) to exercise the
    # generic confirmation card; the sensitive-field gate was removed.
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    last = _last_assistant(chat.get("chat_messages") or [])
    assert last is not None
    attachment = last.get("attachment") or {}
    assert attachment.get("type") == "confirmation"
    assert attachment.get("options"), "confirmation card must carry yes/no options"


def test_typed_yes_resolves_pending_and_appends_message(chat: Chat) -> None:
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    before = len(chat.get("chat_messages") or [])
    chat.send("yes, export")
    after = chat.get("chat_messages") or []
    assert len(after) > before, "confirmation resolution must add new chat turns"
    assert (chat.memory().get("pending_action") or {}) == {}


def test_typed_no_cancels_and_appends_message(chat: Chat) -> None:
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("no")
    last = _last_assistant(chat.get("chat_messages") or [])
    assert last is not None
    assert "cancelled" in last["content"].lower() or "okay" in last["content"].lower()
    assert (chat.memory().get("pending_action") or {}) == {}


def test_new_unrelated_request_after_pending_clears_pending(chat: Chat) -> None:
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    assert chat.memory().get("pending_action")
    chat.send("How many students are in each department?")
    # Pending should be expired and the new question handled.
    assert not chat.memory().get("pending_action")


# J.7 — clear filters vs start over -----------------------------------------


def test_clear_filters_keeps_history(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    chat.send("clear that")
    assert chat.memory()["active_filters"] == []
    assert len(chat.get("chat_messages") or []) >= 2, "history should remain"


def test_start_over_via_sidebar_resets_history_and_filters(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    assert chat.get("chat_messages")
    start_over = next(
        (b for b in chat.at.sidebar.button if b.label == "Start over"), None
    )
    assert start_over is not None, "sidebar must expose a Start over button"
    start_over.click()
    chat.at.session_state["workbook_upload"] = FakeUpload(FIXTURE)
    chat.at.run()
    # Start-over keeps the workbook but wipes the chat. The polish-phase
    # upload greeting is appended ONCE per workbook (when its file_name
    # changes), and start-over doesn't change the file_name — so the chat
    # remains empty (no re-greeting on rerun).
    assert chat.get("chat_messages") == []
    memory = chat.memory()
    assert not memory.get("active_filters")
    assert not memory.get("pending_action")


def test_uploading_a_different_workbook_starts_fresh_chat() -> None:
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    at.session_state["current_user"] = {"username": "t", "role": "Editor"}
    at.session_state["workbook_upload"] = FakeUpload(FIXTURE)
    at.run()
    at.chat_input[0].set_value("Show me Accounting students")
    at.session_state["workbook_upload"] = FakeUpload(FIXTURE)
    at.run()
    assert at.session_state["chat_messages"]

    # Simulate uploading a different workbook by swapping in an upload whose
    # name differs from the cached fingerprint. _maybe_reset_for_new_workbook
    # compares the loaded workbook's file_name to the one cached on the
    # previous run, so a new file_name is sufficient.
    class RenamedUpload(FakeUpload):
        def __init__(self) -> None:
            super().__init__(FIXTURE)
            self.name = "different_workbook.xlsx"

    at.session_state["workbook_upload"] = RenamedUpload()
    # Invalidate the load cache so the renamed upload is parsed fresh.
    at.session_state["cached_load_key"] = None
    at.session_state["cached_loaded"] = None
    at.session_state["cached_profile"] = None
    at.session_state["cached_diagnostics"] = None
    at.run()
    # A new workbook clears prior conversation and the polish-phase upload
    # greeting fires once. So the only message present is the assistant
    # greeting — any prior user turn was wiped.
    messages = at.session_state["chat_messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    assert "I found" in messages[0]["content"]


# J.8 — chat input always available -----------------------------------------


def test_chat_input_remains_after_first_response(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    # AppTest should still expose a chat_input widget.
    assert len(chat.at.chat_input) == 1


def test_chat_input_remains_after_confirmation_card(chat: Chat) -> None:
    chat.send("show me all student emails and GPAs")
    assert len(chat.at.chat_input) == 1


# J.3 — active result is separate from chat ---------------------------------


def test_latest_result_attached_to_assistant_message(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    last = _last_assistant(chat.get("chat_messages") or [])
    assert last is not None
    assert (last.get("attachment") or {}).get("type") == "result"
    # When we ask a different question, the new assistant message carries the
    # new result; the old message keeps its prior result attached. Filter to
    # query-result assistant turns so the polish-phase upload greeting (no
    # attachment) doesn't confuse the index math.
    chat.send("How many students are in each department?")
    assistants_with_results = [
        m for m in _assistants(chat.get("chat_messages") or [])
        if (m.get("attachment") or {}).get("type") == "result"
    ]
    assert len(assistants_with_results) >= 2
    assert (assistants_with_results[-1].get("attachment") or {}).get("type") == "result"
    assert (assistants_with_results[0].get("attachment") or {}).get("type") == "result"
    # The two results are not the same object.
    assert assistants_with_results[0]["attachment"] is not assistants_with_results[-1]["attachment"]
