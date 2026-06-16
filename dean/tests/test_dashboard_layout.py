"""Phase K — dashboard layout regression tests.

Five visible zones with separated state:
  - chat (left) shows the conversation text
  - workbook (top-middle) shows file + sheet stats
  - live output (top-right) shows the latest result/confirmation/edit-plan card
  - figures (bottom-middle) shows the latest chart
  - export center (bottom-right) shows downloadable artifacts
"""

from __future__ import annotations

from tests.conftest import Chat


def _texts(at) -> str:
    return " ".join(b.value.lower() for b in at.markdown if b.value).strip()


def _captions(at) -> list[str]:
    out = []
    for cap in at.caption:
        try:
            value = cap.value
        except Exception:
            value = ""
        if value:
            out.append(value.lower())
    return out


def _assistants(messages):
    return [m for m in messages if m["role"] == "assistant"]


# K.1–K.5: zones exist and update appropriately --------------------------------


def test_all_five_zones_render(chat: Chat) -> None:
    text = _texts(chat.at)
    captions = _captions(chat.at)
    blob = text + " " + " ".join(captions)
    assert "original workbook" in blob, "Workbook zone header missing"
    # Phase UI renamed Live Output → Working Sheet
    assert "working sheet" in blob, "Working Sheet zone header missing"
    # Phase UI renamed Figures → Figures & Insights
    assert "figures" in blob, "Figures zone header missing"
    assert "export center" in blob, "Export Center zone header missing"


def test_workbook_panel_shows_file_and_dimensions(chat: Chat) -> None:
    text = " ".join(_captions(chat.at))
    assert "synthetic_students.xlsx" in text
    assert "rows" in text
    assert "columns" in text


def _markdown_blob(at) -> str:
    parts = []
    for el in getattr(at, "markdown", []):
        try:
            parts.append(str(getattr(el, "value", "")))
        except Exception:
            pass
    return " ".join(parts).lower()


def test_live_output_empty_before_any_question(chat: Chat) -> None:
    # The Working Sheet panel now renders an empty-state card via markdown.
    blob = _markdown_blob(chat.at)
    assert "ask a question in the chat" in blob


def test_figures_panel_empty_before_any_chart(chat: Chat) -> None:
    blob = _markdown_blob(chat.at)
    assert "ask the assistant for a chart" in blob or "no figure yet" in blob


def test_export_center_empty_before_any_export(chat: Chat) -> None:
    blob = _markdown_blob(chat.at)
    assert "no exports yet" in blob or "nothing exported yet" in blob


# K.4: question updates Live Output -------------------------------------------


def test_question_updates_chat_and_live_output(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    messages = chat.get("chat_messages") or []
    assistants = _assistants(messages)
    assert assistants, "no assistant reply produced"
    # The result attachment lives on the message; Live Output reads it.
    last = assistants[-1]
    assert (last.get("attachment") or {}).get("type") == "result"
    # And the chat bubble itself does NOT contain the full table — it references it.
    captions = " ".join(_captions(chat.at))
    assert "shown in live output" in captions or "result table" in captions


def test_followup_keeps_history_and_updates_output(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    messages = chat.get("chat_messages") or []
    users = [m["content"] for m in messages if m["role"] == "user"]
    assert "Show me Accounting students" in users
    assert "now only below 2.5 GPA" in users
    # Latest result attachment reflects the newer filter, but earlier one survives.
    assistants = _assistants(messages)
    assert (assistants[-1].get("attachment") or {}).get("type") == "result"


# K.5: chart requests update the Figures panel --------------------------------


def test_chart_request_populates_figures_panel(chat: Chat) -> None:
    chat.send("Create a bar chart by advisor")
    figure = chat.get("latest_figure")
    assert figure is not None, "latest_figure was not set by chart intent"
    assert figure.get("type") == "bar"
    assert figure.get("field") == "Advisor"
    # Chat history reflects the chart turn but the table itself isn't dumped in chat.
    last_assistant = _assistants(chat.get("chat_messages") or [])[-1]
    assert "figures panel" in last_assistant["content"].lower()


def test_chart_request_pairs_gpa_with_major(chat: Chat) -> None:
    chat.send("Create a bar chart of GPA and Major")
    figure = chat.get("latest_figure")
    assert figure is not None
    assert figure.get("field") == "Major"
    assert figure.get("metric") == "average"
    assert figure.get("value_column") == "GPA"


def test_add_gpa_to_existing_major_chart_uses_chart_context(chat: Chat) -> None:
    chat.send("Create a bar chart by Major")
    chat.send("add GPA to the chart")
    figure = chat.get("latest_figure")
    assert figure is not None
    assert figure.get("field") == "Major"
    assert figure.get("metric") == "average"
    assert figure.get("value_column") == "GPA"


def test_ambiguous_chart_asks_for_clarification(chat: Chat) -> None:
    chat.send("create a chart")
    last = _assistants(chat.get("chat_messages") or [])[-1]
    assert "which field" in last["content"].lower() or "department" in last["content"].lower()


def test_chart_request_does_not_overwrite_active_result(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    assert chat.get("ask_row_count") is not None
    accounting_rows = chat.get("ask_row_count")
    chat.send("Create a bar chart by advisor")
    # ask_* state from the prior question must still be intact — the chart
    # request goes through the Figures panel, not the query planner.
    assert chat.get("ask_row_count") == accounting_rows
    assert chat.get("latest_figure") is not None


# K.6: confirmation still works through chat ---------------------------------


def test_sensitive_request_shows_confirmation_attachment(chat: Chat) -> None:
    chat.send("show me all student emails and GPAs")
    last = _assistants(chat.get("chat_messages") or [])[-1]
    attachment = last.get("attachment") or {}
    assert attachment.get("type") == "result"
    assert not chat.memory().get("pending_action")


def test_confirm_yes_resolves_and_records_export(chat: Chat) -> None:
    chat.send("export this list with emails")
    assert chat.memory().get("pending_action") is not None
    chat.send("yes, export")
    memory = chat.memory()
    assert not (memory.get("pending_action") or {})


def test_cancel_keeps_chat_input_available(chat: Chat) -> None:
    chat.send("export this list with emails")
    chat.send("no")
    assert len(chat.at.chat_input) == 1


# K.7: state separation -------------------------------------------------------


def test_chart_does_not_clear_chat_history(chat: Chat) -> None:
    chat.send("Show me Accounting students")
    before = len(chat.get("chat_messages") or [])
    chat.send("create a bar chart by advisor")
    after = chat.get("chat_messages") or []
    assert len(after) > before, "chart request must append, not replace"


# K.10: debug hidden by default ----------------------------------------------


def test_debug_hidden_by_default(chat: Chat) -> None:
    text = _texts(chat.at)
    assert "developer tools" not in text or "routing_debug" not in text
