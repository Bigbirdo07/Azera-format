"""Alternative + suggestion chips wire through `route_message` and attribute
clicks correctly (alternative clicks become corrections of the prior turn).

The chip buttons live inside the chat panel's result card. We test the
underlying contract here (attachment shape + correction routing) without
having to simulate a real button click, then verify chip routing end-to-end
in the Chat fixture.
"""

from __future__ import annotations

import json

import pytest

import ui.chat_panel as chat_panel


# ---- attachment carries alternatives, suggestions, and entry_id ------------


def test_result_attachment_carries_chips_and_entry_id(chat):
    """A medium-band turn should write an attachment with alternatives,
    suggestions, assumption_note, band, and an entry_id stamped post-log."""
    chat.send("show me struggling students")
    messages = chat.get("chat_messages") or []
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    assert assistant_messages, "expected an assistant turn"
    attachment = assistant_messages[-1].get("attachment") or {}
    assert attachment.get("type") == "result"
    assert attachment.get("band") == "medium"
    assert attachment.get("alternatives"), "medium-band attachment must offer alternatives"
    assert attachment.get("suggestions"), "result attachment must include next-move suggestions"
    assert attachment.get("entry_id"), (
        "entry id must be stamped onto the attachment so chips can correct it"
    )


# ---- clicking an alternative routes the chip text + attributes correction --


def test_alternative_chip_click_routes_and_corrects(chat, monkeypatch):
    """Simulate the chip click handler: set _force_correction_target, then
    route the chip label. The next turn should be logged with corrects_entry_id
    pointing at the original turn."""
    chat.send("show me struggling students")
    first_messages = chat.get("chat_messages") or []
    first_attachment = next(
        (m["attachment"] for m in reversed(first_messages)
         if m.get("role") == "assistant" and m.get("attachment", {}).get("type") == "result"),
        None,
    )
    assert first_attachment and first_attachment.get("entry_id")
    original_entry_id = first_attachment["entry_id"]

    # Mimic what the chip button does: stash the force target, then send the chip text.
    chat.at.session_state["_force_correction_target"] = original_entry_id
    chip_label = first_attachment["alternatives"][0]
    chat.send(chip_label)

    # The just-logged turn must reference the original entry as the correction target.
    from core.interaction_logger import DEFAULT_LOG_PATH

    records = [json.loads(line) for line in DEFAULT_LOG_PATH.read_text().splitlines() if line.strip()]
    assert records[-1]["corrects_entry_id"] == original_entry_id, (
        f"Chip click must log a correction of the original entry. Got: {records[-1]}"
    )
    assert records[-1]["user_corrected"] is True


def test_suggestion_chip_click_does_not_attribute_correction(chat):
    """Suggestion chips (Group by Advisor, Export, etc.) should NOT mark the
    next turn as a correction of the prior assumption — they are next moves,
    not different interpretations."""
    chat.send("show me struggling students")
    first_messages = chat.get("chat_messages") or []
    first_attachment = next(
        (m["attachment"] for m in reversed(first_messages)
         if m.get("role") == "assistant" and m.get("attachment", {}).get("type") == "result"),
        None,
    )
    assert first_attachment and first_attachment.get("suggestions")
    chip_label = first_attachment["suggestions"][0]
    # No _force_correction_target — that's the suggestion-chip code path.
    chat.send(chip_label)

    from core.interaction_logger import DEFAULT_LOG_PATH

    records = [json.loads(line) for line in DEFAULT_LOG_PATH.read_text().splitlines() if line.strip()]
    assert records[-1]["corrects_entry_id"] in (None, ""), (
        f"Suggestion chip should not be a correction. Got: {records[-1]}"
    )


# ---- chat history is preserved -------------------------------------------


def test_chat_history_preserved_across_chip_clicks(chat):
    chat.send("show me struggling students")
    initial_messages = list(chat.get("chat_messages") or [])
    assert initial_messages

    first_attachment = next(
        (m["attachment"] for m in reversed(initial_messages)
         if m.get("role") == "assistant" and m.get("attachment", {}).get("type") == "result"),
        None,
    )
    chip_label = first_attachment["alternatives"][0]
    chat.send(chip_label)

    after = chat.get("chat_messages") or []
    # The original user/assistant pair must still be in chat history.
    assert len(after) > len(initial_messages)
    # And the original turn's content is unchanged.
    for original_msg in initial_messages:
        assert any(m.get("content") == original_msg.get("content") for m in after), (
            "Original chat history must be preserved verbatim"
        )


# ---- privacy: chip payload contains no rows ------------------------------


def test_chip_attachment_carries_no_row_data(chat):
    """The alternatives/suggestions list shipped to the UI must not contain
    row content, names, emails, or anything that could leak data."""
    chat.send("show me struggling students")
    messages = chat.get("chat_messages") or []
    last_assistant = next(m for m in reversed(messages)
                          if m.get("role") == "assistant"
                          and m.get("attachment", {}).get("type") == "result")
    attachment = last_assistant["attachment"]
    blob = json.dumps({
        "alternatives": attachment.get("alternatives"),
        "suggestions": attachment.get("suggestions"),
        "assumption_note": attachment.get("assumption_note"),
    })
    for forbidden in ("@example", "555-", "S0001", "Maria Lopez"):
        assert forbidden not in blob


# ---- force-correction-target is single-shot (popped, not lingering) ------


def test_force_correction_target_is_popped_after_use(chat):
    chat.send("show me struggling students")
    chat.at.session_state["_force_correction_target"] = "test-entry-id"
    chat.send("now group these by Advisor")
    # After consumption, it must not linger and pollute subsequent turns.
    assert chat.at.session_state.get("_force_correction_target") is None \
        if "_force_correction_target" in chat.at.session_state else True
