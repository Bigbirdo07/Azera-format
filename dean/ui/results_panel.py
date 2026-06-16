"""Result rendering for the unified assistant.

Three views, matching the three safe paths:
  - Answer only      (ask_question): question interpreted, calculation, result,
                     optional table preview, and an explicit "no changes" note.
  - Needs clarification (clarify): the focused question plus clickable options.
  - Workbook change  (edit_workbook): the plan summary/steps shown alongside the
                     existing confirm-and-run controls in app.render_action_panel.

This module is presentation only. It never executes actions and never writes a
file; the workbook-change controls live in app.py and require user confirmation.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st


def _badge(label: str, tone: str = "gray") -> None:
    st.markdown(
        f'<span class="status-badge badge-{tone}">{label}</span>',
        unsafe_allow_html=True,
    )


def render_answer_view(on_action: Callable[[str], None] | None = None) -> None:
    state = st.session_state
    explanation = state.get("ask_explanation")
    description = state.get("ask_description") or ""
    narration = state.get("ask_narration") or ""
    assumption_note = state.get("ask_assumption_note") or ""
    band = state.get("ask_band") or "high"
    conversation_llm_used = bool(state.get("ask_conversation_llm_used"))
    value = state.get("ask_value")
    row_count = state.get("ask_row_count")

    header = "Result"
    if isinstance(row_count, int):
        header = f"Result · {row_count} matching record(s)"
    st.markdown(f"#### {header}")
    # Medium-confidence bands explicitly call out the assumption; high-confidence
    # reads just show the narration. The conversational LLM (when on) folds the
    # interpretation into its own reply.
    if not conversation_llm_used:
        if band == "medium" and assumption_note:
            st.markdown(f"_{assumption_note}_")
        elif narration:
            st.markdown(f"_{narration}_")
    st.write(explanation or description)
    if value is not None:
        operation = (state.get("ask_operation") or "").replace("_", " ").title()
        st.metric(operation or "Result", value)

    table = state.get("ask_table") or []
    if table:
        frame = pd.DataFrame(table)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if state.get("ask_preview_truncated"):
            st.caption("Showing a preview; more rows match.")
        st.download_button(
            "Download visible result (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="result.csv",
            mime="text/csv",
            key="dl_result",
            use_container_width=True,
        )

    if on_action is not None:
        _render_suggestion_chips(state, on_action)

    redacted = state.get("ask_redacted") or []
    if redacted:
        st.caption("Some sensitive fields are hidden by default.")
    if state.get("ask_review_note"):
        st.caption(state["ask_review_note"])
    st.caption("No workbook changes were made.")


def _render_suggestion_chips(state, on_action: Callable[[str], None]) -> None:
    """Render alternative interpretations (medium confidence) and next moves."""
    alternatives = list(state.get("ask_alternatives") or [])
    suggestions = list(state.get("ask_suggestions") or [])

    if alternatives:
        st.caption("Did I get that right? Try a different definition:")
        alt_cols = st.columns(min(len(alternatives), 3))
        for index, alternative in enumerate(alternatives[:3]):
            if alt_cols[index].button(alternative, key=f"alt_{index}", use_container_width=True):
                on_action(alternative)
                st.rerun()

    if suggestions:
        st.caption("Next moves:")
        sug_cols = st.columns(min(len(suggestions), 3))
        for index, suggestion in enumerate(suggestions[:3]):
            if sug_cols[index].button(suggestion, key=f"sug_{index}", use_container_width=True):
                on_action(suggestion)
                st.rerun()


def render_clarify_view(on_option: Callable[[str], None]) -> None:
    state = st.session_state
    pending = (state.get("assistant_memory") or {}).get("pending_action") or {}
    question = state.get("clarify_question") or "Could you clarify what you would like me to do?"
    options = state.get("clarify_options") or []

    if pending:
        # A confirmation, not a clarification — present it as a clear card.
        with st.container(border=True):
            st.markdown("#### Confirmation needed")
            st.write(question)
            if pending.get("type") in {"export", "note_edit", "field_update"}:
                st.caption("This saves a new workbook and leaves the original unchanged.")
            cols = st.columns(2)
            confirm_label, cancel_label = (options + ["Confirm", "Cancel"])[:2] if options else ("Confirm", "Cancel")
            if cols[0].button(confirm_label, key="confirm_yes", type="primary", use_container_width=True):
                on_option("yes")
                st.rerun()
            if cols[1].button(cancel_label, key="confirm_no", use_container_width=True):
                on_option("no")
                st.rerun()
        return

    with st.container(border=True):
        _badge("Needs clarification", "yellow")
        st.warning(question)
        if state.get("clarify_reason"):
            st.caption(state["clarify_reason"])
        if options:
            for index, option in enumerate(options):
                if st.button(option, key=f"clarify_opt_{index}", use_container_width=True):
                    on_option(option)
                    st.rerun()
        else:
            st.caption("Type a more specific request, for example name the action and the column.")


def render_edit_plan_summary(plan: dict[str, Any]) -> None:
    """Shown inside the workbook-change view. Displays the plan in plain English,
    its assumptions, and (for multi-step plans) each step."""
    if not plan:
        return
    summary = plan.get("plain_english_summary")
    if summary:
        st.write("Planned change")
        st.info(summary)

    assumptions = plan.get("assumptions") or []
    if assumptions:
        st.caption("Assumptions: " + "; ".join(assumptions))

    commands = plan.get("commands") or []
    if len(commands) > 1:
        st.write(f"Steps ({len(commands)})")
        for index, command in enumerate(commands, 1):
            action = command.get("action", "?")
            sheet = command.get("sheet", "")
            out = command.get("output_sheet", "")
            target = f" → {out}" if out else ""
            st.caption(f"{index}. {action} on {sheet}{target}")
