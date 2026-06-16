from __future__ import annotations

import json
from typing import Any

import streamlit as st

from core.command_schema import SUPPORTED_ACTIONS
from core.correction_manager import (
    export_learning_pack,
    import_learning_pack,
    save_correction,
)
from core.logger import log_user_request
from core.privacy_controls import load_privacy_settings


FEEDBACK_OPTIONS = [
    "Yes",
    "No, wrong action",
    "No, wrong column",
    "No, wrong condition",
    "No, wrong formula/chart/report",
    "Other",
]


def render_feedback_screen(pending_feedback: dict[str, Any], columns: list[str]) -> None:
    if load_privacy_settings()["logging_mode"] == "disabled":
        st.info("Logging is disabled. Feedback will not be stored.")
        return

    st.subheader("Action Feedback")
    st.write("Did I understand your request correctly?")

    feedback = st.radio(
        "Feedback",
        FEEDBACK_OPTIONS,
        horizontal=False,
        key="feedback_choice",
    )

    if feedback == "Yes":
        if st.button("Save feedback", key="save_positive_feedback"):
            log_user_request(
                file_name=pending_feedback["file_name"],
                sheet_name=pending_feedback.get("sheet_name"),
                original_request=pending_feedback.get("original_request", ""),
                generated_command=pending_feedback.get("command", {}),
                parser_confidence=pending_feedback.get("parser_confidence"),
                parser_source=pending_feedback.get("parser_source"),
                action_type=pending_feedback.get("action_type"),
                success=True,
            )
            st.session_state.pop("pending_feedback", None)
            st.success("Saved locally as a successful example.")
        return

    st.write("Correction")
    intended_action = st.selectbox(
        "Intended action",
        sorted(SUPPORTED_ACTIONS),
        index=_action_index(pending_feedback.get("action_type")),
    )
    correct_column = st.selectbox("Correct column", ["", *columns])
    corrected_condition = st.text_input(
        "Correct condition",
        placeholder="Example: Balance Due greater_than 0",
    )
    better_phrase = st.text_input(
        "Better phrase or synonym",
        value=pending_feedback.get("original_request", ""),
    )
    mapped_concept = st.text_input(
        "Map phrase to concept or column",
        value=correct_column or "",
        placeholder="Example: FAFSA Status",
    )
    corrected_json = st.text_area(
        "Corrected JSON command",
        value=json.dumps(
            _default_corrected_command(
                pending_feedback.get("command", {}),
                intended_action,
                correct_column,
            ),
            indent=2,
        ),
        height=260,
    )

    if st.button("Save correction locally", key="save_negative_feedback"):
        try:
            corrected_command = json.loads(corrected_json)
        except json.JSONDecodeError as exc:
            st.error(f"Corrected JSON is invalid: {exc}")
            return

        request_id = log_user_request(
            file_name=pending_feedback["file_name"],
            sheet_name=pending_feedback.get("sheet_name"),
            original_request=pending_feedback.get("original_request", ""),
            generated_command=pending_feedback.get("command", {}),
            parser_confidence=pending_feedback.get("parser_confidence"),
            parser_source=pending_feedback.get("parser_source"),
            action_type=pending_feedback.get("action_type"),
            success=False,
            error_message=feedback,
        )
        save_correction(
            request_id=request_id,
            original_request=pending_feedback.get("original_request", ""),
            incorrect_command=pending_feedback.get("command", {}),
            corrected_command=corrected_command,
            correction_type=feedback,
            better_phrase=better_phrase,
            mapped_concept=mapped_concept,
            raw_column_name=correct_column,
        )
        st.session_state.pop("pending_feedback", None)
        st.success("Correction saved locally. Future parsing can use this learned mapping.")

    if corrected_condition:
        st.caption("Use the corrected JSON field above for executable condition changes.")


def render_learning_admin() -> None:
    with st.expander("Local learning admin"):
        st.caption("Exports and imports contain correction metadata only, not spreadsheet rows.")
        admin_enabled = st.checkbox("Enable admin learning tools")
        if not admin_enabled:
            return

        if st.button("Export local learning pack"):
            export_path = export_learning_pack()
            st.success(f"Exported local learning pack to `{export_path}`")

        uploaded_pack = st.file_uploader(
            "Import local learning pack",
            type=["json"],
            accept_multiple_files=False,
            key="learning_pack_import",
        )
        if uploaded_pack is not None and st.button("Import learning pack"):
            try:
                pack = json.loads(uploaded_pack.getvalue().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                st.error(f"Could not read learning pack: {exc}")
                return

            import_learning_pack(pack)
            st.success("Imported local learning pack.")


def _default_corrected_command(
    generated_command: dict[str, Any],
    intended_action: str,
    correct_column: str,
) -> dict[str, Any]:
    corrected = dict(generated_command)
    corrected["action"] = intended_action

    if correct_column:
        if corrected.get("conditions"):
            corrected["conditions"][0]["column"] = correct_column
        elif intended_action in {"sum_column", "detect_missing_values"}:
            corrected["column"] = correct_column
        elif intended_action == "count_by_group":
            corrected["group_by"] = correct_column
        elif intended_action == "create_chart":
            corrected["group_by"] = correct_column

    return corrected


def _action_index(action_type: str | None) -> int:
    actions = sorted(SUPPORTED_ACTIONS)
    if action_type in actions:
        return actions.index(action_type)
    return 0
