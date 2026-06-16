from __future__ import annotations

import streamlit as st

from core.privacy_controls import (
    LOGGING_MODES,
    clear_all_local_logs,
    clear_learned_corrections,
    export_privacy_report,
    load_privacy_settings,
    save_privacy_settings,
)


def render_privacy_admin() -> dict:
    settings = load_privacy_settings()
    with st.expander("Privacy and safety controls"):
        logging_mode = st.selectbox(
            "Logging mode",
            LOGGING_MODES,
            index=LOGGING_MODES.index(settings["logging_mode"]),
        )
        block_delete = st.checkbox(
            "Block delete-row actions by default",
            value=bool(settings["block_delete_row_actions"]),
        )
        warn_threshold = st.number_input(
            "Warn when an action affects this many rows",
            min_value=1,
            value=int(settings["warn_row_threshold"]),
            step=10,
        )
        next_settings = {
            "logging_mode": logging_mode,
            "block_delete_row_actions": block_delete,
            "warn_row_threshold": int(warn_threshold),
        }
        if next_settings != settings:
            save_privacy_settings(next_settings)
            settings = next_settings
            st.success("Privacy settings saved locally.")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Clear all local logs"):
                clear_all_local_logs()
                st.success("Local logs cleared.")
        with col_b:
            if st.button("Clear learned corrections"):
                clear_learned_corrections()
                st.success("Learned corrections cleared.")
        with col_c:
            if st.button("Export privacy report"):
                path = export_privacy_report()
                st.success(f"Privacy report exported to `{path}`")
    return settings
