from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from nlp.local_model import get_ollama_status, test_ollama_connection
from nlp.model_prompt import OLLAMA_URL
from core.institution_context import INSTITUTION_MODE_LABELS, ROLE_LABELS, InstitutionMode, Role
from core.interaction_logger import DEFAULT_LOG_PATH, read_records
from core.risk_settings import RiskSettings, load_risk_settings, save_risk_settings
from core.privacy_guard import local_only_security_summary


SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"
DEFAULT_SETTINGS = {
    "strict_privacy_mode": True,
    "use_local_llm": True,
    "ollama_model": "llama3.2:3b",
    "llm_explanations_enabled": False,
    "conversation_llm_enabled": False,
    "planner_full_row_access": False,
    "local_llm_full_row_access": False,
    "local_llm_all_matching_rows": False,
    "interaction_logging_enabled": True,
}
MODEL_OPTIONS = ["llama3.2:3b"]


def load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        save_settings(DEFAULT_SETTINGS)
        return dict(DEFAULT_SETTINGS)
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(DEFAULT_SETTINGS)
    return {**DEFAULT_SETTINGS, **settings}


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _render_interaction_logging_section(settings: dict[str, Any]) -> dict[str, Any]:
    """Toggle + analyzer button for the sanitized interaction learning log."""
    st.write("Interaction learning log")
    st.caption(
        "Local, sanitized record of how requests are phrased and resolved. "
        "Used to spot candidate deterministic rules. Audit logs and learning "
        "logs are separate files."
    )
    enabled = st.checkbox(
        "Log sanitized interaction patterns",
        value=bool(settings.get("interaction_logging_enabled", True)),
        help="No row content, names, emails, IDs, notes, financial values, or "
        "row previews are ever written. PII patterns in the typed message are "
        "redacted before the record is saved.",
        key="interaction_logging_toggle",
    )
    if enabled != bool(settings.get("interaction_logging_enabled", True)):
        settings = {**settings, "interaction_logging_enabled": enabled}
        save_settings(settings)

    record_count = len(read_records(DEFAULT_LOG_PATH))
    st.caption(f"Records on disk: **{record_count}** at `{DEFAULT_LOG_PATH}`")

    if st.button("Generate rule-mining report",
                 disabled=record_count == 0,
                 help="Reads the log and writes outputs/interaction_learning_report.md."):
        _run_log_analyzer()
    return settings


def _run_log_analyzer() -> None:
    """Run the analyzer script and surface the result inline."""
    try:
        # Lazy import to avoid pulling pandas/streamlit-unrelated deps on load.
        from scripts.analyze_interaction_logs import (
            DEFAULT_REPORT_PATH,
            render_report,
            summarize,
        )

        records = read_records(DEFAULT_LOG_PATH)
        if not records:
            st.info("No records to analyze yet.")
            return
        summary = summarize(records)
        report = render_report(summary)
        DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_REPORT_PATH.write_text(report, encoding="utf-8")
        st.success(f"Wrote report → {DEFAULT_REPORT_PATH}")
        with st.expander("Report preview", expanded=False):
            st.markdown(report)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not analyze interaction log: {exc}")


def render_settings_panel() -> dict[str, Any]:
    settings = load_settings()
    risk_settings = load_risk_settings(st.session_state)
    with st.expander("Settings"):
        mode_labels = list(INSTITUTION_MODE_LABELS.values())
        role_labels = list(ROLE_LABELS.values())
        mode_value = st.session_state.get("institution_mode", InstitutionMode.GENERIC.value)
        role_value = st.session_state.get("user_role", Role.ADMIN.value)
        try:
            mode_default = INSTITUTION_MODE_LABELS[InstitutionMode(mode_value)]
        except Exception:
            mode_default = INSTITUTION_MODE_LABELS[InstitutionMode.GENERIC]
        try:
            role_default = ROLE_LABELS[Role(role_value)]
        except Exception:
            role_default = ROLE_LABELS[Role.ADMIN]
        mode = st.selectbox(
            "Institution mode",
            mode_labels,
            index=mode_labels.index(mode_default),
        )
        role = st.selectbox(
            "Role view",
            role_labels,
            index=role_labels.index(role_default),
        )
        st.session_state["institution_mode"] = InstitutionMode.from_label(mode).value
        st.session_state["user_role"] = Role.from_label(role).value

        st.write("Parser mode")
        st.info(
            "Default mode uses the built-in rule-based parser. No model setup is required."
        )
        strict_privacy_mode = st.checkbox(
            "Maximum privacy mode",
            value=bool(settings.get("strict_privacy_mode", True)),
            help="Restricts the local model to schema/summary-level access only — no "
            "full workbook rows or hidden fields. The local model itself stays on. "
            "Recommended for sensitive student data.",
        )
        if strict_privacy_mode != bool(settings.get("strict_privacy_mode", True)):
            settings = {**settings, "strict_privacy_mode": strict_privacy_mode}
            if strict_privacy_mode:
                settings["planner_full_row_access"] = False
                settings["local_llm_full_row_access"] = False
                settings["local_llm_all_matching_rows"] = False
            save_settings(settings)

        # Interaction logging is independent of LLM availability. The log is
        # sanitized at write time (see core/interaction_logger), so it remains
        # available even under maximum privacy mode (Option A).
            settings = _render_interaction_logging_section(settings)
        with st.expander("Risk Settings", expanded=False):
            st.caption("Defaults are shown below; adjust them to match your school.")
            risk_settings = RiskSettings(
                gpa_risk_threshold=float(st.number_input("GPA risk", value=float(risk_settings.gpa_risk_threshold))),
                attendance_risk_threshold=float(st.number_input("Attendance risk", value=float(risk_settings.attendance_risk_threshold))),
                severe_attendance_risk_threshold=float(st.number_input("Severe attendance risk", value=float(risk_settings.severe_attendance_risk_threshold))),
                unexcused_absence_concern=int(st.number_input("Unexcused absence concern", value=int(risk_settings.unexcused_absence_concern))),
                tardy_concern=int(st.number_input("Tardy concern", value=int(risk_settings.tardy_concern))),
                high_risk_signal_count=int(st.number_input("High risk signal count", value=int(risk_settings.high_risk_signal_count))),
                moderate_risk_signal_count=int(st.number_input("Moderate risk signal count", value=int(risk_settings.moderate_risk_signal_count))),
                sat_math_benchmark_threshold=_optional_float(st.text_input("SAT Math benchmark threshold", value=_optional_text(risk_settings.sat_math_benchmark_threshold))),
                sat_ebrw_benchmark_threshold=_optional_float(st.text_input("SAT EBRW benchmark threshold", value=_optional_text(risk_settings.sat_ebrw_benchmark_threshold))),
                psat_math_benchmark_threshold=_optional_float(st.text_input("PSAT Math benchmark threshold", value=_optional_text(risk_settings.psat_math_benchmark_threshold))),
                psat_reading_writing_benchmark_threshold=_optional_float(st.text_input("PSAT Reading/Writing benchmark threshold", value=_optional_text(risk_settings.psat_reading_writing_benchmark_threshold))),
            )
            save_risk_settings(st.session_state, risk_settings)

        strict = bool(settings.get("strict_privacy_mode", True))
        if strict and (
            settings.get("planner_full_row_access")
            or settings.get("local_llm_full_row_access")
            or settings.get("local_llm_all_matching_rows")
        ):
            settings = {
                **settings,
                "planner_full_row_access": False,
                "local_llm_full_row_access": False,
                "local_llm_all_matching_rows": False,
            }
            save_settings(settings)
        if strict:
            st.caption("Maximum privacy mode is on: the local model can plan, explain, and "
                       "converse, but never receives full workbook rows or hidden fields.")

        show_advanced_llm = st.checkbox(
            "Show optional local model setup",
            value=bool(settings.get("use_local_llm", False)),
            help="Admin-only setup for Ollama. Most users can leave this closed.",
        )
        if not show_advanced_llm:
            if settings.get("use_local_llm"):
                settings = {
                    **settings,
                    "use_local_llm": False,
                    "planner_full_row_access": False,
                    "local_llm_full_row_access": False,
                    "local_llm_all_matching_rows": False,
                }
                save_settings(settings)
            st.caption("Local model fallback is off.")
            return settings

        use_local_llm = st.checkbox(
            "Enable local LLM fallback",
            value=bool(settings.get("use_local_llm", False)),
            help="Leave this off unless Ollama is installed and running locally.",
        )
        current_model = str(settings.get("ollama_model", DEFAULT_SETTINGS["ollama_model"]))
        model_options = MODEL_OPTIONS if current_model in MODEL_OPTIONS else [current_model, *MODEL_OPTIONS]
        selected_model = st.selectbox(
            "Local Ollama model",
            model_options,
            index=model_options.index(current_model),
        )
        custom_model = st.text_input(
            "Custom model name",
            value=selected_model,
            help="Default: llama3.2:3b (~2 GB, fast).",
        )
        llm_explanations = st.checkbox(
            "Enable local LLM explanations",
            value=bool(settings.get("llm_explanations_enabled", False)) and use_local_llm,
            disabled=not use_local_llm,
            help="When on, the local model rephrases verified results in plain English. "
            "It only sees the computed summary, never student rows.",
        )
        conversation_llm = st.checkbox(
            "Enable conversational local LLM",
            value=bool(settings.get("conversation_llm_enabled", False)) and use_local_llm,
            disabled=not use_local_llm,
            help="When on, every validated result is passed to the local model so it "
            "can reply conversationally — interpretation, result, and next-move hints "
            "in 1–3 sentences.",
        )
        planner_full_rows = st.checkbox(
            "Give planner LLM workbook rows",
            value=bool(settings.get("planner_full_row_access", False)) and use_local_llm and not strict,
            disabled=not use_local_llm or strict,
            help="When on, the local Ollama planner receives workbook rows, not just schema. "
            "Plans are still validated before pandas executes them. Disabled under maximum "
            "privacy mode.",
        )
        full_row_access = st.checkbox(
            "Give local LLM full matching-row samples",
            value=bool(settings.get("local_llm_full_row_access", False)) and use_local_llm and conversation_llm and not strict,
            disabled=not (use_local_llm and conversation_llm) or strict,
            help="When on, the local Ollama narrator can see a bounded sample of matching rows, "
            "including names and hidden columns, so it can reason over concrete students. "
            "The connection is still restricted to localhost. Disabled under maximum privacy mode.",
        )
        all_matching_rows = st.checkbox(
            "Send all matching rows to conversational LLM",
            value=bool(settings.get("local_llm_all_matching_rows", False)) and use_local_llm and conversation_llm and full_row_access and not strict,
            disabled=not (use_local_llm and conversation_llm and full_row_access) or strict,
            help="When on, the local narrator receives all matched rows instead of the "
            "default bounded sample. This stays local, but large result sets can slow the model. "
            "Disabled under maximum privacy mode.",
        )
        next_settings = {
            "strict_privacy_mode": bool(settings.get("strict_privacy_mode", False)),
            "use_local_llm": use_local_llm,
            "ollama_model": custom_model.strip() or selected_model,
            "llm_explanations_enabled": bool(use_local_llm and llm_explanations),
            "conversation_llm_enabled": bool(use_local_llm and conversation_llm),
            "planner_full_row_access": bool(use_local_llm and planner_full_rows),
            "local_llm_full_row_access": bool(use_local_llm and conversation_llm and full_row_access),
            "local_llm_all_matching_rows": bool(
                use_local_llm and conversation_llm and full_row_access and all_matching_rows
            ),
            "interaction_logging_enabled": bool(settings.get("interaction_logging_enabled", True)),
        }
        if next_settings != settings:
            save_settings(next_settings)
            settings = next_settings

        if settings["use_local_llm"]:
            st.warning(
                "When enabled, requests are sent only to your local Ollama service at localhost. "
                "If planner row access is enabled, planner calls also receive workbook rows. "
                "If full matching-row samples are enabled, the conversational narrator receives "
                "matching rows locally; all-row mode removes the sample cap."
            )
        else:
            st.info("Local LLM fallback is off. Users can continue with the built-in rule-based parser.")

        st.write("Security status")
        for item in local_only_security_summary(
            bool(settings.get("use_local_llm", False)),
            OLLAMA_URL,
        ):
            st.caption(item)

        # Expose the raw LLM Runtime Status details for diagnostic transparency
        from nlp.local_model_manager import get_ollama_manager
        runtime_status = get_ollama_manager().status
        st.write("LLM Runtime Status")
        st.code(
            f"Mode: {runtime_status.mode}\n"
            f"Endpoint: {runtime_status.endpoint}\n"
            f"Model: {runtime_status.model_name}\n"
            f"Server Running: {runtime_status.server_running}\n"
            f"Model Available: {runtime_status.model_available}\n"
            f"Fallback Used: {runtime_status.fallback_used}\n"
            f"Privacy Stance: {runtime_status.privacy_status}\n"
            f"Failure Details: {runtime_status.error_message or 'None'}",
            language="text"
        )

        test_disabled = not settings["use_local_llm"]
        if st.button(
            "Test Ollama connection",
            disabled=test_disabled,
            help="Enable local LLM fallback first. If Ollama is not ready, the app still works without it.",
        ):
            ok, message = test_ollama_connection(settings["ollama_model"])
            if ok:
                st.success(message)
            else:
                st.info(message)

    return settings


def _optional_text(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _optional_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
