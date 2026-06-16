from __future__ import annotations

import json
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import streamlit as st

from core.action_engine import ActionResult, execute_command, execute_plan
from core.action_safety import (
    affected_columns,
    affected_row_count,
    change_summary,
    is_delete_row_action,
)
from core.audit_logger import log_audit_event
from core.command_schema import OPERATORS_WITHOUT_VALUE, SUPPORTED_ACTIONS, VALID_OPERATORS, default_command
from core.correction_manager import save_correction, sync_learning_files
from core.excel_loader import load_excel_workbook
from core.exporter import export_edited_workbook, export_workbook_copy
from core.logger import log_user_request
from core.privacy_controls import load_privacy_settings
from core.validator import validate_command
from core.workbook_diagnostics import (
    diagnose_workbook,
    diagnostics_for_sheet,
    sheet_requires_edit_warning,
)
from core.institution_context import InstitutionMode, Role
from core.workbook_profiler import profile_workbook
from database.db import initialize_database
from nlp.llm_json_parser import command_to_confirmation
from nlp.synonym_mapper import load_json, match_column_by_terms, match_column_for_concept, normalize_text
from core.attendance import load_attendance_file, match_attendance_to_roster
from core.data_sources import AttendanceSource, DataSourceRegistry
from core.excel_loader import LoadedWorkbook
from ui.chat_panel import (
    ensure_session_workbook,
    render_chat_panel,
    render_live_output_panel,
    route_message,
)
from ui.correction_screen import render_feedback_screen, render_learning_admin
from ui.auth_screen import render_user_admin, render_user_bar, require_login
from ui.figures_panel import (
    ChartIntent,
    build_altair_chart,
    export_latest_figure_csv,
    render_figures_panel,
)
from ui.privacy_admin import render_privacy_admin
from ui.health_check import render_workbook_health_check
from ui.results_panel import render_answer_view, render_clarify_view, render_edit_plan_summary
from ui.settings_panel import render_settings_panel
from core.risk_settings import load_risk_settings
from ui.workbook_workspace import render_workbook_workspace


st.set_page_config(
    page_title="Dean Assistant",
    page_icon="📘",
    layout="wide",
)


def render_sheet_metadata(sheet_profile) -> None:
    with st.expander(sheet_profile.name, expanded=True):
        st.metric("Rows", sheet_profile.row_count)

        if sheet_profile.columns:
            st.write("Columns")
            st.code("\n".join(sheet_profile.columns), language="text")
        else:
            st.info("No columns found on this sheet.")


def render_action_builder(profile) -> dict[str, Any]:
    st.write("Action Builder")

    action = st.selectbox("Action", sorted(SUPPORTED_ACTIONS))
    sheet = st.selectbox("Sheet", profile.sheet_names)
    sheet_profile = next(item for item in profile.sheets if item.name == sheet)
    columns = sheet_profile.columns

    command: dict[str, Any] = {
        "action": action,
        "sheet": sheet,
    }

    if action == "create_chart":
        chart_type = st.selectbox(
            "Chart type",
            ["bar", "column", "line", "pie", "stacked_bar"],
        )
        metric = st.selectbox(
            "Metric",
            ["count_rows", "sum", "count_missing"],
        )
        command["chart_type"] = chart_type
        command["metric"] = metric
        command["group_by"] = st.selectbox("Group by", columns or [""])

        if metric in {"sum", "count_missing"}:
            command["value_column"] = st.selectbox("Value column", columns or [""])

        default_sheet_name = f"{command['group_by']} Chart" if command.get("group_by") else "Chart"
        command["output_sheet"] = st.text_input("Output sheet", value=default_sheet_name)
        command["title"] = st.text_input("Chart title", value=default_sheet_name)

    elif action == "create_report":
        report_type = st.selectbox(
            "Report type",
            [
                "enrollment_summary",
                "missing_fafsa",
                "outstanding_balance",
                "inactive_withdrawn",
                "program_enrollment",
                "advisor_caseload",
                "registration_status",
                "missing_fafsa_and_balance",
            ],
        )
        command["report_type"] = report_type
        command["include_summary"] = st.checkbox("Include summary", value=True)
        command["include_chart"] = st.checkbox("Include chart", value=True)
        command["output_sheet"] = st.text_input("Output sheet", value=report_type.replace("_", " ").title())
        st.caption("Use Natural language or Manual JSON to add report-specific filters.")

    elif action == "create_data_quality_report":
        command["output_sheet"] = st.text_input("Output sheet", value="Data Quality Report")
        st.caption("Scans common enrollment data issues and writes issue summaries with example row numbers only.")

    elif action == "create_formula":
        formula_type = st.selectbox(
            "Formula type",
            ["IF", "SUM", "AVERAGE", "COUNT", "COUNTA", "COUNTIF", "SUMIF", "XLOOKUP", "CONCAT", "TEXT", "TODAY"],
        )
        command["formula_type"] = formula_type

        if formula_type == "IF":
            new_column = st.text_input("New column", value="Formula Flag")
            condition_col, operator_col, value_col = st.columns([2, 2, 2])
            with condition_col:
                condition_column = st.selectbox("Condition column", columns)
            with operator_col:
                operator = st.selectbox("Formula operator", sorted(VALID_OPERATORS))
            with value_col:
                value = st.text_input("Formula value", value="0")
            command["new_column"] = new_column
            command["logic"] = {
                "condition_column": condition_column,
                "operator": operator,
                "true_value": "Yes",
                "false_value": "No",
            }
            if operator not in {"is_missing", "is_not_missing"}:
                command["logic"]["value"] = _parse_json_value(value)

        elif formula_type in {"SUM", "AVERAGE", "COUNT", "COUNTA", "TEXT"}:
            command["column"] = st.selectbox("Formula column", columns or [""])
            if formula_type == "TEXT":
                command["new_column"] = st.text_input("New column", value=f"{command['column']} Text")
                command["format_text"] = st.text_input("Excel TEXT format", value="0")

        elif formula_type in {"COUNTIF", "SUMIF"}:
            condition_col, operator_col, value_col = st.columns([2, 2, 2])
            with condition_col:
                criteria_column = st.selectbox("Criteria column", columns)
            with operator_col:
                operator = st.selectbox("Criteria operator", sorted(VALID_OPERATORS))
            with value_col:
                value = st.text_input("Criteria value", value="0")
            criterion = {"column": criteria_column, "operator": operator}
            if operator not in {"is_missing", "is_not_missing"}:
                criterion["value"] = _parse_json_value(value)
            command["criteria"] = [criterion]
            if formula_type == "SUMIF":
                command["sum_column"] = st.selectbox("Sum column", columns or [""])

        elif formula_type == "XLOOKUP":
            command["new_column"] = st.text_input("New column", value="Lookup Result")
            lookup_sheets = [name for name in profile.sheet_names if name != sheet]
            if lookup_sheets:
                lookup_sheet = st.selectbox("Lookup sheet", lookup_sheets)
                lookup_columns = _selected_sheet_columns(profile, lookup_sheet)
                command["lookup"] = {
                    "lookup_sheet": lookup_sheet,
                    "lookup_value_column": st.selectbox("Value column on this sheet", columns),
                    "lookup_key_column": st.selectbox("Key column on lookup sheet", lookup_columns),
                    "return_column": st.selectbox("Return column", lookup_columns),
                }
            else:
                st.warning("Add another sheet to build a lookup formula.")

        elif formula_type == "CONCAT":
            command["new_column"] = st.text_input("New column", value="Combined Text")
            command["columns"] = st.multiselect("Columns to concatenate", columns, default=columns[:2])

    elif action in {"filter_rows", "highlight_rows"}:
        if not columns:
            st.warning("This sheet has no columns available for conditions.")
            return command

        condition_col, operator_col, value_col = st.columns([2, 2, 2])
        with condition_col:
            column = st.selectbox("Condition column", columns)
        with operator_col:
            operator = st.selectbox("Operator", sorted(VALID_OPERATORS))
        with value_col:
            value = st.text_input("Value", value="0")

        condition: dict[str, Any] = {
            "column": column,
            "operator": operator,
        }
        if operator not in {"is_missing", "is_not_missing"}:
            condition["value"] = _parse_json_value(value)
        command["conditions"] = [condition]

        if action == "highlight_rows":
            fill_color = st.selectbox(
                "Fill color",
                ["yellow", "green", "red", "blue", "orange"],
            )
            command["format"] = {"fill_color": fill_color}

    elif action == "sum_column":
        numeric_candidates = columns or [""]
        command["column"] = st.selectbox("Column to sum", numeric_candidates)

    elif action == "count_rows":
        use_condition = st.checkbox("Count only rows matching a condition")
        if use_condition and columns:
            condition_col, operator_col, value_col = st.columns([2, 2, 2])
            with condition_col:
                column = st.selectbox("Count condition column", columns)
            with operator_col:
                operator = st.selectbox("Count operator", sorted(VALID_OPERATORS))
            with value_col:
                value = st.text_input("Count value", value="0")
            condition = {
                "column": column,
                "operator": operator,
            }
            if operator not in {"is_missing", "is_not_missing"}:
                condition["value"] = _parse_json_value(value)
            command["conditions"] = [condition]

    elif action == "count_by_group":
        command["group_by"] = st.selectbox("Group by column", columns or [""])

    elif action == "detect_missing_values":
        selected = st.selectbox("Column", ["All columns", *columns])
        if selected != "All columns":
            command["column"] = selected

    elif action == "remove_duplicates":
        selected_columns = st.multiselect(
            "Duplicate check columns",
            columns,
            default=columns,
        )
        if selected_columns:
            command["columns"] = selected_columns

    elif action == "format_report":
        st.caption("Applies header styling, filters, frozen panes, and column widths.")

    return command


def _parse_json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _selected_sheet_columns(profile, sheet_name: str) -> list[str]:
    sheet_profile = next(item for item in profile.sheets if item.name == sheet_name)
    return sheet_profile.columns


PROMPT_SUGGESTIONS = [
    "Highlight students who still owe money",
    "Show students missing FAFSA",
    "Count active students by program",
    "Create a chart of enrollment by program",
    "Check this file for missing data",
    "Format this report",
]

HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
CLARIFICATION_ACTIONS = {
    "filter rows": "filter_rows",
    "highlight rows": "highlight_rows",
    "create formula": "create_formula",
    "create chart": "create_chart",
    "create report": "create_report",
    "count/group data": "count_by_group",
    "format report": "format_report",
}
CLARIFICATION_OPERATORS = [
    "is_missing",
    "is_not_missing",
    "equals",
    "contains",
    "not_equals",
    "greater_than",
    "less_than",
    "greater_or_equal",
    "less_or_equal",
]


def initialize_ui_state() -> None:
    defaults: dict[str, Any] = {
        "chat_messages": [],
        "current_request": "",
        "current_command": {},
        "current_confirmation": None,
        "current_confidence": None,
        "current_source": None,
        "current_validation_error": None,
        "current_can_execute": False,
        "current_clarification": None,
        "current_confidence_level": None,
        "current_warning": None,
        "edit_command_mode": False,
        "show_correction_form": False,
        "action_history": [],
        "latest_output_file": None,
        "latest_result_preview": None,
        "latest_result_message": None,
        "latest_result_sheet": None,
        "assistant_mode": None,
        "current_plan": {},
        "assistant_memory": {},
        "workspace_view": {
            "mode": "upload",
            "workbook_name": None,
            "active_sheet": None,
            "row_count": None,
            "column_count": None,
            "detected_columns": {},
            "original_preview_df": None,
            "result_df": None,
            "pending_preview_df": None,
            "export_preview_df": None,
            "active_filter": None,
            "group_by": None,
            "columns_used": [],
            "pending_action_summary": None,
            "affected_rows": None,
            "export_filename": None,
            "download_path": None,
            "change_summary": []
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def active_theme() -> str:
    """Current UI theme. Light is the professional default; dark is opt-in."""
    theme = st.session_state.get("ui_theme", "light")
    return theme if theme in ("light", "dark") else "light"


_DARK_THEME_CSS = """
        <style>
        /* Workspace background — deep dark academic blue-slate */
        .stApp {
            background: radial-gradient(circle at top left, #0D1625 0%, #080C14 100%) !important;
            color: #E2E8F0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        .block-container { padding-top: 2.75rem; max-width: 100%; }
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stDeployButton"], .stDeployButton, #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #0B0F19;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] * {
            color: #CBD5E1;
            font-family: 'Inter', sans-serif !important;
        }
        [data-testid="stIconMaterial"],
        section[data-testid="stSidebar"] [data-testid="stIconMaterial"],
        span.material-symbols-rounded, span.material-symbols-outlined {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
        }

        /* Base text color & typography */
        body, .stMarkdown, .stMarkdown p, label, .stTextInput label,
        .stSelectbox label, .stCheckbox label {
            color: #E2E8F0 !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Card chrome — translucent slate-800 card styling */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(19, 28, 46, 0.45) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 20px 22px !important;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3) !important;
            backdrop-filter: blur(10px);
            margin-bottom: 8px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(255, 255, 255, 0.12) !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.4) !important;
        }

        /* Card title + subtitle */
        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 2px;
            letter-spacing: -0.01em;
        }
        .card-subtitle {
            font-size: 0.85rem;
            color: #94A3B8;
            margin-bottom: 16px;
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #F8FAFC !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }
        h1 {
            font-size: 1.75rem !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #F8FAFC 0%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .app-tagline { color: #64748B; font-size: 0.95rem; margin-top: 4px; }

        /* Captions — muted slate */
        [data-testid="stCaptionContainer"], .stCaption { color: #64748B !important; }

        /* Buttons — sleek border-outline with blue glow transitions */
        .stButton button {
            background: rgba(30, 41, 59, 0.5);
            color: #E2E8F0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            padding: 6px 16px;
        }
        .stButton button:hover {
            background: #1E3A8A;
            border-color: #3B82F6;
            color: #EFF6FF;
            box-shadow: 0 0 12px rgba(59, 130, 246, 0.3);
            transform: translateY(-1px);
        }
        .stButton button:active {
            transform: translateY(0);
        }
        .stDownloadButton button {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF;
            border: 1px solid #1D4ED8;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        }
        .stDownloadButton button:hover {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
            transform: translateY(-1px);
        }

        /* Inputs — dark fill, light text */
        [data-baseweb="input"], [data-baseweb="select"] > div {
            background-color: #111827 !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
        }
        [data-baseweb="input"] input,
        [data-baseweb="select"] div,
        textarea {
            color: #F8FAFC !important;
        }
        [data-testid="stChatInput"] {
            background-color: #111827 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        [data-testid="stChatInput"] textarea {
            color: #F8FAFC !important;
            background-color: transparent !important;
        }

        /* Status badges */
        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 18px;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(30, 41, 59, 0.4);
            color: #E2E8F0;
            line-height: 1.2;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .badge-green  { background: rgba(34,197,94,0.12);  color: #4ADE80; border-color: rgba(34,197,94,0.30); }
        .badge-yellow { background: rgba(234,179,8,0.12);  color: #FBBF24; border-color: rgba(234,179,8,0.30); }
        .badge-red    { background: rgba(239,68,68,0.12);  color: #F87171; border-color: rgba(239,68,68,0.30); }
        .badge-gray   { background: rgba(30, 41, 59, 0.3); color: #94A3B8; border-color: rgba(255,255,255,0.06); }
        .workflow-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 12px 0;
        }
        .workflow-chip {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0.28rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #CBD5E1;
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.25);
            line-height: 1.2;
        }

        /* Empty-state — dashed dark card */
        .empty-state {
            border: 1px dashed rgba(255, 255, 255, 0.15) !important;
            border-radius: 14px !important;
            padding: 20px;
            color: #94A3B8;
            background: rgba(15, 23, 42, 0.3) !important;
            font-size: 0.92rem;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        .empty-state strong { color: #F8FAFC; }
        .empty-state em { color: #93C5FD; font-style: normal; }
        .empty-state ul { margin: 8px 0 0 20px; padding: 0; }
        .empty-state li { margin: 4px 0; color: #CBD5E1; }

        /* Metrics */
        [data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"] { color: #94A3B8 !important; }

        /* Chat history bubbles — elegant card look */
        [data-testid="stChatMessage"] {
            background-color: rgba(30, 41, 59, 0.25) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 14px !important;
            margin-bottom: 12px !important;
            padding: 14px 16px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }
        [data-testid="stChatMessage"] * { color: #E2E8F0 !important; }

        /* Custom scrollbar for chat history container */
        div[data-testid="stChatMessageContainer"] {
            padding-right: 6px;
        }
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.12);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.25);
        }

        /* Tables (dataframe) — dark rows, light text */
        [data-testid="stDataFrame"] {
            background: #0F172A;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stDataFrame"] * { color: #E2E8F0 !important; }

        /* Expander header */
        [data-testid="stExpander"] details summary {
            color: #F8FAFC;
            font-weight: 600;
        }
        [data-testid="stExpander"] details {
            background: rgba(19, 28, 46, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
        }

        /* Tabs — segment control layout */
        [data-baseweb="tab-list"] {
            background: rgba(15, 23, 42, 0.3) !important;
            border-radius: 10px;
            padding: 4px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 12px;
        }
        [data-baseweb="tab"] {
            color: #94A3B8 !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            border-bottom: none !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            color: #F8FAFC !important;
            background-color: #2563EB !important;
        }

        /* File uploader — keep readable on dark */
        [data-testid="stFileUploader"] section {
            background-color: rgba(17, 24, 39, 0.4) !important;
            border: 2px dashed rgba(255, 255, 255, 0.12) !important;
            border-radius: 12px;
            color: #CBD5E1 !important;
            padding: 16px !important;
            transition: all 0.2s ease;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #3B82F6 !important;
            background-color: rgba(17, 24, 39, 0.6) !important;
        }
        [data-testid="stFileUploader"] section * { color: #CBD5E1 !important; }
        [data-testid="stFileUploader"] button {
            background: #1E3A8A;
            color: #EFF6FF;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 4px 12px;
        }
        [data-testid="stFileUploader"] button:hover {
            background: #2563EB;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        }
        </style>
"""


_LIGHT_THEME_CSS = """
        <style>
        /* Workspace background — soft professional light */
        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
            color: #1E293B;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: #F8FAFC !important;
            color: #0F172A !important;
        }
        [data-testid="stHeader"] {
            box-shadow: none !important;
            border-bottom: 1px solid rgba(226, 232, 240, 0.7);
        }
        .block-container { padding-top: 2.75rem; padding-bottom: 2rem; max-width: 100%; }
        /* Clean product chrome: hide Streamlit's toolbar + rainbow decoration. */
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stDeployButton"], .stDeployButton, #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }
        section[data-testid="stSidebar"] * {
            color: #334155;
            font-family: 'Inter', sans-serif !important;
        }
        /* Never override the Material icon font — doing so spills raw ligature
           text (e.g. "arrow_right") over expander/widget labels. */
        [data-testid="stIconMaterial"],
        section[data-testid="stSidebar"] [data-testid="stIconMaterial"],
        span.material-symbols-rounded, span.material-symbols-outlined {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
        }

        /* Base text color & typography */
        body, .stMarkdown, .stMarkdown p, label, .stTextInput label,
        .stSelectbox label, .stCheckbox label {
            color: #1E293B !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Card chrome — white cards with subtle border */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border: 1px solid rgba(15, 23, 42, 0.08) !important;
            border-radius: 16px !important;
            padding: 20px 22px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 6px 16px -8px rgba(15, 23, 42, 0.12) !important;
            margin-bottom: 8px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(37, 99, 235, 0.25) !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 12px 24px -10px rgba(15, 23, 42, 0.18) !important;
        }

        /* Card title + subtitle */
        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 2px;
            letter-spacing: -0.01em;
        }
        .card-subtitle {
            font-size: 0.85rem;
            color: #64748B;
            margin-bottom: 16px;
        }

        /* Headers — navy/slate */
        h1, h2, h3, h4 {
            color: #0F172A !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 700 !important;
        }
        h1 {
            font-size: 1.75rem !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
            color: #0F172A !important;
        }
        .app-tagline { color: #64748B; font-size: 0.95rem; margin-top: 4px; }
        .app-filename { color: #475569; font-size: 0.95rem; font-weight: 500; }

        /* Captions — muted slate */
        [data-testid="stCaptionContainer"], .stCaption { color: #64748B !important; }

        /* Buttons — calm outline, navy hover */
        .stButton button {
            background: #FFFFFF;
            color: #1E293B;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            font-weight: 500;
            transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
            padding: 6px 16px;
        }
        .stButton button:hover {
            background: #F1F5F9;
            border-color: #2563EB;
            color: #1D4ED8;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.12);
        }
        .stButton button:active { transform: translateY(0); }
        .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF;
            border: 1px solid #1D4ED8;
        }
        .stButton button[kind="primary"]:hover {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            color: #FFFFFF;
        }
        /* Popover trigger (header Upload File) — match light buttons */
        [data-testid="stPopover"] button {
            background: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            font-weight: 500 !important;
        }
        [data-testid="stPopover"] button:hover {
            border-color: #2563EB !important;
            color: #1D4ED8 !important;
        }
        .stDownloadButton button {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF;
            border: 1px solid #1D4ED8;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.15);
        }
        .stDownloadButton button:hover {
            background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
            box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
            transform: translateY(-1px);
        }

        /* Inputs — white fill, dark text */
        [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            color: #0F172A !important;
        }
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea,
        [data-baseweb="select"] div,
        textarea {
            color: #0F172A !important;
            caret-color: #2563EB !important;
            background-color: #FFFFFF !important;
        }
        [data-baseweb="input"]:focus-within,
        [data-baseweb="textarea"]:focus-within {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }
        [data-testid="stChatInput"] {
            display: none !important;
        }
        [data-testid="stForm"] {
            border: 0 !important;
            background: transparent !important;
            padding: 0 !important;
        }
        [data-testid="stForm"] [data-baseweb="textarea"] {
            border-radius: 12px !important;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        }
        [data-testid="stForm"] textarea::placeholder,
        [data-baseweb="input"] input::placeholder {
            color: #94A3B8 !important;
            opacity: 1 !important;
        }

        /* Status badges */
        .status-strip { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            border: 1px solid #E2E8F0;
            background: #F8FAFC;
            color: #334155;
            line-height: 1.2;
        }
        .badge-green  { background: #ECFDF5; color: #047857; border-color: #A7F3D0; }
        .badge-yellow { background: #FFFBEB; color: #B45309; border-color: #FDE68A; }
        .badge-red    { background: #FEF2F2; color: #B91C1C; border-color: #FECACA; }
        .badge-gray   { background: #F1F5F9; color: #64748B; border-color: #E2E8F0; }
        .workflow-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 12px 0;
        }
        .workflow-chip {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 0.28rem 0.68rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #334155;
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            line-height: 1.2;
        }

        [data-testid="stAlert"] {
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            color: #334155 !important;
            padding: 0.75rem 0.95rem !important;
        }
        [data-testid="stAlert"] * {
            color: #334155 !important;
        }
        [data-testid="stAlert"][kind="success"],
        [data-testid="stAlert"][data-baseweb*="notification"] {
            background: #F0FDF4 !important;
            border-color: #BBF7D0 !important;
        }

        /* Pills (filename / view summary) */
        .app-pill {
            display: inline-flex; align-items: center;
            padding: 0.2rem 0.7rem; border-radius: 999px;
            font-size: 0.8rem; font-weight: 500;
            background: #EFF6FF; color: #1D4ED8; border: 1px solid #DBEAFE;
        }

        /* Empty-state — dashed light card */
        .empty-state {
            border: 1px dashed #CBD5E1 !important;
            border-radius: 14px !important;
            padding: 20px;
            color: #64748B;
            background: #F8FAFC !important;
            font-size: 0.92rem;
        }
        .empty-state strong { color: #0F172A; }
        .empty-state em { color: #2563EB; font-style: normal; }
        .empty-state ul { margin: 8px 0 0 20px; padding: 0; }
        .empty-state li { margin: 4px 0; color: #475569; }

        /* Metrics */
        [data-testid="stMetricValue"] { color: #0F172A !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"] { color: #64748B !important; }

        /* Chat history bubbles */
        [data-testid="stChatMessage"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
            padding: 14px 16px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        }
        [data-testid="stChatMessage"] * { color: #1E293B !important; }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(15, 23, 42, 0.18); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(15, 23, 42, 0.32); }

        /* Tables (dataframe) — clean light grid */
        [data-testid="stDataFrame"] {
            background: #FFFFFF;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stDataFrame"] * {
            color: #0F172A !important;
        }
        [data-testid="stDataFrame"] canvas {
            color-scheme: light !important;
        }

        /* Expander */
        [data-testid="stExpander"] details summary { color: #0F172A; font-weight: 600; }
        [data-testid="stExpander"] details {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
        }

        /* Tabs + segmented control — light segment look */
        [data-baseweb="tab-list"] {
            background: #F1F5F9 !important;
            border-radius: 10px;
            padding: 4px;
            border: 1px solid #E2E8F0;
            margin-bottom: 12px;
        }
        [data-baseweb="tab"] {
            color: #64748B !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            border-bottom: none !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
            color: #FFFFFF !important;
            background-color: #2563EB !important;
        }
        [data-testid="stSegmentedControl"] {
            background: #F1F5F9;
            border-radius: 10px;
            padding: 4px;
            border: 1px solid #E2E8F0;
        }

        /* File uploader — light dashed dropzone */
        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 2px dashed #CBD5E1 !important;
            border-radius: 12px;
            color: #475569 !important;
            padding: 16px !important;
            transition: all 0.2s ease;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: #2563EB !important;
            background-color: #EFF6FF !important;
        }
        [data-testid="stFileUploader"] section * { color: #475569 !important; }
        [data-testid="stFileUploader"] button {
            background: #2563EB;
            color: #FFFFFF;
            border: 1px solid #1D4ED8;
            border-radius: 8px;
            padding: 4px 12px;
        }
        [data-testid="stFileUploader"] button:hover { background: #1D4ED8; }
        [data-testid="stPopoverBody"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px !important;
            box-shadow: 0 18px 40px -18px rgba(15, 23, 42, 0.35) !important;
        }
        [data-testid="stPopoverBody"] * {
            color: #0F172A !important;
        }
        </style>
"""


def render_product_styles(theme: str | None = None) -> None:
    """Inject the active theme's CSS.

    Light is the professional default for school deployments; the original dark
    academic workspace is preserved as an opt-in theme (sidebar > Appearance).
    """
    theme = theme or active_theme()
    css = _DARK_THEME_CSS if theme == "dark" else _LIGHT_THEME_CSS
    st.markdown(css, unsafe_allow_html=True)


def badge(label: str, tone: str = "gray") -> None:
    st.markdown(
        f'<span class="status-badge badge-{tone}">{label}</span>',
        unsafe_allow_html=True,
    )


from contextlib import contextmanager


@contextmanager
def render_card(title: str, subtitle: str | None = None):
    """Open a styled workspace card. Use as a context manager:

        with render_card("Original Workbook", "Read-only source of truth"):
            st.file_uploader(...)
    """
    container = st.container(border=True)
    with container:
        if title:
            st.markdown(f'<div class="card-title">{title}</div>',
                        unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="card-subtitle">{subtitle}</div>',
                        unsafe_allow_html=True)
        yield container


def render_empty_state(text: str, suggestions: list[str] | None = None) -> None:
    """A subtle dashed-border empty state — replaces big blue st.info alerts."""
    body = f'<div class="empty-state"><strong>{text}</strong>'
    if suggestions:
        body += "<ul>" + "".join(f"<li>{s}</li>" for s in suggestions) + "</ul>"
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)


def confidence_tone(confidence: float | None) -> tuple[str, str]:
    if confidence is None:
        return "Not interpreted", "gray"
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "High confidence", "green"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "Medium confidence", "yellow"
    return "Low confidence", "red"


def confidence_level(confidence: float | None) -> str:
    if confidence is None:
        return "none"
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return "medium"
    return "low"


def workbook_status(profile, diagnostics) -> tuple[str, str]:
    if profile is None:
        return "No file uploaded", "gray"
    has_columns = any(sheet.columns for sheet in profile.sheets)
    has_warnings = any(sheet.data_quality_warnings for sheet in diagnostics.sheets)
    if not has_columns:
        return "Needs column mapping", "yellow"
    if has_warnings:
        return "File loaded", "yellow"
    return "Ready", "green"


def selected_sheet_profile(profile, sheet_name: str):
    return next(item for item in profile.sheets if item.name == sheet_name)


def column_profile(dataframe) -> list[dict[str, Any]]:
    rows = []
    for column in dataframe.columns:
        series = dataframe[column]
        examples = [
            str(value)
            for value in series.dropna().astype(str).head(3).tolist()
            if str(value).strip()
        ]
        missing = int((series.isna() | (series.astype(str).str.strip() == "")).sum())
        rows.append(
            {
                "Column": str(column),
                "Inferred type": str(series.dtype),
                "Missing values": missing,
                "Example values": ", ".join(examples),
            }
        )
    return rows


def explain_command(command: dict[str, Any]) -> str:
    if not command:
        return "No command has been interpreted yet."
    if command.get("action") == "clarify":
        return command.get("question", "The request needs clarification.")
    return command_to_confirmation(command).replace(" Continue?", "")


def reset_current_interpretation() -> None:
    st.session_state["current_request"] = ""
    st.session_state["current_command"] = {}
    st.session_state["current_plan"] = {}
    st.session_state["assistant_mode"] = None
    st.session_state["current_confirmation"] = None
    st.session_state["current_confidence"] = None
    st.session_state["current_source"] = None
    st.session_state["current_validation_error"] = None
    st.session_state["current_can_execute"] = False
    st.session_state["current_clarification"] = None
    st.session_state["current_confidence_level"] = None
    st.session_state["current_warning"] = None
    st.session_state["edit_command_mode"] = False
    st.session_state["show_correction_form"] = False


def _learned_synonym_map() -> dict[str, list[str]]:
    learned = load_json("learned_synonyms.json")
    mapped: dict[str, list[str]] = {}
    if isinstance(learned, list):
        for item in learned:
            phrase = str(item.get("phrase", "")).strip()
            concept = str(item.get("mapped_concept", "")).strip()
            if phrase and concept:
                mapped.setdefault(concept, []).append(phrase)
    return mapped


def _synonyms_with_learned() -> dict[str, list[str]]:
    synonyms = load_json("synonyms.json")
    learned = _learned_synonym_map()
    for concept, phrases in learned.items():
        synonyms.setdefault(concept, [])
        synonyms[concept].extend(phrase for phrase in phrases if phrase not in synonyms[concept])
    return synonyms


def _fuzzy_phrase_in_text(phrase: str, text: str) -> bool:
    phrase_tokens = normalize_text(phrase).split()
    text_tokens = normalize_text(text).split()
    if not phrase_tokens or not text_tokens:
        return False

    for phrase_token in phrase_tokens:
        if phrase_token in text_tokens:
            continue
        if not any(SequenceMatcher(None, phrase_token, text_token).ratio() >= 0.78 for text_token in text_tokens):
            return False
    return True


def likely_action_label(command: dict[str, Any], request: str = "") -> str:
    action = command.get("action")
    reverse = {value: key for key, value in CLARIFICATION_ACTIONS.items()}
    if action in reverse:
        return reverse[action]

    text = normalize_text(request)
    if any(term in text for term in ["highlight", "mark", "color"]):
        return "highlight rows"
    if any(term in text for term in ["chart", "graph", "plot"]):
        return "create chart"
    if any(term in text for term in ["formula", "column that says", "flag"]):
        return "create formula"
    if any(term in text for term in ["report", "summary"]):
        return "create report"
    if any(term in text for term in ["count", "how many", "by"]):
        return "count/group data"
    if any(term in text for term in ["format", "clean"]):
        return "format report"
    return "filter rows"


def suggested_columns_for_request(request: str, columns: list[str], limit: int = 6) -> list[str]:
    if not columns:
        return ["Other"]

    suggestions: list[str] = []
    synonyms = _synonyms_with_learned()
    text = normalize_text(request)

    direct_column, direct_score = match_column_by_terms([request], columns)
    if direct_column and direct_score >= 0.35:
        suggestions.append(direct_column)

    for concept, phrases in synonyms.items():
        if (
            concept in text
            or any(normalize_text(phrase) in text for phrase in phrases)
            or any(_fuzzy_phrase_in_text(phrase, request) for phrase in phrases)
        ):
            column, score = match_column_for_concept(concept, columns, synonyms)
            if column and score >= 0.35:
                suggestions.append(column)

    request_tokens = set(text.split())
    scored_columns: list[tuple[float, str]] = []
    for column in columns:
        column_tokens = set(normalize_text(column).split())
        overlap = len(request_tokens & column_tokens)
        score = overlap / max(len(request_tokens | column_tokens), 1)
        if score > 0:
            scored_columns.append((score, column))

    for _, column in sorted(scored_columns, reverse=True):
        suggestions.append(column)

    unique = []
    for column in suggestions:
        if column not in unique:
            unique.append(column)
    for column in columns:
        if len(unique) >= limit:
            break
        if column not in unique:
            unique.append(column)

    return [*unique[:limit], "Other"]


def default_operator_and_value(request: str) -> tuple[str, str]:
    text = normalize_text(request)
    if any(term in text for term in ["missing", "blank", "empty", "didnt", "didn t", "not submitted", "not complete"]):
        return "is_missing", ""
    if any(term in text for term in ["owe", "owes", "owed", "balance due", "money due"]):
        return "greater_than", "0"
    if "active" in text:
        return "equals", "Active"
    if "inactive" in text:
        return "equals", "Inactive"
    if "withdrawn" in text:
        return "equals", "Withdrawn"
    return "contains", ""


def _condition_from_clarification(column: str, operator: str, raw_value: str) -> dict[str, Any]:
    condition: dict[str, Any] = {"column": column, "operator": operator}
    if operator not in OPERATORS_WITHOUT_VALUE:
        condition["value"] = _parse_json_value(raw_value) if raw_value.strip() else ""
    return condition


def build_clarified_command(
    *,
    request: str,
    action_label: str,
    sheet: str,
    column: str,
    operator: str,
    raw_value: str,
) -> dict[str, Any]:
    action = CLARIFICATION_ACTIONS[action_label]
    command: dict[str, Any] = {"action": action, "sheet": sheet}
    has_column = bool(column and column != "Other")

    if action in {"filter_rows", "highlight_rows"}:
        if has_column:
            command["conditions"] = [_condition_from_clarification(column, operator, raw_value)]
        if action == "highlight_rows":
            command["format"] = {"fill_color": "yellow"}
    elif action == "count_by_group":
        if has_column:
            if " by " in f" {normalize_text(request)} ":
                command["group_by"] = column
            else:
                command["action"] = "count_rows"
                command["conditions"] = [_condition_from_clarification(column, operator, raw_value)]
    elif action == "create_formula":
        if has_column:
            command.update(
                {
                    "new_column": f"{column} Flag",
                    "formula_type": "IF",
                    "logic": {
                        "condition_column": column,
                        "operator": operator,
                        "true_value": "Yes",
                        "false_value": "No",
                    },
                }
            )
            if operator not in OPERATORS_WITHOUT_VALUE:
                command["logic"]["value"] = _parse_json_value(raw_value) if raw_value.strip() else ""
    elif action == "create_chart":
        if has_column:
            command.update(
                {
                    "chart_type": "bar",
                    "group_by": column,
                    "metric": "count_rows",
                    "output_sheet": f"{column} Chart",
                    "title": f"{column} Chart",
                }
            )
    elif action == "create_report":
        command.update(
            {
                "report_type": "enrollment_summary",
                "include_summary": True,
                "include_chart": True,
                "output_sheet": "Clarified Report",
            }
        )
        if has_column:
            command["conditions"] = [_condition_from_clarification(column, operator, raw_value)]
    elif action == "format_report":
        pass

    return command


def save_clarified_request(
    *,
    request: str,
    incorrect_command: dict[str, Any],
    corrected_command: dict[str, Any],
    selected_column: str,
) -> None:
    if not request:
        return
    try:
        save_correction(
            request_id=None,
            original_request=request,
            incorrect_command=incorrect_command or {},
            corrected_command=corrected_command,
            correction_type="clarification",
            better_phrase=request,
            mapped_concept=selected_column if selected_column != "Other" else corrected_command.get("action"),
            raw_column_name=selected_column if selected_column != "Other" else None,
        )
    except Exception as exc:
        st.warning(f"Command rebuilt, but the local correction example could not be saved: {exc}")


def render_workbook_panel(uploaded_file, loaded, profile, diagnostics) -> str | None:
    st.subheader("Excel Workbook")
    uploaded_file = st.file_uploader(
        "Upload Excel workbook",
        type=["xlsx"],
        accept_multiple_files=False,
        key="workbook_upload",
    )

    if uploaded_file is None:
        badge("No file uploaded", "gray")
        st.info("Upload a `.xlsx` file to preview sheets and start asking questions.")
        return None

    if loaded is None or profile is None:
        return None

    status_label, status_tone = workbook_status(profile, diagnostics)
    badge(status_label, status_tone)
    st.write(f"File: `{profile.file_name}`")

    selected_sheet = st.selectbox(
        "Sheet",
        profile.sheet_names,
        key=f"sheet_selector_{profile.file_name}",
    )
    sheet_profile = selected_sheet_profile(profile, selected_sheet)
    dataframe = loaded.sheets[selected_sheet]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Sheets", len(profile.sheet_names))
    metric_cols[1].metric("Rows", sheet_profile.row_count)
    metric_cols[2].metric("Columns", len(sheet_profile.columns))
    metric_cols[3].metric("Preview rows", min(len(dataframe.index), 100))

    st.write("Preview")
    st.dataframe(dataframe.head(100), use_container_width=True)

    with st.expander("Workbook metadata", expanded=False):
        st.write("Column names")
        if sheet_profile.columns:
            st.code("\n".join(sheet_profile.columns), language="text")
        else:
            st.info("No columns found on the selected sheet.")

    with st.expander("Column Profile", expanded=False):
        profile_rows = column_profile(dataframe)
        if profile_rows:
            st.dataframe(profile_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No columns are available to profile.")

    with st.expander("Workbook Health Check", expanded=False):
        render_workbook_health_check(diagnostics)

    return selected_sheet


def render_result_preview() -> None:
    preview = st.session_state.get("latest_result_preview")
    if preview is not None:
        st.write("Latest result preview")
        st.dataframe(preview.head(50), use_container_width=True)


def render_download_latest() -> None:
    output_file = st.session_state.get("latest_output_file")
    if not output_file:
        return
    path = Path(output_file)
    if not path.exists():
        st.warning("The latest output file is no longer available.")
        return
    st.download_button(
        "Download latest edited Excel file",
        data=path.read_bytes(),
        file_name=path.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def run_current_action(
    *,
    loaded,
    current_user,
    permissions: dict[str, bool],
    privacy_settings: dict[str, Any],
) -> None:
    command = st.session_state.get("current_command") or {}
    plan = st.session_state.get("current_plan") or {}
    request = st.session_state.get("current_request", "")
    source = st.session_state.get("current_source")
    confidence = st.session_state.get("current_confidence")

    plan_commands = plan.get("commands") or ([command] if command else [])

    if not permissions["can_execute"]:
        st.error("Your role can preview and ask questions but cannot run actions.")
        return
    if not command and not plan_commands:
        st.error("No interpreted command is ready to run.")
        return
    if st.session_state.get("current_validation_error"):
        st.error("Resolve validation issues before running this action.")
        return
    if any(is_delete_row_action(step) for step in plan_commands) and privacy_settings["block_delete_row_actions"]:
        st.error("This row-removal action is blocked by the current safety settings.")
        return

    try:
        # Multi-step plans and planner-only actions run through execute_plan;
        # legacy single commands (clarification rebuilds, manual JSON) keep the
        # original single-command path.
        if plan.get("commands"):
            plan_result = execute_plan(plan, loaded.workbook, loaded.sheets, loaded.file_name)
            result = ActionResult(
                message=plan_result.message,
                preview=plan_result.preview,
                result_sheet=plan_result.result_sheet,
            )
        else:
            result = execute_command(command, loaded.workbook, loaded.sheets, loaded.file_name)
        export_path = export_edited_workbook(loaded.workbook, loaded.file_name)
        log_audit_event(
            username=current_user.username,
            user_role=current_user.role,
            action_type=command.get("action"),
            columns_affected=affected_columns(command),
            row_count_affected=affected_row_count(command, loaded.sheets),
            success=True,
            source=source,
        )
        st.session_state["latest_output_file"] = str(export_path)
        st.session_state["latest_result_preview"] = result.preview
        st.session_state["latest_result_message"] = result.message
        st.session_state["latest_result_sheet"] = result.result_sheet
        st.session_state.setdefault("export_history", []).insert(0, Path(export_path).name)
        action_name = command.get("action") if isinstance(command, dict) else None
        if action_name in {"create_chart", "create_report", "count_by_group", "summarize_missing"} and result.preview is not None:
            st.session_state.setdefault("figures_history", []).insert(
                0,
                {"sheet": result.result_sheet, "action": action_name, "preview": result.preview, "message": result.message},
            )
        st.session_state["action_history"].insert(
            0,
            {
                "request": request,
                "action": command.get("action"),
                "confidence": confidence,
                "success": True,
                "output": str(export_path),
            },
        )
        st.session_state["pending_feedback"] = {
            "file_name": loaded.file_name,
            "sheet_name": command.get("sheet"),
            "original_request": request,
            "command": command,
            "parser_confidence": confidence,
            "parser_source": source,
            "action_type": command.get("action"),
        }
        st.success(result.message)
    except Exception as exc:
        log_user_request(
            file_name=loaded.file_name,
            sheet_name=command.get("sheet") if isinstance(command, dict) else None,
            original_request=request,
            generated_command=command if isinstance(command, dict) else {},
            parser_confidence=confidence,
            parser_source=source,
            action_type=command.get("action") if isinstance(command, dict) else None,
            success=False,
            error_message=str(exc),
        )
        st.session_state["action_history"].insert(
            0,
            {
                "request": request,
                "action": command.get("action"),
                "confidence": confidence,
                "success": False,
                "output": "",
            },
        )
        st.error(f"Action failed: {exc}")


def render_clarification_panel(loaded, profile) -> None:
    request = st.session_state.get("current_request", "")
    original_command = st.session_state.get("current_command") or {}
    sheet_name = original_command.get("sheet") or profile.sheet_names[0]
    columns = _selected_sheet_columns(profile, sheet_name)
    likely_action = likely_action_label(original_command, request)
    column_options = suggested_columns_for_request(request, columns)
    default_operator, default_value = default_operator_and_value(request)

    st.warning("I need a little more information before I run anything.")
    with st.container(border=True):
        st.write("Help me understand your request.")
        st.caption(f"Detected intent: {likely_action if likely_action else 'Not sure yet'}")

        action_index = list(CLARIFICATION_ACTIONS).index(likely_action) if likely_action in CLARIFICATION_ACTIONS else 0
        action_label = st.selectbox(
            "What should I do?",
            list(CLARIFICATION_ACTIONS),
            index=action_index,
            key="clarify_action_label",
        )

        column_choice = st.selectbox(
            "Which column should I use?",
            column_options,
            key="clarify_column_choice",
        )
        if column_choice == "Other":
            selected_column = st.selectbox(
                "Choose a column",
                columns or ["Other"],
                key="clarify_other_column",
            )
        else:
            selected_column = column_choice

        operator_index = CLARIFICATION_OPERATORS.index(default_operator) if default_operator in CLARIFICATION_OPERATORS else 0
        operator = st.selectbox(
            "Condition",
            CLARIFICATION_OPERATORS,
            index=operator_index,
            key="clarify_operator",
        )

        raw_value = ""
        if operator not in OPERATORS_WITHOUT_VALUE:
            raw_value = st.text_input(
                "Value",
                value=default_value,
                key="clarify_value",
                placeholder="Example: Missing, Active, 0",
            )

        if st.button("Rebuild Command", type="primary", use_container_width=True):
            clarified = build_clarified_command(
                request=request,
                action_label=action_label,
                sheet=sheet_name,
                column=selected_column,
                operator=operator,
                raw_value=raw_value,
            )
            validation_error = None
            try:
                validate_command(clarified, loaded.sheets, loaded.file_name)
            except Exception as exc:
                validation_error = str(exc)

            if validation_error:
                st.session_state["current_command"] = clarified
                st.session_state["current_plan"] = {"plan_type": "single_action", "commands": [clarified]}
                st.session_state["current_validation_error"] = validation_error
                st.session_state["current_can_execute"] = False
            else:
                save_clarified_request(
                    request=request,
                    incorrect_command=original_command,
                    corrected_command=clarified,
                    selected_column=selected_column,
                )
                st.session_state["current_command"] = clarified
                st.session_state["current_plan"] = {"plan_type": "single_action", "commands": [clarified]}
                st.session_state["current_confirmation"] = command_to_confirmation(clarified)
                st.session_state["current_confidence"] = 0.90
                st.session_state["current_confidence_level"] = "high"
                st.session_state["current_source"] = "clarification"
                st.session_state["current_validation_error"] = None
                st.session_state["current_can_execute"] = True
                st.session_state["current_clarification"] = None
                st.session_state["current_warning"] = None
                st.session_state["medium_review_confirmed"] = False
                st.session_state["chat_messages"].append(
                    {
                        "role": "assistant",
                        "content": f"Thanks. I rebuilt the command using {selected_column}. Review it on the right before running.",
                    }
                )
            st.rerun()


def render_modifications_panel(loaded, profile, current_user, permissions, privacy_settings) -> None:
    """Right-top panel: the modification the assistant interpreted from the request,
    along with the run/cancel/correct controls. Excludes figures and exports — those
    live in their own panels."""
    st.subheader("Modifications")
    command = st.session_state.get("current_command") or {}
    confidence = st.session_state.get("current_confidence")
    level = st.session_state.get("current_confidence_level") or confidence_level(confidence)
    label, tone = confidence_tone(confidence)
    badge(label, tone)

    if not loaded:
        st.info("Upload a workbook and ask a question to see the interpreted action.")
        return

    if confidence is not None:
        st.metric("Confidence", f"{confidence:.0%}")
    if st.session_state.get("current_warning"):
        st.warning(st.session_state["current_warning"])

    plan = st.session_state.get("current_plan") or {}
    if plan.get("commands"):
        render_edit_plan_summary(plan)
    elif command:
        st.write("What the assistant understood")
        st.info(explain_command(command))
    else:
        st.caption("Ask a question on the left, then your modification will appear here.")

    validation_error = st.session_state.get("current_validation_error")
    if command:
        if validation_error:
            badge("Needs review", "red")
            st.error(validation_error)
        else:
            badge("Validated", "green")
            try:
                rows = affected_row_count(command, loaded.sheets)
            except Exception:
                rows = 0
            st.caption(f"Estimated affected rows: {rows}")
            if rows >= int(privacy_settings["warn_row_threshold"]):
                st.warning(f"This action may affect {rows} rows.")

        with st.expander("Generated JSON command", expanded=False):
            st.json(command)

        if level == "low":
            render_clarification_panel(loaded, profile)

        if st.session_state.get("edit_command_mode"):
            raw_command = st.text_area(
                "Edit command JSON",
                value=json.dumps(command, indent=2),
                height=260,
            )
            if st.button("Apply edited command", use_container_width=True):
                try:
                    edited = json.loads(raw_command)
                    validate_command(edited, loaded.sheets, loaded.file_name)
                except Exception as exc:
                    st.session_state["current_validation_error"] = str(exc)
                else:
                    st.session_state["current_command"] = edited
                    st.session_state["current_plan"] = {"plan_type": "single_action", "commands": [edited]}
                    st.session_state["current_confirmation"] = command_to_confirmation(edited)
                    st.session_state["current_confidence"] = 0.90
                    st.session_state["current_confidence_level"] = "high"
                    st.session_state["current_source"] = "manual_edit"
                    st.session_state["current_validation_error"] = None
                    st.session_state["current_can_execute"] = True
                    st.session_state["current_clarification"] = None
                    st.session_state["current_warning"] = None
                    st.session_state["edit_command_mode"] = False
                st.rerun()

        medium_confirmed = True
        if level == "medium":
            medium_confirmed = st.checkbox(
                "I reviewed this interpretation and want to allow running it.",
                key="medium_review_confirmed",
            )

        button_cols = st.columns(2)
        with button_cols[0]:
            if st.button(
                "Run Action",
                type="primary",
                use_container_width=True,
                disabled=(
                    bool(validation_error)
                    or not st.session_state.get("current_can_execute")
                    or level == "low"
                    or not medium_confirmed
                ),
            ):
                run_current_action(
                    loaded=loaded,
                    current_user=current_user,
                    permissions=permissions,
                    privacy_settings=privacy_settings,
                )
                st.rerun()
            if st.button("Edit Interpretation", use_container_width=True):
                st.session_state["edit_command_mode"] = True
                st.rerun()
        with button_cols[1]:
            if st.button("Correct Interpretation", use_container_width=True):
                st.session_state["show_correction_form"] = True
                st.rerun()
            if st.button("Cancel", use_container_width=True):
                reset_current_interpretation()
                st.rerun()

    latest_message = st.session_state.get("latest_result_message")
    if latest_message:
        st.success(latest_message)
        if st.session_state.get("latest_result_sheet"):
            st.caption(f"Result sheet: {st.session_state['latest_result_sheet']}")

    with st.expander("Session history", expanded=False):
        history = st.session_state.get("action_history", [])
        if not history:
            st.caption("No actions have run in this session.")
        else:
            st.dataframe(history[:10], use_container_width=True, hide_index=True)

    pending_feedback = st.session_state.get("pending_feedback")
    if pending_feedback and profile is not None:
        with st.expander("After-action feedback", expanded=False):
            feedback_sheet = pending_feedback.get("sheet_name") or profile.sheet_names[0]
            render_feedback_screen(
                pending_feedback,
                _selected_sheet_columns(profile, feedback_sheet),
            )


def _load_or_use_cache(uploaded_file):
    """Load the uploaded workbook once and cache it across reruns.

    Streamlit reruns the whole script on every interaction. Parsing the .xlsx on
    each rerun is what makes follow-up questions feel like the app 'froze' —
    and on flaky uploader state can also cause `loaded` to silently drop. We
    cache the LoadedWorkbook keyed by file name + byte length and only reparse
    when those change.
    """
    if uploaded_file is None:
        view = st.session_state.get("workspace_view")
        mode = view.get("mode", "upload") if isinstance(view, dict) else "upload"
        if mode != "upload" and st.session_state.get("cached_loaded") is not None:
            return (
                st.session_state.get("cached_loaded"),
                st.session_state.get("cached_profile"),
                st.session_state.get("cached_diagnostics"),
                st.session_state.get("cached_load_error"),
            )
        for key in ("cached_loaded", "cached_profile", "cached_diagnostics", "cached_load_key", "cached_load_error"):
            st.session_state.pop(key, None)
        return None, None, None, None

    try:
        file_name = getattr(uploaded_file, "name", "")
        file_bytes = uploaded_file.getvalue()
        cache_key = f"{file_name}:{len(file_bytes)}"
    except Exception:
        cache_key = None

    if cache_key and st.session_state.get("cached_load_key") == cache_key:
        return (
            st.session_state.get("cached_loaded"),
            st.session_state.get("cached_profile"),
            st.session_state.get("cached_diagnostics"),
            st.session_state.get("cached_load_error"),
        )

    # If the uploader's bytes are unavailable but we have a cached workbook from a
    # previous run, keep using it rather than dropping the user back to "no workbook."
    if cache_key is None and st.session_state.get("cached_loaded") is not None:
        return (
            st.session_state.get("cached_loaded"),
            st.session_state.get("cached_profile"),
            st.session_state.get("cached_diagnostics"),
            st.session_state.get("cached_load_error"),
        )

    loaded = None
    profile = None
    diagnostics = None
    load_error = None
    try:
        loaded = load_excel_workbook(uploaded_file)
        profile = profile_workbook(loaded.file_name, loaded.sheets)
        diagnostics = diagnose_workbook(loaded.workbook)
        if not profile.sheet_names:
            load_error = "No usable visible sheets were found in this workbook."
    except Exception as exc:
        load_error = f"Could not read this workbook: {exc}"

    st.session_state["cached_loaded"] = loaded
    st.session_state["cached_profile"] = profile
    st.session_state["cached_diagnostics"] = diagnostics
    st.session_state["cached_load_key"] = cache_key
    st.session_state["cached_load_error"] = load_error
    return loaded, profile, diagnostics, load_error


def main() -> None:
    initialize_database()
    sync_learning_files()
    initialize_ui_state()
    render_product_styles()

    current_user = require_login()
    if current_user is None:
        return

    uploaded_file = st.session_state.get("workbook_upload")
    roster_loaded, _profile_pre, diagnostics, load_error = _load_or_use_cache(uploaded_file)

    # Bind the roster to the per-session DataSourceRegistry, then re-derive an
    # enriched LoadedWorkbook (roster + attendance metrics + combined risk) so
    # every downstream component (chat, session workbook, suggestions, planner)
    # sees the same column-augmented view.
    registry = _ensure_data_source_registry(roster_loaded)
    _ingest_pending_attendance_upload(registry, roster_loaded)
    loaded, profile = _build_enriched_view(roster_loaded, registry)

    _maybe_reset_for_new_workbook(loaded)
    # Bind/refresh the session workbook to whatever is currently loaded. This
    # idempotently triggers reset_for_new_source on a different upload (v0.3
    # "reset only on new upload" contract); a no-op when the user re-uploads
    # the same file.
    ensure_session_workbook(loaded, profile)
    _maybe_post_upload_greeting(loaded, registry)

    privacy_settings = load_privacy_settings()

    with st.sidebar:
        permissions = render_user_bar(current_user)
        if loaded is not None:
            render_sidebar_resets(loaded, profile, "")
            render_sidebar_session_workbook()
        render_sidebar_failure_log()
        settings, privacy_settings, show_debug = render_sidebar_advanced(permissions, privacy_settings)

    # Update and synchronize local LLM manager state
    from nlp.local_model_manager import get_ollama_manager
    manager = get_ollama_manager()
    manager.update_state(settings, show_debug)
    settings_status = manager.status

    if settings.get("strict_privacy_mode", True):
        settings = {**settings, "use_local_llm": False, "llm_explanations_enabled": False}

    # The active sheet (the one the chat planner runs queries against) is the
    # last source sheet the user picked. Generated and chart sheets are views
    # only — they never become the planner's target.
    active_sheet = ""
    if loaded is not None and profile is not None and profile.sheet_names:
        persisted = st.session_state.get(f"active_source_sheet_{profile.file_name}")
        active_sheet = persisted if persisted in profile.sheet_names else profile.sheet_names[0]

    # Sync workspace view state
    view = st.session_state.get("workspace_view")
    if isinstance(view, dict):
        if loaded is not None and profile is not None and active_sheet:
            view["workbook_name"] = profile.file_name
            view["active_sheet"] = active_sheet
            if view.get("mode") == "upload":
                view["mode"] = "original"
        else:
            view["mode"] = "upload"


    render_app_header(loaded, profile, active_sheet)

    # Display warning if local LLM was requested but is currently running in rule fallback mode
    if settings.get("use_local_llm", False) and not settings.get("strict_privacy_mode", True):
        if settings_status.mode == "rule_only" and settings_status.error_message:
            st.warning(f"Local LLM is unavailable: {settings_status.error_message}. Operating in rule-only mode.")

    if load_error:
        st.error(load_error)
        if diagnostics:
            render_workbook_health_check(diagnostics)

    if loaded is not None and active_sheet:
        _handle_inline_edit_actions(loaded, current_user, permissions, privacy_settings)

    # Two-panel workspace: chat on the left, unified workbook view on the
    # right (source sheets + LLM-generated sheets + chart sheets, all in one
    # picker — so results land inline with the source roster).
    chat_col, workbook_col = st.columns([0.36, 0.64], gap="large")

    with chat_col:
        with render_card("Assistant", "Ask questions about your roster."):
            render_chat_panel(
                loaded if active_sheet else None,
                profile if active_sheet else None,
                active_sheet,
                settings,
            )

    with workbook_col:
        render_workbook_workspace(loaded, profile, diagnostics, settings)

    pending_feedback = st.session_state.get("pending_feedback")
    if pending_feedback and profile is not None:
        with st.expander("After-action feedback", expanded=False):
            feedback_sheet = pending_feedback.get("sheet_name") or profile.sheet_names[0]
            render_feedback_screen(
                pending_feedback,
                _selected_sheet_columns(profile, feedback_sheet),
            )

    if loaded and show_debug:
        render_debug_panel(loaded, active_sheet)


def _ensure_data_source_registry(roster_loaded) -> DataSourceRegistry:
    """Return the per-session DataSourceRegistry, freshly created on first
    use and re-bound to the active roster on every rerun."""
    registry: DataSourceRegistry | None = st.session_state.get("data_source_registry")
    if registry is None:
        registry = DataSourceRegistry()
        st.session_state["data_source_registry"] = registry
    registry.risk_settings = load_risk_settings(st.session_state)
    registry.set_roster(roster_loaded)
    return registry


def _ingest_pending_attendance_upload(registry: DataSourceRegistry, roster_loaded) -> None:
    """If the user uploaded a new attendance file this rerun, parse it,
    match against the roster, and bind to the registry.

    We compare upload bytes to a cached key so a rerun without a new upload
    is a no-op (and so re-uploading the same file is also a no-op).
    """
    uploaded = st.session_state.get("attendance_upload")
    if uploaded is None:
        return
    if roster_loaded is None:
        # No roster yet → don't parse attendance (can't match Student IDs).
        return
    try:
        cache_key = f"{getattr(uploaded, 'name', '')}:{len(uploaded.getvalue())}"
    except Exception:
        cache_key = None
    if cache_key and st.session_state.get("attendance_cache_key") == cache_key:
        return
    try:
        loaded_attendance = load_attendance_file(uploaded)
    except Exception as exc:
        st.session_state["attendance_load_error"] = (
            f"Could not read the attendance file: {exc}"
        )
        return
    st.session_state["attendance_load_error"] = None
    target_sheet = registry.enriched_roster_sheet
    roster_frame = roster_loaded.sheets.get(target_sheet) if target_sheet else None
    matched, unmatched, unmatched_ids = (0, 0, [])
    if roster_frame is not None:
        matched, unmatched, unmatched_ids = match_attendance_to_roster(
            loaded_attendance.frame, roster_frame,
        )
    registry.set_attendance(AttendanceSource(
        file_name=loaded_attendance.file_name,
        frame=loaded_attendance.frame,
        matched_count=matched,
        unmatched_count=unmatched,
        unmatched_ids=unmatched_ids,
        warnings=loaded_attendance.warnings,
    ))
    st.session_state["attendance_cache_key"] = cache_key


def _build_enriched_view(
    roster_loaded, registry: DataSourceRegistry,
) -> tuple[Any | None, Any | None]:
    """Re-derive a (LoadedWorkbook, WorkbookProfile) pair whose sheets dict
    includes attendance metrics + combined-risk columns merged onto the
    roster. Returns (None, None) when no roster is loaded.
    """
    if roster_loaded is None:
        return None, None
    enriched_sheets = registry.enriched_sheets() or dict(roster_loaded.sheets)
    enriched_loaded = LoadedWorkbook(
        file_name=roster_loaded.file_name,
        workbook=roster_loaded.workbook,
        sheets=enriched_sheets,
        warnings=list(roster_loaded.warnings or []),
    )
    enriched_profile = profile_workbook(enriched_loaded.file_name, enriched_sheets)
    return enriched_loaded, enriched_profile


def render_data_sources_panel(
    loaded, profile, diagnostics, settings, registry: DataSourceRegistry,
) -> str | None:
    """Right-panel renderer. Tabs: Roster / Attendance / Assessments / Results.

    Returns the active *roster* sheet name so the caller can use it as the
    chat planner's target. Generated / chart sheets live in the Results tab
    (they're views only; the planner stays on the roster).
    """
    tabs = st.tabs(["Academic Workbook", "Attendance", "Assessments", "Results"])
    active_sheet: str | None = None
    with tabs[0]:
        active_sheet = _render_roster_tab(loaded, profile, registry)
    with tabs[1]:
        _render_attendance_tab(registry)
    with tabs[2]:
        _render_assessments_tab(registry)
    with tabs[3]:
        _render_results_tab(loaded, profile, diagnostics, settings)
    return active_sheet


def _render_roster_tab(loaded, profile, registry: DataSourceRegistry | None = None) -> str | None:
    """Academic Workbook tab — capability-first layout.

    Display order (capabilities first; raw schema last and collapsed):
      1. File header + protection badges + attendance badge.
      2. Detected capabilities checklist.
      3. Detected fields, grouped (Roster / Performance / Attendance / Actions / Export).
      4. Missing-field helpful notes.
      5. Raw sheet preview (collapsed expander — for verifying the actual data).
    """
    st.file_uploader(
        "Upload academic workbook .xlsx",
        type=["xlsx"],
        accept_multiple_files=False,
        key="workbook_upload",
        help=("One file. The assistant detects roster columns and attendance "
              "data (inline columns or a sibling Attendance sheet) "
              "automatically. The original file is never modified."),
    )
    if loaded is None or profile is None or not profile.sheet_names:
        render_empty_state(
            "No workbook loaded.",
            suggestions=["Upload an .xlsx workbook to begin.",
                         "I'll detect teacher, department, GPA, academic "
                         "standing — and attendance if it's present."],
        )
        return None

    persist_key = f"active_source_sheet_{profile.file_name}"
    options = list(profile.sheet_names)
    default = st.session_state.get(persist_key)
    if default not in options:
        default = options[0]
    if len(options) > 1:
        active = st.selectbox(
            "Roster sheet", options,
            index=options.index(default), key=f"roster_sheet_{profile.file_name}",
        )
    else:
        active = options[0]
    st.session_state[persist_key] = active

    sheet_profile = next(item for item in profile.sheets if item.name == active)

    # 1. Header — file name + sheet + counts in school-office wording.
    st.markdown(
        f"**Academic Workbook Loaded**  \n"
        f"`{profile.file_name}`  \n"
        f"{active} sheet · {sheet_profile.row_count} students · "
        f"{len(sheet_profile.columns)} columns"
    )
    badge_cols = st.columns(3)
    with badge_cols[0]:
        badge("Original workbook protected", "green")
    with badge_cols[1]:
        badge("Read-only source", "green")
    with badge_cols[2]:
        if registry is not None and registry.attendance_available():
            badge("Attendance available", "green")
        else:
            badge("No attendance fields detected", "yellow")

    # 2 + 3 + 4. Capabilities, detected fields, missing notes.
    attendance_available = bool(registry is not None and registry.attendance_available())
    _render_capability_summary(
        sheet_profile.columns,
        attendance_available,
        frame=loaded.sheets.get(active),
    )

    # 5. The raw sheet — useful for verifying the actual rows, but it's NOT
    # the first thing the user sees. Collapsed by default.
    with st.expander("View the full sheet", expanded=False):
        _render_source_sheet_full(loaded, profile, active)
    return active


def _render_capability_summary(columns: list[str], attendance_available: bool, frame=None) -> None:
    """Render the Detected Capabilities checklist + Detected Fields blocks +
    helpful missing-field notes. Pure presentation — all logic lives in
    ``core.workbook_capabilities``."""
    from core.workbook_capabilities import (
        CATEGORY_ORDER,
        detect_capabilities,
        group_detected_fields,
        missing_field_messages,
    )

    mode = InstitutionMode.from_label(st.session_state.get("institution_mode", InstitutionMode.GENERIC.value))
    caps = detect_capabilities(columns, attendance_available=attendance_available, mode=mode)
    grouped = group_detected_fields(columns)
    messages = missing_field_messages(columns, attendance_available=attendance_available, mode=mode)

    st.markdown("#### Detected Capabilities")
    available_caps = [c for c in caps if c.available]
    unavailable_caps = [c for c in caps if not c.available]
    for cap in available_caps:
        if cap.note:
            st.markdown(f"✓ **{cap.title}** — {cap.note}")
        else:
            st.markdown(f"✓ **{cap.title}**")
    for cap in unavailable_caps:
        st.markdown(
            f"<span style='opacity:0.55;'>○ {cap.title}"
            f"{f' — {cap.note}' if cap.note else ''}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Detected Fields")
    has_any = False
    for category in CATEGORY_ORDER:
        labels = grouped.get(category) or []
        if not labels:
            continue
        has_any = True
        st.markdown(f"**{category}:** {' · '.join(labels)}")
    if not has_any:
        st.caption("No recognised academic fields yet.")

    if messages:
        with st.expander(f"Notes on missing fields ({len(messages)})", expanded=False):
            for note in messages:
                st.markdown(f"• {note}")
    with st.expander("Workbook Readiness", expanded=False):
        from core.workbook_capabilities import readiness_checks, readiness_issues
        for label, status in readiness_checks(columns):
            symbol = "✓" if status == "found" else ("⚠" if status == "issue found" else "○")
            st.markdown(f"{symbol} {label}: {status}")
        for issue in readiness_issues(frame if frame is not None else columns):
            st.markdown(f"⚠ {issue}")
    with st.expander("Workflow Templates", expanded=False):
        from core.institution_context import workflow_templates, role_prompt_snippets
        role = Role.from_label(st.session_state.get("user_role", Role.ADMIN.value))
        for title, body in workflow_templates(mode):
            st.markdown(f"**{title}**")
            st.caption(body)
        st.caption("Suggested workflow focus: " + ", ".join(role_prompt_snippets(role, mode)))




def _render_attendance_tab(registry: DataSourceRegistry) -> None:
    """Attendance status panel.

    Leads with whichever attendance source was auto-detected inside the
    uploaded workbook (inline columns or a sibling sheet). Falls back to a
    friendly "no attendance" state for roster-only workbooks. The external
    attendance uploader is kept as an Advanced expander — it's the right
    tool when a school exports daily attendance as a separate file but
    should NOT be the primary path the user has to find.
    """
    if registry.roster is None:
        render_empty_state(
            "Upload a roster first.",
            suggestions=["Attendance is detected automatically when the workbook "
                         "carries it (inline columns or a sibling Attendance sheet)."],
        )
        return

    summary = registry.summary()
    detection = summary.get("workbook_attendance") or {}
    mode = detection.get("mode", "none")

    if registry.attendance is not None:
        # External upload (Advanced) is the active source.
        badge("Attendance available · external file", "green")
    elif mode == "inline":
        badge("Attendance available · workbook columns", "green")
    elif mode == "sheet":
        badge(f"Attendance available · workbook sheet ‘{detection.get('attendance_sheet')}’",
              "green")
    else:
        badge("No attendance fields detected", "yellow")

    if mode == "inline":
        cols = detection.get("inline_columns") or []
        st.caption(
            f"Detected attendance columns on the roster: {', '.join(cols)}."
        )
    elif mode == "sheet":
        sheet_name = detection.get("attendance_sheet")
        st.caption(
            f"Detected sibling sheet **{sheet_name}** — attendance metrics "
            "are computed and merged into the roster automatically."
        )
    elif registry.attendance is None:
        st.markdown(
            "I do not see attendance data in this workbook. I can still help "
            "with GPA, academic standing, teacher, department, and Academic "
            "Watch workflows. If you have a separate daily-attendance file, "
            "you can load it under **Advanced** below."
        )

    if registry.attendance is not None:
        info = summary.get("attendance") or {}
        metric_cols = st.columns(3)
        metric_cols[0].metric("Attendance rows", info.get("rows", 0))
        metric_cols[1].metric("Matched students", info.get("matched", 0))
        metric_cols[2].metric("Unmatched IDs", info.get("unmatched", 0))
        st.caption(
            f"**{info.get('file_name', '')}** · "
            f"{info.get('rows', 0)} rows · {len(info.get('columns') or [])} columns"
        )
        unmatched_sample = info.get("unmatched_sample") or []
        if unmatched_sample:
            st.warning(
                "Some Student IDs in the attendance file don't appear in the roster: "
                + ", ".join(unmatched_sample)
                + (" …" if (info.get("unmatched", 0) > len(unmatched_sample)) else "")
            )
        for note in info.get("warnings") or []:
            st.caption(f"• {note}")
        if not registry.attendance.frame.empty:
            st.dataframe(registry.attendance.frame.head(200),
                         use_container_width=True, hide_index=True, height=240)

    with st.expander("Advanced: load a separate attendance file", expanded=False):
        st.caption(
            "Use this only when the daily attendance file is separate from "
            "the academic workbook. An external file takes priority over "
            "anything auto-detected inside the workbook."
        )
        st.file_uploader(
            "Upload attendance .xlsx",
            type=["xlsx"],
            accept_multiple_files=False,
            key="attendance_upload",
            help="Long-format: one row per (Student ID, Date). Matched on Student ID.",
        )
        load_error = st.session_state.get("attendance_load_error")
        if load_error:
            st.error(load_error)


def _render_assessments_tab(registry: DataSourceRegistry) -> None:
    """PSAT/SAT assessment status from the uploaded academic workbook."""
    if registry.roster is None:
        render_empty_state(
            "Upload a roster first.",
            suggestions=["Assessment fields are detected automatically when the workbook carries them."],
        )
        return
    summary = registry.summary()
    detection = summary.get("workbook_assessment") or {}
    mode = detection.get("mode", "none")
    if mode == "inline":
        badge("Assessments available · workbook columns", "green")
        cols = detection.get("inline_columns") or []
        st.caption("Detected assessment fields: " + ", ".join(cols))
        return
    if mode == "sheet":
        badge("Assessments available · workbook sheet", "green")
        st.caption(f"Assessment sheet: **{detection.get('assessment_sheet', '')}**")
        return
    if mode == "ambiguous":
        badge("Assessment sheet needs review", "yellow")
        st.warning("Multiple possible assessment sheets found: " + ", ".join(detection.get("candidate_sheets") or []))
        return
    render_empty_state(
        "Assessment scores not detected.",
        suggestions=[
            "You can still review GPA, attendance, standing, teacher, department, and watch workflows.",
            "Add SAT/PSAT columns or an Assessments sheet inside the workbook to unlock benchmark review.",
        ],
    )


def _render_results_tab(loaded, profile, diagnostics, settings) -> None:
    """Generated session sheets + chart sheets (the previous unified view)."""
    render_unified_workbook_panel(loaded, profile, diagnostics, settings,
                                  hide_uploader=True, hide_source_sheets=True)


def render_unified_workbook_panel(loaded, profile, diagnostics, settings,
                                  *, hide_uploader: bool = False,
                                  hide_source_sheets: bool = False) -> str | None:
    """Single workbook view that lists source sheets, generated session
    sheets, and chart sheets in one picker. Returns the active *source* sheet
    name — the planner always operates on a source sheet, even when the user
    is currently viewing a generated or chart sheet.
    """
    if not hide_uploader:
        st.file_uploader(
            "Upload .xlsx roster",
            type=["xlsx"],
            accept_multiple_files=False,
            key="workbook_upload",
            help="The original workbook is never modified.",
        )
    if loaded is None or profile is None or not profile.sheet_names:
        render_empty_state(
            "No workbook loaded.",
            suggestions=["Upload an .xlsx roster to begin.",
                         "Examples: students, advisors, GPA, standing."],
        )
        return None

    if hide_source_sheets:
        source_options: list[tuple[str, str, str]] = []
    else:
        source_options = [("source", name, name) for name in profile.sheet_names]

    workbook = st.session_state.get("session_workbook")
    generated_options: list[tuple[str, str, str]] = []
    if workbook is not None:
        for record in workbook.sheets:
            generated_options.append(("generated", record.name, f"Generated · {record.name}"))

    figures_history = st.session_state.get("figures_history") or []
    figure_options: list[tuple[str, str, str]] = []
    for idx, fig in enumerate(figures_history):
        title = fig.get("title") or f"Chart {idx + 1}"
        figure_options.append(("figure", str(idx), f"Chart · {title}"))

    all_options = source_options + generated_options + figure_options
    labels = [opt[2] for opt in all_options]
    if not labels:
        # Results tab with nothing generated yet — friendly empty state.
        render_empty_state(
            "No results yet.",
            suggestions=["Ask the assistant a question to create a generated sheet.",
                         "Charts the assistant builds will also appear here."],
        )
        return None

    # Auto-jump: the most recent chat turn or chart can request a specific
    # view via _pending_view_sheet. Consume it once so subsequent reruns
    # leave the user where they last clicked.
    pending = st.session_state.pop("_pending_view_sheet", None)
    selector_key = f"unified_view_{profile.file_name}"

    target_label: str | None = None
    if pending:
        kind, key = pending
        for opt in all_options:
            if opt[0] == kind and opt[1] == key:
                target_label = opt[2]
                break
    existing = st.session_state.get(selector_key)
    if target_label is None and existing in labels:
        target_label = existing
    if target_label is None:
        target_label = labels[0]
    st.session_state[selector_key] = target_label

    selected_label = st.selectbox(
        "View",
        labels,
        key=selector_key,
        help="Switch between the uploaded sheets, results created by the assistant, and chart sheets.",
    )
    selected_kind, selected_key, _ = all_options[labels.index(selected_label)]

    # Persist the last-picked source sheet for the chat planner. When the
    # user is viewing a generated/chart sheet, the planner stays on whatever
    # source sheet they were last on.
    persist_key = f"active_source_sheet_{profile.file_name}"
    if selected_kind == "source":
        st.session_state[persist_key] = selected_key
        active_source = selected_key
    else:
        active_source = st.session_state.get(persist_key) or profile.sheet_names[0]

    if selected_kind == "source":
        _render_source_sheet_full(loaded, profile, active_source)
    elif selected_kind == "generated":
        _render_generated_sheet(workbook, selected_key)
    elif selected_kind == "figure":
        _render_chart_sheet(figures_history, int(selected_key))

    warnings = list(getattr(loaded, "warnings", None) or [])
    with st.expander(f"Workbook notes ({len(warnings)})", expanded=False):
        if warnings:
            for note in warnings:
                st.caption(f"• {note}")
        else:
            st.caption("No loader warnings.")
    return active_source


def _render_source_sheet_full(loaded, profile, sheet_name: str) -> None:
    """Full sheet view of the uploaded workbook (no 5-row preview limit).

    Sensitive columns are still hidden by default per core.privacy, matching
    the redact_table behavior used in the old preview.
    """
    import pandas as pd
    from core.privacy import redact_table

    sheet_profile = next((item for item in profile.sheets if item.name == sheet_name), None)
    if sheet_profile is None:
        st.caption(f"Sheet '{sheet_name}' is not in the loaded workbook.")
        return

    cols = st.columns(3)
    cols[0].metric("Rows", sheet_profile.row_count)
    cols[1].metric("Columns", len(sheet_profile.columns))
    cols[2].metric("Sheets", len(profile.sheet_names))

    badge_cols = st.columns(2)
    with badge_cols[0]:
        badge("Original file protected", "green")
    with badge_cols[1]:
        badge("Read-only", "green")

    st.caption(
        f"**{profile.file_name}** · {sheet_name} · "
        f"{sheet_profile.row_count} rows · {len(sheet_profile.columns)} columns"
    )

    frame = loaded.sheets.get(sheet_name)
    if frame is None or frame.empty:
        st.caption("This sheet has no rows.")
        return

    rows = frame.to_dict(orient="records")
    redacted_rows, hidden = redact_table(rows, list(frame.columns))
    if hidden:
        st.caption(f"Sensitive columns hidden by default: {', '.join(hidden)}")
    st.dataframe(
        pd.DataFrame(redacted_rows) if redacted_rows else frame,
        use_container_width=True,
        hide_index=True,
        height=520,
    )


def _render_generated_sheet(workbook, sheet_name: str) -> None:
    """Render a sheet produced by a chat turn (filter, sort, aggregate)."""
    import pandas as pd

    if workbook is None:
        render_empty_state("No generated sheets yet.")
        return
    record = next((s for s in workbook.sheets if s.name == sheet_name), None)
    if record is None:
        st.caption(f"Generated sheet '{sheet_name}' is no longer in this session.")
        return

    badge("Generated by the assistant", "blue")
    if record.user_message:
        st.caption(f"From: {record.user_message}")
    if record.conclusion:
        st.markdown(f"**Conclusion:** {record.conclusion}")
    if record.sensitive_redacted:
        st.caption(f"Sensitive columns hidden by default: {', '.join(record.sensitive_redacted)}")

    if record.data:
        frame = pd.DataFrame(record.data)
        st.dataframe(frame, use_container_width=True, hide_index=True, height=520)
        st.download_button(
            "Download this sheet (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"{sheet_name}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_generated_{sheet_name}",
        )
    else:
        st.caption("No rows in this generated sheet.")


def _render_chart_sheet(figures_history, index: int) -> None:
    """Render a chart sheet — the altair chart plus its summary table."""
    import pandas as pd

    if not figures_history or index < 0 or index >= len(figures_history):
        render_empty_state("That chart is no longer available.")
        return
    fig = figures_history[index]

    badge("Chart sheet", "blue")
    title = fig.get("title") or "Chart"
    st.markdown(f"**{title}**")

    preview = fig.get("preview")
    summary = preview if isinstance(preview, pd.DataFrame) else pd.DataFrame(preview or [])
    if summary.empty:
        st.caption("No data for this chart.")
        return

    intent = ChartIntent(
        chart_type=str(fig.get("type", "bar")),
        field=str(fig.get("field", "")),
        metric=str(fig.get("metric", "count")),
        value_column=str(fig.get("value_column", "")),
    )
    chart = build_altair_chart(intent, summary, title)
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download chart data (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"chart_{index}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"dl_chart_{index}",
    )


def render_export_center() -> None:
    """Bottom-right card content (title supplied by render_card wrapper)."""
    output_file = st.session_state.get("latest_output_file")
    figure = st.session_state.get("latest_figure")
    history = st.session_state.get("export_history") or []

    if not output_file and not figure and not history:
        render_empty_state(
            "No exports yet.",
            suggestions=["Exports and modified workbooks will appear here.",
                         "Ask the assistant to export this list."],
        )
        return

    if output_file:
        path = Path(output_file)
        if path.exists():
            with st.container(border=True):
                st.markdown("<div style='color:#15803D; font-weight:bold; font-size:1.1rem; margin-bottom:6px;'>Export Ready!</div>", unsafe_allow_html=True)
                st.markdown(f"**Filename:** `{path.name}`")
                
                changes = st.session_state.get("latest_output_changes")
                if changes:
                    st.markdown(f"**Changes:** {changes}")
                    
                st.info("Data Safety Note: The original uploaded workbook remains unmodified. This download is a separate, edited working copy.")
                st.download_button(
                    "Download Edited Workbook (.xlsx)",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="export_workbook",
                    type="primary",
                )
        else:
            st.caption("The latest workbook output is no longer available.")

    # Latest result CSV
    latest_result_attachment = _latest_result_attachment()
    if latest_result_attachment and latest_result_attachment.get("table"):
        import pandas as pd

        frame = pd.DataFrame(latest_result_attachment["table"])
        st.download_button(
            "Download latest result (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="latest_result.csv",
            mime="text/csv",
            use_container_width=True,
            key="export_result_csv",
        )

    if figure:
        figure_csv = export_latest_figure_csv()
        if figure_csv is not None:
            file_name, payload = figure_csv
            st.download_button(
                f"Latest figure data (CSV): {figure.get('field', 'figure')}",
                data=payload,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True,
                key="export_figure_csv",
            )

    if history:
        st.caption("Session outputs:")
        for entry in history[:8]:
            st.caption(f"• {entry}")


def _latest_result_attachment() -> dict[str, Any] | None:
    messages = st.session_state.get("chat_messages") or []
    for index in range(len(messages) - 1, -1, -1):
        attachment = messages[index].get("attachment") or {}
        if attachment.get("type") == "result":
            return attachment
    return None


def render_sidebar_workbook(loaded, profile) -> str | None:
    """Sidebar: file uploader plus a compact workbook summary."""
    st.markdown("### Workbook")
    st.file_uploader(
        "Upload Excel workbook",
        type=["xlsx"],
        accept_multiple_files=False,
        key="workbook_upload",
    )
    if loaded is None or profile is None or not profile.sheet_names:
        st.caption("No workbook loaded yet.")
        return None

    selected_sheet = st.selectbox(
        "Sheet",
        profile.sheet_names,
        key=f"sheet_selector_{profile.file_name}",
    )
    sheet_profile = next(item for item in profile.sheets if item.name == selected_sheet)
    st.caption(f"**{profile.file_name}**")
    st.caption(f"{selected_sheet} · {sheet_profile.row_count} rows · {len(sheet_profile.columns)} columns")

    warnings = getattr(loaded, "warnings", None)
    if warnings:
        with st.expander(f"Workbook notes ({len(warnings)})", expanded=False):
            for note in warnings:
                st.caption(f"• {note}")
    return selected_sheet


def render_sidebar_resets(loaded, profile, selected_sheet: str) -> None:
    """Sidebar buttons for Clear filters and Start over.

    Clear filters keeps the chat history; Start over wipes the conversation
    state but keeps the uploaded workbook so the user can begin a fresh thread.
    """
    st.markdown("### Conversation")
    cols = st.columns(2)
    if cols[0].button("Clear filters", key="side_clear", use_container_width=True):
        from ui.chat_panel import route_message  # avoid circular at module load

        route_message(
            request="clear that",
            selected_sheet=selected_sheet,
            loaded=loaded,
            profile=profile,
            settings={"strict_privacy_mode": True},
        )
        st.rerun()
    if cols[1].button("Start over", key="side_reset", use_container_width=True):
        _start_over_keep_workbook()
        st.rerun()


def render_sidebar_session_workbook() -> None:
    """Sidebar panel for the accumulating session workbook (v0.3).

    Shows the sheets recorded so far, a download button for the .xlsx, a
    suppress-last-turn action ([Don't save this]), and an explicit
    'Start new investigation' button that opens a fresh workbook.
    """
    workbook = st.session_state.get("session_workbook")
    if workbook is None:
        return

    st.markdown("### Session workbook")
    sheets = workbook.list_sheets()
    if not sheets:
        st.caption("No investigations recorded yet. Each question you ask becomes a sheet in this workbook.")
    else:
        st.caption(f"{len(sheets)} sheet{'s' if len(sheets) != 1 else ''} · `{workbook.path.name}`")
        for index, entry in enumerate(sheets, start=1):
            st.caption(f"{index}. {entry['name']} — {entry['row_count']} row{'s' if entry['row_count'] != 1 else ''}")

    cols = st.columns(2)
    download_disabled = not sheets or not workbook.path.exists()
    if download_disabled:
        cols[0].button("Download", key="sw_download", disabled=True, use_container_width=True)
    else:
        try:
            payload = workbook.path.read_bytes()
        except OSError:
            payload = b""
        cols[0].download_button(
            label="Download",
            data=payload,
            file_name=workbook.path.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="sw_download",
            use_container_width=True,
            disabled=not payload,
        )

    if cols[1].button("Don't save last", key="sw_suppress", disabled=not sheets,
                      use_container_width=True):
        if workbook.suppress_last_turn():
            st.toast("Removed the most recent sheet.")
        st.rerun()

    if st.button("Start new investigation", key="sw_new_investigation",
                 use_container_width=True, disabled=not sheets):
        workbook.reset_for_new_source(
            source_file_name=workbook.source_file_name,
            schema_hash=workbook.schema_hash,
        )
        st.toast("Started a fresh session workbook.")
        st.rerun()


def render_sidebar_failure_log() -> None:
    """Show recent asks the assistant couldn't handle.

    Each entry has a timestamp, the user's exact message, and a one-line
    reason. The point isn't to apologize to the user — it's to give the
    developer (us) a triage queue when iterating on rule coverage.
    """
    from core.failure_log import clear_failures, read_failures

    recent = read_failures(limit=25)
    label = f"Things I couldn't answer ({len(recent)})" if recent else "Things I couldn't answer"
    with st.expander(label, expanded=False):
        if not recent:
            st.caption("No failed asks recorded yet.")
            return
        st.caption(
            "Every clarify / unsupported / failed query lands here so we can "
            "see what real phrasings the planner still needs to learn."
        )
        for entry in recent[:10]:
            ts = (entry.get("timestamp") or "").replace("T", " ")
            intent = entry.get("intent") or "?"
            message = (entry.get("user_message") or "").strip()
            reason = (entry.get("reason") or "").strip()
            sheet = entry.get("sheet") or ""
            st.markdown(
                f"**{message or '(empty)'}**\n\n"
                f"<span style='font-size:0.78rem; opacity:0.65;'>"
                f"{ts} · {intent}{f' · {sheet}' if sheet else ''}</span>",
                unsafe_allow_html=True,
            )
            if reason:
                st.caption(reason[:160] + ("…" if len(reason) > 160 else ""))
            st.divider()
        if st.button("Clear failure log", key="failure_log_clear",
                     use_container_width=True):
            clear_failures()
            st.toast("Failure log cleared.")
            st.rerun()


def _start_over_keep_workbook() -> None:
    """Wipe conversation state but keep the uploaded workbook + cached load."""
    st.session_state["chat_messages"] = []
    st.session_state["assistant_memory"] = {}
    st.session_state["assistant_mode"] = None
    st.session_state["current_command"] = {}
    st.session_state["current_plan"] = {}
    st.session_state["current_confirmation"] = None
    st.session_state["current_confidence"] = None
    st.session_state["current_source"] = None
    st.session_state["current_validation_error"] = None
    st.session_state["current_can_execute"] = False
    st.session_state["current_clarification"] = None
    st.session_state["current_warning"] = None
    st.session_state["latest_output_file"] = None
    st.session_state["latest_result_preview"] = None
    st.session_state["latest_result_message"] = None
    st.session_state["latest_result_sheet"] = None
    st.session_state["figures_history"] = []
    st.session_state["action_history"] = []
    st.session_state["export_history"] = []
    st.session_state["latest_figure"] = None
    st.session_state["routing_debug"] = None
    st.session_state["pending_feedback"] = None
    for key in (
        "ask_question",
        "ask_operation",
        "ask_description",
        "ask_explanation",
        "ask_value",
        "ask_row_count",
        "ask_table",
        "ask_columns_used",
        "ask_confidence",
        "ask_source",
        "ask_preview_truncated",
        "ask_redacted",
        "ask_review_note",
        "clarify_question",
        "clarify_reason",
        "clarify_options",
    ):
        st.session_state.pop(key, None)


def _maybe_reset_for_new_workbook(loaded) -> None:
    """If the user uploaded a different workbook than the previous run, treat it
    as the start of a fresh conversation (J.7 'New workbook upload')."""
    file_name = getattr(loaded, "file_name", None) if loaded is not None else None
    previous = st.session_state.get("_last_loaded_file_name")
    if file_name == previous:
        return
    st.session_state["_last_loaded_file_name"] = file_name
    if file_name is None or previous is None:
        # Initial load or workbook removal — don't wipe a fresh user's empty chat.
        if previous is not None:
            _start_over_keep_workbook()
        return
    _start_over_keep_workbook()


def _maybe_post_upload_greeting(loaded, registry) -> None:
    """Append ONE assistant greeting per uploaded workbook.

    Keyed by ``loaded.file_name`` so Streamlit's per-interaction reruns
    don't keep prepending the same greeting; a different upload resets
    the key (via ``_maybe_reset_for_new_workbook`` clearing chat first).
    """
    if loaded is None or registry is None or registry.roster is None:
        return
    from core.workbook_capabilities import upload_assistant_message

    file_name = getattr(loaded, "file_name", "") or ""
    already_greeted = st.session_state.get("_greeted_workbook")
    if already_greeted == file_name:
        return
    sheet = registry.enriched_roster_sheet or ""
    frame = loaded.sheets.get(sheet)
    columns = list(frame.columns) if frame is not None else []
    if not columns:
        return
    greeting = upload_assistant_message(
        columns,
        attendance_available=registry.attendance_available(),
        mode=InstitutionMode.from_label(st.session_state.get("institution_mode", InstitutionMode.GENERIC.value)),
    )
    messages = st.session_state.setdefault("chat_messages", [])
    messages.append({
        "role": "assistant",
        "content": greeting,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    st.session_state["_greeted_workbook"] = file_name


def _handle_inline_edit_actions(loaded, current_user, permissions, privacy_settings) -> None:
    """The inline edit-plan card renders Run/Cancel buttons whose presses come
    back to us on the next rerun. We dispatch them here so the chat history can
    grow naturally with a success/failure assistant message."""
    from ui.chat_panel import append_assistant_message, mark_latest_edit_plan_resolved

    message_index = st.session_state.get("_chat_edit_plan_message_index")
    if message_index is None:
        return

    if st.session_state.get(f"edit_cancel_{message_index}"):
        mark_latest_edit_plan_resolved("cancelled")
        reset_current_interpretation()
        append_assistant_message("Cancelled. No workbook changes were made.")
        st.session_state.pop("_chat_edit_plan_message_index", None)
        return

    if not st.session_state.get(f"edit_run_{message_index}"):
        return

    # Run the planned action and post the result back into chat.
    mark_latest_edit_plan_resolved("executed")
    st.session_state.pop("_chat_edit_plan_message_index", None)
    success, message, output_file = _execute_current_plan(
        loaded=loaded,
        current_user=current_user,
        permissions=permissions,
        privacy_settings=privacy_settings,
    )
    if success and output_file:
        from pathlib import Path
        view = st.session_state.get("workspace_view")
        if isinstance(view, dict):
            view["mode"] = "export_ready"
            view["export_filename"] = Path(output_file).name
            view["download_path"] = output_file
            view["change_summary"] = [message]
    attachment = {"type": "download", "path": output_file} if (success and output_file) else None
    append_assistant_message(message, attachment=attachment)



def _execute_current_plan(*, loaded, current_user, permissions, privacy_settings):
    """Wrap run_current_action so the chat panel can run an edit plan without
    rendering inline Streamlit alerts (those flash and disappear on rerun)."""
    command = st.session_state.get("current_command") or {}
    plan = st.session_state.get("current_plan") or {}
    request = st.session_state.get("current_request", "")
    source = st.session_state.get("current_source")
    confidence = st.session_state.get("current_confidence")

    plan_commands = plan.get("commands") or ([command] if command else [])

    if not permissions["can_execute"]:
        return False, "Your role can preview and ask questions but cannot run actions.", None
    if not command and not plan_commands:
        return False, "No interpreted command is ready to run.", None
    if st.session_state.get("current_validation_error"):
        return False, "Resolve validation issues before running this action.", None
    if any(is_delete_row_action(step) for step in plan_commands) and privacy_settings["block_delete_row_actions"]:
        return False, "This row-removal action is blocked by the current safety settings.", None

    try:
        if plan.get("commands"):
            plan_result = execute_plan(plan, loaded.workbook, loaded.sheets, loaded.file_name)
            result = ActionResult(
                message=plan_result.message,
                preview=plan_result.preview,
                result_sheet=plan_result.result_sheet,
            )
        else:
            result = execute_command(command, loaded.workbook, loaded.sheets, loaded.file_name)
        export_path = export_edited_workbook(loaded.workbook, loaded.file_name)
        log_audit_event(
            username=current_user.username,
            user_role=current_user.role,
            action_type=command.get("action"),
            columns_affected=affected_columns(command),
            row_count_affected=affected_row_count(command, loaded.sheets),
            success=True,
            source=source,
        )
        st.session_state["latest_output_file"] = str(export_path)
        st.session_state["latest_result_preview"] = result.preview
        st.session_state["latest_result_message"] = result.message
        st.session_state["latest_result_sheet"] = result.result_sheet
        st.session_state.setdefault("export_history", []).insert(0, Path(export_path).name)
        action_name = command.get("action") if isinstance(command, dict) else None
        if action_name in {"create_chart", "create_report", "count_by_group", "summarize_missing"} and result.preview is not None:
            st.session_state.setdefault("figures_history", []).insert(
                0,
                {"sheet": result.result_sheet, "action": action_name, "preview": result.preview, "message": result.message},
            )
        st.session_state["action_history"].insert(
            0,
            {
                "request": request,
                "action": command.get("action"),
                "confidence": confidence,
                "success": True,
                "output": str(export_path),
            },
        )
        st.session_state["pending_feedback"] = {
            "file_name": loaded.file_name,
            "sheet_name": command.get("sheet"),
            "original_request": request,
            "command": command,
            "parser_confidence": confidence,
            "parser_source": source,
            "action_type": command.get("action"),
        }
        reset_current_interpretation()
        return True, result.message, str(export_path)
    except Exception as exc:
        log_user_request(
            file_name=loaded.file_name,
            sheet_name=command.get("sheet") if isinstance(command, dict) else None,
            original_request=request,
            generated_command=command if isinstance(command, dict) else {},
            parser_confidence=confidence,
            parser_source=source,
            action_type=command.get("action") if isinstance(command, dict) else None,
            success=False,
            error_message=str(exc),
        )
        st.session_state["action_history"].insert(
            0,
            {
                "request": request,
                "action": command.get("action"),
                "confidence": confidence,
                "success": False,
                "output": "",
            },
        )
        return False, f"Action failed: {exc}", None


def render_sidebar_advanced(permissions, privacy_settings):
    """Advanced controls live in collapsed sidebar expanders. Returns
    (settings, privacy_settings, show_debug)."""
    if permissions["can_admin"]:
        settings = render_settings_panel()
        privacy_settings = render_privacy_admin()
        render_learning_admin()
        render_user_admin()
    else:
        settings = {"strict_privacy_mode": True, "use_local_llm": False, "ollama_model": "llama3.2:3b"}
        st.caption("Privacy mode is on. Ask an admin to change advanced settings.")
    with st.expander("Appearance", expanded=False):
        current = active_theme()
        choice = st.radio(
            "Theme", ["Light", "Dark"],
            index=0 if current == "light" else 1,
            key="ui_theme_choice", horizontal=True,
        )
        new_theme = "dark" if choice == "Dark" else "light"
        if new_theme != st.session_state.get("ui_theme", "light"):
            st.session_state["ui_theme"] = new_theme
            st.rerun()
        st.session_state["ui_theme"] = new_theme
    show_debug = st.checkbox("Show developer/debug info", value=False, key="show_debug_toggle")
    return settings, privacy_settings, show_debug


def render_app_header(loaded, profile, active_sheet: str = "") -> None:
    """Clean product header: title, file summary, and Upload / Export actions.

    Diagnostics (privacy, runtime, mappings) are intentionally NOT here — they
    live in the Advanced details drawer at the bottom of the page.
    """
    left, right = st.columns([0.66, 0.34])
    with left:
        st.markdown("<h1>Dean Assistant</h1>", unsafe_allow_html=True)
        if loaded is not None and profile is not None:
            rows = cols = 0
            if active_sheet and active_sheet in loaded.sheets:
                df = loaded.sheets[active_sheet]
                rows, cols = int(df.shape[0]), int(df.shape[1])
            st.markdown(
                f'<div class="app-filename">{profile.file_name}</div>'
                f'<span class="app-pill">{rows:,} rows · {cols} columns</span>'
                f'<span class="status-badge badge-green" style="margin-left:8px;">'
                f'Workbook loaded · {rows:,} students</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="app-tagline">Upload a roster to begin. '
                'All processing stays local.</div>',
                unsafe_allow_html=True,
            )
    with right:
        up_col, ex_col = st.columns(2)
        with up_col:
            with st.popover("Upload File", use_container_width=True):
                st.file_uploader(
                    "Upload academic workbook (.xlsx)",
                    type=["xlsx"],
                    accept_multiple_files=False,
                    key="workbook_upload",
                    help="One file. Detected and processed locally; the original is never modified.",
                )
        with ex_col:
            if st.button("Export", use_container_width=True, key="header_export_btn",
                         disabled=loaded is None):
                st.session_state["workspace_segment"] = "Export"
                st.rerun()


def render_status_badges(loaded, settings: dict[str, Any], active_sheet: str = "", profile = None) -> None:
    """Render the active session status, privacy stance, LLM mode, and key mapped columns."""
    is_strict = settings.get("strict_privacy_mode", True)
    privacy_label = "Privacy Protected" if is_strict else "Privacy Mode Off"
    privacy_tone = "green" if is_strict else "yellow"
    
    from nlp.local_model_manager import get_ollama_manager
    status = get_ollama_manager().status
    if status.mode == "disabled":
        llm_label = "Local LLM: Disabled"
        llm_tone = "gray"
    elif status.mode == "rule_only":
        llm_label = "Local LLM: Failed (Rule-only)"
        llm_tone = "red"
    elif status.mode == "bundled_ollama":
        llm_label = f"Local LLM: Bundled ({status.model_name})"
        llm_tone = "green"
    elif status.mode == "system_ollama":
        llm_label = f"Local LLM: System ({status.model_name})"
        llm_tone = "yellow"
    else:
        llm_label = "Local LLM: Off"
        llm_tone = "gray"

    with st.container(border=True):
        st.markdown(
            '<div style="font-weight:600; font-size:1.05rem; margin-bottom:10px; color:#1E293B;">Session Profile & Privacy Safeguards</div>',
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Workbook:** `{loaded.file_name}`" if loaded else "**Workbook:** _None uploaded_")
            st.markdown(f"**Active Sheet:** `{active_sheet}`" if active_sheet else "**Active Sheet:** _None_")
        with col2:
            if loaded and active_sheet in loaded.sheets:
                df = loaded.sheets[active_sheet]
                st.markdown(f"**Dimensions:** `{df.shape[0]} rows × {df.shape[1]} columns`")
            else:
                st.markdown("**Dimensions:** _N/A_")
            st.markdown(
                f"**Privacy Stance:** <span class='status-badge badge-{privacy_tone}'>{privacy_label}</span>",
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"**Runtime Status:** <span class='status-badge badge-{llm_tone}'>{llm_label}</span>",
                unsafe_allow_html=True
            )
            st.markdown("**Data Integrity:** `Original workbook remains unmodified`")
            
        if loaded and active_sheet in loaded.sheets:
            from core.schema import canonical_map
            mapped = canonical_map(list(loaded.sheets[active_sheet].columns))
            if mapped:
                mapped_str = ", ".join([f"**{k.replace('_', ' ').title()}**: `{v}`" for k, v in mapped.items()])
                st.markdown(
                    f"<div style='font-size:0.82rem; margin-top:8px; border-top:1px solid #E2E8F0; padding-top:8px; color:#475569;'>"
                    f"🔑 **Detected Key Columns:** {mapped_str}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='font-size:0.82rem; margin-top:8px; border-top:1px solid #E2E8F0; padding-top:8px; color:#94A3B8;'>"
                    "🔑 **Detected Key Columns:** None detected</div>",
                    unsafe_allow_html=True
                )


def render_debug_panel(loaded, selected_sheet: str) -> None:
    from core.schema import build_debug_state, build_workbook_schema

    with st.expander("Developer / debug info", expanded=True):
        try:
            schema = build_workbook_schema(loaded.sheets)
            state = build_debug_state(
                st.session_state.get("assistant_memory") or {},
                schema,
                selected_sheet or (loaded.sheets and next(iter(loaded.sheets))) or "",
                routing=st.session_state.get("routing_debug"),
            )
            st.caption("Detected sheets, normalized schema, sensitivity, and live conversation state. No raw sensitive rows are shown.")
            st.json(state)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not build debug state: {exc}")

        _render_analyst_trace(st.session_state.get("analyst_trace_debug"))


def _render_analyst_trace(trace: dict | None) -> None:
    """Show the free-form analyst's last run: the code it wrote, what each step
    printed, and the confidence/verification signals. Dev-facing only."""
    if not trace:
        return
    st.divider()
    st.markdown("#### 🧮 Code Analyst trace (last answer)")
    st.caption(f"Question: {trace.get('question', '')}")
    verified = trace.get("verified")
    verified_label = {True: "✅ agreed", False: "⚠️ disagreed", None: "— not checked"}.get(verified, "—")
    cols = st.columns(4)
    cols[0].metric("Confidence", str(trace.get("confidence", "—")))
    cols[1].metric("Cross-check", verified_label)
    cols[2].metric("Grounded", "yes" if trace.get("grounded") else "no")
    cols[3].metric("Code steps", str(trace.get("iterations", "—")))
    if trace.get("plan"):
        st.caption("Stated plan")
        st.text(trace["plan"])
    for index, step in enumerate(trace.get("steps", []), start=1):
        st.caption(f"Step {index} — code")
        st.code(step.get("code", ""), language="python")
        st.caption(f"Step {index} — output")
        st.text((step.get("output") or "(no output)")[:1500])


if __name__ == "__main__":
    main()
