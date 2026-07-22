from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from core.action_engine import ActionResult, execute_command, execute_plan
from core.action_safety import (
    affected_columns,
    affected_row_count,
    is_delete_row_action,
)
from core.audit_logger import log_audit_event
from core.command_schema import OPERATORS_WITHOUT_VALUE
from core.correction_manager import save_correction, sync_learning_files
from core.excel_loader import load_excel_workbook
from core.exporter import export_edited_workbook
from core.logger import log_user_request
from core.privacy_controls import load_privacy_settings
from core.validator import validate_command
from core.workbook_diagnostics import diagnose_workbook
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
    route_message,
)
from ui.correction_screen import render_feedback_screen, render_learning_admin
from ui.auth_screen import render_user_admin, render_user_bar, require_login
from ui.figures_panel import (
    ChartIntent,
    build_altair_chart,
    render_figures_panel,
)
from ui.privacy_admin import render_privacy_admin
from ui.health_check import render_workbook_health_check
from ui.settings_panel import load_settings, render_settings_panel
from ui.system_resources_panel import render_system_resources_panel
from core.risk_settings import load_risk_settings
from ui.workbook_workspace import render_workbook_workspace


st.set_page_config(
    page_title="Dean Assistant",
    page_icon="📘",
    layout="wide",
)


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

    # Warm the local model once per session so the first question doesn't eat the
    # 25-40s cold-start. Backgrounded and best-effort; skipped when the local LLM
    # is off. Strict privacy mode restricts row/field access (see core.llm_config),
    # it does not disable the model itself.
    if settings.get("use_local_llm") and not st.session_state.get("_ollama_warmed"):
        from nlp.local_model import warm_model

        warm_model(settings.get("ollama_model") or settings.get("planner_model") or "llama3.2:3b")
        st.session_state["_ollama_warmed"] = True

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
    if settings.get("use_local_llm", False):
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
        render_system_resources_panel()
    else:
        # Non-admins can't EDIT the settings panel, but they must still get
        # whatever the admin actually configured -- not a hardcoded
        # strict-mode override. Previously this silently forced
        # use_local_llm=False for every non-admin session regardless of the
        # saved config, which meant the AI layer was effectively "admin-only"
        # even when a site admin had deliberately turned it on for everyone.
        settings = load_settings()
        st.caption("Using this workspace's configured settings. Ask an admin to change them.")
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
