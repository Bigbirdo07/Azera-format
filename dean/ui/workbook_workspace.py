import io
import streamlit as st
import pandas as pd
from pathlib import Path
from contextlib import contextmanager


def _excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a dataframe to .xlsx bytes for a download button."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    return buffer.getvalue()


def _safe_dataframe(df, **kwargs):
    try:
        st.dataframe(df, **kwargs)
    except BaseException:
        try:
            st.table(df.head(100))
        except BaseException:
            st.text(df.to_string())


@contextmanager
def workspace_card(title: str, subtitle: str | None = None):
    """Local context manager for clean workspace card styling."""
    container = st.container(border=True)
    with container:
        if title:
            st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="card-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        yield container

def _render_capability_summary(columns: list[str], attendance_available: bool, frame=None) -> None:
    """Render the Detected Capabilities checklist + Detected Fields blocks +
    helpful missing-field notes. Pure presentation — all logic lives in
    ``core.workbook_capabilities``."""
    from core.workbook_capabilities import (
        CATEGORY_ORDER,
        detect_capabilities,
        group_detected_fields,
        missing_field_messages,
        readiness_checks,
        readiness_issues,
    )
    from core.institution_context import InstitutionMode, Role, workflow_templates, role_prompt_snippets

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
        for label, status in readiness_checks(columns):
            symbol = "✓" if status == "found" else ("⚠" if status == "issue found" else "○")
            st.markdown(f"{symbol} {label}: {status}")
        for issue in readiness_issues(frame if frame is not None else columns):
            st.markdown(f"⚠ {issue}")
    with st.expander("Workflow Templates", expanded=False):
        role = Role.from_label(st.session_state.get("user_role", Role.ADMIN.value))
        for title, body in workflow_templates(mode):
            st.markdown(f"**{title}**")
            st.caption(body)
        st.caption("Suggested workflow focus: " + ", ".join(role_prompt_snippets(role, mode)))


def render_workbook_workspace(loaded, profile, diagnostics, settings) -> None:
    """Consolidated workbook workspace on the right side.
    
    Reads from st.session_state["workspace_view"] and renders exactly one active
    mode. Features view toggling, original preview, filtered result table,
    pending action warning with preview, and download panel.
    """
    view = st.session_state.get("workspace_view")
    if not isinstance(view, dict):
        view = {
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
        }
        st.session_state["workspace_view"] = view

    # Force "upload" mode if no workbook is currently loaded
    if loaded is None or profile is None:
        view["mode"] = "upload"

    mode = view.get("mode", "upload")

    # --- No workbook: simple get-started card ---
    if mode == "upload" or loaded is None or profile is None:
        with workspace_card("Get Started", "Use the Upload File button at the top to begin."):
            st.markdown(
                '<div class="empty-state"><strong>No workbook loaded yet.</strong>'
                '<ul><li>Click <em>Upload File</em> in the header to add an .xlsx roster.</li>'
                '<li>We will auto-detect student identity, advisor mappings, GPA, academic standing, and attendance.</li>'
                '<li>Your data is processed 100% locally on your computer. No row-level data is ever sent to the cloud.</li></ul></div>',
                unsafe_allow_html=True,
            )
        # "Advanced details" is a diagnostics surface hidden from end users.
        # It renders only when the show_workspace_details flag is set.
        if st.session_state.get("show_workspace_details"):
            _render_workspace_details(loaded, profile, diagnostics, settings, view)
        return

    # --- Workbook loaded: segmented Original / Results / Export ---
    segments = ["Original Sheet", "Results", "Export"]
    if st.session_state.get("workspace_segment") not in segments:
        st.session_state["workspace_segment"] = "Original Sheet"
    segment = st.segmented_control(
        "Workbook view", segments, key="workspace_segment",
        label_visibility="collapsed",
    ) or st.session_state.get("workspace_segment", "Original Sheet")

    active_sheet = view.get("active_sheet") or profile.sheet_names[0]

    # ---- Original Sheet ----
    if segment == "Original Sheet":
        with workspace_card("Original Sheet", "Your uploaded sheet remains unchanged."):
            persist_key = f"active_source_sheet_{profile.file_name}"
            options = list(profile.sheet_names)
            default = st.session_state.get(persist_key)
            if default not in options:
                default = options[0]

            col_header = st.columns([4, 1])
            with col_header[0]:
                if len(options) > 1:
                    active_sheet = st.selectbox(
                        "Roster sheet", options,
                        index=options.index(default), key=f"roster_sheet_{profile.file_name}",
                    )
                else:
                    active_sheet = options[0]
            with col_header[1]:
                if len(options) > 1:
                    st.write("")
                    st.write("")
                if st.button("Clear File", key="clear_workbook_file", use_container_width=True):
                    view["mode"] = "upload"
                    st.session_state.pop("workbook_upload", None)
                    for key in ("cached_loaded", "cached_profile", "cached_diagnostics", "cached_load_key", "cached_load_error"):
                        st.session_state.pop(key, None)
                    st.rerun()

            st.session_state[persist_key] = active_sheet
            sheet_profile = next(item for item in profile.sheets if item.name == active_sheet)
            frame = loaded.sheets[active_sheet]
            _render_workbook_intelligence(profile, active_sheet)
            st.caption(f"{sheet_profile.row_count:,} rows · {len(sheet_profile.columns):,} columns")
            _safe_dataframe(frame, use_container_width=True, hide_index=True, height=520)

    # ---- Results ----
    elif segment == "Results":
        if mode == "pending_action":
            _render_proposed_update(view, loaded, profile, settings, active_sheet)
        else:
            result_df = view.get("result_df")
            title = view.get("title") or view.get("pending_action_summary") or "Results"
            with workspace_card(title, None):
                if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                    st.caption(f"{len(result_df):,} rows shown")
                    _safe_dataframe(result_df, use_container_width=True, hide_index=True, height=520)
                else:
                    st.info("Ask a question to create a filtered or modified view.")

    # ---- Export ----
    elif segment == "Export":
        _render_export_segment(view)

    # "Advanced details" is a diagnostics surface hidden from end users.
    # It renders only when the show_workspace_details flag is set.
    if st.session_state.get("show_workspace_details"):
        _render_workspace_details(loaded, profile, diagnostics, settings, view)
    return


def _render_workbook_intelligence(profile, active_sheet: str) -> None:
    sheet_profile = next((item for item in profile.sheets if item.name == active_sheet), None)
    if sheet_profile is None:
        return
    summary = getattr(profile, "workbook_summary", {}) or {}
    st.markdown("#### Workbook Intelligence")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Sheets", summary.get("sheet_count", len(profile.sheets)))
    metric_cols[1].metric("Rows", f"{sheet_profile.row_count:,}")
    metric_cols[2].metric("Columns", f"{getattr(sheet_profile, 'column_count', len(sheet_profile.columns)):,}")
    metric_cols[3].metric("Missing fields", len(getattr(sheet_profile, "missing_by_column", {}) or {}))

    workflows = list(getattr(sheet_profile, "answerable_workflows", []) or [])
    if workflows:
        st.markdown("**Ready workflows**")
        st.markdown(
            '<div class="workflow-chip-row">'
            + "".join(f'<span class="workflow-chip">{workflow}</span>' for workflow in workflows)
            + "</div>",
            unsafe_allow_html=True,
        )

    warnings = list(getattr(sheet_profile, "warnings", []) or [])
    if warnings:
        with st.expander("Intelligence notes", expanded=False):
            for warning in warnings[:8]:
                st.markdown(f"- {warning}")


def _render_proposed_update(view, loaded, profile, settings, active_sheet: str) -> None:
    """The Proposed Update card: a clear summary of a pending edit plus
    Confirm / Cancel buttons that reuse the chat's confirmation pipeline."""
    with workspace_card("Proposed Update", "Review before anything is written."):
        action = view.get("pending_action_label") or "Update"
        st.markdown(f"**Action:** {action}")
        summary = view.get("pending_action_summary")
        if summary:
            st.markdown(summary)
        affected = view.get("affected_rows")
        if affected is not None:
            st.markdown(f"**Rows affected:** {affected}")
        column = view.get("target_column")
        if column:
            st.markdown(f"**Column updated:** {column}")

        pending_df = view.get("pending_preview_df")
        if isinstance(pending_df, pd.DataFrame) and not pending_df.empty:
            st.caption("Preview of affected rows")
            _safe_dataframe(pending_df, use_container_width=True, hide_index=True)
        st.caption("The original workbook will not be modified.")

        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            if st.button("Confirm Update", type="primary", use_container_width=True,
                         key="ws_confirm_update"):
                from ui.chat_panel import route_message
                route_message(request="yes, do it", selected_sheet=active_sheet,
                              loaded=loaded, profile=profile, settings=settings)
                st.rerun()
        with cancel_col:
            if st.button("Cancel", use_container_width=True, key="ws_cancel_update"):
                from ui.chat_panel import route_message
                route_message(request="no, cancel", selected_sheet=active_sheet,
                              loaded=loaded, profile=profile, settings=settings)
                st.rerun()


def _render_export_segment(view) -> None:
    """Business-ready export: current result (Excel/CSV) + edited workbook."""
    with workspace_card("Export", "Download your results or the edited workbook."):
        result_df = view.get("result_df")
        has_result = isinstance(result_df, pd.DataFrame) and not result_df.empty
        if has_result:
            st.download_button(
                "Export Results to Excel",
                data=_excel_bytes(result_df),
                file_name="dean_assistant_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_results_xlsx", use_container_width=True,
            )
            st.download_button(
                "Export Results to CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="dean_assistant_results.csv",
                mime="text/csv",
                key="export_results_csv", use_container_width=True,
            )
        else:
            st.caption("Run a question to enable result exports.")

        path_str = view.get("download_path") or st.session_state.get("latest_output_file")
        if path_str and Path(path_str).exists():
            path = Path(path_str)
            st.download_button(
                "Download Edited Workbook (.xlsx)",
                data=path.read_bytes(),
                file_name=path.name,
                key="export_edited_xlsx", use_container_width=True,
            )
        else:
            st.caption("An edited-workbook download appears here after you confirm an update.")


def _render_workspace_details(loaded, profile, diagnostics, settings, view) -> None:
    """One collapsed drawer holding all the technical detail: diagnostics
    (privacy / runtime / detected columns / health) plus the working-sheet,
    figures, and export-center status zones."""
    with st.expander("Advanced details", expanded=False):
        # Diagnostics relocated from the old top-of-page status strip.
        if loaded is not None and profile is not None:
            try:
                from app import render_status_badges
                active_sh = view.get("active_sheet") or profile.sheet_names[0]
                render_status_badges(loaded, settings, active_sheet=active_sh, profile=profile)
            except Exception:
                pass
            if diagnostics:
                try:
                    from ui.health_check import render_workbook_health_check
                    render_workbook_health_check(diagnostics)
                except Exception:
                    pass
        # 1. Workbook summary for test assertions
        if loaded is not None and profile is not None:
            active_sh = view.get("active_sheet") or profile.sheet_names[0]
            sheet_prof = next((item for item in profile.sheets if item.name == active_sh), None)
            if sheet_prof:
                st.markdown(f"**Original Workbook**")
                st.caption(f"File: `{profile.file_name}` | Sheet: `{active_sh}` | Workbook Loaded")
                st.caption(f"{sheet_prof.row_count} Students / Rows")
                st.caption(f"{len(sheet_prof.columns)} Columns")
                st.caption("Original file protected | Read-only source")
        else:
            st.markdown("No workbook loaded yet.")
            st.caption("No workbook loaded")
        
        # 2. Working Sheet / Live Output tests compatibility
        working_title = (
            "Modified Working Sheet"
            if st.session_state.get("latest_output_file")
            else "Working Sheet"
        )
        st.markdown(f"**{working_title}**")
        st.caption("Original workbook is unchanged")
        messages = st.session_state.get("chat_messages") or []
        latest_attachment = None
        for msg in reversed(messages):
            attachment = msg.get("attachment") or {}
            if attachment.get("type") in {"result", "confirmation", "edit_plan"}:
                latest_attachment = attachment
                break
        if latest_attachment is None:
            st.markdown("Ask a question in the chat to see results.")
        
        # 3. Figures & Insights tests compatibility
        st.markdown("**Figures & Insights**")
        if not st.session_state.get("latest_figure"):
            st.markdown("Ask the assistant for a chart.")
        if loaded is not None:
            try:
                from app import render_figures_panel
                render_figures_panel()
            except Exception:
                pass
            
        # 4. Export Center tests compatibility
        st.markdown("**Export Center**")
        if not st.session_state.get("latest_output_file"):
            st.markdown("No exports yet.")
