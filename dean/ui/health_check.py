from __future__ import annotations

import pandas as pd
import streamlit as st

from core.workbook_diagnostics import WorkbookDiagnostics


def render_workbook_health_check(diagnostics: WorkbookDiagnostics) -> None:
    st.subheader("Workbook Health Check")
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Sheet count", diagnostics.sheet_count)
    metric_b.metric("Hidden sheets", len(diagnostics.hidden_sheets))

    if diagnostics.hidden_sheets:
        st.warning(f"Hidden sheets detected: {', '.join(diagnostics.hidden_sheets)}")

    rows = []
    for sheet in diagnostics.sheets:
        rows.append(
            {
                "Sheet": sheet.name,
                "State": sheet.state,
                "Protected": sheet.is_protected,
                "Likely header row": sheet.likely_header_row,
                "Merged ranges": len(sheet.merged_cell_ranges),
                "Formulas": sheet.formula_count,
                "Blank rows": sheet.blank_row_count,
                "Blank columns": sheet.blank_column_count,
                "Duplicate columns": ", ".join(sheet.duplicate_column_names),
                "Tables": sheet.table_count,
                "Filter": sheet.has_existing_filter,
                "Charts": sheet.chart_count,
                "Pivot tables": sheet.pivot_table_count,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    warnings = []
    for sheet in diagnostics.sheets:
        for warning in sheet.data_quality_warnings:
            warnings.append(f"{sheet.name}: {warning}")
        for warning in sheet.complex_formatting_warnings:
            warnings.append(f"{sheet.name}: {warning}")

    if warnings:
        with st.expander("Data quality and formatting warnings", expanded=True):
            for warning in warnings:
                st.warning(warning)
    else:
        st.success("No major workbook health issues detected.")
