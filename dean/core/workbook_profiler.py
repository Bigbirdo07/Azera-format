from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.schema import canonical_for


@dataclass(frozen=True)
class SheetProfile:
    name: str
    row_count: int
    columns: list[str]
    column_count: int = 0
    canonical_fields: dict[str, str] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    missing_by_column: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    answerable_workflows: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkbookProfile:
    file_name: str
    sheet_names: list[str]
    sheets: list[SheetProfile]
    primary_sheet: str = ""
    workbook_summary: dict[str, Any] = field(default_factory=dict)


def profile_workbook(file_name: str, sheets: dict[str, pd.DataFrame]) -> WorkbookProfile:
    sheet_profiles = [
        _profile_sheet(sheet_name, dataframe)
        for sheet_name, dataframe in sheets.items()
    ]
    primary_sheet = _primary_sheet(sheet_profiles)

    return WorkbookProfile(
        file_name=file_name,
        sheet_names=list(sheets.keys()),
        sheets=sheet_profiles,
        primary_sheet=primary_sheet,
        workbook_summary=_workbook_summary(sheet_profiles, primary_sheet),
    )


def _profile_sheet(sheet_name: str, dataframe: pd.DataFrame) -> SheetProfile:
    columns = [str(column) for column in dataframe.columns]
    canonical_fields = {
        column: canonical
        for column in columns
        if (canonical := canonical_for(column))
    }
    numeric_columns = [
        column for column in columns
        if pd.to_numeric(dataframe[column], errors="coerce").notna().mean() >= 0.75
    ]
    categorical_columns = [
        column for column in columns
        if column not in numeric_columns and dataframe[column].nunique(dropna=True) <= 50
    ]
    missing_by_column = {
        column: int(_blank_mask(dataframe[column]).sum())
        for column in columns
        if int(_blank_mask(dataframe[column]).sum()) > 0
    }
    duplicate_rows = int(dataframe.duplicated(keep=False).sum())
    canonicals = set(canonical_fields.values())
    return SheetProfile(
        name=sheet_name,
        row_count=len(dataframe.index),
        columns=columns,
        column_count=len(columns),
        canonical_fields=canonical_fields,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        missing_by_column=missing_by_column,
        duplicate_rows=duplicate_rows,
        answerable_workflows=_answerable_workflows(canonicals),
        warnings=_profile_warnings(canonicals, missing_by_column, duplicate_rows),
    )


def _answerable_workflows(canonicals: set[str]) -> list[str]:
    workflows: list[str] = []
    if {"student_id", "student"}.intersection(canonicals) or "gpa" in canonicals:
        workflows.append("Student roster questions")
    if "advisor" in canonicals or "teacher" in canonicals:
        workflows.append("Advisor caseload analysis")
    if "gpa" in canonicals and ("academic_status" in canonicals or "academic_watch" in canonicals):
        workflows.append("Academic risk review")
    if {"attendance_rate", "attendance_category", "days_absent", "attendance_risk"}.intersection(canonicals):
        workflows.append("Attendance risk review")
    if {"sat_total", "sat_math", "sat_ebrw", "psat_total", "psat_math", "psat_reading_writing"}.intersection(canonicals):
        workflows.append("Assessment readiness review")
    if "major" in canonicals or "discipline" in canonicals:
        workflows.append("Program and discipline comparison")
    if "location" in canonicals:
        workflows.append("Campus/location comparison")
    workflows.append("Data quality review")
    return list(dict.fromkeys(workflows))


def _profile_warnings(
    canonicals: set[str],
    missing_by_column: dict[str, int],
    duplicate_rows: int,
) -> list[str]:
    warnings: list[str] = []
    if "gpa" not in canonicals:
        warnings.append("GPA field not detected; academic-performance summaries will be limited.")
    if not {"advisor", "teacher"}.intersection(canonicals):
        warnings.append("Advisor/teacher field not detected; caseload grouping will be limited.")
    if not {"attendance_rate", "attendance_category", "days_absent", "attendance_risk"}.intersection(canonicals):
        warnings.append("Attendance fields not detected; attendance-risk questions will be limited.")
    if missing_by_column:
        worst_column, worst_count = max(missing_by_column.items(), key=lambda item: item[1])
        warnings.append(f"{len(missing_by_column)} column(s) contain blanks; {worst_column} has {worst_count}.")
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} fully duplicated row(s) detected.")
    return warnings


def _primary_sheet(sheet_profiles: list[SheetProfile]) -> str:
    if not sheet_profiles:
        return ""
    named = next(
        (sheet.name for sheet in sheet_profiles if sheet.name.lower() in {"student roster", "students"}),
        "",
    )
    if named:
        return named
    return max(sheet_profiles, key=lambda sheet: sheet.row_count).name


def _workbook_summary(sheet_profiles: list[SheetProfile], primary_sheet: str) -> dict[str, Any]:
    primary = next((sheet for sheet in sheet_profiles if sheet.name == primary_sheet), None)
    total_rows = sum(sheet.row_count for sheet in sheet_profiles)
    total_columns = sum(sheet.column_count or len(sheet.columns) for sheet in sheet_profiles)
    workflows = []
    warnings = []
    for sheet in sheet_profiles:
        workflows.extend(sheet.answerable_workflows)
        warnings.extend(sheet.warnings)
    return {
        "sheet_count": len(sheet_profiles),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "primary_sheet": primary_sheet,
        "primary_rows": primary.row_count if primary else 0,
        "primary_columns": (primary.column_count or len(primary.columns)) if primary else 0,
        "answerable_workflows": list(dict.fromkeys(workflows)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _blank_mask(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.casefold()
    return series.isna() | text.isin({"", "nan", "none", "null", "n/a", "na"})
