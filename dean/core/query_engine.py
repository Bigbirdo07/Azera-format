"""Read-only query engine for Ask Mode.

Every exact number the assistant reports is computed here with pandas, never by
the LLM. The engine takes a validated ask_question query plan and returns a
structured, factual result. The LLM may later phrase the result in plain
English, but it is not trusted to do arithmetic.

This module never mutates the workbook and never writes a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.schema import canonical_for


# Values that count as a "missing / incomplete" status for membership filters.
_BLANK_TOKENS = {"", "nan", "none", "null", "n/a", "na"}

# Canonical concepts for imported status/standing labels (not computed by
# Dean) that commonly get crossed against a numeric metric in a groupby or
# pivot. These correlate loosely with metrics like GPA in real data but are
# never a strict cutoff -- a school's own SIS sets them from a mix of
# factors (credits, discipline, a prior term's probation, etc.), so a
# high-GPA student can still carry a "Bad Standing" label. Without this note
# that reads as a data error rather than the normal case it is.
_IMPORTED_STATUS_CANONICALS = {"academic_status", "conduct_status", "enrollment_status"}
_IMPORTED_STATUS_NAME_HINTS = ("standing", "status")


def _status_column_caveat(frame: pd.DataFrame, column: str) -> str:
    """Caveat to append to a result description when `column` is an imported
    status/standing label being crossed against a numeric metric, and the
    workbook has no accompanying "<...> Reason" column to explain outliers."""
    if not column:
        return ""
    canonical = canonical_for(column)
    normalized = column.strip().lower()
    is_status_like = canonical in _IMPORTED_STATUS_CANONICALS or any(
        hint in normalized for hint in _IMPORTED_STATUS_NAME_HINTS
    )
    if not is_status_like:
        return ""
    # Only a reason column tied to *this* status field counts -- a workbook
    # can have "Risk Reason" (Dean's own combined-risk explanation) without
    # that saying anything about why an imported "Standing" label disagrees
    # with GPA. Requiring the column name as a prefix keeps this specific.
    has_reason_column = any(
        str(c).strip().lower().startswith(normalized) and "reason" in str(c).strip().lower()
        for c in frame.columns
    )
    if has_reason_column:
        return ""
    return (
        f" Note: {column} is an existing label from your workbook, not derived from this "
        f"metric -- individual students may not follow the overall pattern (no {column} "
        "Reason field is present to explain exceptions)."
    )

ASK_OPERATIONS = {
    "count_rows",
    "count_unique",
    "list_unique",
    "percent_rows",
    "sum_column",
    "average_column",
    "min_column",
    "max_column",
    "groupby_count",
    "groupby_sum",
    "groupby_average",
    "missing_summary",
    "duplicate_check",
    "filtered_preview",
    "data_quality_summary",
    "cohort_summary",
    "cohort_comparison",
    "student_intervention_summary",
    "advisor_outcome_summary",
    "pivot_table_summary",
    "trend_summary",
}

MAX_PREVIEW_ROWS = 500


class QueryExecutionError(ValueError):
    pass


@dataclass
class QueryResult:
    operation: str
    description: str  # factual, plain sentence the LLM may rephrase
    value: float | int | None = None
    row_count: int | None = None
    table: list[dict[str, Any]] = field(default_factory=list)
    columns_used: list[str] = field(default_factory=list)
    preview_truncated: bool = False


def run_query(query: dict[str, Any], sheets: dict[str, pd.DataFrame]) -> QueryResult:
    """Execute an ask_question query plan against in-memory DataFrames."""
    operation = query.get("operation")
    if operation not in ASK_OPERATIONS:
        raise QueryExecutionError(f"Unsupported ask operation: {operation!r}")

    sheet_name = query.get("sheet") or _default_sheet(sheets)
    if sheet_name not in sheets:
        raise QueryExecutionError(f"Unknown sheet: {sheet_name!r}")
    frame = sheets[sheet_name]

    filters = query.get("filters") or []
    filter_mode = (query.get("filter_mode") or "all").lower()
    if filter_mode not in {"all", "any"}:
        filter_mode = "all"
    group_by = query.get("group_by") or ""
    value_column = query.get("value_column") or ""
    sort_by = query.get("sort_by") or ""
    sort = query.get("sort") or None
    raw_limit = query.get("limit", 10)
    if operation == "filtered_preview" and "limit" in query and raw_limit is None:
        preview_limit = min(len(frame), MAX_PREVIEW_ROWS)
    else:
        preview_limit = int(raw_limit or 10)
    group_limit = None if "limit" in query and raw_limit is None else int(raw_limit or 10)
    select_columns = [c for c in (query.get("select_columns") or []) if isinstance(c, str) and c]

    _assert_columns_exist(frame, filters, group_by, value_column)
    if sort and sort.get("column") and sort["column"] not in frame.columns:
        group_sort_columns = set()
        if operation == "groupby_count":
            group_sort_columns.add("Count")
        elif operation in {"groupby_sum", "groupby_average"} and value_column:
            group_sort_columns.add(value_column)
        elif operation == "advisor_outcome_summary":
            group_sort_columns.add("Outcome Score")
        elif operation == "student_intervention_summary":
            group_sort_columns.add("Intervention Signals")
        if sort["column"] in group_sort_columns:
            pass
        else:
            raise QueryExecutionError(f"Unknown column: {sort['column']!r}")
    for column in select_columns:
        if column not in frame.columns:
            raise QueryExecutionError(f"Unknown column: {column!r}")

    if operation == "count_rows":
        return _count_rows(frame, filters, filter_mode)
    if operation == "percent_rows":
        return _percent_rows(frame, filters, filter_mode)
    if operation == "count_unique":
        return _count_unique(frame, value_column or group_by, filters, filter_mode)
    if operation == "list_unique":
        return _list_unique(frame, value_column or group_by, filters, preview_limit, filter_mode)
    if operation == "sum_column":
        return _aggregate_column(frame, filters, value_column, "sum", filter_mode)
    if operation == "average_column":
        return _aggregate_column(frame, filters, value_column, "mean", filter_mode)
    if operation == "min_column":
        return _aggregate_column(frame, filters, value_column, "min", filter_mode)
    if operation == "max_column":
        return _aggregate_column(frame, filters, value_column, "max", filter_mode)
    if operation in {"groupby_count", "groupby_sum", "groupby_average"}:
        return _group_by(frame, filters, group_by, value_column, operation, sort_by, group_limit, sort, filter_mode)
    if operation == "missing_summary":
        return _missing_summary(frame)
    if operation == "duplicate_check":
        return _duplicate_check(frame, value_column or group_by)
    if operation == "filtered_preview":
        return _filtered_preview(frame, filters, preview_limit, sort, select_columns, filter_mode)
    if operation == "data_quality_summary":
        return _data_quality_summary(frame)
    if operation == "cohort_summary":
        return _cohort_summary(frame, filters, filter_mode)
    if operation == "cohort_comparison":
        return _cohort_comparison(frame, filters, group_by, filter_mode)
    if operation == "student_intervention_summary":
        return _student_intervention_summary(frame, filters, preview_limit, filter_mode)
    if operation == "advisor_outcome_summary":
        return _advisor_outcome_summary(frame, filters, group_limit, sort, filter_mode)
    if operation == "pivot_table_summary":
        return _pivot_table_summary(frame, query, filters, filter_mode)
    if operation == "trend_summary":
        return _trend_summary(frame, filters, filter_mode)

    raise QueryExecutionError(f"Unhandled operation: {operation!r}")


# Operations ------------------------------------------------------------------


def _count_rows(
    frame: pd.DataFrame, filters: list[dict[str, Any]], filter_mode: str = "all",
) -> QueryResult:
    mask = _build_mask(frame, filters, filter_mode)
    count = int(mask.sum())
    return QueryResult(
        operation="count_rows",
        value=count,
        row_count=count,
        table=[{"Metric": "Matching records", "Value": count}],
        description=f"{count} row(s) match the conditions.",
        columns_used=[f["column"] for f in filters if f.get("column")],
    )


def _percent_rows(
    frame: pd.DataFrame, filters: list[dict[str, Any]], filter_mode: str = "all",
) -> QueryResult:
    """Share of rows matching the filter, expressed as a percent of total."""
    total = int(len(frame))
    if total == 0:
        return QueryResult(
            operation="percent_rows", value=0.0, row_count=0,
            description="The sheet is empty, so the share is 0%.",
            columns_used=[f["column"] for f in filters if f.get("column")],
        )
    mask = _build_mask(frame, filters, filter_mode)
    matched = int(mask.sum())
    pct = round((matched / total) * 100, 2)
    return QueryResult(
        operation="percent_rows",
        value=pct,
        row_count=matched,
        description=f"{matched} of {total} row(s) match — {pct}% of the sheet.",
        columns_used=[f["column"] for f in filters if f.get("column")],
    )


def _count_unique(
    frame: pd.DataFrame, column: str,
    filters: list[dict[str, Any]] | None = None, filter_mode: str = "all",
) -> QueryResult:
    if not column or column not in frame.columns:
        raise QueryExecutionError("count_unique requires a known column.")
    filters = filters or []
    mask = _build_mask(frame, filters, filter_mode)
    count = int(frame.loc[mask, column].nunique(dropna=True))
    return QueryResult(
        operation="count_unique",
        value=count,
        row_count=count,
        description=f"There are {count} distinct {column} value(s).",
        columns_used=[column] + [f["column"] for f in filters if f.get("column")],
    )


def _list_unique(
    frame: pd.DataFrame, column: str, filters: list[dict[str, Any]] | None = None,
    limit: int = 10, filter_mode: str = "all",
) -> QueryResult:
    """Return the distinct values of a column (sorted) as the result table."""
    if not column or column not in frame.columns:
        raise QueryExecutionError("list_unique requires a known column.")
    filters = filters or []
    mask = _build_mask(frame, filters, filter_mode)
    series = frame.loc[mask, column].dropna()
    try:
        values = sorted(series.astype(str).unique().tolist())
    except TypeError:
        values = list(dict.fromkeys(series.astype(str).tolist()))
    total = len(values)
    truncated = isinstance(limit, int) and limit > 0 and total > limit
    visible = values[:limit] if isinstance(limit, int) and limit > 0 else values
    table = [{column: v} for v in visible]
    return QueryResult(
        operation="list_unique",
        value=total,
        row_count=total,
        table=table,
        description=f"{total} distinct {column} value(s).",
        columns_used=[column] + [f["column"] for f in filters if f.get("column")],
        preview_truncated=truncated,
    )


def _aggregate_column(
    frame: pd.DataFrame, filters: list[dict[str, Any]], value_column: str, how: str,
    filter_mode: str = "all",
) -> QueryResult:
    if not value_column:
        raise QueryExecutionError(f"{how} requires a value_column.")
    mask = _build_mask(frame, filters, filter_mode)
    numeric = pd.to_numeric(frame.loc[mask, value_column], errors="coerce")
    result = getattr(numeric, how)()
    value = None if pd.isna(result) else round(float(result), 4)
    label = {"sum": "Total", "mean": "Average", "min": "Minimum", "max": "Maximum"}[how]
    return QueryResult(
        operation=f"{'average' if how == 'mean' else how}_column",
        value=value,
        row_count=int(mask.sum()),
        description=f"{label} of {value_column} across matching rows is {value}.",
        columns_used=[value_column] + [f["column"] for f in filters if f.get("column")],
    )


def _group_by(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    group_by: str,
    value_column: str,
    operation: str,
    sort_by: str,
    limit: int | None,
    sort: dict[str, Any] | None = None,
    filter_mode: str = "all",
) -> QueryResult:
    if not group_by:
        raise QueryExecutionError("group operations require a group_by column.")
    mask = _build_mask(frame, filters, filter_mode)
    subset = frame.loc[mask]

    if operation == "groupby_count":
        grouped = subset.groupby(group_by, dropna=False).size().reset_index(name="Count")
        metric_column = "Count"
    else:
        if not value_column:
            raise QueryExecutionError(f"{operation} requires a value_column.")
        numeric = pd.to_numeric(subset[value_column], errors="coerce")
        agg = "sum" if operation == "groupby_sum" else "mean"
        grouped = (
            subset.assign(**{value_column: numeric})
            .groupby(group_by, dropna=False)[value_column]
            .agg(agg)
            .round(4)
            .reset_index()
        )
        metric_column = value_column

    # Sort direction: prefer the structured sort dict (which carries direction)
    # over the legacy sort_by string. Default to descending so existing callers
    # that only set sort_by keep their "top N" semantics.
    sort_column = sort_by if sort_by in grouped.columns else metric_column
    direction = "desc"
    if isinstance(sort, dict) and sort.get("column") in grouped.columns:
        sort_column = sort["column"]
        direction = str(sort.get("direction") or "desc").lower()
    ascending = direction == "asc"
    grouped = grouped.sort_values(sort_column, ascending=ascending)
    if limit is not None:
        grouped = grouped.head(limit)

    table = grouped.to_dict(orient="records")
    top = table[0] if table else {}
    superlative = "lowest" if ascending else "highest"
    top_desc = (
        f"{top.get(group_by)} has the {superlative} {metric_column} ({top.get(metric_column)})."
        if top
        else "No rows matched."
    )
    caveat = _status_column_caveat(frame, group_by) if operation != "groupby_count" else ""
    return QueryResult(
        operation=operation,
        row_count=int(mask.sum()),
        table=table,
        description=f"Grouped {metric_column} by {group_by}. {top_desc}{caveat}",
        columns_used=[group_by] + ([value_column] if value_column else []),
    )


def _missing_summary(frame: pd.DataFrame) -> QueryResult:
    total = len(frame)
    rows = []
    for column in frame.columns:
        missing = int(_blank_mask(frame[column]).sum())
        if missing:
            rows.append(
                {
                    "Column": column,
                    "Missing": missing,
                    "Missing %": round(100 * missing / total, 1) if total else 0.0,
                }
            )
    rows.sort(key=lambda item: item["Missing"], reverse=True)
    if rows:
        worst = rows[0]
        desc = f"{len(rows)} column(s) have missing values; {worst['Column']} has the most ({worst['Missing']})."
    else:
        desc = "No columns have missing values."
    return QueryResult(
        operation="missing_summary",
        table=rows,
        row_count=total,
        description=desc,
        columns_used=[r["Column"] for r in rows],
    )


def _duplicate_check(frame: pd.DataFrame, column: str) -> QueryResult:
    if column and column in frame.columns:
        dup_mask = frame[column].duplicated(keep=False) & frame[column].notna()
        dup_count = int(dup_mask.sum())
        examples = (
            frame.loc[dup_mask, column].astype(str).value_counts().head(10).reset_index()
        )
        examples.columns = [column, "Occurrences"]
        return QueryResult(
            operation="duplicate_check",
            value=dup_count,
            row_count=dup_count,
            table=examples.to_dict(orient="records"),
            description=f"{dup_count} row(s) share a duplicated {column} value.",
            columns_used=[column],
        )
    dup_count = int(frame.duplicated(keep=False).sum())
    return QueryResult(
        operation="duplicate_check",
        value=dup_count,
        row_count=dup_count,
        description=f"{dup_count} fully duplicated row(s) found across all columns.",
        columns_used=list(frame.columns),
    )


def _filtered_preview(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    limit: int,
    sort: dict[str, Any] | None = None,
    select_columns: list[str] | None = None,
    filter_mode: str = "all",
) -> QueryResult:
    mask = _build_mask(frame, filters, filter_mode)
    matched = int(mask.sum())
    subset = frame.loc[mask]
    limit = min(int(limit or MAX_PREVIEW_ROWS), MAX_PREVIEW_ROWS)
    sort_note = ""
    if sort and sort.get("column") in frame.columns:
        ascending = str(sort.get("direction", "asc")).lower() != "desc"
        column = sort["column"]
        # Sort numerically when possible so GPA-like columns order correctly.
        numeric = pd.to_numeric(subset[column], errors="coerce")
        if numeric.notna().any():
            subset = subset.assign(_sort=numeric).sort_values("_sort", ascending=ascending).drop(columns="_sort")
        else:
            subset = subset.sort_values(column, ascending=ascending)
        sort_note = f", sorted by {column} {'ascending' if ascending else 'descending'}"
    preview = subset.head(limit)
    projection_note = ""
    columns_used = [f["column"] for f in filters if f.get("column")]
    if select_columns:
        kept = [c for c in select_columns if c in preview.columns]
        if kept:
            preview = preview[kept]
            projection_note = f", columns: {', '.join(kept)}"
            columns_used = list(dict.fromkeys(columns_used + kept))
    if matched <= limit:
        description = f"{matched} matching row(s)."
        truncated = False
    else:
        description = f"{matched} row(s) match; showing the first {limit}{sort_note}{projection_note}."
        truncated = True

    return QueryResult(
        operation="filtered_preview",
        row_count=matched,
        table=preview.to_dict(orient="records"),
        description=description,
        columns_used=columns_used,
        preview_truncated=truncated,
    )


def _data_quality_summary(frame: pd.DataFrame) -> QueryResult:
    total = len(frame)
    missing = _missing_summary(frame)
    full_dupes = int(frame.duplicated(keep=False).sum())
    rows = list(missing.table)
    # None (not "") so pandas keeps "Missing %" a single numeric dtype --
    # mixing a string into an otherwise-float column breaks Streamlit's
    # Arrow serialization of the result table.
    rows.append({"Column": "(fully duplicated rows)", "Missing": full_dupes, "Missing %": None})
    desc = (
        f"{total} rows scanned. {len(missing.table)} column(s) have missing values; "
        f"{full_dupes} fully duplicated row(s)."
    )
    return QueryResult(
        operation="data_quality_summary",
        row_count=total,
        table=rows,
        description=desc,
        columns_used=list(frame.columns),
    )


def _cohort_summary(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    filter_mode: str = "all",
) -> QueryResult:
    """Compact factual profile for a filtered student cohort."""
    mask = _build_mask(frame, filters, filter_mode)
    subset = frame.loc[mask]
    row_count = int(len(subset))
    rows: list[dict[str, Any]] = [{"Metric": "Students", "Value": row_count}]
    columns_used = [f["column"] for f in filters if f.get("column")]

    gpa_col = _find_column(frame, ("gpa", "grade point average"))
    if gpa_col:
        gpa_values = pd.to_numeric(subset[gpa_col], errors="coerce")
        gpa = gpa_values.mean()
        if not pd.isna(gpa):
            rows.append({"Metric": f"Average {gpa_col}", "Value": round(float(gpa), 3)})
            rows.append({"Metric": f"Median {gpa_col}", "Value": round(float(gpa_values.median()), 3)})
            columns_used.append(gpa_col)
        low_gpa = int((gpa_values < 2.5).sum())
        rows.append({"Metric": f"{gpa_col} below 2.5", "Value": low_gpa})
        rows.append({"Metric": f"{gpa_col} below 2.0", "Value": int((gpa_values < 2.0).sum())})

    attendance_col = _find_column(frame, ("attendance rate", "attendance %", "attendance"))
    if attendance_col:
        attendance_values = pd.to_numeric(subset[attendance_col], errors="coerce")
        attendance = attendance_values.mean()
        if not pd.isna(attendance):
            rows.append({"Metric": f"Average {attendance_col}", "Value": round(float(attendance), 2)})
            columns_used.append(attendance_col)
        threshold = _scaled_numeric_value(subset[attendance_col], 0.9)
        attendance_support = int((attendance_values < threshold).sum())
        rows.append({"Metric": f"{attendance_col} below 90%", "Value": attendance_support})

    attendance_category_col = _find_column(frame, ("attendance category",))
    if attendance_category_col:
        support = int(
            subset[attendance_category_col]
            .astype(str)
            .str.casefold()
            .eq("needs attendance support")
            .sum()
        )
        rows.append({"Metric": "Needs Attendance Support", "Value": support})
        columns_used.append(attendance_category_col)

    standing_col = _find_column(frame, ("standing", "academic standing", "academic status"))
    if standing_col:
        bad_standing = int(
            subset[standing_col].astype(str).str.casefold().eq("bad standing").sum()
        )
        rows.append({"Metric": "Bad Standing", "Value": bad_standing})
        columns_used.append(standing_col)

    for score_col in ("PSAT Math", "PSAT English", "PSAT Total", "SAT Math", "SAT English", "SAT Total"):
        column = _find_column(frame, (score_col,))
        if not column:
            continue
        avg = pd.to_numeric(subset[column], errors="coerce").mean()
        if not pd.isna(avg):
            rows.append({"Metric": f"Average {column}", "Value": round(float(avg), 2)})
            columns_used.append(column)

    days_absent_col = _find_column(frame, ("days absent", "absences", "total absences"))
    if days_absent_col:
        absent_values = pd.to_numeric(subset[days_absent_col], errors="coerce")
        avg_absent = absent_values.mean()
        if not pd.isna(avg_absent):
            rows.append({"Metric": f"Average {days_absent_col}", "Value": round(float(avg_absent), 2)})
            rows.append({"Metric": f"{days_absent_col} above 10", "Value": int((absent_values > 10).sum())})
            columns_used.append(days_absent_col)

    for label in ("Standing", "Year", "Major", "Discipline", "Location", "Advisor"):
        column = _find_column(frame, (label,))
        if not column:
            continue
        counts = subset[column].fillna("(missing)").astype(str).value_counts().head(4)
        for value, count in counts.items():
            rows.append({"Metric": f"{column}: {value}", "Value": int(count)})
        columns_used.append(column)

    description = _cohort_summary_description(rows, row_count)
    return QueryResult(
        operation="cohort_summary",
        value=row_count,
        row_count=row_count,
        table=rows,
        description=description,
        columns_used=list(dict.fromkeys(columns_used)),
    )


def _cohort_comparison(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    group_by: str,
    filter_mode: str = "all",
) -> QueryResult:
    if not group_by or group_by not in frame.columns:
        raise QueryExecutionError("cohort_comparison requires a known group_by column.")
    mask = _build_mask(frame, filters, filter_mode)
    subset = frame.loc[mask]
    if subset.empty:
        return QueryResult(
            operation="cohort_comparison",
            row_count=0,
            table=[],
            description="No rows matched the requested comparison.",
            columns_used=[group_by] + [f["column"] for f in filters if f.get("column")],
        )

    gpa_col = _find_column(frame, ("gpa", "grade point average"))
    standing_col = _find_column(frame, ("standing", "academic standing", "academic status"))
    attendance_col = _find_column(frame, ("attendance rate", "attendance %", "attendance"))
    attendance_category_col = _find_column(frame, ("attendance category",))
    major_col = _find_column(frame, ("major",))
    year_col = _find_column(frame, ("year", "class year", "grade"))
    location_col = _find_column(frame, ("location", "campus"))
    score_cols = [
        column for label in ("PSAT Total", "SAT Total")
        if (column := _find_column(frame, (label,)))
    ]

    rows: list[dict[str, Any]] = []
    for group_value, group in subset.groupby(group_by, dropna=False):
        row: dict[str, Any] = {group_by: group_value, "Students": int(len(group))}
        if gpa_col:
            gpa = pd.to_numeric(group[gpa_col], errors="coerce")
            row["Average GPA"] = None if pd.isna(gpa.mean()) else round(float(gpa.mean()), 3)
            row["GPA below 2.5"] = int((gpa < 2.5).sum())
        if standing_col:
            row["Bad Standing"] = int(
                group[standing_col].astype(str).str.casefold().eq("bad standing").sum()
            )
        if attendance_col:
            attendance = pd.to_numeric(group[attendance_col], errors="coerce")
            row["Average Attendance Rate"] = None if pd.isna(attendance.mean()) else round(float(attendance.mean()), 3)
            row["Attendance Rate below 90%"] = int((attendance < _scaled_numeric_value(group[attendance_col], 0.9)).sum())
        if attendance_category_col:
            row["Needs Attendance Support"] = int(
                group[attendance_category_col]
                .astype(str)
                .str.casefold()
                .eq("needs attendance support")
                .sum()
            )
        for column in score_cols:
            score = pd.to_numeric(group[column], errors="coerce")
            row[f"Average {column}"] = None if pd.isna(score.mean()) else round(float(score.mean()), 2)
        if major_col:
            top = group[major_col].dropna().astype(str).value_counts().head(1)
            row["Top Major"] = "" if top.empty else str(top.index[0])
        if year_col:
            top = group[year_col].dropna().astype(str).value_counts().head(1)
            row["Top Year"] = "" if top.empty else str(top.index[0])
        if location_col:
            top = group[location_col].dropna().astype(str).value_counts().head(1)
            row["Top Location"] = "" if top.empty else str(top.index[0])
        rows.append(row)

    rows.sort(key=lambda item: str(item.get(group_by, "")))
    compared = len(rows)
    return QueryResult(
        operation="cohort_comparison",
        value=compared,
        row_count=int(len(subset)),
        table=rows,
        description=f"Compared {compared} {group_by} group(s) across {len(subset)} matching row(s).",
        columns_used=list(dict.fromkeys(
            [group_by] + [f["column"] for f in filters if f.get("column")]
            + [c for c in [gpa_col, standing_col, attendance_col, attendance_category_col, major_col, year_col, location_col, *score_cols] if c]
        )),
    )


def _student_intervention_summary(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    limit: int,
    filter_mode: str = "all",
) -> QueryResult:
    """Rank students by transparent workbook risk indicators.

    This does not decide intervention clinically or administratively. It
    compiles the rows whose workbook indicators suggest staff review.
    """
    mask = _build_mask(frame, filters, filter_mode)
    subset = frame.loc[mask].copy()
    if subset.empty:
        return QueryResult(
            operation="student_intervention_summary",
            value=0,
            row_count=0,
            table=[],
            description="No rows matched the current scope for intervention review.",
            columns_used=[f["column"] for f in filters if f.get("column")],
        )

    score = pd.Series(0, index=subset.index, dtype="int64")
    reasons = pd.Series("", index=subset.index, dtype="object")
    columns_used = [f["column"] for f in filters if f.get("column")]

    def add_signal(condition: pd.Series, label: str, column: str | None) -> None:
        nonlocal score, reasons, columns_used
        condition = condition.fillna(False)
        score = score + condition.astype(int)
        reasons = reasons.where(~condition, reasons.apply(lambda text: f"{text}; {label}" if text else label))
        if column:
            columns_used.append(column)

    gpa_col = _find_column(frame, ("gpa", "grade point average"))
    if gpa_col:
        gpa = pd.to_numeric(subset[gpa_col], errors="coerce")
        add_signal(gpa < 2.5, "GPA below 2.5", gpa_col)
        add_signal(gpa < 2.0, "GPA below 2.0", gpa_col)

    attendance_col = _find_column(frame, ("attendance rate", "attendance %", "attendance"))
    if attendance_col:
        attendance = pd.to_numeric(subset[attendance_col], errors="coerce")
        add_signal(attendance < _scaled_numeric_value(subset[attendance_col], 0.9), "Attendance below 90%", attendance_col)
        add_signal(attendance < _scaled_numeric_value(subset[attendance_col], 0.8), "Attendance below 80%", attendance_col)

    standing_col = _find_column(frame, ("standing", "academic standing", "academic status"))
    if standing_col:
        standing = subset[standing_col].astype(str).str.strip().str.casefold()
        add_signal(
            standing.isin({"bad standing", "warning", "probation", "at risk", "academic warning", "academic probation"}),
            "Standing indicates risk",
            standing_col,
        )

    attendance_category_col = _find_column(frame, ("attendance category",))
    if attendance_category_col:
        category = subset[attendance_category_col].astype(str).str.strip().str.casefold()
        add_signal(category.eq("needs attendance support"), "Needs attendance support", attendance_category_col)

    days_absent_col = _find_column(frame, ("days absent", "absences", "total absences"))
    if days_absent_col:
        absent = pd.to_numeric(subset[days_absent_col], errors="coerce")
        add_signal(absent > 10, "More than 10 days absent", days_absent_col)

    review = subset.assign(**{"Intervention Signals": score, "Review Reason": reasons})
    review = review[review["Intervention Signals"] > 0]
    review = review.sort_values("Intervention Signals", ascending=False)
    total = int(len(review))
    visible = review.head(limit or MAX_PREVIEW_ROWS)

    preferred = [
        "Student ID", "Name", "Advisor", "Year", "Major", "Discipline", "Location",
        "GPA", "Standing", "Attendance Rate", "Attendance Category", "Days Absent",
        "Intervention Signals", "Review Reason",
    ]
    kept = [column for column in preferred if column in visible.columns]
    if not kept:
        kept = list(visible.columns)
    table = visible[kept].to_dict(orient="records")
    description = (
        f"{total} student(s) match intervention-review indicators. "
        "Indicators used: GPA below 2.5/2.0, attendance below 90%/80%, risky standing, "
        "attendance-support category, and days absent above 10 when those columns exist."
    )
    return QueryResult(
        operation="student_intervention_summary",
        value=total,
        row_count=total,
        table=table,
        description=description,
        columns_used=list(dict.fromkeys(columns_used + kept)),
        preview_truncated=total > len(table),
    )


def _advisor_outcome_summary(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    limit: int | None,
    sort: dict[str, Any] | None,
    filter_mode: str = "all",
) -> QueryResult:
    """Rank advisors by measurable student-outcome indicators."""
    advisor_col = _find_column(frame, ("advisor", "teacher", "professor"))
    if not advisor_col:
        raise QueryExecutionError("advisor_outcome_summary requires an Advisor/Teacher column.")

    mask = _build_mask(frame, filters, filter_mode)
    subset = frame.loc[mask].copy()
    if subset.empty:
        return QueryResult(
            operation="advisor_outcome_summary",
            value=0,
            row_count=0,
            table=[],
            description="No rows matched the current scope for advisor outcome review.",
            columns_used=[advisor_col] + [f["column"] for f in filters if f.get("column")],
        )

    gpa_col = _find_column(frame, ("gpa", "grade point average"))
    attendance_col = _find_column(frame, ("attendance rate", "attendance %", "attendance"))
    standing_col = _find_column(frame, ("standing", "academic standing", "academic status"))
    attendance_category_col = _find_column(frame, ("attendance category",))
    days_absent_col = _find_column(frame, ("days absent", "absences", "total absences"))

    rows: list[dict[str, Any]] = []
    for advisor, group in subset.groupby(advisor_col, dropna=False):
        count = int(len(group))
        row: dict[str, Any] = {advisor_col: advisor, "Students": count}

        avg_gpa = None
        low_gpa_pct = 0.0
        if gpa_col:
            gpa = pd.to_numeric(group[gpa_col], errors="coerce")
            avg_gpa = None if pd.isna(gpa.mean()) else round(float(gpa.mean()), 3)
            row["Average GPA"] = avg_gpa
            row["GPA below 2.5"] = int((gpa < 2.5).sum())
            low_gpa_pct = _pct(row["GPA below 2.5"], count)
            row["GPA below 2.5 %"] = low_gpa_pct

        avg_attendance = None
        low_attendance_pct = 0.0
        if attendance_col:
            attendance = pd.to_numeric(group[attendance_col], errors="coerce")
            avg_attendance = None if pd.isna(attendance.mean()) else round(float(attendance.mean()), 3)
            row["Average Attendance Rate"] = avg_attendance
            low_attendance = int((attendance < _scaled_numeric_value(group[attendance_col], 0.9)).sum())
            row["Attendance below 90%"] = low_attendance
            low_attendance_pct = _pct(low_attendance, count)
            row["Attendance below 90% %"] = low_attendance_pct

        bad_standing_pct = 0.0
        if standing_col:
            standing = group[standing_col].astype(str).str.strip().str.casefold()
            bad_standing = int(standing.isin({"bad standing", "warning", "probation", "at risk", "academic warning", "academic probation"}).sum())
            row["Risky Standing"] = bad_standing
            bad_standing_pct = _pct(bad_standing, count)
            row["Risky Standing %"] = bad_standing_pct

        support_pct = 0.0
        if attendance_category_col:
            category = group[attendance_category_col].astype(str).str.strip().str.casefold()
            support = int(category.eq("needs attendance support").sum())
            row["Needs Attendance Support"] = support
            support_pct = _pct(support, count)
            row["Needs Attendance Support %"] = support_pct

        if days_absent_col:
            absent = pd.to_numeric(group[days_absent_col], errors="coerce")
            avg_absent = absent.mean()
            row["Average Days Absent"] = None if pd.isna(avg_absent) else round(float(avg_absent), 2)

        row["Outcome Score"] = _advisor_outcome_score(
            avg_gpa=avg_gpa,
            avg_attendance=avg_attendance,
            low_gpa_pct=low_gpa_pct,
            low_attendance_pct=low_attendance_pct,
            bad_standing_pct=bad_standing_pct,
            support_pct=support_pct,
        )
        rows.append(row)

    direction = str((sort or {}).get("direction") or "desc").lower()
    rows.sort(key=lambda item: item.get("Outcome Score", 0), reverse=direction != "asc")
    if limit is not None:
        rows = rows[:limit]
    top = rows[0] if rows else {}
    description = (
        "Ranked advisor groups using workbook indicators: average GPA, average attendance, "
        "low-GPA share, low-attendance share, risky standing, and attendance-support counts. "
        f"{top.get(advisor_col)} has the {'lowest' if direction == 'asc' else 'highest'} outcome score "
        f"({top.get('Outcome Score')}) among the displayed groups."
        if top
        else "No advisor groups were available to rank."
    )
    used = [advisor_col] + [f["column"] for f in filters if f.get("column")]
    used.extend(c for c in [gpa_col, attendance_col, standing_col, attendance_category_col, days_absent_col] if c)
    return QueryResult(
        operation="advisor_outcome_summary",
        value=len(rows),
        row_count=int(len(subset)),
        table=rows,
        description=description,
        columns_used=list(dict.fromkeys(used)),
    )


def _pivot_table_summary(
    frame: pd.DataFrame,
    query: dict[str, Any],
    filters: list[dict[str, Any]],
    filter_mode: str = "all",
) -> QueryResult:
    row_column = query.get("pivot_rows") or query.get("group_by") or ""
    column_column = query.get("pivot_columns") or ""
    value_column = query.get("value_column") or ""
    metric = query.get("metric") or ("average" if value_column else "count")
    if not row_column or row_column not in frame.columns:
        raise QueryExecutionError("pivot_table_summary requires a known row/group column.")
    if column_column and column_column not in frame.columns:
        raise QueryExecutionError(f"Unknown pivot column: {column_column!r}")
    if value_column and value_column not in frame.columns:
        raise QueryExecutionError(f"Unknown pivot value column: {value_column!r}")

    mask = _build_mask(frame, filters, filter_mode)
    subset = frame.loc[mask]
    if subset.empty:
        return QueryResult(
            operation="pivot_table_summary",
            row_count=0,
            table=[],
            description="No rows matched the current scope for the pivot table.",
            columns_used=[row_column, column_column, value_column],
        )

    if column_column:
        if metric == "average" and value_column:
            table_df = pd.pivot_table(
                subset,
                index=row_column,
                columns=column_column,
                values=value_column,
                aggfunc=lambda values: round(float(pd.to_numeric(values, errors="coerce").mean()), 4),
                fill_value=0,
                dropna=False,
            )
        elif metric == "sum" and value_column:
            table_df = pd.pivot_table(
                subset,
                index=row_column,
                columns=column_column,
                values=value_column,
                aggfunc=lambda values: round(float(pd.to_numeric(values, errors="coerce").sum()), 4),
                fill_value=0,
                dropna=False,
            )
        else:
            table_df = pd.crosstab(subset[row_column].fillna("(missing)"), subset[column_column].fillna("(missing)"))
        table_df = table_df.reset_index()
    else:
        if metric == "average" and value_column:
            numeric = pd.to_numeric(subset[value_column], errors="coerce")
            table_df = (
                subset.assign(**{value_column: numeric})
                .groupby(row_column, dropna=False)[value_column]
                .mean()
                .round(4)
                .reset_index(name=f"Average {value_column}")
            )
        elif metric == "sum" and value_column:
            numeric = pd.to_numeric(subset[value_column], errors="coerce")
            table_df = (
                subset.assign(**{value_column: numeric})
                .groupby(row_column, dropna=False)[value_column]
                .sum()
                .round(4)
                .reset_index(name=f"Sum {value_column}")
            )
        else:
            table_df = subset.groupby(row_column, dropna=False).size().reset_index(name="Count")

    table_df.columns = [str(column) for column in table_df.columns]
    metric_label = f"{metric} {value_column}".strip() if value_column else "count"
    by_label = f"{row_column} by {column_column}" if column_column else row_column
    caveat = ""
    if value_column:
        caveat = _status_column_caveat(frame, row_column) or _status_column_caveat(frame, column_column)
    return QueryResult(
        operation="pivot_table_summary",
        value=len(table_df.index),
        row_count=int(len(subset)),
        table=table_df.to_dict(orient="records"),
        description=f"Created a pivot-style summary of {metric_label} across {by_label}.{caveat}",
        columns_used=list(dict.fromkeys(
            [c for c in [row_column, column_column, value_column] if c]
            + [f["column"] for f in filters if f.get("column")]
        )),
    )


def _trend_summary(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    filter_mode: str = "all",
) -> QueryResult:
    mask = _build_mask(frame, filters, filter_mode)
    subset = frame.loc[mask]
    if subset.empty:
        return QueryResult(
            operation="trend_summary",
            row_count=0,
            table=[],
            description="No rows matched the current scope for trend analysis.",
            columns_used=[f["column"] for f in filters if f.get("column")],
        )

    numeric_columns = _trend_numeric_columns(subset)
    categorical_columns = _trend_categorical_columns(subset, numeric_columns)
    rows: list[dict[str, Any]] = []

    for category in categorical_columns:
        counts = subset[category].dropna().astype(str).value_counts()
        if len(counts.index) < 2:
            continue
        group_sizes = subset.groupby(category, dropna=False).size()
        largest_group = str(group_sizes.sort_values(ascending=False).index[0])
        rows.append({
            "Trend": "Largest group",
            "Group By": category,
            "Metric": "Count",
            "Highest Group": largest_group,
            "Highest Value": int(group_sizes.max()),
            "Lowest Group": str(group_sizes.sort_values(ascending=True).index[0]),
            "Lowest Value": int(group_sizes.min()),
            "Difference": int(group_sizes.max() - group_sizes.min()),
        })
        for metric in numeric_columns:
            numeric = pd.to_numeric(subset[metric], errors="coerce")
            grouped = subset.assign(**{metric: numeric}).groupby(category, dropna=False)[metric].mean().dropna()
            if len(grouped.index) < 2:
                continue
            highest = grouped.sort_values(ascending=False)
            lowest = grouped.sort_values(ascending=True)
            high_value = float(highest.iloc[0])
            low_value = float(lowest.iloc[0])
            rows.append({
                "Trend": "Metric gap",
                "Group By": category,
                "Metric": f"Average {metric}",
                "Highest Group": str(highest.index[0]),
                "Highest Value": round(high_value, 4),
                "Lowest Group": str(lowest.index[0]),
                "Lowest Value": round(low_value, 4),
                "Difference": round(high_value - low_value, 4),
            })

    rows.sort(key=lambda item: abs(float(item.get("Difference") or 0)), reverse=True)
    rows = rows[:25]
    if rows:
        top = rows[0]
        description = (
            f"Found {len(rows)} notable workbook trend(s). Strongest displayed gap: "
            f"{top['Metric']} by {top['Group By']} ({top['Highest Group']} vs {top['Lowest Group']})."
        )
    else:
        description = "No strong category-level trends were detected from the available columns."
    return QueryResult(
        operation="trend_summary",
        value=len(rows),
        row_count=int(len(subset)),
        table=rows,
        description=description,
        columns_used=list(dict.fromkeys(
            [f["column"] for f in filters if f.get("column")]
            + categorical_columns
            + numeric_columns
        )),
    )


def _trend_numeric_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    preferred_tokens = (
        "gpa", "attendance", "absent", "present", "sat", "psat", "score",
        "rate", "days", "credits", "balance",
    )
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().mean() < 0.75:
            continue
        normalized = str(column).strip().casefold()
        if any(token in normalized for token in preferred_tokens):
            columns.append(column)
    return columns[:8]


def _trend_categorical_columns(frame: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    numeric_set = set(numeric_columns)
    columns: list[str] = []
    for column in frame.columns:
        if column in numeric_set:
            continue
        normalized = str(column).strip().casefold()
        if any(token in normalized for token in ("id", "name", "email", "phone", "note", "address")):
            continue
        unique = int(frame[column].nunique(dropna=True))
        if 2 <= unique <= 30:
            columns.append(column)
    preferred = []
    for target in ("Advisor", "Major", "Discipline", "Year", "Standing", "Location", "Attendance Category"):
        found = _find_column(frame, (target,))
        if found and found in columns and found not in preferred:
            preferred.append(found)
    return list(dict.fromkeys(preferred + columns))[:8]


def _pct(part: int | float, total: int | float) -> float:
    return round((float(part) / float(total)) * 100.0, 1) if total else 0.0


def _advisor_outcome_score(
    *,
    avg_gpa: float | None,
    avg_attendance: float | None,
    low_gpa_pct: float,
    low_attendance_pct: float,
    bad_standing_pct: float,
    support_pct: float,
) -> float:
    score = 50.0
    if avg_gpa is not None:
        score += min(max(float(avg_gpa), 0.0), 4.0) / 4.0 * 25.0
    if avg_attendance is not None:
        attendance_value = float(avg_attendance)
        if attendance_value <= 1.0:
            attendance_value *= 100.0
        score += min(max(attendance_value, 0.0), 100.0) / 100.0 * 25.0
    score -= low_gpa_pct * 0.18
    score -= low_attendance_pct * 0.16
    score -= bad_standing_pct * 0.14
    score -= support_pct * 0.12
    return round(min(max(score, 0.0), 100.0), 2)


def _cohort_summary_description(rows: list[dict[str, Any]], row_count: int) -> str:
    metrics = {str(row.get("Metric")): row.get("Value") for row in rows}
    pieces = [f"Cohort summary for {row_count} matching row(s)."]
    if "Average GPA" in metrics:
        pieces.append(f"Average GPA is {metrics['Average GPA']}.")
    if "Bad Standing" in metrics:
        pieces.append(f"Bad Standing: {metrics['Bad Standing']}.")
    if "Needs Attendance Support" in metrics:
        pieces.append(f"Needs Attendance Support: {metrics['Needs Attendance Support']}.")
    return " ".join(pieces)


def _find_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    expanded = set(names)
    for name in names:
        expanded.update(_column_aliases(name))
    targets = {str(name).strip().lower() for name in expanded}
    for column in frame.columns:
        normalized = str(column).strip().lower()
        if normalized in targets:
            return column
    for column in frame.columns:
        normalized = str(column).strip().lower()
        if any(target and (target in normalized or normalized in target) for target in targets):
            return column
    return None


def _column_aliases(name: str) -> tuple[str, ...]:
    normalized = str(name).strip().lower()
    aliases = {
        "advisor": ("advisor/counselor", "counselor", "school counselor", "advisor counselor", "teacher", "professor"),
        "teacher": ("advisor", "advisor/counselor", "counselor", "professor"),
        "professor": ("advisor", "teacher", "counselor"),
        "major": ("major/program", "program", "department", "discipline"),
        "program": ("major", "major/program", "department", "discipline"),
        "department": ("major", "program", "discipline"),
        "discipline": ("major", "program", "department"),
        "year": ("grade", "grade level", "yr/grade", "class year"),
        "grade": ("year", "grade level", "yr/grade"),
        "standing": ("academic standing", "academic status", "status"),
        "academic standing": ("standing", "academic status"),
        "academic status": ("standing", "academic standing"),
        "gpa": ("current gpa", "cumulative gpa", "grade point average"),
        "attendance rate": ("attendance %", "attendance percent", "attendance percentage"),
        "attendance %": ("attendance rate", "attendance percent", "attendance percentage"),
        "days absent": ("absences", "total absences"),
        "sat math": ("sat-math",),
        "sat ebrw": ("sat-reading/writing", "sat english", "sat reading writing"),
        "psat math": ("psat_m", "psat-math"),
        "psat ebrw": ("psat english score", "psat english", "psat reading writing"),
    }
    return aliases.get(normalized, ())


# Masking ---------------------------------------------------------------------


def _build_mask(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    mode: str = "all",
) -> pd.Series:
    """Combine each filter's mask under ``mode`` ("all" = AND, "any" = OR).

    Per-condition masks are computed in isolation so OR composes correctly —
    accumulating with ``mask |= ...`` against a True seed would always be True.
    """
    masks: list[pd.Series] = []
    for condition in filters:
        masks.append(_condition_mask(frame, condition))
    if not masks:
        return pd.Series(True, index=frame.index)
    if mode == "any":
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
    else:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
    return combined.fillna(False)


def _condition_mask(frame: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    """Compute the boolean mask for a single filter condition."""
    column = condition.get("column")
    operator = condition.get("operator")
    value = condition.get("value")
    series = frame[column]

    if operator == "in":
        return _membership_mask(series, value)
    if operator == "equals":
        return _text_eq(series, value)
    if operator == "not_equals":
        return ~_text_eq(series, value)
    if operator == "greater_than":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric > _scaled_numeric_value(series, value)
    if operator == "greater_or_equal":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric >= _scaled_numeric_value(series, value)
    if operator == "less_than":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric < _scaled_numeric_value(series, value)
    if operator == "less_or_equal":
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric <= _scaled_numeric_value(series, value)
    if operator == "contains":
        return series.astype(str).str.contains(str(value), case=False, na=False)
    if operator == "contains_any":
        terms = value if isinstance(value, list) else [value]
        return series.astype(str).apply(
            lambda item: any(str(t).casefold() in item.casefold() for t in terms)
        )
    if operator == "not_contains":
        return ~series.astype(str).str.contains(str(value), case=False, na=False)
    if operator == "contains_text":
        # Semantically tagged for free-text columns. Same matching logic
        # as 'contains' (case-insensitive substring, NaN-safe) but the
        # downstream privacy layer treats matched rows differently —
        # the note column is hidden by default; only a 'Matched Notes'
        # indicator is shown unless the user confirms reveal.
        return series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
    if operator == "not_contains_text":
        return ~series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
    if operator == "starts_with":
        return series.astype(str).str.casefold().str.startswith(str(value).casefold()) & series.notna()
    if operator == "ends_with":
        return series.astype(str).str.casefold().str.endswith(str(value).casefold()) & series.notna()
    if operator == "between":
        low, high = _between_bounds(value)
        numeric = pd.to_numeric(series, errors="coerce")
        return (numeric >= low) & (numeric <= high)
    if operator == "not_in":
        return ~_membership_mask(series, value)
    if operator == "is_missing" or operator == "is_blank":
        return _blank_mask(series)
    if operator == "is_not_missing" or operator == "is_not_blank":
        return ~_blank_mask(series)
    raise QueryExecutionError(f"Unsupported operator: {operator!r}")


def _scaled_numeric_value(series: pd.Series, value: Any) -> float:
    """Compare rate/percent columns on the scale used by the workbook.

    User phrasing like "below 90%" is parsed as 0.9. Some workbooks store
    rates as fractions (0.85), others as percents (85.0). Use the column's
    observed scale so the same plan works for both.
    """
    numeric_value = float(value)
    column_name = str(getattr(series, "name", "") or "").casefold()
    if not any(token in column_name for token in ("rate", "percent", "percentage", "%")):
        return numeric_value
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return numeric_value
    max_value = float(numeric_series.max())
    if max_value > 1.0 and 0.0 <= numeric_value <= 1.0:
        return numeric_value * 100.0
    if max_value <= 1.0 and 1.0 < numeric_value <= 100.0:
        return numeric_value / 100.0
    return numeric_value


def _between_bounds(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise QueryExecutionError("between requires a [low, high] value.")
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise QueryExecutionError("between bounds must be numeric.") from exc
    return (low, high) if low <= high else (high, low)


def _membership_mask(series: pd.Series, value: Any) -> pd.Series:
    values = value if isinstance(value, list) else [value]
    tokens = {str(v).strip().casefold() for v in values}
    wants_blank = bool(tokens & _BLANK_TOKENS)
    normalized = series.astype(str).str.strip().str.casefold()
    member = normalized.isin(tokens)
    if wants_blank:
        member = member | _blank_mask(series)
    return member


def _text_eq(series: pd.Series, value: Any) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.astype(str).str.casefold() == str(value).casefold()
    return series == value


def _blank_mask(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    return series.isna() | normalized.isin(_BLANK_TOKENS)


# Helpers ---------------------------------------------------------------------


def _default_sheet(sheets: dict[str, pd.DataFrame]) -> str:
    return next(iter(sheets), "")


def _assert_columns_exist(
    frame: pd.DataFrame,
    filters: list[dict[str, Any]],
    group_by: str,
    value_column: str,
) -> None:
    referenced = [f.get("column") for f in filters if f.get("column")]
    if group_by:
        referenced.append(group_by)
    if value_column:
        referenced.append(value_column)
    for column in referenced:
        if column not in frame.columns:
            raise QueryExecutionError(f"Unknown column: {column!r}")
