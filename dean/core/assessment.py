"""PSAT/SAT assessment support inside the uploaded academic workbook.

Assessment data is optional and can appear as inline roster columns or as a
sibling sheet in the same workbook. Matching is by Student ID only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.schema import canonical_for


ASSESSMENT_CANONICALS = {
    "test_type",
    "test_date",
    "psat_math",
    "psat_reading_writing",
    "psat_total",
    "sat_math",
    "sat_ebrw",
    "sat_total",
    "math_score",
    "reading_writing_score",
    "total_score",
    "math_benchmark_met",
    "reading_benchmark_met",
    "benchmark_status",
    "college_readiness",
    "assessment_risk",
}

BENCHMARK_NOT_MET_VALUES = {
    "below benchmark",
    "did not meet",
    "did not meet benchmark",
    "not met",
    "no",
    "false",
    "n",
    "0",
    "below",
    "at risk",
    "risk",
}
BENCHMARK_MET_VALUES = {"met", "meets", "yes", "true", "y", "1", "college ready", "ready"}


@dataclass(frozen=True)
class AssessmentDetection:
    mode: str = "none"  # none | inline | sheet | ambiguous
    inline_columns: list[str] = field(default_factory=list)
    inline_fields: list[str] = field(default_factory=list)
    assessment_sheet: str = ""
    candidate_sheets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_workbook_assessments(
    sheets: dict[str, pd.DataFrame],
    roster_sheet: str,
) -> AssessmentDetection:
    roster = sheets.get(roster_sheet)
    if roster is not None:
        inline = _assessment_columns(roster.columns)
        if inline:
            return AssessmentDetection(
                mode="inline",
                inline_columns=inline,
                inline_fields=list(dict.fromkeys(canonical_for(c) for c in inline if canonical_for(c))),
            )

    candidates: list[tuple[str, int]] = []
    for name, frame in sheets.items():
        if name == roster_sheet or frame is None or frame.empty:
            continue
        cols = set(_assessment_columns(frame.columns))
        score = len(cols)
        normalized_name = str(name).strip().lower()
        if any(token in normalized_name for token in ("assessment", "assessments", "psat", "sat")):
            score += 3
        if score >= 3:
            candidates.append((name, score))

    if not candidates:
        return AssessmentDetection()
    candidates.sort(key=lambda item: item[1], reverse=True)
    top_score = candidates[0][1]
    top = [name for name, score in candidates if score == top_score]
    if len(top) > 1:
        return AssessmentDetection(
            mode="ambiguous",
            candidate_sheets=top,
            warnings=["Multiple possible assessment sheets found. Choose one before matching assessments."],
        )
    return AssessmentDetection(mode="sheet", assessment_sheet=top[0], candidate_sheets=[name for name, _ in candidates])


def canonicalise_assessment_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if df is None or df.empty:
        return pd.DataFrame(), ["Assessment sheet is empty."]

    mapping: dict[str, str] = {}
    for column in df.columns:
        canonical = canonical_for(column)
        target = _standard_name(canonical)
        if target and target not in mapping:
            mapping[target] = column

    if "Student ID" not in mapping:
        warnings.append("Student ID missing; assessment matching is unavailable.")
        return pd.DataFrame(), warnings

    out = pd.DataFrame()
    for standard, source in mapping.items():
        out[standard] = df[source]

    for column in ("Test Type", "Benchmark Status", "College Readiness"):
        if column in out.columns:
            out[column] = out[column].astype(str).str.strip()
    if "Test Date" in out.columns:
        out["Test Date"] = pd.to_datetime(out["Test Date"], errors="coerce")
    for column in ("Math Score", "Reading/Writing Score", "Total Score"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out, warnings


def match_assessments_to_roster(roster_df: pd.DataFrame, assessment_df: pd.DataFrame) -> tuple[int, int, list[str]]:
    roster_id = _find_student_id_column(roster_df.columns)
    if roster_id is None or "Student ID" not in assessment_df.columns:
        return 0, int(len(assessment_df)), []
    roster_ids = set(roster_df[roster_id].dropna().astype(str).str.strip())
    assessment_ids = assessment_df["Student ID"].dropna().astype(str).str.strip()
    matched_mask = assessment_ids.isin(roster_ids)
    unmatched_ids = sorted(set(assessment_ids[~matched_mask].tolist()))
    return int(matched_mask.sum()), int((~matched_mask).sum()), unmatched_ids


def latest_assessment_by_student(assessment_df: pd.DataFrame) -> pd.DataFrame:
    if assessment_df is None or assessment_df.empty or "Student ID" not in assessment_df.columns:
        return pd.DataFrame()
    out = assessment_df.copy()
    if "Test Date" in out.columns:
        out["_sort_date"] = pd.to_datetime(out["Test Date"], errors="coerce")
    else:
        out["_sort_date"] = pd.NaT
    out["_row_order"] = range(len(out))
    out = out.sort_values(["Student ID", "_sort_date", "_row_order"], na_position="first")
    latest = out.groupby("Student ID", as_index=False).tail(1).drop(columns=["_sort_date", "_row_order"])
    return latest.reset_index(drop=True)


def compute_assessment_metrics(assessment_df: pd.DataFrame, *, risk_settings: Any = None) -> pd.DataFrame:
    latest = latest_assessment_by_student(assessment_df)
    if latest.empty:
        return latest

    rows: list[dict[str, Any]] = []
    for _, row in latest.iterrows():
        test_type = str(row.get("Test Type", "") or "").strip().upper()
        math_score = row.get("Math Score")
        reading_score = row.get("Reading/Writing Score")
        total_score = row.get("Total Score")
        math_met = _benchmark_bool(row.get("Math Benchmark Met"))
        reading_met = _benchmark_bool(row.get("Reading Benchmark Met"))
        status_risk, status_reason = _status_risk(row.get("Benchmark Status"))
        threshold_risks = _threshold_risks(test_type, math_score, reading_score, risk_settings)
        risk_reasons: list[str] = []
        if status_risk:
            risk_reasons.append(status_reason or "Assessment below benchmark")
        if math_met is False:
            risk_reasons.append(_score_label(test_type, "Math benchmark not met"))
        if reading_met is False:
            risk_reasons.append(_score_label(test_type, "Reading benchmark not met"))
        risk_reasons.extend(threshold_risks)
        risk = bool(risk_reasons)

        item: dict[str, Any] = {
            "Student ID": row.get("Student ID"),
            "Latest Test Type": row.get("Test Type", ""),
            "Latest Test Date": row.get("Test Date"),
            "Latest Math Score": math_score,
            "Latest Reading/Writing Score": reading_score,
            "Latest Total Score": total_score,
            "Benchmark Status": row.get("Benchmark Status", ""),
            "Math Benchmark Met": math_met if math_met is not None else row.get("Math Benchmark Met", ""),
            "Reading Benchmark Met": reading_met if reading_met is not None else row.get("Reading Benchmark Met", ""),
            "College Readiness": row.get("College Readiness", ""),
            "Assessment Risk": risk,
            "Assessment Reason": "; ".join(dict.fromkeys(risk_reasons)),
        }
        if test_type == "SAT":
            item.update({"SAT Math": math_score, "SAT EBRW": reading_score, "SAT Total": total_score})
        if test_type == "PSAT":
            item.update({"PSAT Math": math_score, "PSAT Reading/Writing": reading_score, "PSAT Total": total_score})
        rows.append(item)
    return pd.DataFrame(rows)


def attach_assessment_metrics(roster_df: pd.DataFrame, assessment_metrics: pd.DataFrame) -> pd.DataFrame:
    if roster_df is None or roster_df.empty or assessment_metrics is None or assessment_metrics.empty:
        return roster_df
    roster_id = _find_student_id_column(roster_df.columns)
    if roster_id is None or "Student ID" not in assessment_metrics.columns:
        return roster_df
    return roster_df.merge(
        assessment_metrics,
        how="left",
        left_on=roster_id,
        right_on="Student ID",
        suffixes=("", "_assessment"),
    ).drop(columns=["Student ID_assessment"], errors="ignore")


def assessment_available_columns(columns) -> list[str]:
    return _assessment_columns(columns)


def _assessment_columns(columns) -> list[str]:
    out = []
    for column in columns:
        canonical = canonical_for(column)
        if canonical in ASSESSMENT_CANONICALS:
            out.append(column)
    return out


def _standard_name(canonical: str | None) -> str | None:
    return {
        "student_id": "Student ID",
        "test_type": "Test Type",
        "test_date": "Test Date",
        "psat_math": "Math Score",
        "sat_math": "Math Score",
        "math_score": "Math Score",
        "psat_reading_writing": "Reading/Writing Score",
        "sat_ebrw": "Reading/Writing Score",
        "reading_writing_score": "Reading/Writing Score",
        "psat_total": "Total Score",
        "sat_total": "Total Score",
        "total_score": "Total Score",
        "math_benchmark_met": "Math Benchmark Met",
        "reading_benchmark_met": "Reading Benchmark Met",
        "benchmark_status": "Benchmark Status",
        "college_readiness": "College Readiness",
        "assessment_risk": "Assessment Risk",
    }.get(canonical or "")


def _benchmark_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in BENCHMARK_MET_VALUES:
        return True
    if normalized in BENCHMARK_NOT_MET_VALUES:
        return False
    return None


def _status_risk(value: Any) -> tuple[bool, str]:
    if value is None or pd.isna(value):
        return False, ""
    normalized = str(value).strip().lower()
    if normalized in BENCHMARK_NOT_MET_VALUES or "below" in normalized or "not meet" in normalized:
        return True, f"Benchmark Status = {value}"
    return False, ""


def _threshold_risks(test_type: str, math_score: Any, reading_score: Any, settings: Any) -> list[str]:
    if settings is None:
        return []
    out: list[str] = []
    if test_type == "SAT":
        out.extend(_score_threshold_reason(math_score, getattr(settings, "sat_math_benchmark_threshold", None), "SAT Math below configured benchmark"))
        out.extend(_score_threshold_reason(reading_score, getattr(settings, "sat_ebrw_benchmark_threshold", None), "SAT EBRW below configured benchmark"))
    elif test_type == "PSAT":
        out.extend(_score_threshold_reason(math_score, getattr(settings, "psat_math_benchmark_threshold", None), "PSAT Math below configured benchmark"))
        out.extend(_score_threshold_reason(reading_score, getattr(settings, "psat_reading_writing_benchmark_threshold", None), "PSAT Reading/Writing below configured benchmark"))
    return out


def _score_threshold_reason(score: Any, threshold: Any, label: str) -> list[str]:
    if threshold in (None, "") or pd.isna(score):
        return []
    try:
        return [label] if float(score) < float(threshold) else []
    except (TypeError, ValueError):
        return []


def _score_label(test_type: str, label: str) -> str:
    prefix = test_type if test_type in {"SAT", "PSAT"} else "Assessment"
    return f"{prefix} {label}"


def _find_student_id_column(columns) -> str | None:
    for column in columns:
        if canonical_for(column) == "student_id":
            return column
    return None
