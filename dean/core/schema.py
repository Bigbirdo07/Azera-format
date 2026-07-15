"""Schema layer: canonical column mapping, type inference, and debug state.

This sits on top of the loaded (cleaned) workbook. It does not modify the
original file. Canonical names are exposed for understanding/display/debug; the
real (cleaned) column names remain the working identifiers everywhere else.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.privacy import detect_sensitive_columns, get_default_visible_columns
from nlp.synonym_mapper import normalize_text


_BLANK_TOKENS = {"", "nan", "none", "null", "n/a", "na"}

# canonical -> normalized variant phrases. Ordered so specific names win.
CANONICAL_FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("cumulative_gpa", ("cumulative gpa", "cum gpa", "overall gpa")),
    ("term_gpa", ("term gpa", "semester gpa")),
    ("gpa", ("gpa", "g p a", "grade point average", "grade average")),
    ("student_id", ("student id", "student number", "banner id", "id number", "id")),
    ("first_name", ("first name", "given name", "fname")),
    ("last_name", ("last name", "surname", "family name", "lname")),
    ("full_name", ("full name", "student name", "name")),
    ("email", ("email", "e mail", "email address")),
    ("phone", ("phone", "phone number", "mobile", "cell", "telephone")),
    ("parent_guardian_contact", ("parent guardian contact", "parent/guardian contact",
                                 "guardian contact", "parent contact",
                                 "guardian phone", "parent phone",
                                 "guardian email", "parent email")),
    ("department", ("department", "dept", "division", "subject", "academic department", "teaching department")),
    ("major", ("major", "academic program", "program of study", "concentration", "academic concentration")),
    ("program", ("program", "concentration track")),
    ("concentration", ("concentration",)),
    # Teacher / professor / instructor are the same concept on a school roster —
    # alias them all to the canonical 'teacher'.
    # canonical_for() reads column names: "Professor Name" should map to
    # `teacher`. Keep the descriptive variants here. They are NOT in the chat
    # synonyms.json because there they would cause fuzzy false-positives
    # (e.g. "teacher name" landing on a literal Name column).
    ("teacher", ("teacher", "professor", "instructor", "faculty", "faculty member",
                 "teacher name", "professor name", "instructor name")),
    ("academic_watch", ("academic watch", "watch list", "watchlist", "flagged",
                        "intervention needed", "on watch")),
    ("follow_up_needed", ("follow up needed", "follow up", "advisor follow up",
                          "follow-up needed", "needs follow up", "needs follow-up")),
    ("course", ("course", "class", "section", "course name", "class name")),
    ("class_level", ("class level", "student level", "level")),
    ("year", ("year", "class year", "class", "academic year")),
    ("credits_completed", ("credits completed", "credits earned", "completed credits", "earned credits", "credits")),
    ("credits_attempted", ("credits attempted", "attempted credits")),
    ("advisor", ("advisor", "adviser", "faculty advisor", "counselor", "advisor name")),
    ("academic_status", ("academic status", "academic standing", "standing", "status")),
    ("graduation_status", ("graduation status", "grad status", "graduated")),
    ("enrollment_status", ("enrollment status", "registration status", "enrolled", "registered")),
    ("financial_aid_status", ("financial aid status", "financial aid", "fafsa", "aid status")),
    ("conduct_status", ("conduct status", "conduct", "disciplinary status", "discipline",
                        "discipline information", "discipline record", "discipline incidents",
                        "behavior incidents", "office referrals")),
    ("notes", ("notes", "note", "comments", "remarks")),
    ("date_of_birth", ("date of birth", "dob", "birth date", "birthdate")),
    ("school", ("school", "college")),
    # Skyward-export fields (Class Roster / Student Information reports).
    ("grad_year", ("grad year", "graduation year", "grad yr", "expected graduation")),
    ("entry_date", ("entry date", "enrollment date", "date enrolled", "start date")),
    ("withdrawal_date", ("withdrawal date", "exit date", "date withdrawn", "dropped date")),
    ("home_address", ("home address",)),
    ("mailing_address", ("mailing address",)),
    ("emergency_contact", ("emergency contact", "emergency contact name",
                          "emergency contact phone", "emergency contacts")),
    # Attendance-derived (read-only) metrics. Variants here are exhaustive so
    # a school that exports "Attendance %", "Total Absences", or "Tardies"
    # gets recognised without the user having to rename anything.
    ("attendance_rate", ("attendance rate", "attendance pct", "attendance percent",
                         "attendance percentage", "attendance %", "present rate",
                         "attendance")),
    ("days_present", ("days present", "present days")),
    ("days_absent", ("days absent", "absent days", "days missed",
                     "absences", "total absences", "absence count",
                     "number of absences")),
    ("days_tardy", ("days tardy", "tardies", "tardy days", "tardy count",
                    "number of tardies")),
    ("unexcused_absences", ("unexcused absences", "unexcused absence",
                            "unexcused days", "unexcused")),
    ("excused_absences", ("excused absences", "excused absence",
                          "excused days", "excused")),
    ("recent_absences", ("recent absences", "absences last 14 days",
                         "absences this month")),
    ("attendance_status", ("attendance status",)),
    ("attendance_risk", ("attendance risk", "chronic absence",
                         "chronic absenteeism", "chronically absent")),
    ("severe_attendance_risk", ("severe attendance risk",)),
    ("attendance_watch", ("attendance watch", "attendance flag",
                          "attendance intervention")),
    # Combined risk (read-only derived) — write protection only.
    ("risk_signals", ("risk signals", "risk count", "risk score")),
    ("risk_level", ("risk level", "risk band", "overall risk")),
    ("gpa_risk", ("gpa risk",)),
    ("standing_risk", ("standing risk", "academic standing risk")),
    ("assessment_risk", ("assessment risk", "psat risk", "sat risk")),
    # Assessment (PSAT/SAT) — protected as scores are uploaded, not edited.
    ("test_type", ("test type", "assessment", "exam", "psat sat", "assessment type")),
    ("test_date", ("test date", "assessment date", "exam date")),
    ("psat_math", ("psat math", "psat math score")),
    ("psat_reading_writing", ("psat reading", "psat reading writing",
                              "psat reading/writing",
                              "psat evidence based reading and writing",
                              "psat ebrw", "psat verbal")),
    ("psat_total", ("psat total", "psat total score", "total psat")),
    ("sat_math", ("sat math", "sat math score")),
    ("sat_ebrw", ("sat ebrw", "sat reading", "sat reading writing",
                  "sat reading/writing",
                  "sat evidence based reading and writing", "sat verbal")),
    ("sat_total", ("sat total", "sat total score", "total sat", "sat score", "sat scores")),
    ("math_benchmark_met", ("math benchmark", "math benchmark met",
                            "sat math benchmark", "psat math benchmark",
                            "math ready")),
    ("reading_benchmark_met", ("reading benchmark", "reading benchmark met",
                               "ebrw benchmark", "sat reading benchmark",
                               "psat reading benchmark", "reading ready")),
    ("college_readiness", ("college readiness", "college ready", "readiness level")),
    ("math_score", ("math score",)),
    ("reading_writing_score", ("reading writing score", "reading/writing score",
                               "evidence based reading writing", "ebrw",
                               "reading score")),
    ("total_score", ("total score", "assessment score", "exam score")),
    ("benchmark_status", ("benchmark status", "psat benchmark",
                          "sat benchmark", "assessment benchmark",
                          "benchmark met", "college readiness benchmark",
                          "readiness status")),
]

_DATE_NAME_HINTS = ("date", "dob", "birth", "graduat", "enroll", "admit", "start")


def canonical_for(column_name: str) -> str | None:
    """Map a (possibly messy) column name to a canonical field, or None."""
    normalized = normalize_text(column_name)
    if not normalized:
        return None
    for canonical, variants in CANONICAL_FIELDS:
        if normalized in variants:
            return canonical
    for canonical, variants in CANONICAL_FIELDS:
        for variant in variants:
            if variant and _contains_phrase(normalized, variant):
                return canonical
    return None


def canonical_map(columns: list[str]) -> dict[str, str]:
    """Map canonical field -> actual column name (first match wins)."""
    mapping: dict[str, str] = {}
    for column in columns:
        canonical = canonical_for(column)
        if canonical and canonical not in mapping:
            mapping[canonical] = column
    return mapping


def infer_column_types(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-column analysis type + coercion confidence (non-destructive)."""
    info: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        series = frame[column]
        original_dtype = str(series.dtype)
        non_blank = series[~_blank_mask(series)]
        count = len(non_blank)
        analysis_dtype = "text"
        success_rate = 1.0
        warning: str | None = None

        if count == 0:
            analysis_dtype = "empty"
        else:
            numeric_rate = pd.to_numeric(non_blank, errors="coerce").notna().mean()
            if numeric_rate >= 0.9:
                analysis_dtype = "numeric"
                success_rate = round(float(numeric_rate), 3)
                if not pd.api.types.is_numeric_dtype(series):
                    warning = f"'{column}' is stored as text but was interpreted numerically ({success_rate:.0%})."
            elif _looks_dateish(column, non_blank):
                date_rate = pd.to_datetime(non_blank, errors="coerce", format="mixed").notna().mean()
                if date_rate >= 0.9:
                    analysis_dtype = "date"
                    success_rate = round(float(date_rate), 3)
                else:
                    analysis_dtype = _category_or_text(non_blank)
            else:
                analysis_dtype = _category_or_text(non_blank)
            # Partial numeric coercion is worth flagging even if not chosen.
            if analysis_dtype == "numeric" and 0.9 <= numeric_rate < 1.0 and warning is None:
                warning = f"Some '{column}' values are not numeric ({success_rate:.0%} parsed)."

        semantic_role = _detect_semantic_role(column, non_blank, analysis_dtype)

        info[column] = {
            "original_dtype": original_dtype,
            "analysis_dtype": analysis_dtype,
            "coercion_success_rate": success_rate,
            "coercion_warning": warning,
            "semantic_role": semantic_role,
        }
    return info


# Column-name patterns that strongly indicate a free-text narrative field.
_FREE_TEXT_NAME_PATTERNS = (
    "notes", "note", "comment", "comments", "remark", "remarks", "memo",
    "follow up", "followup", "follow-up", "advisor notes", "advising notes",
    "counselor notes", "case notes", "internal notes", "description",
    "reason", "explanation", "narrative", "summary", "feedback",
)

# Below this average length, even a 'notes'-named column is more likely a tag
# than a narrative. Above it, even a non-notes column reads like prose.
_FREE_TEXT_MIN_AVG_LEN = 25


def _detect_semantic_role(column: str, non_blank: pd.Series, analysis_dtype: str) -> str | None:
    """Return a semantic role tag for the column, or None.

    Currently only emits 'free_text' for narrative note/comment columns.
    A column qualifies if its name matches a notes-like pattern OR it's a
    text-typed column whose average non-blank value is long enough to look
    like prose (≥25 chars) and has high uniqueness (≥70% of rows distinct).
    """
    if analysis_dtype not in {"text", "category"}:
        return None
    name_norm = normalize_text(column)
    name_match = any(pattern in name_norm for pattern in _FREE_TEXT_NAME_PATTERNS)

    avg_len = 0.0
    unique_rate = 0.0
    if len(non_blank) > 0:
        as_str = non_blank.astype(str)
        avg_len = float(as_str.str.len().mean())
        unique_rate = float(as_str.nunique() / len(non_blank))

    if name_match:
        return "free_text"
    if analysis_dtype == "text" and avg_len >= _FREE_TEXT_MIN_AVG_LEN and unique_rate >= 0.7:
        return "free_text"
    return None


def build_workbook_schema(sheets: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for sheet_name, frame in sheets.items():
        columns = [str(c) for c in frame.columns]
        column_types = infer_column_types(frame)
        free_text_columns = [c for c, meta in column_types.items()
                             if meta.get("semantic_role") == "free_text"]
        schema[sheet_name] = {
            "row_count": int(len(frame.index)),
            "columns": columns,
            "canonical_map": canonical_map(columns),
            "column_types": column_types,
            "sensitive": detect_sensitive_columns(columns),
            "default_visible": get_default_visible_columns(columns),
            "free_text_columns": free_text_columns,
        }
    return schema


def schema_warnings(sheets: dict[str, pd.DataFrame]) -> list[str]:
    """Human-facing warnings derived from analysis (numeric-as-text, empty)."""
    warnings: list[str] = []
    for sheet_name, frame in sheets.items():
        if frame.empty or not list(frame.columns):
            warnings.append(f"Sheet '{sheet_name}' is empty and was ignored.")
            continue
        for column, meta in infer_column_types(frame).items():
            if meta["coercion_warning"]:
                warnings.append(meta["coercion_warning"])
    return warnings


def build_debug_state(
    memory: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    active_sheet: str,
    routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Privacy-safe developer view of current assistant state. Never includes
    raw sensitive rows — only metadata, plans, and summaries."""
    pending = dict(memory.get("pending_action") or {})
    pending.pop("query", None)  # may hold filter values; keep type/reason only
    return {
        "sheets": list(schema.keys()),
        "active_sheet": active_sheet,
        "normalized_schema": {
            "columns": schema.get(active_sheet, {}).get("columns", []),
            "canonical_map": schema.get(active_sheet, {}).get("canonical_map", {}),
            "column_types": schema.get(active_sheet, {}).get("column_types", {}),
        },
        "sensitive_columns": schema.get(active_sheet, {}).get("sensitive", {}),
        "active_filters": memory.get("active_filters", []),
        "active_sort": memory.get("active_sort", {}),
        "active_group_by": memory.get("active_group_by", ""),
        "active_limit": memory.get("active_limit"),
        "pending_action": pending,
        "last_plan": memory.get("last_query_plan", {}),
        "last_result_summary": memory.get("last_result_description", ""),
        "last_operation": memory.get("last_operation", ""),
        "routing": {
            "plan_source": (routing or {}).get("plan_source", "rules"),
            "llm_used": (routing or {}).get("llm_used", False),
            "validation_status": (routing or {}).get("validation_status", ""),
            "fallback_reason": (routing or {}).get("fallback_reason"),
        },
    }


# Helpers ---------------------------------------------------------------------


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _blank_mask(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    return series.isna() | normalized.isin(_BLANK_TOKENS)


def _looks_dateish(column: str, values: pd.Series) -> bool:
    if any(hint in normalize_text(column) for hint in _DATE_NAME_HINTS):
        return True
    return False


def _category_or_text(values: pd.Series) -> str:
    unique = values.astype(str).nunique()
    return "category" if unique <= max(20, int(0.5 * len(values))) else "text"
