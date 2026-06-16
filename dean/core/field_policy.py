"""Which workbook fields the assistant may edit.

Safe fields are operational flags staff add (notes, follow-up). Protected fields
are the academic/identity/contact record itself, which this assistant never
edits — they require a manual, higher-privilege process that is not implemented.
"""

from __future__ import annotations

from core.privacy import classify_sensitivity
from core.schema import canonical_for
from nlp.synonym_mapper import normalize_text


SAFE_EDITABLE_FIELDS = (
    "notes",
    "note",
    "follow up needed",
    "followup needed",
    "follow-up needed",
    "advisor follow up",
    "advisor followup",
    "academic watch",
    "attendance watch",
    "watch list",
    "watchlist",
    "intervention needed",
    "internal flag",
    "outreach status",
    "review status",
)

# Canonical concepts that must never be edited by the assistant.
# Attendance Rate, Days Absent, and SAT/PSAT scores are *derived* from
# uploaded data sources — editing them would silently desynchronise the
# computed metrics from their source files.
_PROTECTED_CANONICALS = {
    "student_id",
    "first_name",
    "last_name",
    "full_name",
    "email",
    "phone",
    "date_of_birth",
    "gpa",
    "cumulative_gpa",
    "term_gpa",
    "academic_status",
    "conduct_status",
    "financial_aid_status",
    # Computed metrics (from core.attendance / core.combined_risk).
    "attendance_rate",
    "days_present",
    "days_absent",
    "days_tardy",
    "unexcused_absences",
    "recent_absences",
    "attendance_risk",
    "severe_attendance_risk",
    "risk_signals",
    "risk_level",
    "gpa_risk",
    "standing_risk",
    "assessment_risk",
    # PSAT/SAT score fields (deferred ingestion; reserved as protected).
    "test_type",
    "test_date",
    "psat_math",
    "psat_reading_writing",
    "psat_total",
    "sat_math",
    "sat_ebrw",
    "sat_total",
    "math_benchmark_met",
    "reading_benchmark_met",
    "college_readiness",
    "math_score",
    "reading_writing_score",
    "total_score",
    "benchmark_status",
}


def field_status(column_name: str) -> str:
    """Return 'safe', 'protected', or 'unknown' for an edit target."""
    normalized = normalize_text(column_name)
    if normalized in {normalize_text(f) for f in SAFE_EDITABLE_FIELDS}:
        return "safe"
    canonical = canonical_for(column_name)
    if canonical in _PROTECTED_CANONICALS:
        return "protected"
    sensitive, _ = classify_sensitivity(column_name)
    if sensitive:
        return "protected"
    return "unknown"


def is_safe_editable(column_name: str) -> bool:
    return field_status(column_name) == "safe"


def is_protected(column_name: str) -> bool:
    return field_status(column_name) == "protected"
