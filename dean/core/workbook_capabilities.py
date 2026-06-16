"""School-office-language view of an academic workbook.

Turns raw detected canonical fields into the four things a dean / counsellor
actually wants to see after upload:

  1. ``group_detected_fields(columns)``  → fields bucketed into
     Roster / Performance / Attendance / Actions / Export with clean labels.
  2. ``detect_capabilities(columns, ...)`` → checklist of workflows the
     workbook supports ("GPA performance review", "Attendance-risk review").
  3. ``missing_field_messages(columns, ...)`` → friendly notes for the gaps
     ("Attendance not detected. You can still review GPA, standing, …").
  4. ``upload_assistant_message(...)`` → one-shot greeting the chat panel
     appends exactly once per workbook upload.

This module never reads row data and never touches the LLM — it operates on
column names + canonical resolution from ``core.schema.canonical_for``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from core.institution_context import InstitutionMode, capability_title, label_for
from core.schema import canonical_for


# ---- display labels --------------------------------------------------------


# Canonical → label displayed in the UI. Keys are the canonical names that
# core.schema.canonical_for() returns; values are the labels a counsellor
# would expect to see ("Academic Standing" beats "academic_status").
_FIELD_LABELS: dict[str, str] = {
    # Roster.
    "teacher": "Teacher",
    "department": "Department",
    "discipline": "Department",
    "student_id": "Student ID",
    "full_name": "Student",
    "first_name": "Student First Name",
    "last_name": "Student Last Name",
    "parent_guardian_contact": "Parent/guardian contact",
    "major": "Major",
    "program": "Program",
    "concentration": "Concentration",
    "advisor": "Advisor",
    "year": "Year",
    "class_level": "Class Level",
    "test_type": "Test Type",
    "test_date": "Test Date",
    "psat_math": "PSAT Math",
    "psat_reading_writing": "PSAT Reading/Writing",
    "psat_total": "PSAT Total",
    "sat_math": "SAT Math",
    "sat_ebrw": "SAT EBRW",
    "sat_total": "SAT Total",
    "math_benchmark_met": "Math Benchmark Met",
    "reading_benchmark_met": "Reading Benchmark Met",
    "benchmark_status": "Benchmark Status",
    "college_readiness": "College Readiness",
    "assessment_risk": "Assessment Risk",
    # Performance.
    "gpa": "GPA",
    "cumulative_gpa": "Cumulative GPA",
    "term_gpa": "Term GPA",
    "academic_status": "Academic Standing",
    "credits_completed": "Credits Completed",
    # Attendance.
    "attendance_rate": "Attendance Rate",
    "days_present": "Days Present",
    "days_absent": "Days Absent",
    "days_tardy": "Days Tardy",
    "unexcused_absences": "Unexcused Absences",
    "excused_absences": "Excused Absences",
    "attendance_status": "Attendance Status",
    "attendance_risk": "Attendance Risk",
    "severe_attendance_risk": "Severe Attendance Risk",
    "recent_absences": "Recent Absences",
    # Actions.
    "academic_watch": "Academic Watch",
    "attendance_watch": "Attendance Watch",
    "follow_up_needed": "Follow Up Needed",
    "notes": "Notes",
}

# Which bucket each canonical belongs to. Categories follow the spec's
# Roster / Performance / Attendance / Actions / Export framing.
_FIELD_CATEGORIES: dict[str, str] = {
    # Roster.
    "teacher": "Roster",
    "department": "Roster",
    "discipline": "Roster",
    "student_id": "Roster",
    "full_name": "Roster",
    "first_name": "Roster",
    "last_name": "Roster",
    "parent_guardian_contact": "Roster",
    "major": "Roster",
    "program": "Roster",
    "concentration": "Roster",
    "advisor": "Roster",
    "year": "Roster",
    "class_level": "Roster",
    "test_type": "Assessments",
    "test_date": "Assessments",
    "psat_math": "Assessments",
    "psat_reading_writing": "Assessments",
    "psat_total": "Assessments",
    "sat_math": "Assessments",
    "sat_ebrw": "Assessments",
    "sat_total": "Assessments",
    "math_benchmark_met": "Assessments",
    "reading_benchmark_met": "Assessments",
    "benchmark_status": "Assessments",
    "college_readiness": "Assessments",
    "assessment_risk": "Assessments",
    # Performance.
    "gpa": "Performance",
    "cumulative_gpa": "Performance",
    "term_gpa": "Performance",
    "academic_status": "Performance",
    "credits_completed": "Performance",
    # Attendance.
    "attendance_rate": "Attendance",
    "days_present": "Attendance",
    "days_absent": "Attendance",
    "days_tardy": "Attendance",
    "unexcused_absences": "Attendance",
    "excused_absences": "Attendance",
    "attendance_status": "Attendance",
    "attendance_risk": "Attendance",
    "severe_attendance_risk": "Attendance",
    "recent_absences": "Attendance",
    # Actions.
    "academic_watch": "Actions",
    "attendance_watch": "Actions",
    "follow_up_needed": "Actions",
    "notes": "Actions",
}

CATEGORY_ORDER = ("Roster", "Performance", "Attendance", "Assessments", "Actions", "Export")


# ---- field grouping --------------------------------------------------------


def group_detected_fields(columns: Iterable[str]) -> dict[str, list[str]]:
    """Bucket detected columns into school-office categories.

    Returns a dict keyed by ``CATEGORY_ORDER`` plus ``"Missing Important
    Fields"``. Each value is the de-duplicated list of clean labels found
    in that bucket. ``"Export"`` is always present (the app can always
    export) so the panel can render it as an available capability.
    """
    out: dict[str, list[str]] = {category: [] for category in CATEGORY_ORDER}
    out["Missing Important Fields"] = []
    seen_per_category: dict[str, set[str]] = {c: set() for c in out}

    for column in columns:
        canonical = canonical_for(column)
        if canonical is None:
            continue
        category = _FIELD_CATEGORIES.get(canonical)
        label = _FIELD_LABELS.get(canonical)
        if category is None or label is None:
            continue
        if label in seen_per_category[category]:
            continue
        seen_per_category[category].add(label)
        out[category].append(label)

    # "Export updated workbook" is always available regardless of detected
    # fields — the action layer can write a new workbook from any roster.
    out["Export"] = ["Export updated workbook", "Export filtered list"]

    # Helpful, non-blocking note about important academic fields the
    # workbook *didn't* include.
    out["Missing Important Fields"] = _missing_important_field_labels(columns)
    return out


def readiness_checks(columns: Iterable[str]) -> list[tuple[str, str]]:
    canonicals = {canonical_for(col) for col in columns}
    canonicals.discard(None)
    checks = [
        ("Student ID", "found" if "student_id" in canonicals else "issue found"),
        ("Student Name", "found" if any(c in canonicals for c in ("full_name", "first_name", "last_name")) else "missing but optional"),
        ("Teacher / Professor", "found" if any(c in canonicals for c in ("teacher", "advisor")) else "issue found"),
        ("Department", "found" if any(c in canonicals for c in ("department", "discipline")) else "missing but optional"),
        ("GPA", "found" if any(c in canonicals for c in ("gpa", "cumulative_gpa", "term_gpa")) else "issue found"),
        ("Academic Standing", "found" if "academic_status" in canonicals else "missing but optional"),
    ]
    has_attendance = _has_any_attendance(canonicals)
    has_watch = any(c in canonicals for c in ("academic_watch", "attendance_watch"))
    checks.append(("Attendance fields", "found" if has_attendance else "missing but optional"))
    checks.append(("Watch fields", "found" if has_watch else "missing but optional"))
    checks.append(("Attendance Watch", "found" if "attendance_watch" in canonicals else "missing but optional"))
    checks.append(("Academic Watch", "found" if "academic_watch" in canonicals else "missing but optional"))
    return checks


def readiness_issues(frame) -> list[str]:
    issues: list[str] = []
    if isinstance(frame, (list, tuple)):
        columns = list(frame)
        values = None
    elif frame is None:
        columns = []
        values = None
    elif getattr(frame, "empty", True):
        columns = list(getattr(frame, "columns", []))
        values = None
    else:
        columns = list(getattr(frame, "columns", frame))
        values = frame
    student_id_col = None
    norm_map = {str(c).strip().lower(): c for c in columns}
    for key in ("student id", "id"):
        if key in norm_map:
            student_id_col = norm_map[key]
            break
    if not student_id_col:
        issues.append("missing Student ID column")
        return issues
    if values is None:
        return issues
    ids = values[student_id_col].dropna().astype(str).str.strip()
    ids = ids[ids != ""]
    if ids.empty:
        issues.append("blank Student IDs")
    if ids.duplicated().any():
        issues.append("duplicate Student IDs")
    return issues


def _missing_important_field_labels(columns: Iterable[str]) -> list[str]:
    """Return a sorted list of "headline" fields that the workbook does NOT
    contain (display-only — never blocks workflow)."""
    canonicals_present = {canonical_for(col) for col in columns}
    canonicals_present.discard(None)
    important = (
        ("teacher", "Teacher"),
        ("department", "Department"),
        ("gpa", "GPA"),
        ("academic_status", "Academic Standing"),
        ("attendance_rate", "Attendance"),
        ("academic_watch", "Academic Watch"),
        ("attendance_watch", "Attendance Watch"),
    )
    missing: list[str] = []
    for canonical, label in important:
        if canonical in canonicals_present:
            continue
        # Attendance is covered if ANY attendance column is present.
        if canonical == "attendance_rate" and _has_any_attendance(canonicals_present):
            continue
        # Teacher is covered if an advisor is present (same workflow role).
        if canonical == "teacher" and "advisor" in canonicals_present:
            continue
        # Academic Watch + Attendance Watch are "creatable" — they belong
        # in the missing list so the UI can render the helpful "I'll create
        # the column on export" message.
        missing.append(label)
    return missing


def _has_any_attendance(canonicals: set[str]) -> bool:
    return any(c in canonicals for c in (
        "attendance_rate", "days_absent", "days_tardy",
        "unexcused_absences", "excused_absences",
        "attendance_status", "attendance_risk",
    ))


# ---- capabilities ----------------------------------------------------------


@dataclass(frozen=True)
class Capability:
    """One workflow the assistant can perform on the detected workbook."""
    key: str           # stable identifier for tests / UI
    title: str         # checkmark label (e.g. "GPA performance review")
    available: bool    # rendered with ✓ when True
    note: str = ""     # short hint for unavailable / creatable capabilities


def detect_capabilities(
    columns: Iterable[str],
    *,
    attendance_available: bool = False,
    mode: InstitutionMode = InstitutionMode.GENERIC,
) -> list[Capability]:
    """Return the workflow checklist the user sees as 'this workbook supports …'.

    ``attendance_available`` lets the caller indicate that attendance was
    auto-detected outside the column list (e.g. from a sibling sheet
    inside the same workbook). When True, the attendance-risk review
    capability is unlocked even if the roster sheet itself doesn't carry
    the canonical attendance columns yet.
    """
    canonicals = {canonical_for(col) for col in columns}
    canonicals.discard(None)

    has_teacher = "teacher" in canonicals or "advisor" in canonicals
    has_department = "department" in canonicals or "discipline" in canonicals
    has_gpa = any(c in canonicals for c in ("gpa", "cumulative_gpa", "term_gpa"))
    has_major = "major" in canonicals or "program" in canonicals
    has_standing = "academic_status" in canonicals
    has_attendance = attendance_available or _has_any_attendance(canonicals)
    has_assessment = _has_any_assessment(canonicals)
    has_benchmark = _has_assessment_benchmark(canonicals)
    has_academic_watch_column = "academic_watch" in canonicals
    has_attendance_watch_column = "attendance_watch" in canonicals

    caps: list[Capability] = []

    caps.append(Capability(
        key="teacher_department",
        title=capability_title("teacher_department", mode),
        available=has_teacher and has_department,
        note=("" if has_teacher and has_department
              else "Teacher or department field not detected."),
    ))
    caps.append(Capability(
        key="gpa_performance",
        title=capability_title("gpa_performance", mode),
        available=has_gpa,
        note=("" if has_gpa else "GPA field not detected."),
    ))
    caps.append(Capability(
        key="major_grouping",
        title=capability_title("major_grouping", mode),
        available=has_major,
        note=("" if has_major else "Major field not detected."),
    ))
    caps.append(Capability(
        key="academic_standing",
        title=capability_title("academic_standing", mode),
        available=has_standing,
        note=("" if has_standing else "Academic Standing field not detected."),
    ))
    caps.append(Capability(
        key="attendance_risk",
        title=capability_title("attendance_risk", mode),
        available=has_attendance,
        note=("" if has_attendance else "Attendance fields not detected."),
    ))
    caps.append(Capability(
        key="assessment_review",
        title=capability_title("assessment_review", mode),
        available=has_assessment,
        note=("" if has_assessment else "Assessment scores not detected."),
    ))
    caps.append(Capability(
        key="benchmark_risk",
        title=capability_title("benchmark_risk", mode),
        available=has_benchmark or "assessment_risk" in canonicals,
        note=("" if has_benchmark or "assessment_risk" in canonicals
              else "Benchmark-risk questions require benchmark fields or configured thresholds."),
    ))
    caps.append(Capability(
        key="combined_academic_risk",
        title=capability_title("combined_academic_risk", mode),
        available=has_gpa or has_attendance or has_standing or has_assessment,
    ))
    # Watch columns are CREATABLE — the action writes them into a new
    # workbook on export — so the capability is always available. The note
    # explains the "we'll create the column on export" behaviour when the
    # source workbook didn't include it.
    caps.append(Capability(
        key="academic_watch_updates",
        title=capability_title("academic_watch_updates", mode),
        available=True,
        note=("" if has_academic_watch_column
              else "Academic Watch column will be created in the exported workbook."),
    ))
    caps.append(Capability(
        key="attendance_watch_updates",
        title=capability_title("attendance_watch_updates", mode),
        available=True,
        note=("" if has_attendance_watch_column
              else "Attendance Watch column will be created in the exported workbook."),
    ))
    caps.append(Capability(
        key="export",
        title=capability_title("export", mode),
        available=True,
    ))
    return caps


# ---- missing-field messages -----------------------------------------------


def missing_field_messages(
    columns: Iterable[str],
    *,
    attendance_available: bool = False,
    mode: InstitutionMode = InstitutionMode.GENERIC,
) -> list[str]:
    """Friendly notes for fields that *aren't* present — never errors.

    Each message tells the user what's still possible, so a roster-only
    workbook doesn't feel broken when attendance is absent.
    """
    canonicals = {canonical_for(col) for col in columns}
    canonicals.discard(None)
    out: list[str] = []

    if not (attendance_available or _has_any_attendance(canonicals)):
        out.append(
            f"Attendance not detected. You can still review GPA, standing, "
            f"{label_for('teacher', mode).lower()}, department, and Academic Watch workflows."
        )
    if not any(c in canonicals for c in ("gpa", "cumulative_gpa", "term_gpa")):
        out.append(
            "GPA not detected. I can still help with teacher, department, "
            "attendance, and standing workflows."
        )
    if "teacher" not in canonicals and "advisor" not in canonicals:
        out.append(
            f"{label_for('teacher', mode)}/professor field not detected. I can still group by "
            "department, major, GPA, standing, or attendance."
        )
    if "academic_status" not in canonicals:
        out.append(
            "Academic standing not detected. I can still use GPA and "
            "attendance fields for risk review if they are available."
        )
    if not _has_any_assessment(canonicals):
        out.append(
            "Assessment scores not detected. You can still review GPA, attendance, "
            "standing, teacher, department, and watch workflows."
        )
    elif not _has_assessment_benchmark(canonicals) and "assessment_risk" not in canonicals:
        out.append(
            "Assessment scores detected. Benchmark-risk questions require benchmark "
            "fields or configured thresholds."
        )
    if "academic_watch" not in canonicals:
        out.append(
            "Academic Watch field not found. If you mark students Academic "
            "Watch, I’ll create that column in the exported workbook."
        )
    if "attendance_watch" not in canonicals:
        out.append(
            "Attendance Watch field not found. If you mark students "
            "Attendance Watch, I’ll create that column in the exported workbook."
        )
    return out


# ---- one-time upload greeting ---------------------------------------------


def upload_assistant_message(
    columns: Iterable[str],
    *,
    attendance_available: bool = False,
    mode: InstitutionMode = InstitutionMode.GENERIC,
) -> str:
    """Build the friendly one-line greeting the chat panel appends ONCE per
    workbook upload. Adapts to what was actually detected so a roster-only
    workbook doesn't get told attendance exists."""
    canonicals = {canonical_for(col) for col in columns}
    canonicals.discard(None)

    detected_phrases: list[str] = []
    if "teacher" in canonicals:
        detected_phrases.append(label_for("teacher", mode).lower())
    elif "advisor" in canonicals:
        detected_phrases.append("advisor")
    if "department" in canonicals or "discipline" in canonicals:
        detected_phrases.append("department")
    if any(c in canonicals for c in ("student_id", "full_name", "first_name", "last_name")):
        detected_phrases.append("student")
    if any(c in canonicals for c in ("gpa", "cumulative_gpa", "term_gpa")):
        detected_phrases.append("GPA")
    if "major" in canonicals:
        detected_phrases.append("major")
    if "academic_status" in canonicals:
        detected_phrases.append("academic standing")

    has_attendance = attendance_available or _has_any_attendance(canonicals)

    if not detected_phrases:
        # No recognised academic fields — the user uploaded something we
        # can't drive workflows against. Be honest but not alarming.
        return (
            "I don't recognise the columns in this workbook as academic "
            "roster fields. Try uploading a workbook with at least a "
            "Student column and one of GPA, Teacher, Professor, or Department."
        )

    fields_summary = _humanise_list(detected_phrases)
    if has_attendance:
        return (
            f"I found {fields_summary}, and attendance fields in this "
            "workbook. I can help you review performance, identify "
            "attendance or GPA risk, group students by teacher or "
            "department, mark Academic Watch or Attendance Watch, and "
            "export a new workbook."
        )
    return (
        f"I found {fields_summary} fields. I do not see attendance "
        "fields, so attendance-risk questions are not available yet. "
        "I can still help with GPA, standing, teacher, department, "
        "Academic Watch, and export workflows."
    )


def _humanise_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _has_any_assessment(canonicals: set[str]) -> bool:
    return any(c in canonicals for c in (
        "test_type", "test_date", "psat_math", "psat_reading_writing",
        "psat_total", "sat_math", "sat_ebrw", "sat_total",
        "math_score", "reading_writing_score", "total_score",
        "benchmark_status", "math_benchmark_met", "reading_benchmark_met",
        "college_readiness", "assessment_risk",
    ))


def _has_assessment_benchmark(canonicals: set[str]) -> bool:
    return any(c in canonicals for c in (
        "benchmark_status", "math_benchmark_met", "reading_benchmark_met",
        "college_readiness",
    ))
