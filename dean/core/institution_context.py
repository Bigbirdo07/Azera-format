"""School-context layer — institution mode + role.

Both settings are UI-only overlays. They never change which planner runs,
which filters resolve, which actions execute, or which workbook is written.
They only:

  - swap field labels (Teacher ↔ Professor / Instructor; Grade ↔ Class Year)
  - reorder / filter the suggested-question list
  - retune the capability-checklist titles
  - choose role-appropriate workflow-template wording

Stored in ``st.session_state`` under ``institution_mode`` / ``user_role``.
Defaults: Generic Academic + Administrator/Dean.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InstitutionMode(str, Enum):
    PK12 = "pk12"
    COLLEGE = "college"
    GENERIC = "generic"

    @classmethod
    def from_label(cls, label: str) -> "InstitutionMode":
        norm = (label or "").strip().lower()
        if norm in {"pk-12", "pk12", "k-12", "k12"}:
            return cls.PK12
        if norm in {"college", "college / university", "university", "higher education"}:
            return cls.COLLEGE
        return cls.GENERIC


class Role(str, Enum):
    ADMIN = "admin"
    COUNSELOR = "counselor"
    TEACHER = "teacher"
    REGISTRAR = "registrar"

    @classmethod
    def from_label(cls, label: str) -> "Role":
        norm = (label or "").strip().lower()
        if "counsel" in norm or "advisor" in norm:
            return cls.COUNSELOR
        if "teacher" in norm or "professor" in norm or "instructor" in norm:
            return cls.TEACHER
        if "registrar" in norm or "data" in norm:
            return cls.REGISTRAR
        return cls.ADMIN


INSTITUTION_MODE_LABELS = {
    InstitutionMode.PK12: "PK–12",
    InstitutionMode.COLLEGE: "College / University",
    InstitutionMode.GENERIC: "Generic Academic",
}

ROLE_LABELS = {
    Role.ADMIN: "Administrator / Dean",
    Role.COUNSELOR: "Counselor / Advisor",
    Role.TEACHER: "Teacher / Professor",
    Role.REGISTRAR: "Registrar / Data Staff",
}


@dataclass(frozen=True)
class InstitutionContext:
    """The active mode + role for the session. Used by every helper that
    needs to translate labels."""
    mode: InstitutionMode = InstitutionMode.GENERIC
    role: Role = Role.ADMIN


# ---- label mapping --------------------------------------------------------


# Per-mode display label for each canonical concept. When a concept isn't in
# the inner dict, the helper falls back to the canonical's GENERIC label.
_LABEL_BY_MODE: dict[InstitutionMode, dict[str, str]] = {
    InstitutionMode.PK12: {
        "teacher": "Teacher",
        "advisor": "Counselor",
        "year": "Grade",
        "class_level": "Grade Level",
        "academic_status": "Academic Standing",
        "major": "Subject / Program",
        "department": "Department",
        "full_name": "Student",
        "student_id": "Student ID",
        "parent_guardian_contact": "Parent/guardian contact",
        "gpa": "GPA",
        "academic_watch": "Academic Watch",
        "attendance_watch": "Attendance Watch",
        "follow_up_needed": "Follow Up Needed",
    },
    InstitutionMode.COLLEGE: {
        "teacher": "Professor",
        "advisor": "Advisor",
        "year": "Class Year",
        "class_level": "Class Level",
        "academic_status": "Academic Standing",
        "major": "Major",
        "department": "Department",
        "full_name": "Student",
        "student_id": "Student ID",
        "parent_guardian_contact": "Emergency / Guardian Contact",
        "gpa": "GPA",
        "academic_watch": "Academic Watch",
        "attendance_watch": "Attendance Watch",
        "follow_up_needed": "Advisor Follow Up",
    },
    InstitutionMode.GENERIC: {
        "teacher": "Teacher / Instructor",
        "advisor": "Advisor",
        "year": "Year",
        "class_level": "Class Level",
        "academic_status": "Academic Standing",
        "major": "Major",
        "department": "Department",
        "full_name": "Student",
        "student_id": "Student ID",
        "parent_guardian_contact": "Guardian Contact",
        "gpa": "GPA",
        "academic_watch": "Academic Watch",
        "attendance_watch": "Attendance Watch",
        "follow_up_needed": "Follow Up Needed",
    },
}


def label_for(canonical: str, mode: InstitutionMode = InstitutionMode.GENERIC) -> str:
    """Return the user-facing label for a canonical field in a given mode.

    Falls back to a title-cased canonical when no override is registered.
    """
    by_mode = _LABEL_BY_MODE.get(mode, _LABEL_BY_MODE[InstitutionMode.GENERIC])
    if canonical in by_mode:
        return by_mode[canonical]
    return canonical.replace("_", " ").title()


# ---- capability titles per mode -------------------------------------------


# Mode-specific titles for each capability key from
# core.workbook_capabilities.detect_capabilities. Missing keys fall back to
# the GENERIC text.
_CAPABILITY_TITLES_BY_MODE: dict[InstitutionMode, dict[str, str]] = {
    InstitutionMode.PK12: {
        "teacher_department": "Teacher and grade-level questions",
        "gpa_performance": "GPA and grade-level review",
        "major_grouping": "Subject / program grouping",
        "academic_standing": "Academic standing review",
        "attendance_risk": "Attendance-risk review",
        "assessment_review": "PSAT/SAT assessment review",
        "benchmark_risk": "Benchmark-risk review",
        "combined_academic_risk": "Combined academic risk review",
        "academic_watch_updates": "Intervention and watch-list updates",
        "attendance_watch_updates": "Attendance Watch updates",
        "export": "Counselor follow-up exports",
    },
    InstitutionMode.COLLEGE: {
        "teacher_department": "Professor and department questions",
        "gpa_performance": "GPA performance review",
        "major_grouping": "Major-based grouping",
        "academic_standing": "Academic standing and retention review",
        "attendance_risk": "Attendance-risk review",
        "assessment_review": "PSAT/SAT assessment review",
        "benchmark_risk": "Benchmark and retention-risk review",
        "combined_academic_risk": "Combined academic risk review",
        "academic_watch_updates": "Advisor outreach updates",
        "attendance_watch_updates": "Attendance Watch updates",
        "export": "Retention-risk exports",
    },
    InstitutionMode.GENERIC: {
        "teacher_department": "Teacher/instructor and department questions",
        "gpa_performance": "GPA performance review",
        "major_grouping": "Major-based grouping",
        "academic_standing": "Academic standing review",
        "attendance_risk": "Attendance-risk review",
        "assessment_review": "PSAT/SAT assessment review",
        "benchmark_risk": "Benchmark-risk review",
        "combined_academic_risk": "Combined academic risk review",
        "academic_watch_updates": "Academic Watch updates",
        "attendance_watch_updates": "Attendance Watch updates",
        "export": "Export updated workbook",
    },
}


def capability_title(key: str, mode: InstitutionMode) -> str:
    """Return the mode-appropriate title for a capability key."""
    by_mode = _CAPABILITY_TITLES_BY_MODE.get(
        mode, _CAPABILITY_TITLES_BY_MODE[InstitutionMode.GENERIC],
    )
    return by_mode.get(key) or _CAPABILITY_TITLES_BY_MODE[InstitutionMode.GENERIC].get(
        key, key.replace("_", " ").title(),
    )


# ---- role-aware suggestion priorities -------------------------------------


# Per-role *intent groups* that should be promoted to the top of the
# Suggested Questions panel. Each group is a list of suggestion-id prefixes
# (as emitted by nlp.dynamic_suggestions). Anything outside the priority
# list still appears, just lower.
ROLE_PRIORITY_PREFIXES: dict[Role, list[str]] = {
    Role.ADMIN: [
        "dyn_avg_", "dyn_list_unique_", "dyn_top_n_", "dyn_at_risk_count",
        "dyn_data_quality",
    ],
    Role.COUNSELOR: [
        "dyn_at_risk_count", "dyn_low_gpa_low_attendance",
        "dyn_attendance_below_90", "dyn_mark_academic_watch_export",
        "dyn_mark_attendance_watch_export", "dyn_missing_",
    ],
    Role.TEACHER: [
        "dyn_top_n_", "dyn_bottom_n_", "dyn_filter_",
        "dyn_attendance_below_90",
    ],
    Role.REGISTRAR: [
        "dyn_data_quality", "dyn_missing_", "dyn_list_unique_",
    ],
}


# ---- workflow templates ----------------------------------------------------


WORKFLOW_TEMPLATES_BY_MODE: dict[InstitutionMode, list[tuple[str, str]]] = {
    InstitutionMode.PK12: [
        ("Assessment Review", "Find students below benchmark. Group by teacher or department. Mark Follow Up Needed or Academic Watch. Export workbook."),
        ("Academic Watch Review", "Find students below GPA threshold. Group by teacher. Mark Academic Watch. Export workbook."),
        ("Attendance Watch Review", "Find students below attendance threshold. Group by teacher. Mark Attendance Watch. Export workbook."),
        ("Combined Risk Review", "Find students with 2 or more risk signals. Group by teacher or department. Mark Follow Up Needed. Export workbook."),
    ],
    InstitutionMode.COLLEGE: [
        ("Assessment Review", "Find students below benchmark. Group by professor or department. Mark Follow Up Needed or Academic Watch. Export workbook."),
        ("Academic Watch Review", "Find students below GPA threshold. Group by professor or department. Mark Academic Watch. Export workbook."),
        ("Attendance Watch Review", "Find students below attendance threshold. Group by professor or advisor. Mark Attendance Watch. Export workbook."),
        ("Combined Risk Review", "Find students with 2 or more risk signals. Group by professor or department. Mark Follow Up Needed. Export workbook."),
    ],
    InstitutionMode.GENERIC: [
        ("Assessment Review", "Find students below benchmark. Group by teacher or department. Mark Follow Up Needed or Academic Watch. Export workbook."),
        ("Academic Watch Review", "Find students below GPA threshold. Group by teacher or department. Mark Academic Watch. Export workbook."),
        ("Attendance Watch Review", "Find students below attendance threshold. Group by teacher or department. Mark Attendance Watch. Export workbook."),
        ("Combined Risk Review", "Find students with 2 or more risk signals. Group by teacher or department. Mark Follow Up Needed. Export workbook."),
    ],
}


def workflow_templates(mode: InstitutionMode) -> list[tuple[str, str]]:
    return WORKFLOW_TEMPLATES_BY_MODE.get(mode, WORKFLOW_TEMPLATES_BY_MODE[InstitutionMode.GENERIC])


def role_prompt_snippets(role: Role, mode: InstitutionMode) -> list[str]:
    if role == Role.ADMIN:
        return [
            "department summaries",
            f"{label_for('teacher', mode).lower()} groupings",
            "export reports",
        ]
    if role == Role.COUNSELOR:
        return [
            "student risk lists",
            "follow-up needed",
            "attendance watch",
            "intervention list",
        ]
    if role == Role.TEACHER:
        return [
            "students under my classes",
            "low GPA / attendance risk in my group",
        ]
    return [
        "missing IDs",
        "duplicate IDs",
        "unmatched attendance records",
        "export validation",
    ]
