"""DataSourceRegistry — the per-session bundle of academic data sources.

The roster is the anchor (one ``LoadedWorkbook`` from ``core.excel_loader``);
attendance and assessments are optional sibling sources that get *joined into*
the roster as derived columns. The chat planner only ever sees the enriched
roster + an unmodified long-format attendance frame, so every existing intent
(filter, top-N, projection, OR, percent, groupby) works on attendance and
assessment fields without parser changes.

Guarantees:
- Each uploaded source is held verbatim — no LLM ever sees row data.
- The original roster bytes are never modified; the registry only produces
  derived in-memory views.
- A new roster upload resets the entire registry (attendance and assessments
  are bound to one roster snapshot, so a fresh roster must clear them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.excel_loader import LoadedWorkbook


@dataclass
class AttendanceSource:
    """Holds the raw long-format attendance frame plus the matching report.

    ``frame`` is the canonicalised attendance table — one row per (Student ID,
    Date) with columns Student ID / Date / Attendance Status / Excused /
    Tardy / Period? / Teacher? / Course?. ``unmatched_ids`` is the list of
    Student IDs in the attendance file that do NOT appear in the roster (so
    the UI can warn the user).
    """
    file_name: str
    frame: pd.DataFrame
    matched_count: int = 0
    unmatched_count: int = 0
    unmatched_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AssessmentSource:
    """Stub for PSAT/SAT uploads. Implementation deferred until attendance is
    complete (see core.assessment for the schema)."""
    file_name: str
    frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


@dataclass
class DataSourceRegistry:
    """All data sources for one chat session.

    The roster is required; attendance can come from three places (in
    priority order):
      1. An external attendance file the user uploaded as Advanced.
      2. Attendance columns already on the roster sheet.
      3. A sibling "Attendance" sheet inside the uploaded workbook.

    Whichever wins becomes the source for the enriched-roster view that
    every existing intent (filter, top-N, projection, OR, percent, groupby,
    list-unique, watch) operates on.
    """
    roster: LoadedWorkbook | None = None
    attendance: AttendanceSource | None = None
    assessments: AssessmentSource | None = None
    # The roster sheet name that the planner treats as authoritative — the
    # one we enrich. Defaults to the first sheet of the roster workbook.
    enriched_roster_sheet: str = ""
    # Populated by set_roster() from detect_workbook_attendance.
    workbook_attendance: Any = None  # AttendanceDetection from core.attendance
    workbook_assessment: Any = None  # AssessmentDetection from core.assessment
    risk_settings: Any = None

    # ---- lifecycle ----------------------------------------------------------

    def set_roster(self, roster: LoadedWorkbook | None) -> None:
        """Bind a roster. Clears the external attendance upload — attendance
        is tied to one roster snapshot (matching by Student ID is only
        meaningful within the same student population). Re-runs attendance
        detection against the new workbook."""
        from core.attendance import detect_workbook_attendance
        from core.assessment import detect_workbook_assessments

        previous = self.roster.file_name if self.roster else None
        self.roster = roster
        if roster is None:
            self.enriched_roster_sheet = ""
            self.attendance = None
            self.assessments = None
            self.workbook_attendance = None
            self.workbook_assessment = None
            return
        if previous != roster.file_name:
            self.attendance = None
            self.assessments = None
        if roster.sheets:
            self.enriched_roster_sheet = _select_roster_sheet(roster.sheets)
        self.workbook_attendance = detect_workbook_attendance(
            roster.sheets, self.enriched_roster_sheet,
        )
        self.workbook_assessment = detect_workbook_assessments(
            roster.sheets, self.enriched_roster_sheet,
        )

    def set_attendance(self, attendance: AttendanceSource | None) -> None:
        self.attendance = attendance

    def clear(self) -> None:
        self.roster = None
        self.attendance = None
        self.assessments = None
        self.enriched_roster_sheet = ""

    # ---- derived views ------------------------------------------------------

    def enriched_sheets(self) -> dict[str, pd.DataFrame]:
        """Return the sheets the chat planner should query against.

        Priority order for attendance enrichment:
          1. External upload (Advanced).
          2. A sibling Attendance sheet inside the workbook.
          3. Inline attendance columns already on the roster — no
             recomputation needed.
          4. No attendance — roster passes through unchanged (combined-risk
             still attaches whatever non-attendance signals exist).
        """
        from core.attendance import compute_attendance_metrics
        from core.assessment import (
            attach_assessment_metrics,
            canonicalise_assessment_frame,
            compute_assessment_metrics,
        )
        from core.combined_risk import attach_combined_risk

        if self.roster is None:
            return {}
        sheets = dict(self.roster.sheets)
        target = self.enriched_roster_sheet or next(iter(sheets), "")
        if not target or target not in sheets:
            return sheets

        roster_frame = sheets[target].copy()

        if self.attendance is not None and not self.attendance.frame.empty:
            # Priority 1: external upload wins.
            metrics = compute_attendance_metrics(
                self.attendance.frame,
                risk_threshold=_risk_setting(self.risk_settings, "attendance_risk_threshold"),
                severe_threshold=_risk_setting(self.risk_settings, "severe_attendance_risk_threshold"),
            )
            roster_frame = _merge_metrics(roster_frame, metrics)
            sheets["Attendance"] = self.attendance.frame
        elif (self.workbook_attendance is not None
              and self.workbook_attendance.mode == "sheet"
              and self.workbook_attendance.attendance_sheet in sheets):
            # Priority 2: sibling sheet inside the workbook.
            from core.attendance import canonicalise_attendance_frame
            attendance_raw = sheets[self.workbook_attendance.attendance_sheet]
            canonical_attendance, _ = canonicalise_attendance_frame(attendance_raw)
            if not canonical_attendance.empty:
                metrics = compute_attendance_metrics(
                    canonical_attendance,
                    risk_threshold=_risk_setting(self.risk_settings, "attendance_risk_threshold"),
                    severe_threshold=_risk_setting(self.risk_settings, "severe_attendance_risk_threshold"),
                )
                roster_frame = _merge_metrics(roster_frame, metrics)
        # Priority 3 (inline columns) requires nothing — the roster frame
        # already carries them. Priority 4 (none) also requires nothing.

        if (self.workbook_assessment is not None
                and self.workbook_assessment.mode == "sheet"
                and self.workbook_assessment.assessment_sheet in sheets):
            assessment_raw = sheets[self.workbook_assessment.assessment_sheet]
            canonical_assessment, warnings = canonicalise_assessment_frame(assessment_raw)
            if not canonical_assessment.empty:
                metrics = compute_assessment_metrics(
                    canonical_assessment,
                    risk_settings=self.risk_settings,
                )
                roster_frame = attach_assessment_metrics(roster_frame, metrics)
                sheets[self.workbook_assessment.assessment_sheet] = canonical_assessment

        roster_frame = attach_combined_risk(
            roster_frame,
            gpa_threshold=_risk_setting(self.risk_settings, "gpa_risk_threshold"),
            attendance_threshold=_risk_setting(self.risk_settings, "attendance_risk_threshold"),
            high_count=_risk_setting(self.risk_settings, "high_risk_signal_count"),
            moderate_count=_risk_setting(self.risk_settings, "moderate_risk_signal_count"),
            sat_math_threshold=_risk_setting(self.risk_settings, "sat_math_benchmark_threshold"),
            sat_ebrw_threshold=_risk_setting(self.risk_settings, "sat_ebrw_benchmark_threshold"),
            psat_math_threshold=_risk_setting(self.risk_settings, "psat_math_benchmark_threshold"),
            psat_reading_writing_threshold=_risk_setting(self.risk_settings, "psat_reading_writing_benchmark_threshold"),
        )
        sheets[target] = roster_frame
        return sheets

    def enriched_columns(self) -> dict[str, list[str]]:
        return {name: list(frame.columns) for name, frame in self.enriched_sheets().items()}

    def summary(self) -> dict[str, Any]:
        """Compact human-readable snapshot for the Data Sources panel.

        ``workbook_attendance`` describes whichever attendance signal was
        auto-detected inside the uploaded workbook (inline columns / sibling
        sheet / none). It is independent of ``attendance``, which is set
        only when the user used the Advanced external uploader.
        """
        out: dict[str, Any] = {
            "roster": None,
            "attendance": None,
            "assessments": None,
            "workbook_attendance": None,
            "workbook_assessment": None,
        }
        if self.roster:
            target = self.enriched_roster_sheet or next(iter(self.roster.sheets), "")
            frame = self.roster.sheets.get(target)
            out["roster"] = {
                "file_name": self.roster.file_name,
                "active_sheet": target,
                "rows": int(len(frame)) if frame is not None else 0,
                "columns": list(frame.columns) if frame is not None else [],
                "warnings": list(self.roster.warnings or []),
            }
        if self.workbook_attendance is not None:
            wa = self.workbook_attendance
            out["workbook_attendance"] = {
                "mode": wa.mode,
                "inline_columns": list(wa.inline_columns),
                "inline_fields": list(wa.inline_fields),
                "attendance_sheet": wa.attendance_sheet,
            }
        if self.workbook_assessment is not None:
            wa = self.workbook_assessment
            out["workbook_assessment"] = {
                "mode": wa.mode,
                "inline_columns": list(wa.inline_columns),
                "inline_fields": list(wa.inline_fields),
                "assessment_sheet": wa.assessment_sheet,
                "candidate_sheets": list(wa.candidate_sheets),
                "warnings": list(wa.warnings),
            }
        if self.attendance:
            out["attendance"] = {
                "file_name": self.attendance.file_name,
                "rows": int(len(self.attendance.frame)),
                "columns": list(self.attendance.frame.columns),
                "matched": self.attendance.matched_count,
                "unmatched": self.attendance.unmatched_count,
                "unmatched_sample": self.attendance.unmatched_ids[:5],
                "warnings": list(self.attendance.warnings or []),
            }
        if self.assessments:
            out["assessments"] = {
                "file_name": self.assessments.file_name,
                "rows": int(len(self.assessments.frame)),
                "columns": list(self.assessments.frame.columns),
                "warnings": list(self.assessments.warnings or []),
            }
        return out

    def attendance_available(self) -> bool:
        """True when *any* attendance signal is present (external upload,
        inline columns, or sibling sheet)."""
        if self.attendance is not None and not self.attendance.frame.empty:
            return True
        wa = self.workbook_attendance
        return wa is not None and wa.mode in {"inline", "sheet"}


def _merge_metrics(roster: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Merge per-student metrics onto the roster by Student ID.

    Resolves the roster's student-id column case/space-insensitively so a
    column named "Student ID" or "StudentID" or "student id" all work.
    """
    if metrics.empty or "Student ID" not in metrics.columns:
        return roster
    id_col = _find_student_id_column(roster.columns)
    if id_col is None:
        return roster
    merged = roster.merge(
        metrics, how="left",
        left_on=id_col, right_on="Student ID",
        suffixes=("", "_attendance"),
    )
    # Drop the redundant join key when its name differs from the roster's.
    if id_col != "Student ID" and "Student ID" in merged.columns:
        merged = merged.drop(columns=["Student ID"])
    return merged


def _select_roster_sheet(sheets: dict[str, pd.DataFrame]) -> str:
    """Pick the student roster sheet for enrichment.

    Most uploads are single-sheet files, but multi-sheet workbooks often carry
    sibling Attendance/Assessment tabs. Prefer a sheet with roster-like fields
    instead of blindly choosing the first visible sheet.
    """
    if not sheets:
        return ""
    scored: list[tuple[int, int, str]] = []
    for index, (name, frame) in enumerate(sheets.items()):
        if frame is None or frame.empty:
            continue
        score = 0
        name_norm = str(name).strip().lower()
        if any(token in name_norm for token in ("roster", "student", "students")):
            score += 4
        if _find_student_id_column(frame.columns) is not None:
            score += 3
        lowered = {_norm_column(column) for column in frame.columns}
        for variants, weight in (
            (("studentname", "fulllegalname", "name"), 2),
            (("gpa", "currentgpa", "cumulativegpa"), 2),
            (("advisor", "advisorcounselor", "counselor"), 2),
            (("major", "majorprogram", "program"), 1),
            (("academicsanding", "academicstanding", "standing"), 1),
            (("gradelevel", "yrgrade", "year"), 1),
        ):
            if lowered.intersection(variants):
                score += weight
        scored.append((score, -index, name))
    if not scored:
        return next(iter(sheets))
    scored.sort(reverse=True)
    return scored[0][2]


def _risk_setting(settings: Any, name: str):
    return getattr(settings, name, None) if settings is not None else None


def _find_student_id_column(columns) -> str | None:
    for col in columns:
        norm = str(col).strip().lower().replace(" ", "").replace("_", "")
        if norm == "studentid":
            return col
    return None


def _norm_column(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "").replace("/", "")
