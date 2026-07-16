"""Vague-phrase resolver for the medium-confidence path.

The rule planner alone produces a no-filter `filtered_preview` when the user
says something it doesn't recognize (e.g. "show me struggling students").
Running that bare plan returns the whole sheet — but the assistant just
*claimed* it interpreted "struggling" as a risk definition. That's the gap
this module closes.

For known vague phrases (struggling, at risk, needs help, concerning,
falling behind, overloaded advisors, no advisor, best students, …) the
resolver either:

1. Returns a concrete validated plan when the supporting columns exist
   (Academic Status, GPA, Advisor), with an explicit assumption note and
   alternative-interpretation chips, OR
2. Returns a clarification question when no supporting column exists.

The resolver never sees row data. It looks at column names and the
workbook-wide set of categorical values that the planner already gathers
for the LLM prompt (small cardinality, no PII).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nlp.synonym_mapper import normalize_text


# Vague phrases the assistant should resolve to a concrete at-risk filter.
VAGUE_RISK_PHRASES: tuple[str, ...] = (
    "struggling",
    "students who are struggling",
    "at risk",
    "at-risk",
    "students at risk",
    "needs help",
    "needs advisor attention",
    "needs attention",
    "concerning",
    "falling behind",
    "underperforming",
)

VAGUE_ADVISOR_LOAD_PHRASES: tuple[str, ...] = (
    "overloaded advisor",
    "overloaded advisors",
    "advisor load",
    "advisor workload",
    "overworked advisor",
    "overworked advisors",
)

VAGUE_NO_ADVISOR_PHRASES: tuple[str, ...] = (
    "no advisor",
    "without an advisor",
    "without advisor",
    "missing advisor",
    "no assigned advisor",
    "students with no advisor",
)

VAGUE_TOP_PHRASES: tuple[str, ...] = (
    "best students",
    "top students",
    "doing well",
    "high performing",
)

# Safe at-risk values to look for in an Academic Status column. We deliberately
# do NOT include disciplinary, financial, health, or notes fields — those need
# explicit user permission and are sensitive per the privacy layer.
RISK_STATUS_VALUES: tuple[str, ...] = (
    "Warning",
    "Probation",
    "At Risk",
    "Academic Warning",
    "Academic Probation",
)


@dataclass(frozen=True)
class VagueResolution:
    matched_phrase: str
    category: str
    query: dict[str, Any] | None
    assumption: str
    alternatives: list[str] = field(default_factory=list)
    clarification: str | None = None

    @property
    def has_plan(self) -> bool:
        return self.query is not None


def resolve_vague_term(
    *,
    message: str,
    sheet: str,
    columns: list[str],
    categorical_values: dict[str, list[str]] | None = None,
) -> VagueResolution | None:
    """Return a vague-term resolution or None when no vague phrase is found."""
    text = f" {normalize_text(message)} "

    risk = _find_phrase(text, VAGUE_RISK_PHRASES)
    if risk:
        return _resolve_risk(risk, sheet, columns, categorical_values or {}, text=text)

    advisor_load = _find_phrase(text, VAGUE_ADVISOR_LOAD_PHRASES)
    if advisor_load:
        return _resolve_advisor_load(advisor_load, sheet, columns)

    no_advisor = _find_phrase(text, VAGUE_NO_ADVISOR_PHRASES)
    if no_advisor:
        return _resolve_no_advisor(no_advisor, sheet, columns)

    top = _find_phrase(text, VAGUE_TOP_PHRASES)
    if top:
        return _resolve_top(top, sheet, columns)

    return None


# Internal helpers -----------------------------------------------------------


def _find_phrase(padded_text: str, phrases: tuple[str, ...]) -> str | None:
    for phrase in phrases:
        target = f" {phrase} "
        if target in padded_text:
            return phrase
    return None


def _find_column(columns: list[str], *targets: str) -> str | None:
    """Pick the first column whose normalized name matches any target."""
    needles = {normalize_text(t) for t in targets}
    for column in columns:
        if normalize_text(column) in needles:
            return column
    return None


def _matched_risk_values(values: list[str]) -> list[str]:
    risk = {normalize_text(v) for v in RISK_STATUS_VALUES}
    return [v for v in values if normalize_text(v) in risk]


_ATTENDANCE_QUALIFIER_WORDS = ("attendance", "absent", "absences", "attending")
_GPA_EXCLUSION_PHRASES = ("not gpa", "not by gpa", "not on gpa", "not grades", "not grade")


def _resolve_risk(phrase, sheet, columns, categorical_values, *, text: str = "") -> VagueResolution:
    # A qualifier in the surrounding message ("struggling WITH ATTENDANCE",
    # "... not gpa") should redirect the definition instead of always
    # defaulting to Academic Status / GPA. Caught live: "no sorry I meant
    # struggling with attendance not gpa" kept resolving to the GPA<2.5
    # definition from the turn before -- the correction was silently ignored
    # because this function had no attendance-flavored branch at all.
    if text and any(word in text for word in _ATTENDANCE_QUALIFIER_WORDS):
        attendance_resolution = _resolve_attendance_risk(phrase, sheet, columns)
        if attendance_resolution is not None:
            return attendance_resolution

    gpa_excluded = bool(text) and any(p in text for p in _GPA_EXCLUSION_PHRASES)

    status_column = _find_column(columns, "Academic Status", "Status")
    gpa_column = None if gpa_excluded else _find_column(columns, "GPA")

    if status_column:
        # If we can see the column's distinct values, narrow to ones that
        # match. Otherwise fall back to the full list — the query engine will
        # simply return zero rows for values that aren't present.
        available = categorical_values.get(status_column, [])
        matched = _matched_risk_values(available) or list(RISK_STATUS_VALUES)
        assumption = (
            f"I interpreted '{phrase}' as students with {status_column} in "
            f"{', '.join(matched)}."
        )
        alternatives = ["Now use GPA below 2.5 instead", "Now use GPA below 2.0 instead",
                        "Use Probation only"]
        if not gpa_column:
            alternatives = [a for a in alternatives if "GPA" not in a]
            alternatives.append("Use Warning only")
        return VagueResolution(
            matched_phrase=phrase, category="risk",
            query={
                "request_type": "ask_question",
                "operation": "filtered_preview",
                "sheet": sheet,
                "filters": [{"column": status_column, "operator": "in", "value": matched}],
                "group_by": "", "value_column": "", "sort": None, "limit": None,
                "plain_english_question": f"students who appear {phrase}",
                "confidence": 0.7,
            },
            assumption=assumption,
            alternatives=alternatives[:3],
        )

    if gpa_column:
        assumption = f"I interpreted '{phrase}' as students with {gpa_column} below 2.5."
        return VagueResolution(
            matched_phrase=phrase, category="risk",
            query={
                "request_type": "ask_question",
                "operation": "filtered_preview",
                "sheet": sheet,
                "filters": [{"column": gpa_column, "operator": "less_than", "value": 2.5}],
                "group_by": "", "value_column": "", "sort": None, "limit": None,
                "plain_english_question": f"students who appear {phrase}",
                "confidence": 0.7,
            },
            assumption=assumption,
            alternatives=["Now use GPA below 2.0 instead", "Now use GPA below 3.0 instead"],
        )

    return VagueResolution(
        matched_phrase=phrase, category="unsupported", query=None,
        assumption="",
        clarification=(
            f"I can help with '{phrase}', but I need a definition. "
            "Should I use GPA, academic status, credits, advisor notes, or another column?"
        ),
    )


def _resolve_attendance_risk(phrase, sheet, columns) -> VagueResolution | None:
    """Attendance-flavored definition of a vague risk phrase. Same column
    preference as core.query_planner's _attendance_predicate_filter: a
    precomputed category/risk column over a raw rate threshold, so both
    paths agree on what "attendance risk" means for a given workbook."""
    category_column = _find_column(columns, "Attendance Category")
    if category_column:
        return VagueResolution(
            matched_phrase=phrase, category="risk",
            query={
                "request_type": "ask_question",
                "operation": "filtered_preview",
                "sheet": sheet,
                "filters": [{"column": category_column, "operator": "equals",
                            "value": "Needs Attendance Support"}],
                "group_by": "", "value_column": "", "sort": None, "limit": None,
                "plain_english_question": f"students who appear {phrase} (attendance)",
                "confidence": 0.7,
            },
            assumption=f"I interpreted '{phrase}' as students with {category_column} = Needs Attendance Support.",
            alternatives=["Now use GPA below 2.5 instead", "Now use Attendance Rate below 90% instead"],
        )
    risk_column = _find_column(columns, "Attendance Risk")
    if risk_column:
        return VagueResolution(
            matched_phrase=phrase, category="risk",
            query={
                "request_type": "ask_question",
                "operation": "filtered_preview",
                "sheet": sheet,
                "filters": [{"column": risk_column, "operator": "equals", "value": True}],
                "group_by": "", "value_column": "", "sort": None, "limit": None,
                "plain_english_question": f"students who appear {phrase} (attendance)",
                "confidence": 0.7,
            },
            assumption=f"I interpreted '{phrase}' as students flagged {risk_column}.",
            alternatives=["Now use GPA below 2.5 instead"],
        )
    rate_column = _find_column(columns, "Attendance Rate", "Attendance %")
    if rate_column:
        return VagueResolution(
            matched_phrase=phrase, category="risk",
            query={
                "request_type": "ask_question",
                "operation": "filtered_preview",
                "sheet": sheet,
                "filters": [{"column": rate_column, "operator": "less_than", "value": 90}],
                "group_by": "", "value_column": "", "sort": None, "limit": None,
                "plain_english_question": f"students who appear {phrase} (attendance)",
                "confidence": 0.7,
            },
            assumption=f"I interpreted '{phrase}' as students with {rate_column} below 90%.",
            alternatives=["Now use GPA below 2.5 instead", "Now use 80% instead"],
        )
    return None


def _resolve_advisor_load(phrase, sheet, columns) -> VagueResolution:
    advisor_column = _find_column(columns, "Advisor")
    if not advisor_column:
        return VagueResolution(
            matched_phrase=phrase, category="unsupported", query=None,
            assumption="",
            clarification=(
                f"I can summarize '{phrase}', but I do not see an Advisor column. "
                "Which column tracks advisors?"
            ),
        )
    return VagueResolution(
        matched_phrase=phrase, category="advisor_load",
        query={
            "request_type": "ask_question",
            "operation": "groupby_count",
            "sheet": sheet,
            "filters": [],
            "group_by": advisor_column,
            "value_column": "",
            "sort": {"column": advisor_column, "direction": "descending"},
            "limit": None,
            "plain_english_question": "advisor load by student count",
            "confidence": 0.7,
        },
        assumption=(
            f"I interpreted '{phrase}' as: group by {advisor_column} and count "
            "students, sorted descending."
        ),
        alternatives=["Show only the top 10 advisors",
                     "Show advisors with more than 25 students"],
    )


def _resolve_no_advisor(phrase, sheet, columns) -> VagueResolution:
    advisor_column = _find_column(columns, "Advisor")
    if not advisor_column:
        return VagueResolution(
            matched_phrase=phrase, category="unsupported", query=None,
            assumption="",
            clarification=(
                f"I can list students with no advisor, but I do not see an Advisor "
                "column. Which column tracks advisors?"
            ),
        )
    return VagueResolution(
        matched_phrase=phrase, category="no_advisor",
        query={
            "request_type": "ask_question",
            "operation": "filtered_preview",
            "sheet": sheet,
            "filters": [{"column": advisor_column, "operator": "is_blank"}],
            "group_by": "", "value_column": "", "sort": None, "limit": None,
            "plain_english_question": f"students with no {advisor_column}",
            "confidence": 0.8,
        },
        assumption=f"I interpreted '{phrase}' as: {advisor_column} is blank.",
        alternatives=[],
    )


def _resolve_top(phrase, sheet, columns) -> VagueResolution:
    gpa_column = _find_column(columns, "GPA")
    if not gpa_column:
        return VagueResolution(
            matched_phrase=phrase, category="unsupported", query=None,
            assumption="",
            clarification=(
                f"I can find '{phrase}', but I do not see a GPA column. "
                "Which column should I use to rank?"
            ),
        )
    return VagueResolution(
        matched_phrase=phrase, category="top",
        query={
            "request_type": "ask_question",
            "operation": "filtered_preview",
            "sheet": sheet,
            "filters": [{"column": gpa_column, "operator": "greater_or_equal", "value": 3.5}],
            "group_by": "", "value_column": "",
            "sort": {"column": gpa_column, "direction": "descending"},
            "limit": None,
            "plain_english_question": f"top students by {gpa_column}",
            "confidence": 0.7,
        },
        assumption=f"I interpreted '{phrase}' as students with {gpa_column} of 3.5 or higher.",
        alternatives=["Use GPA above 3.7 only", "Show the top 10 by GPA"],
    )
