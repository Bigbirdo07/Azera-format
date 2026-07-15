"""Workbook-aware suggested questions.

Given the actual sheet a counselor just loaded, build a short, ordered list of
``Question`` objects that exercise the planner's range — top-N row previews,
list-unique catalogs, value-filtered slices, group-by aggregates, missing-data
finders, column projections. Each suggestion uses the user's REAL column
names (and, when relevant, a real value sniffed from the data) so the panel
reflects this workbook, not a generic college example.

Read-only: never mutates the DataFrame, never calls the LLM. Output slots
straight into ``AskableCategory.questions`` for ``render_suggested_questions_panel``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from nlp.question_library import Question
from nlp.synonym_mapper import load_json, load_synonyms_with_learned, match_column_for_concept
from core.institution_context import InstitutionMode, Role, role_prompt_snippets


# Concepts we treat as "row entities" — never useful as a list_unique target
# (listing every Student ID is not what the counselor wants).
_ROW_ENTITY_CONCEPTS = ("student_id", "student")

# Concepts we treat as "class identities" — the right list_unique targets.
_CLASS_IDENTITY_CONCEPTS = ("teacher", "advisor", "department", "discipline",
                            "major", "program", "course")

# Concepts whose column is a numeric metric worth sorting/averaging.
_SCORE_CONCEPTS = (
    "gpa", "credits", "balance_due",
    "sat_math", "sat_ebrw", "sat_total",
    "psat_math", "psat_reading_writing", "psat_total",
)

# Concepts that frequently have meaningful "missing" rows worth surfacing.
_MISSING_PRONE_CONCEPTS = ("fafsa_status", "advisor", "balance_due")

# Attendance concepts that unlock attendance-themed suggestions.
_ATTENDANCE_CONCEPTS = (
    "attendance_rate", "days_absent", "days_tardy",
    "unexcused_absences", "attendance_risk",
)

_ASSESSMENT_CONCEPTS = (
    "sat_math", "sat_ebrw", "sat_total",
    "psat_math", "psat_reading_writing", "psat_total",
    "benchmark_status", "math_benchmark_met", "reading_benchmark_met",
    "assessment_risk",
)

# Cardinality ceiling for a column to count as "low-card categorical" for the
# sniffed-value filter suggestion ("Show me Accounting students").
_MAX_FILTER_VALUE_CARDINALITY = 25

# Max suggestions to return — the panel renders one button per row, so keep
# this small enough to scan in a couple of seconds. 14 = enough headroom for
# the north-star action prompts (Mark Academic/Attendance Watch + export)
# *and* a sniffed categorical filter like "Show me Biology students".
_MAX_SUGGESTIONS = 14


def build_dynamic_suggestions(
    frame: pd.DataFrame | None,
    columns: list[str],
    *,
    synonyms: dict[str, list[str]] | None = None,
    limit: int = _MAX_SUGGESTIONS,
    mode: InstitutionMode = InstitutionMode.GENERIC,
    role: Role | None = None,
) -> list[Question]:
    """Return a workbook-tailored list of Question objects.

    Caller passes the active sheet's DataFrame (for value sniffing) and its
    column names. The function falls back to column-only generation when
    frame is None — useful in tests or when the sheet is empty.
    """
    if not columns:
        return []
    synonyms = synonyms if synonyms is not None else load_synonyms_with_learned()

    column_set = set(columns)
    resolved: dict[str, str] = {}
    # Attendance concepts go through canonical_for (strict) so the
    # "Attendance Watch" column doesn't fuzzy-match as if it were
    # "Attendance Rate" — they're different workflows.
    from core.schema import canonical_for
    attendance_by_canonical: dict[str, str] = {}
    for column in columns:
        canonical = canonical_for(column)
        if canonical in _ATTENDANCE_CONCEPTS and canonical not in attendance_by_canonical:
            attendance_by_canonical[canonical] = column
    resolved.update(attendance_by_canonical)

    for concept in (
        *_SCORE_CONCEPTS, *_CLASS_IDENTITY_CONCEPTS,
        *_ROW_ENTITY_CONCEPTS, *_MISSING_PRONE_CONCEPTS,
        *_ASSESSMENT_CONCEPTS,
        "year", "academic_status",
    ):
        if concept in resolved:
            continue
        column, score = match_column_for_concept(concept, columns, synonyms)
        if column and score >= 0.55 and column in column_set:
            resolved[concept] = column

    out: list[Question] = []
    seen_ids: set[str] = set()

    def add(qid: str, text: str, requires: Iterable[str] = ()) -> None:
        if qid in seen_ids or len(out) >= limit:
            return
        seen_ids.add(qid)
        out.append(Question(
            id=qid, text=text,
            requires_columns=tuple(requires), follow_ups=(),
        ))

    # A. Top-N for the primary score column.
    score_col = resolved.get("gpa") or resolved.get("credits")
    if score_col:
        add(f"dyn_top_n_{_slug(score_col)}",
            f"Top 10 students by {score_col}", [score_col])
        add(f"dyn_bottom_n_{_slug(score_col)}",
            f"Bottom 10 students by {score_col}", [score_col])

    # B. Year / Grade sniffed value — picks one real value the workbook has,
    # so a K-12 sheet says "5th graders" and a college sheet says "Freshmen".
    year_col = resolved.get("year")
    if year_col:
        sample_year = _sniff_grade_value(frame, year_col)
        if sample_year is not None:
            add(f"dyn_year_filter_{_slug(str(sample_year))}",
                _phrase_grade_filter(sample_year),
                [year_col])
            add(f"dyn_list_by_{_slug(year_col)}",
            f"List students by {year_col}", [year_col])

    # C. At-risk shortcut.
    if "academic_status" in resolved:
        add("dyn_at_risk_count",
            "How many students are at academic risk?",
            [resolved["academic_status"]])

    # C2. Attendance-themed prompts move up high — they're the most useful
    # questions to put in front of a counsellor once attendance is detected.
    rate_col_early = resolved.get("attendance_rate")
    if rate_col_early:
        add("dyn_attendance_below_90",
            "Show students with attendance below 90%", [rate_col_early])
        if score_col and score_col != rate_col_early:
            add("dyn_low_gpa_low_attendance",
                f"Show students with {score_col} below 2.0 and attendance below 90%",
            [score_col, rate_col_early])
    absences_col_early = resolved.get("days_absent")
    if absences_col_early and not rate_col_early:
        add("dyn_absences_threshold",
            f"Which students have more than 5 {absences_col_early.lower()}",
            [absences_col_early])

    # C3. Action prompts — Academic Watch / Attendance Watch + export.
    # Creatable on export even when the columns are absent.
    if score_col or rate_col_early or "academic_status" in resolved:
        add("dyn_mark_academic_watch_export",
            "Mark these students Academic Watch and export",
            ())
    if rate_col_early:
        add("dyn_mark_attendance_watch_export",
            "Mark these students Attendance Watch and export",
            [rate_col_early])

    sat_math_col = resolved.get("sat_math")
    psat_total_col = resolved.get("psat_total")
    assessment_risk_col = resolved.get("assessment_risk")
    benchmark_col = resolved.get("benchmark_status") or assessment_risk_col
    if sat_math_col:
        add("dyn_sat_math_threshold",
            "Show students below SAT Math threshold", [sat_math_col])
        teacher_col = resolved.get("teacher") or resolved.get("advisor")
        if teacher_col:
            add("dyn_avg_sat_math_by_teacher",
                f"Show average SAT Math by {teacher_col}", [sat_math_col, teacher_col])
        dept_col = resolved.get("department") or resolved.get("discipline")
        if dept_col:
            add("dyn_lowest_sat_math_by_department",
                f"Which department has the lowest average SAT Math?", [sat_math_col, dept_col])
    if psat_total_col:
        dept_col = resolved.get("department") or resolved.get("discipline")
        if dept_col:
            add("dyn_avg_psat_total_by_department",
                f"Show average PSAT total by {dept_col}", [psat_total_col, dept_col])
        add("dyn_low_psat_total",
            "Which students have low PSAT total scores?", [psat_total_col])
    if benchmark_col:
        add("dyn_below_benchmark",
            "Show students below benchmark", [benchmark_col])
        teacher_col = resolved.get("teacher") or resolved.get("advisor")
        if teacher_col:
            add("dyn_teacher_benchmark_risk",
                f"Which {teacher_col}s have the most students below benchmark?", [benchmark_col, teacher_col])
        add("dyn_high_risk_assessment",
            "Show high-risk students including assessment risk", [benchmark_col])
        if score_col:
            add("dyn_low_gpa_benchmark",
                f"Show students with {score_col} below 2.0 and below benchmark", [score_col, benchmark_col])
        if rate_col_early:
            add("dyn_low_attendance_benchmark",
                "Show students with poor attendance and below benchmark", [rate_col_early, benchmark_col])

    # D. List-unique for class-identity catalogues. Skip if the resolved column
    # also resolves to a row-entity concept (synonym fuzz can pull "Name" in
    # under "course") or if the column has too many distinct values to be a
    # useful catalog (a name column with one row per student is not a list).
    row_entity_columns = {
        resolved[c] for c in _ROW_ENTITY_CONCEPTS if c in resolved
    }
    catalog_seen: set[str] = set()
    for concept in _CLASS_IDENTITY_CONCEPTS:
        col = resolved.get(concept)
        if not col or col in row_entity_columns or col in catalog_seen:
            continue
        if not _looks_like_catalog(frame, col):
            continue
        catalog_seen.add(col)
        add(f"dyn_list_unique_{_slug(col)}",
            f"List every {col}", [col])

    # E. Average score by class-identity (e.g. "Average GPA by Department").
    if score_col:
        for concept in ("department", "discipline", "major", "advisor"):
            group_col = resolved.get(concept)
            if not group_col:
                continue
            add(f"dyn_avg_{_slug(score_col)}_by_{_slug(group_col)}",
                f"Average {score_col} by {group_col}",
                [score_col, group_col])
            break  # one is enough — the user can ask for others

    # F. Sniffed categorical filter — "Show me Accounting students". Dedup by
    # the picked value so the same word doesn't get suggested for two columns
    # that happen to share it (Department + Major both contain "Accounting").
    if frame is not None and not frame.empty:
        used_values: set[str] = set()
        for cat_col, sample_value in _categorical_samples(frame, exclude=set(resolved.get("year") or [])):
            if cat_col == resolved.get("year"):
                continue
            key = str(sample_value).strip().lower()
            if key in used_values:
                continue
            used_values.add(key)
            add(f"dyn_filter_{_slug(cat_col)}_{_slug(str(sample_value))}",
                f"Show me {sample_value} students",
                [cat_col])
            if len(out) >= limit:
                break

    # G. Column projection example: only when both a Name-like and a score-like
    # column are present, so the question shows two real columns.
    name_col = _name_like_column(columns)
    if name_col and score_col:
        add(f"dyn_project_{_slug(name_col)}_{_slug(score_col)}",
            f"Show me just {name_col} and {score_col}",
            [name_col, score_col])

    # H. Missing-data prompts.
    for concept in _MISSING_PRONE_CONCEPTS:
        col = resolved.get(concept)
        if not col:
            continue
        add(f"dyn_missing_{_slug(col)}",
            f"Who is missing {col}?", [col])
        if len(out) >= limit:
            break

    # I. Data quality summary is always safe.
    add("dyn_data_quality", "Show me the data quality summary", ())

    # Mode-specific phrasing overlay: keep the same planner intents, but swap
    # the wording the user sees in the suggested buttons.
    if mode == InstitutionMode.PK12:
        for q in out:
            q.text = _replace_words(q.text, {"Professor": "Teacher", "Advisor": "Counselor", "Retention": "Intervention"})
    elif mode == InstitutionMode.COLLEGE:
        for q in out:
            q.text = _replace_words(q.text, {"Teacher": "Professor", "Counselor": "Advisor", "Intervention": "Retention"})
    if role:
        snippets = role_prompt_snippets(role, mode)
        if snippets:
            out.sort(key=lambda q: 0 if any(s.lower() in q.text.lower() for s in snippets[:2]) else 1)

    return out[:limit]


# ---- helpers ---------------------------------------------------------------


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


def _looks_like_catalog(frame: pd.DataFrame | None, column: str) -> bool:
    """True when the column has few enough distinct values for `list every X`
    to be useful. A Name column with one value per row is NOT a catalog.
    Defaults to True when frame is None so column-only callers don't drop
    the entire class-identity loop."""
    if frame is None or column not in frame.columns:
        return True
    series = frame[column].dropna()
    if series.empty:
        return False
    nunique = int(series.nunique())
    total = int(len(series))
    if nunique > 50:
        return False
    if total >= 20 and nunique / total > 0.5:
        return False
    return True


def _name_like_column(columns: list[str]) -> str | None:
    """Return the column the counselor would call "the student's name."""
    # Prefer an exact "Name" before tokens like "First Name" / "Last Name",
    # because the planner's row-output reads more naturally with the full name.
    for column in columns:
        if column.strip().lower() == "name":
            return column
    for column in columns:
        lower = column.lower()
        if "full name" in lower or "student name" in lower:
            return column
    for column in columns:
        if "name" in column.lower() and "first" not in column.lower() and "last" not in column.lower():
            return column
    return None


def _categorical_samples(
    frame: pd.DataFrame, *, exclude: set[str]
) -> list[tuple[str, str]]:
    """Yield (column, sample_value) pairs for low-cardinality text columns.

    Picks the most common non-null value so the suggestion is the slice most
    likely to have rows. Filters out boolean-looking, numeric, name-like, and
    contact-info columns where "Show me <value> students" would be nonsense.
    """
    out: list[tuple[str, str]] = []
    skip_substrings = ("name", "email", "phone", "id", "date", "address", "notes",
                       "first", "last", "birth", "ssn", "zip")
    for column in frame.columns:
        if column in exclude:
            continue
        lower = column.lower()
        if any(token in lower for token in skip_substrings):
            continue
        series = frame[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        ):
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        clean = series.dropna().astype(str).str.strip()
        clean = clean[clean != ""]
        if clean.empty:
            continue
        uniq = clean.unique()
        if len(uniq) == 0 or len(uniq) > _MAX_FILTER_VALUE_CARDINALITY:
            continue
        # Skip boolean / yes-no / status-like categories — those are better
        # surfaced via the at-risk and missing-data prompts.
        normalized = {str(v).strip().lower() for v in uniq}
        if normalized <= {"yes", "no", "true", "false", "y", "n", "0", "1"}:
            continue
        # The "Show me <Senior> students" form would duplicate the Year sniff.
        if normalized <= {"freshman", "sophomore", "junior", "senior"}:
            continue
        most_common = clean.mode()
        sample = str(most_common.iloc[0]) if not most_common.empty else str(uniq[0])
        out.append((column, sample))
    return out


def _sniff_grade_value(frame: pd.DataFrame | None, year_col: str) -> str | None:
    """Pick one real value from the Year/Grade column.

    Prefers "Freshman" for college rosters and a numbered grade for K-12;
    falls back to whatever value has the most rows.
    """
    if frame is None or year_col not in frame.columns:
        return None
    clean = frame[year_col].dropna().astype(str).str.strip()
    clean = clean[clean != ""]
    if clean.empty:
        return None
    values = clean.unique().tolist()
    lower = {v.lower(): v for v in values}
    for preferred in ("freshman", "freshmen"):
        if preferred in lower:
            return lower[preferred]
    # K-12: kindergarten is the youngest grade — prefer it over numeric "1"
    # when both are present.
    for preferred in ("k", "kindergarten"):
        if preferred in lower:
            return lower[preferred]
    # Otherwise, pick the smallest numeric grade we see.
    numbered = []
    for v in values:
        stripped = v.lstrip("0")
        if stripped.isdigit():
            numbered.append((int(stripped), v))
    if numbered:
        numbered.sort()
        return numbered[0][1]
    if "K" in values:
        return "K"
    most_common = clean.mode()
    return str(most_common.iloc[0]) if not most_common.empty else values[0]


def _phrase_grade_filter(value: str) -> str:
    """Turn a Year/Grade value into a natural-language suggestion."""
    norm = value.strip()
    lower = norm.lower()
    if lower in ("freshman", "freshmen"):
        return "Show me freshmen"
    if lower in ("sophomore", "sophomores"):
        return "Show me sophomores"
    if lower in ("junior", "juniors"):
        return "Show me juniors"
    if lower in ("senior", "seniors"):
        return "Show me seniors"
    if lower == "k" or lower == "kindergarten":
        return "Show me kindergarten students"
    stripped = norm.lstrip("0")
    if stripped.isdigit():
        suffix = _ordinal_suffix(int(stripped))
        return f"Show me {stripped}{suffix} graders"
    return f"Show me {norm} students"


def _ordinal_suffix(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _replace_words(text: str, replacements: dict[str, str]) -> str:
    out = text
    for src, dst in replacements.items():
        out = out.replace(src, dst)
        out = out.replace(src.lower(), dst.lower())
    return out
