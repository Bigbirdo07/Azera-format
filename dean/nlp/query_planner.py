"""Ask Mode query planner.

Turns a natural-language question into a structured ask_question query plan that
core.query_engine can execute with pandas. Rule-based first; falls back to the
local Ollama model only when the rules cannot resolve the question. The plan is
metadata only (operation, columns, filters) — no spreadsheet rows are involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from core.schema import _FREE_TEXT_NAME_PATTERNS
from nlp.local_model import plan_query_from_local_model
from nlp.synonym_mapper import (
    load_json,
    load_synonyms_with_learned,
    match_column_by_terms,
    match_column_for_concept,
    match_column_for_concept_with_fallback,
    normalize_text,
)


HIGH = 0.85
MEDIUM = 0.7
LOW = 0.5

# Status values that mean "missing / incomplete" for membership filters.
_MISSING_STATUS_VALUES = ["Missing", "Incomplete", "Not Submitted", ""]

# Concept -> the value-bearing numeric column we'd sum/average. Order matters
# for the bare-comparison fallback: when the user types "5 absences" without
# naming a column, the resolver should find Days Absent before falling
# through to the GPA-range default.
_NUMERIC_CONCEPTS = (
    "balance_due", "gpa", "attendance_rate",
    "days_absent", "days_tardy", "unexcused_absences",
    "sat_math", "sat_ebrw", "sat_total",
    "psat_math", "psat_reading_writing", "psat_total",
    "math_score", "reading_writing_score", "total_score",
)

_COMPARISON_WORDS = {
    "greater_than": [
        "greater than", "more than", "higher than", "larger than",
        "over", "above", "exceeds", "exceeding",
    ],
    "greater_or_equal": ["at least", "or more", "or higher", "or above", "or greater", "and up", "and above", "and over"],
    "less_than": [
        "less than", "lower than", "smaller than", "fewer than",
        "under", "below",
    ],
    "less_or_equal": ["at most", "or less", "or lower", "or below", "or fewer", "and down", "and below", "up to"],
}

# Symbolic comparisons ("gpa >= 2", "gpa < 3") map to operators directly.
# Order matters: two-character ops come first so ">=" doesn't get caught by ">".
_SYMBOL_COMPARISONS: tuple[tuple[str, str], ...] = (
    (">=", "greater_or_equal"),
    ("<=", "less_or_equal"),
    (">", "greater_than"),
    ("<", "less_than"),
)

# Suffix forms — "2.5+", "2.5 plus" — read as ">=" with the number to the left.
_SUFFIX_COMPARISONS: tuple[tuple[str, str], ...] = (
    ("+", "greater_or_equal"),
    (" plus", "greater_or_equal"),
)


@dataclass
class QueryPlanResult:
    query: dict[str, Any]
    confidence: float
    source: str  # "rule" | "local_llm"
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_options: list[str] = field(default_factory=list)
    columns_used: list[str] = field(default_factory=list)


def plan_query(
    *,
    user_request: str,
    selected_sheet: str,
    sheet_columns: dict[str, list[str]],
    use_local_llm: bool = False,
    ollama_model: str = "llama3.2:3b",
    frame=None,
) -> QueryPlanResult:
    columns = sheet_columns.get(selected_sheet, [])
    rule = _rule_plan(user_request, selected_sheet, columns, frame)
    if rule and rule.confidence >= MEDIUM:
        return rule

    if use_local_llm:
        llm = _llm_plan(
            user_request=user_request,
            selected_sheet=selected_sheet,
            sheet_columns=sheet_columns,
            ollama_model=ollama_model,
        )
        if llm and llm.confidence >= LOW:
            return llm

    if rule:
        return rule
    return QueryPlanResult(
        query={},
        confidence=0.0,
        source="rule",
        needs_clarification=True,
        clarification_question="What would you like to know about this sheet, and which column should I use?",
    )


# Rule-based planning ---------------------------------------------------------


def _rule_plan(user_request: str, sheet: str, columns: list[str], frame=None) -> QueryPlanResult | None:
    text = normalize_text(user_request)
    synonyms = load_synonyms_with_learned()
    if (
        ("academic watch" in text or "attendance watch" in text)
        and not _watch_column_available(text, columns)
    ):
        field = "Academic Watch" if "academic watch" in text else "Attendance Watch"
        return QueryPlanResult(
            query={},
            confidence=0.9,
            source="rule",
            needs_clarification=True,
            clarification_question=(
                f"The uploaded workbook does not include {field}, "
                "so I can't answer that from the available records."
            ),
        )

    comparison_query = _detect_cohort_comparison(text, sheet, columns, frame, synonyms)
    if comparison_query:
        return QueryPlanResult(
            query=comparison_query,
            confidence=HIGH,
            source="rule",
            columns_used=list(dict.fromkeys(
                [comparison_query.get("group_by")]
                + [f.get("column") for f in comparison_query.get("filters", [])]
            )),
        )

    ambiguity = _ambiguous_value_clarification(user_request, text, frame, columns)
    if ambiguity:
        return ambiguity

    sort = _detect_sort(text, columns, synonyms)
    sort_column = sort["column"] if sort else None
    group_by = _detect_group_by(text, columns, synonyms, exclude_column=sort_column)
    operation = _detect_operation(text, group_by, has_sort=bool(sort))
    value_column = _detect_value_column(text, columns, synonyms, operation)
    # Pass the original (un-normalized) text so the notes filter can preserve
    # quoted phrases verbatim — normalize_text strips quotes and punctuation.
    filters = _detect_filters(text, columns, synonyms, original_text=user_request)

    # Categorical value filters detected from the actual data ("Accounting" ->
    # Department, "seniors" -> Year), for columns not already filtered.
    if frame is not None:
        from nlp.conversation import detect_value_filters

        already = {f.get("column") for f in filters}
        advisor_filter = _partial_advisor_name_filter(text, frame, columns, synonyms)
        if advisor_filter and advisor_filter["column"] not in already:
            filters.append(advisor_filter)
            already.add(advisor_filter["column"])
        major_filter = _explicit_major_value_filter(text, frame, columns)
        if major_filter and major_filter["column"] not in already:
            filters.append(major_filter)
            already.add(major_filter["column"])
        for condition in detect_value_filters(user_request, frame):
            if _is_assessment_term_value_filter(text, condition):
                continue
            if condition["column"] not in already:
                filters.append(condition)
                already.add(condition["column"])

        # K-12 grade tokens ("5th grade", "grade 5", "5th graders", "K",
        # "kindergarten") against an actual Grade/Year column. Runs after the
        # generic value-filter pass so it can still inject when nothing else
        # matched the column.
        grade_filter = _grade_level_filter(user_request, frame, columns, synonyms)
        if grade_filter and grade_filter["column"] not in already:
            filters.append(grade_filter)
            already.add(grade_filter["column"])

    limit = _detect_limit(text)

    # "top 10 students by GPA" / "bottom 5 by attendance" / "10 lowest GPA
    # students" — a row-preview ranked by a numeric column, NOT a min/max
    # aggregate or a groupby. When this matches we override operation, sort,
    # and limit, and clear group_by if it was inferred from the sort target.
    top_n = _detect_top_n(text, columns, synonyms)
    if top_n is not None:
        top_n_sort, top_n_limit = top_n
        if group_by and _is_group_count_top_n(text):
            operation = "groupby_count"
            sort = {"column": "Count", "direction": top_n_sort.get("direction", "desc")}
            limit = top_n_limit
            value_column = ""
        # Skip when the user explicitly asked for an aggregate verb the engine
        # would honor differently (average / sum / count).
        elif not any(p in text for p in ("average", "avg", "mean ", "sum ", "sum of", "total ")):
            operation = "filtered_preview"
            sort = top_n_sort
            limit = top_n_limit
            value_column = ""
            # "top 10 students by GPA" is row-level; any group_by inferred
            # from "students"/"by GPA" was a misread of the same wording.
            group_by = None

    # "list students by year" / "list seniors by GPA" / "show me students by
    # advisor" — when the verb is list/show and there's no aggregator word,
    # treat "by <column>" as a SORT on the listed rows, not a group_by. Keeps
    # the K-12 / college "list one big roster by grade" workflow intact.
    if (
        group_by
        and not sort
        and re.search(r"\b(?:list|show)\b", text)
        and not any(p in text for p in (
            "how many", "count", "average", "avg", "mean ", "sum ", "sum of",
            "total ", "per ", "for each", "in each",
        ))
    ):
        sort = {"column": group_by, "direction": "asc"}
        group_by = None
        if operation in {"count_rows", "groupby_count"}:
            operation = "filtered_preview"

    # "best/top performing" or "lowest/worst performing" means rank by average
    # GPA. With a category ("which major performs best", "what advisor has the
    # lowest performing students") that is an average GPA by group, sorted by
    # the group's mean — the single top/bottom group is reported.
    # Skip when a top-N row preview already committed — "top 3 students with
    # highest GPA" is asking for 3 students, not the single top major.
    performance = _detect_performance_query(text, columns, synonyms)
    if performance and not (top_n is not None and operation == "filtered_preview"):
        group_column, gpa_column, perf_direction = performance
        if group_column:
            operation = "groupby_average"
            group_by = group_column
            value_column = gpa_column
            # Only set sort+limit when the user didn't already specify a sort —
            # respect explicit "rank by X ascending" etc.
            if not sort:
                sort = {"column": gpa_column, "direction": perf_direction}
            if limit is None:
                limit = 1

    # "how many majors are there" means distinct values of a column, not a row
    # count. Override a plain count with no filters/grouping.
    if operation == "count_rows" and not filters and not group_by:
        unique_column = _detect_count_unique(text, columns, synonyms)
        if unique_column:
            operation = "count_unique"
            value_column = unique_column
    # "show me all teachers that teach Biology" → list distinct teachers,
    # filter by Biology. The count_unique override also fires for filtered_
    # preview when the request names a class-identity entity ("teachers" /
    # "professors" / "departments" / "majors") — filters are kept.
    elif operation == "filtered_preview" and not group_by:
        unique_column = _detect_count_unique(text, columns, synonyms)
        if unique_column:
            if "student" not in text:
                operation = "count_unique"
                value_column = unique_column

    # "list all departments" / "what advisors do we have" — the user wants the
    # DISTINCT VALUES, not the count. Promotes a freshly-classified count_unique
    # back to list_unique when the verb is list/show/what (not "how many").
    list_column = _detect_list_unique(text, columns, synonyms)
    if list_column and operation in {"count_unique", "filtered_preview", "count_rows"} and not group_by:
        operation = "list_unique"
        value_column = list_column

    if (
        operation in {"count_unique", "list_unique"}
        and value_column
        and any(f.get("column") == value_column for f in filters)
        and re.search(r"\b(?:show|list|find|pull)\b", text)
        and not re.search(r"\b(?:represented|different|distinct|unique|how many|number of|count)\b", text)
    ):
        operation = "filtered_preview"
        value_column = ""

    # "what percent of students are on probation" / "share of seniors with
    # 90+ credits" — the same filter set as count_rows, but the answer is a
    # percent of the sheet. Promote count_rows / filtered_preview to
    # percent_rows when the question explicitly asks for a share/percent.
    if _is_percent_question(text):
        if operation in {"count_rows", "filtered_preview"}:
            operation = "percent_rows"
            limit = None
            value_column = ""

    # Column projection: "show me just student name and gpa" → narrow to those
    # columns. Only meaningful for row-level previews, so when the projection
    # detector fires on a request the default rules read as a row count, flip
    # to filtered_preview so the engine can actually narrow columns.
    select_columns = _detect_select_columns(text, columns, synonyms)
    if select_columns:
        if operation in {"count_rows", "count_unique"} and not group_by and not filters:
            operation = "filtered_preview"
            value_column = ""
        if operation != "filtered_preview":
            select_columns = []

    if operation == "duplicate_check" and not value_column:
        value_column = _detect_named_column(text, columns)

    if (
        operation in {"count_rows", "data_quality_summary"}
        and _has_person_cohort_filter(filters)
        and _is_broad_summary_request(text)
    ):
        operation = "cohort_summary"
        value_column = ""
        group_by = None

    # "what is each student's housing status" — a lookup for a field the
    # workbook doesn't have. Say so instead of answering a generic count.
    if operation == "count_rows" and not filters and not group_by and not value_column:
        missing_field = _unavailable_field(text, columns, synonyms)
        if missing_field:
            return QueryPlanResult(
                query={},
                confidence=0.9,
                source="rule",
                needs_clarification=True,
                clarification_question=(
                    f"The uploaded workbook does not include {missing_field}, "
                    "so I can't answer that from the available records."
                ),
            )

    if operation is None:
        return None

    if _asks_for_assessment_benchmark(text) and not filters:
        return QueryPlanResult(
            query={},
            confidence=0.9,
            source="rule",
            needs_clarification=True,
            clarification_question=(
                "I found assessment scores, but I do not see benchmark fields or "
                "configured benchmark thresholds. You can ask 'show SAT Math below "
                "500' or set a benchmark threshold in Risk Settings."
            ),
        )

    filter_mode = _detect_filter_mode(text, filters)

    query: dict[str, Any] = {
        "request_type": "ask_question",
        "operation": operation,
        "sheet": sheet,
        "filters": filters,
        "group_by": group_by or "",
        "value_column": value_column or "",
        "sort": sort,
        "sort_by": "",
        "limit": limit,
        "select_columns": list(select_columns),
        "filter_mode": filter_mode,
        "plain_english_question": user_request.strip(),
        "confidence": 0.0,
    }

    confidence, used = _score(operation, group_by, value_column, filters)
    if select_columns:
        # An explicit "just X and Y" projection is a strong, grounded signal —
        # lift confidence so the planner doesn't drop it into clarify.
        confidence = max(confidence, HIGH)
        used = list(dict.fromkeys(used + list(select_columns)))
    query["confidence"] = confidence
    return QueryPlanResult(
        query=query,
        confidence=confidence,
        source="rule",
        columns_used=used,
    )


def _detect_operation(text: str, group_by: str | None, has_sort: bool = False) -> str | None:
    if "duplicate" in text:
        return "duplicate_check"
    if ("missing" in text or "blank" in text or "empty" in text) and "column" in text:
        return "missing_summary"
    if any(p in text for p in ("summarize", "summarise", "summary of", "data quality summary", "quality summary", "overview", "what looks wrong", "anything wrong")):
        return "data_quality_summary"

    wants_average = any(p in text for p in ("average", "avg", "mean "))
    wants_total = _total_means_sum(text)

    if group_by:
        if wants_average:
            return "groupby_average"
        if wants_total:
            return "groupby_sum"
        return "groupby_count"

    if wants_average:
        return "average_column"
    if wants_total:
        return "sum_column"
    # When the user is sorting ("lowest first"/"highest first"), "lowest"/
    # "highest" describe direction, not a min/max aggregate.
    if not has_sort:
        if any(p in text for p in ("minimum", "lowest", "smallest")):
            return "min_column"
        if any(p in text for p in ("maximum", "highest", "largest")):
            return "max_column"
    if any(p in text for p in ("how many", "number of", "count of", "count ", "are there", "how much")):
        return "count_rows"
    if any(p in text for p in ("show", "list", "who ", "which student", "find")):
        return "filtered_preview"
    return "count_rows"


def _total_means_sum(text: str) -> bool:
    if any(p in text for p in ("sum of", "sum ", "combined")):
        return True
    if "total" not in text:
        return False
    # In assessment phrases, Total is part of the score column name.
    if any(p in text for p in ("sat total", "psat total", "total sat", "total psat")):
        return False
    return bool(re.search(r"\btotal\s+(?:balance|amount|due|owed|absences?|days|credits?)\b", text))


def _detect_group_by(
    text: str, columns: list[str], synonyms: dict[str, Any], exclude_column: str | None = None
) -> str | None:
    group_count_rank = re.search(
        r"\b(?:top|bottom|highest|lowest|most|least|fewest)\s+(?:[0-9]+\s+)?([a-z ]+?)\s+by\s+(?:student\s+)?count\b",
        text,
    )
    if group_count_rank:
        column = _resolve_phrase_to_column(group_count_rank.group(1), columns, synonyms, take=2)
        if column and column != exclude_column:
            return column

    # "under each <X>" / "under which <X>" / "under their <X>" — school-roster
    # phrasing that means "group these by X" (typically teacher/professor).
    under_match = re.search(
        r"\bunder\s+(?:each|which|every|their|those|these|the)\s+([a-z]+(?:\s+[a-z]+)?)",
        text,
    )
    if under_match:
        column = _resolve_phrase_to_column(under_match.group(1), columns, synonyms, take=2)
        if column and column != exclude_column:
            return column

    # "by <category>" / "per <category>" / "for/in each <category>".
    match = re.search(r"\b(?:by|per|for each|in each|for every|in every|grouped by|across)\s+([a-z0-9 ]+)", text)
    if match:
        column = _resolve_phrase_to_column(match.group(1), columns, synonyms, take=3)
        # A "by X" that names the sort column is a sort target, not a grouping.
        if column and column != exclude_column:
            return column

    # Subject superlatives: "which major has the most students",
    # "what advisor has the largest caseload".
    subject = re.search(
        r"\b(?:which|what)\s+([a-z ]+?)\s+(?:has|have|had|with)\b.*"
        r"\b(?:most|fewest|highest|lowest|largest|smallest|biggest|least|greatest|best|worst|top|bottom)\b",
        text,
    )
    if subject:
        column = _resolve_phrase_to_column(subject.group(1), columns, synonyms, take=2, from_end=True)
        if column:
            return column

    # "top N <category>" / "top <category>".
    top = re.search(r"\btop\s+(?:[0-9]+\s+)?([a-z ]+?)(?:\s+by\b|\s+with\b|\s+in\b|$)", text)
    if top:
        column = _resolve_phrase_to_column(top.group(1), columns, synonyms, take=2)
        if column:
            return column
    return None


def _resolve_phrase_to_column(
    phrase: str, columns: list[str], synonyms: dict[str, Any], take: int = 3, from_end: bool = False
) -> str | None:
    tokens = phrase.strip().split()
    if from_end:
        tokens = tokens[-take:]
    candidates = []
    for length in range(min(len(tokens), take), 0, -1):
        candidates.append(" ".join(tokens[:length]))
        candidates.append(" ".join(tokens[-length:]))
    for candidate in candidates:
        if normalize_text(candidate) in {"class", "classes"}:
            col, s = match_column_for_concept("year", columns, synonyms)
            if col and s >= 0.55:
                return col
        column, score = match_column_by_terms([candidate, _singularize(candidate)], columns)
        if column and score >= 0.6:
            return column
        concept, _ = _concept_for_phrase(candidate, synonyms)
        if concept:
            col, s, _fallback_from = match_column_for_concept_with_fallback(concept, columns, synonyms)
            if col and s >= 0.55:
                return col
    return None


_PERFORMANCE_PHRASES_DESC = (
    # "best/top/highest" → rank by GPA descending (top performers first)
    "best performing",
    "top performing",
    "highest performing",
    "best performer",
    "top performer",
    "performs best",
    "perform best",
    "performing best",
    "best gpa",
    "best average gpa",
    "highest gpa",
    "highest average gpa",
    "best grades",
    "highest grades",
    "strongest students",
    "academically strongest",
)
_PERFORMANCE_PHRASES_ASC = (
    # "lowest/worst/weakest" → rank by GPA ascending (bottom performers first)
    "lowest performing",
    "worst performing",
    "poorly performing",
    "poor performing",
    "weakest performing",
    "lowest performer",
    "worst performer",
    "performs worst",
    "perform worst",
    "performing worst",
    "performing poorly",
    "lowest gpa",
    "lowest average gpa",
    "worst gpa",
    "worst average gpa",
    "lowest grades",
    "worst grades",
    "weakest students",
    "struggling students",
    "academically weakest",
    "academically struggling",
    "lower gpa",
    "lower grades",
)
_PERFORMANCE_PHRASES = _PERFORMANCE_PHRASES_DESC + _PERFORMANCE_PHRASES_ASC


def _detect_performance_query(
    text: str, columns: list[str], synonyms: dict[str, Any]
) -> tuple[str | None, str, str] | None:
    """Recognize best/top/lowest/worst performing questions and map them to GPA.

    Returns (group_column_or_None, gpa_column, direction) where direction is
    "asc" for lowest/worst phrasings and "desc" for best/top. Returns None if
    not a performance question or GPA isn't available.
    """
    # If the predicate detector already classified this as a student-level
    # filter ("not performing well" / "low performing" / etc.), the question
    # is about a STUDENT subset, not a TEACHER/MAJOR ranking. Defer.
    if any(phrase in text for phrase in _PERFORMANCE_PREDICATE_PHRASES):
        return None
    if _mentions_explicit_non_gpa_metric(text, synonyms):
        return None
    asc_hit = any(phrase in text for phrase in _PERFORMANCE_PHRASES_ASC)
    desc_hit = any(phrase in text for phrase in _PERFORMANCE_PHRASES_DESC)
    bare_perf = ("performing" in text or "performance" in text)
    if not (asc_hit or desc_hit or bare_perf):
        return None
    gpa_column, score = match_column_for_concept("gpa", columns, synonyms)
    if not gpa_column or score < 0.55:
        return None
    # The GPA column is the ranking value, not a grouping candidate — exclude
    # it from both detectors so phrasings like "which major has the worst gpa"
    # don't grab GPA from the "the worst gpa" tail of the sentence before the
    # "which major" qualifier is considered.
    group_column = (
        _detect_group_by(text, columns, synonyms, exclude_column=gpa_column)
        or _detect_category_after_quantifier(text, columns, synonyms, exclude_column=gpa_column)
    )
    # Bare "performing"/"performance" with no qualifier defaults to descending
    # (the historical behavior) — most "X performing" questions mean "top X."
    direction = "asc" if asc_hit and not desc_hit else "desc"
    return (group_column, gpa_column, direction)


def _mentions_explicit_non_gpa_metric(text: str, synonyms: dict[str, Any]) -> bool:
    for concept in _NUMERIC_CONCEPTS:
        if concept == "gpa":
            continue
        terms = [concept.replace("_", " "), *synonyms.get(concept, [])]
        if any(_term_in_text(term, text) for term in terms):
            return True
    return False


def _detect_category_after_quantifier(
    text: str, columns: list[str], synonyms: dict[str, Any],
    exclude_column: str | None = None,
) -> str | None:
    """Resolve the category being ranked in a performance question, e.g.
    'those 29 majors', 'which major', 'best performing advisor'.

    ``exclude_column`` skips resolutions that point at a column we don't want
    as the group (typically the value column being ranked, e.g. GPA — grouping
    by the value column is never the right read).
    """
    patterns = (
        r"\b(?:highest|best|top|lowest|worst|weakest)\s+(?:well\s+)?performing\s+([a-z]+(?:\s+[a-z]+)?)",
        r"\b(?:best|top|highest|lowest|worst|weakest)\s+performing\s+([a-z]+(?:\s+[a-z]+)?)",
        r"\b(?:well\s+)?performing\s+([a-z]+)",
        r"\b(?:which|what)\s+([a-z]+(?:\s+[a-z]+)?)",
        r"\b(?:those|these|all)\s+(?:\d+\s+)?([a-z]+(?:\s+[a-z]+)?)",
        r"\bperforming\s+([a-z]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            column = _resolve_phrase_to_column(match.group(1), columns, synonyms, take=2)
            if column and column != exclude_column:
                return column
    return None


# Class-identity concepts: "show me all <X>" reads as "list the distinct X"
# only when X is one of these (it's a class identifier, not a row identifier).
# 'student' / 'name' don't qualify — those stay row-level lists.
_CLASS_IDENTITY_CONCEPTS = (
    "teacher", "advisor", "department", "major", "course", "program",
    "year", "location", "standing", "attendance category", "attendance categories",
)


def _detect_list_unique(
    text: str, columns: list[str], synonyms: dict[str, Any]
) -> str | None:
    """Detect "list every X" / "what X do we have" / "list all X" — return the
    column whose distinct values the user wants to see, or None.

    Requires a list/show/what verb (NOT "how many"); resolved column must be
    a class-identity concept so we don't list every row identifier.
    """
    if not text:
        return None
    if any(p in text for p in ("how many", "number of", "count of", "count ")):
        return None
    patterns = (
        r"\bwhat\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:are\s+)?represented\b",
        r"\bwhich\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:are\s+)?represented\b",
        r"\blist\s+(?:all\s+|every\s+|the\s+)?([a-z]+(?:\s+[a-z]+)?)(?:\s+(?:that|who|with|in|teaching|for|of|are|do)\b|$)",
        r"\blist\s+(?:out\s+)?(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bshow\s+(?:me\s+)?all\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)(?:\s+(?:that|who|with|in|teaching|for|of)\b|$)",
        r"\bshow\s+(?:me\s+)?every\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bshow\s+(?:me\s+)?(?:all\s+|every\s+|the\s+)?([a-z]+(?:\s+[a-z]+)?)\s+(?:there\s+are|we\s+have)\b",
        r"\bwhat\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:do\s+we\s+have|are\s+(?:there|in|listed)|exist)\b",
        r"\bwhat\s+(?:are\s+the|all)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bwhich\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:are\s+(?:there|in|listed)|do\s+we\s+have|exist)\b",
        r"\btell\s+(?:me\s+)?(?:the|every|all)\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bgive\s+me\s+(?:the\s+)?(?:list\s+of\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        phrase = match.group(1).strip()
        if not phrase:
            continue
        # Must look like a class-identity concept — guards against row entities
        # like 'students' or named-column phrases like 'student name'.
        if not any(concept in phrase for concept in _CLASS_IDENTITY_CONCEPTS):
            continue
        column = _resolve_phrase_to_column(phrase, columns, synonyms, take=2, from_end=True)
        if column:
            return column
    return None


def _detect_count_unique(text: str, columns: list[str], synonyms: dict[str, Any]) -> str | None:
    """Detect "how many <category> are there" / "how many distinct X" and map
    the category to a real column. Returns None for entity nouns like 'students'
    that don't name a column (those stay a row count)."""
    patterns = (
        r"\bhow many\s+(?:different|distinct|unique|separate)\s+([a-z ]+)",
        r"\b(?:number|count)\s+of\s+(?:different|distinct|unique)\s+([a-z ]+)",
        r"\bhow many\s+([a-z ]+?)\s+(?:are there|do we have|exist|are listed|are in (?:the )?(?:sheet|file|data))\b",
        r"\bhow many\s+(?:unique|distinct|different)\s+([a-z ]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            phrase = match.group(1)
            if normalize_text(phrase) in {"student", "students", "people", "learners"}:
                return None
            column = _resolve_phrase_to_column(phrase, columns, synonyms, take=2, from_end=True)
            if column:
                return column
    # "show me all <class> that <verb> X" / "list all <class> with X" — common
    # school-roster phrasing for "list the distinct <class>". Only applies to
    # class-identity nouns (teacher/advisor/department/major/course/program).
    list_patterns = (
        r"\b(?:show|list|tell me|give me)(?:\s+me)?(?:\s+all)?\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)\s+(?:that|who|with|in|teaching|for|of)\b",
        r"\b(?:show|list)(?:\s+me)?\s+all\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
    )
    for pattern in list_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        phrase = match.group(1).strip()
        # Must look like a class-identity concept to qualify.
        if not any(concept in phrase for concept in _CLASS_IDENTITY_CONCEPTS):
            continue
        column = _resolve_phrase_to_column(phrase, columns, synonyms, take=2, from_end=True)
        if column:
            return column
    return None


def _unavailable_field(text: str, columns: list[str], synonyms: dict[str, Any]) -> str | None:
    """If the user asks for a specific attribute that maps to no column, return
    a readable field name to report as unavailable; otherwise None."""
    match = re.search(r"\b([a-z]+(?:\s+[a-z]+)?)\s+(status|information|info|details|record)\b", text)
    suffix = ""
    if match:
        suffix = f" {match.group(2)}"
        phrase = match.group(1)
    else:
        match = re.search(r"\bstudent\s*s?\s+([a-z]+(?:\s+[a-z]+)?)\b", text)
        if not match:
            return None
        phrase = match.group(1)
    cleaned = " ".join(word for word in phrase.split() if len(word) > 1 and word not in {"each", "the", "a", "an", "student", "students"})
    if cleaned in {"are there", "do we have", "exist", "listed"}:
        return None
    if not cleaned:
        return None
    if _resolve_phrase_to_column(cleaned, columns, synonyms, take=2, from_end=True):
        return None
    for concept in synonyms:
        if any(normalize_text(term) in cleaned for term in synonyms.get(concept, [])):
            if match_column_for_concept(concept, columns, synonyms)[0]:
                return None
    return f"{cleaned}{suffix}".strip()


def _detect_named_column(text: str, columns: list[str]) -> str | None:
    """Find a column the user named directly (handles simple plurals)."""
    for column in columns:
        normalized = normalize_text(column)
        if normalized in text or _singularize_words(text).find(normalized) >= 0:
            return column
    column, score = match_column_by_terms([text, _singularize(text)], columns)
    return column if column and score >= 0.6 else None


def _term_in_text(term: str, text: str) -> bool:
    normalized = normalize_text(term)
    if not normalized:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in normalized.split()) + r"(?!\w)"
    return bool(re.search(pattern, text))


def _watch_column_available(text: str, columns: list[str]) -> bool:
    wanted = []
    if "academic watch" in text:
        wanted.append("academic watch")
    if "attendance watch" in text:
        wanted.append("attendance watch")
    available = {normalize_text(column) for column in columns}
    return any(column in available for column in wanted)


def _is_group_count_top_n(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "student count", "students count", "by count", "by student count",
            "by students", "most students", "fewest students", "least students",
        )
    )


def _singularize(phrase: str) -> str:
    return " ".join(_singularize_words(word) for word in phrase.split())


def _singularize_words(text: str) -> str:
    return re.sub(r"\b([a-z]+?)s\b", r"\1", text)


def _detect_value_column(
    text: str, columns: list[str], synonyms: dict[str, Any], operation: str | None
) -> str | None:
    if operation not in {"sum_column", "average_column", "min_column", "max_column", "groupby_sum", "groupby_average"}:
        return None
    for column in columns:
        normalized_column = normalize_text(column)
        if normalized_column in {"gpa", "grade point average"} and _term_in_text(normalized_column, text):
            return column
    for concept in _NUMERIC_CONCEPTS:
        if any(_term_in_text(term, text) for term in [concept.replace("_", " "), *synonyms.get(concept, [])]):
            column, score = match_column_for_concept(concept, columns, synonyms)
            if column and score >= 0.55:
                return column
    return None


def _detect_filters(text: str, columns: list[str], synonyms: dict[str, Any],
                    *, original_text: str | None = None) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []

    # Free-text notes search comes first so its tokens ("notes", "mention",
    # "about") don't get reinterpreted by later detectors.
    notes_filter = _notes_filter(text, columns, original_text=original_text or text)
    if notes_filter:
        filters.append(notes_filter)

    # Numeric comparisons like "gpa below 3" / "balance over 0". Multi-
    # clause asks ("gpa below 2.0 AND attendance below 90%") are split on
    # AND/OR so both filters get parsed — but only when the message doesn't
    # contain "between" (whose 'and' is structural, not a connector).
    numeric = None
    seen_numeric_cols: set[str] = set()
    if " between " in f" {text} ":
        chunks = [text]
    else:
        chunks = re.split(r"\s+(?:and|or|;)\s+", text)
    for chunk in chunks:
        chunk_filter = _numeric_filter(chunk, columns, synonyms)
        if chunk_filter and chunk_filter["column"] not in seen_numeric_cols:
            filters.append(chunk_filter)
            seen_numeric_cols.add(chunk_filter["column"])
            if numeric is None:
                numeric = chunk_filter

    # Academic performance predicates: "students not performing well",
    # "struggling students", "students at risk". Default: GPA < 2.0 unless
    # the user already typed a numeric comparison on GPA.
    perf_filters = _performance_predicate_filters(text, columns, synonyms,
                                                  has_numeric=bool(numeric))
    for pf in perf_filters:
        if pf["column"] not in {f.get("column") for f in filters}:
            filters.append(pf)

    # Attendance predicates: "at attendance risk", "poor attendance",
    # "chronic absenteeism". When the workbook has the computed Attendance
    # Risk boolean column, prefer that (it's already the < 90% threshold);
    # otherwise fall back to an Attendance Rate < 90 comparison.
    attendance_filter = _attendance_predicate_filter(
        text, columns, synonyms, has_numeric=bool(numeric),
        existing_columns={f.get("column") for f in filters},
    )
    if attendance_filter:
        filters.append(attendance_filter)

    assessment_filter = _assessment_benchmark_filter(
        text, columns, existing_columns={f.get("column") for f in filters},
    )
    if assessment_filter:
        filters.append(assessment_filter)

    # "high-risk students" → Risk Level == "High Risk" when the combined
    # risk columns are present.
    risk_level_filter = _risk_level_filter(text, columns)
    if risk_level_filter and risk_level_filter["column"] not in {
        f.get("column") for f in filters
    }:
        filters.append(risk_level_filter)

    # Missing FAFSA / financial aid.
    if "fafsa" in text or "fasfa" in text or "financial aid" in text:
        column, score = match_column_for_concept("fafsa_status", columns, synonyms)
        if column and score >= 0.55 and _has_missing_language(text):
            filters.append({"column": column, "operator": "in", "value": list(_MISSING_STATUS_VALUES)})

    # No advisor / missing advisor.
    if ("no advisor" in text or "without advisor" in text or "missing advisor" in text or "no adviser" in text):
        column, score = match_column_for_concept("advisor", columns, synonyms)
        if column and score >= 0.55:
            filters.append({"column": column, "operator": "is_missing"})

    # Owe money / unpaid balance (> 0).
    if any(_term_in_text(p, text) for p in ("owe", "unpaid", "balance due", "owes money", "outstanding")):
        column, score = match_column_for_concept("balance_due", columns, synonyms)
        if column and score >= 0.55 and not numeric:
            filters.append({"column": column, "operator": "greater_than", "value": 0})

    # Generic "missing / no / without / blank / empty <column>" for any column
    # not already filtered above (e.g. "students missing a Second Major").
    already = {f["column"] for f in filters}
    for column in columns:
        if column in already:
            continue
        normalized = normalize_text(column)
        if any(
            f"{keyword} {article}{normalized}" in text
            for keyword in ("missing", "no", "without", "blank", "empty", "have no", "has no")
            for article in ("", "a ", "an ", "the ")
        ):
            filters.append({"column": column, "operator": "is_missing"})

    already = {f["column"] for f in filters}
    for column in columns:
        if column in already:
            continue
        normalized = normalize_text(column)
        if any(
            f"{keyword} {article}{normalized}" in text
            for keyword in ("have", "has", "with")
            for article in ("", "a ", "an ", "the ")
        ):
            filters.append({"column": column, "operator": "is_not_missing"})

    return filters


def _detect_cohort_comparison(
    text: str,
    sheet: str,
    columns: list[str],
    frame,
    synonyms: dict[str, Any],
) -> dict[str, Any] | None:
    if frame is None or not any(word in text for word in ("compare", "versus", " vs ", "against")):
        return None
    advisor_column, score = match_column_for_concept("advisor", columns, synonyms)
    if not advisor_column or score < 0.55 or advisor_column not in frame.columns:
        return None
    values = _matching_person_values(text, frame[advisor_column])
    if len(values) < 2:
        return None
    return {
        "request_type": "ask_question",
        "operation": "cohort_comparison",
        "sheet": sheet,
        "filters": [{"column": advisor_column, "operator": "in", "value": values}],
        "group_by": advisor_column,
        "value_column": "",
        "sort": None,
        "sort_by": "",
        "limit": None,
        "select_columns": [],
        "filter_mode": "all",
        "plain_english_question": text,
        "confidence": HIGH,
    }


def _matching_person_values(text: str, series) -> list[str]:
    ignored = {"dr", "prof", "professor", "mr", "mrs", "ms", "miss", "students", "student"}
    # (value, surname, is_strong) where strong == both surname and a first name
    # are present in the text. The surname is the strong identifier: matching on
    # a first name alone over-expands the set (e.g. "Victor Ford" pulling in
    # "Victor Chen").
    candidates: list[tuple[str, str, bool]] = []
    for value in series.dropna().astype(str).unique():
        normalized = normalize_text(value)
        tokens = [token for token in normalized.split() if len(token) >= 3 and token not in ignored]
        surname = tokens[-1] if tokens else ""
        surname_match = bool(surname) and _term_in_text(surname, text)
        full_match = bool(normalized) and _term_in_text(normalized, text)
        if not (surname_match or full_match):
            continue
        first_name_match = any(_term_in_text(token, text) for token in tokens[:-1])
        candidates.append((value, surname, surname_match and first_name_match))
    # When a fully-specified name pins a surname (e.g. "Victor Ford"), drop the
    # surname-only siblings sharing it (e.g. "Anna Ford"). A bare surname in the
    # query keeps every person with that surname.
    pinned_surnames = {surname for _, surname, strong in candidates if strong}
    matches = [
        value
        for value, surname, strong in candidates
        if strong or surname not in pinned_surnames
    ]
    return list(dict.fromkeys(matches))


def _ambiguous_value_clarification(
    original_request: str,
    text: str,
    frame,
    columns: list[str],
) -> QueryPlanResult | None:
    if frame is None:
        return None
    if any(word in text for word in ("compare", "majoring")):
        return None
    if not any(word in text for word in ("student", "students", "people", "rows")):
        return None

    explicit_terms = {
        "Major": ("major", "majors", "majoring"),
        "Discipline": ("discipline", "disciplines", "department", "departments"),
        "Year": ("year", "freshman", "freshmen", "sophomore", "junior", "senior"),
        "Location": ("location", "campus", "online"),
        "Standing": ("standing", "status"),
    }
    explicit_columns = {
        column
        for column, terms in explicit_terms.items()
        if any(_term_in_text(term, text) for term in terms)
    }

    categorical_columns = [
        column
        for column in columns
        if normalize_text(column) in {"major", "discipline"}
        and column not in explicit_columns
        and column in frame.columns
    ]
    if len(categorical_columns) < 2:
        return None

    for first_idx, first_col in enumerate(categorical_columns):
        for value in frame[first_col].dropna().astype(str).unique():
            normalized_value = normalize_text(value)
            if len(normalized_value) < 3 or not _term_in_text(normalized_value, text):
                continue
            matches = [first_col]
            for other_col in categorical_columns[first_idx + 1:]:
                other_values = {normalize_text(v) for v in frame[other_col].dropna().astype(str).unique()}
                if normalized_value in other_values:
                    matches.append(other_col)
            if len(matches) < 2:
                continue
            options = _ambiguity_options(original_request, value, matches[:3])
            return QueryPlanResult(
                query={},
                confidence=HIGH,
                source="rule",
                needs_clarification=True,
                clarification_question=(
                    f"'{value}' appears in more than one field. Which meaning should I use?"
                ),
                clarification_options=options,
            )
    return None


def _ambiguity_options(original_request: str, value: str, columns: list[str]) -> list[str]:
    lowered = normalize_text(original_request)
    wants_count = any(phrase in lowered for phrase in ("how many", "count", "number of"))
    group_match = re.search(r"\b(?:group|break down|count)\b.*\bby\s+([a-z0-9 ]+)$", lowered)
    group_phrase = group_match.group(1).strip() if group_match else ""
    tail = _tail_after_ambiguous_value(original_request, value)
    options: list[str] = []
    for column in columns:
        if normalize_text(column) == "major":
            if group_phrase:
                options.append(f"Group students majoring in {value} by {group_phrase}")
            elif wants_count:
                options.append(f"How many students majoring in {value}{tail}")
            else:
                options.append(f"Show students majoring in {value}{tail}")
        else:
            if group_phrase:
                options.append(f"Group students where {column} is {value} by {group_phrase}")
            elif wants_count:
                options.append(f"How many students where {column} is {value}{tail}")
            else:
                options.append(f"Show students where {column} is {value}{tail}")
    return options


def _tail_after_ambiguous_value(original_request: str, value: str) -> str:
    match = re.search(re.escape(value), original_request, flags=re.IGNORECASE)
    if not match:
        return ""
    tail = original_request[match.end():].strip()
    tail = re.sub(r"^(?:students?|people|rows?)\b", "", tail, flags=re.IGNORECASE).strip()
    if not tail or tail.lower().startswith("by "):
        return ""
    return f" {tail}"


def _partial_advisor_name_filter(
    text: str,
    frame,
    columns: list[str],
    synonyms: dict[str, Any],
) -> dict[str, Any] | None:
    if not text or frame is None:
        return None
    advisor_column, score = match_column_for_concept("advisor", columns, synonyms)
    if not advisor_column or score < 0.55 or advisor_column not in frame.columns:
        return None
    if not any(word in text for word in ("student", "students", "advisee", "advisees", "advisor", "adviser")):
        return None

    ignored = {"dr", "prof", "professor", "mr", "mrs", "ms", "miss", "students", "student"}
    matches: list[str] = []
    for value in frame[advisor_column].dropna().astype(str).unique():
        normalized = normalize_text(value)
        tokens = [token for token in normalized.split() if len(token) >= 3 and token not in ignored]
        if not tokens:
            continue
        if any(_term_in_text(token, text) for token in tokens) or _term_in_text(normalized, text):
            matches.append(value)

    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return {"column": advisor_column, "operator": "equals", "value": unique_matches[0]}
    return None


def _explicit_major_value_filter(text: str, frame, columns: list[str]) -> dict[str, Any] | None:
    if not text or frame is None:
        return None
    if not any(_term_in_text(term, text) for term in ("major", "majors", "majoring")):
        return None
    major_column = next((column for column in columns if normalize_text(column) == "major"), None)
    if not major_column or major_column not in frame.columns:
        return None
    matches = []
    for value in frame[major_column].dropna().astype(str).unique():
        normalized_value = normalize_text(value)
        if len(normalized_value) >= 3 and _term_in_text(normalized_value, text):
            matches.append(value)
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return {"column": major_column, "operator": "equals", "value": unique_matches[0]}
    return None


def _has_person_cohort_filter(filters: list[dict[str, Any]]) -> bool:
    person_columns = {"advisor", "adviser", "teacher", "professor", "instructor", "counselor"}
    for condition in filters:
        column = normalize_text(str(condition.get("column") or ""))
        if column in person_columns:
            return True
    return False


def _is_broad_summary_request(text: str) -> bool:
    return any(
        phrase in text
        for phrase in (
            "summary", "summarize", "summarise", "overview",
            "what can you say", "tell me about",
        )
    )


# Phrases that mean "student is performing below standard" — predicate form
# (a filter on student rows), not the group-ranking form handled by
# _detect_performance_query. Default threshold is GPA < 2.0.
_PERFORMANCE_PREDICATE_PHRASES = (
    "not performing well", "not performing", "performing poorly",
    "performing badly", "low performing", "poor performing",
    "needs academic watch", "need academic watch",
    "academically struggling",
    "low gpa", "low gpas", "low grades", "low grade",
    "poor gpa", "poor grades",
)
# Subset that also implies the Academic Standing axis when a standing column
# is available. Kept narrow so the existing vague-term resolver still owns
# "at risk" / "struggling" / "needs attention" (which it handles with the
# assumption_note + alternatives UX).
_PERFORMANCE_PREDICATE_STANDING_PHRASES = (
    "academic watch", "needs academic watch", "need academic watch",
    "academic intervention", "needs intervention",
)
_AT_RISK_STANDING_VALUES = ("Probation", "Warning", "At Risk")
_DEFAULT_GPA_THRESHOLD = 2.0


_ATTENDANCE_PREDICATE_PHRASES = (
    "at attendance risk", "attendance risk", "attendance concern",
    "poor attendance", "low attendance", "bad attendance",
    "need attendance support", "needs attendance support",
    "needing attendance support", "students needing attendance support",
    "chronic absenteeism", "chronically absent",
    "missing too much school", "missing a lot of school",
)
_ATTENDANCE_DEFAULT_THRESHOLD = 90.0
_BELOW_BENCHMARK_VALUES = [
    "Below Benchmark",
    "Did Not Meet",
    "Did Not Meet Benchmark",
    "Not Met",
    "Below",
]


def _attendance_predicate_filter(
    text: str, columns: list[str], synonyms: dict[str, Any],
    *, has_numeric: bool, existing_columns: set[str],
) -> dict[str, Any] | None:
    """Return an Attendance Risk filter when the message implies one.

    Prefers the computed Attendance Risk boolean column when present
    (deterministic threshold lives in core.attendance). Falls back to a
    numeric Attendance Rate < 90 comparison when only the rate column is
    available. Skips when a numeric attendance comparison was already
    parsed — that's the user's explicit threshold and wins.
    """
    if not text:
        return None
    if not any(phrase in text for phrase in _ATTENDANCE_PREDICATE_PHRASES):
        return None
    category_col = _find_named_column(columns, ("attendance category",))
    if category_col and category_col not in existing_columns:
        return {
            "column": category_col,
            "operator": "equals",
            "value": "Needs Attendance Support",
        }
    risk_col = _find_named_column(columns, ("attendance risk",))
    if risk_col and risk_col not in existing_columns and not has_numeric:
        return {"column": risk_col, "operator": "equals", "value": True}
    rate_col = _find_named_column(columns, ("attendance rate",))
    if rate_col and rate_col not in existing_columns and not has_numeric:
        return {
            "column": rate_col,
            "operator": "less_than",
            "value": _ATTENDANCE_DEFAULT_THRESHOLD,
        }
    return None


def _risk_level_filter(text: str, columns: list[str]) -> dict[str, Any] | None:
    """Map "high-risk students" / "moderate-risk" to the Risk Level column."""
    if not text:
        return None
    level_col = _find_named_column(columns, ("risk level",))
    if not level_col:
        return None
    if "high risk" in text or "high-risk" in text:
        return {"column": level_col, "operator": "equals", "value": "High Risk"}
    if "moderate risk" in text or "moderate-risk" in text:
        return {"column": level_col, "operator": "equals", "value": "Moderate Risk"}
    if "low risk" in text or "low-risk" in text:
        return {"column": level_col, "operator": "equals", "value": "Low Risk"}
    return None


def _asks_for_assessment_benchmark(text: str) -> bool:
    return "benchmark" in text and any(
        token in text for token in ("sat", "psat", "assessment", "benchmark")
    )


def _is_assessment_term_value_filter(text: str, condition: dict[str, Any]) -> bool:
    if not any(token in text for token in ("sat", "psat", "benchmark", "assessment")):
        return False
    value = normalize_text(str(condition.get("value") or ""))
    return value in {"math", "reading", "verbal", "ebrw", "benchmark", "sat", "psat"}


def _assessment_benchmark_filter(
    text: str,
    columns: list[str],
    *,
    existing_columns: set[str],
) -> dict[str, Any] | None:
    if not _asks_for_assessment_benchmark(text):
        return None
    wants_math = "math" in text
    wants_reading = "reading" in text or "ebrw" in text or "verbal" in text
    if wants_math:
        col = _find_named_column(columns, (
            "math benchmark met", "sat math benchmark", "psat math benchmark",
        ))
        if col and col not in existing_columns:
            return {"column": col, "operator": "equals", "value": False}
    if wants_reading:
        col = _find_named_column(columns, (
            "reading benchmark met", "ebrw benchmark",
            "sat reading benchmark", "psat reading benchmark",
        ))
        if col and col not in existing_columns:
            return {"column": col, "operator": "equals", "value": False}
    risk_col = _find_named_column(columns, ("assessment risk",))
    if risk_col and risk_col not in existing_columns:
        return {"column": risk_col, "operator": "equals", "value": True}
    status_col = _find_named_column(columns, (
        "benchmark status", "college readiness benchmark", "readiness status",
    ))
    if status_col and status_col not in existing_columns:
        return {"column": status_col, "operator": "in", "value": _BELOW_BENCHMARK_VALUES}
    return None


def _find_named_column(columns: list[str], variants: tuple[str, ...]) -> str | None:
    """Exact-or-loose match of one of the variants to an actual column name."""
    norm_map = {normalize_text(c): c for c in columns}
    for variant in variants:
        if variant in norm_map:
            return norm_map[variant]
    for variant in variants:
        for norm, original in norm_map.items():
            if variant in norm:
                return original
    return None


def _performance_predicate_filters(
    text: str, columns: list[str], synonyms: dict[str, Any],
    *, has_numeric: bool,
) -> list[dict[str, Any]]:
    """Return GPA / Academic Standing filters implied by performance phrases.

    Honors three signals:
      - "based on gpa" / "by gpa" → GPA only, never standing.
      - Bare "at risk" / "on watch" with a Standing column → standing only.
      - Everything else: GPA < 2.0 (with optional standing OR-conjunct that we
        cannot yet express, so we emit GPA-only and leave the suggester to
        offer the standing alternative).
    """
    matched = any(phrase in text for phrase in _PERFORMANCE_PREDICATE_PHRASES)
    if not matched:
        return []
    # If the user has already given an explicit GPA comparison, don't override.
    if has_numeric:
        return []

    gpa_col, gpa_score = match_column_for_concept("gpa", columns, synonyms)
    standing_col, standing_score = match_column_for_concept("academic_status", columns, synonyms)

    standing_focus = any(p in text for p in _PERFORMANCE_PREDICATE_STANDING_PHRASES)
    gpa_focus = "based on gpa" in text or "by gpa" in text or "via gpa" in text

    out: list[dict[str, Any]] = []
    if standing_focus and not gpa_focus and standing_col and standing_score >= 0.55:
        out.append({"column": standing_col, "operator": "in",
                    "value": list(_AT_RISK_STANDING_VALUES)})
        return out

    if gpa_col and gpa_score >= 0.55:
        out.append({"column": gpa_col, "operator": "less_than",
                    "value": _DEFAULT_GPA_THRESHOLD})
    return out


# Verbs that introduce a note-search predicate after the column reference.
# "mention" / "contain" / "say" are positive; "about" / "regarding" are loose
# but still resolve to contains_text.
_NOTES_POSITIVE_VERBS = (
    "mentioning", "mentions", "mention", "containing", "contains", "contain",
    "saying", "says", "say", "regarding", "about", "with",
)
# Phrases that flip the search to "does NOT contain".
_NOTES_NEGATIVE_PREFIXES = (
    "do not mention", "don't mention", "does not mention", "doesn't mention",
    "do not contain", "don't contain", "does not contain", "doesn't contain",
    "not mentioning", "not containing", "without mention of", "without",
)
# Phrases meaning the note field is blank.
_NOTES_BLANK_PHRASES = (
    "no notes", "no advisor notes", "no comments", "have no notes",
    "has no notes", "missing notes", "blank notes", "empty notes",
    "without notes", "no note",
)
_GENERIC_NOTES_TERMS = (
    "notes", "note", "comments", "comment", "remarks", "remark", "memo",
    "case notes", "advisor notes", "advising notes", "counselor notes",
)


def _find_notes_column(text: str, columns: list[str]) -> str | None:
    """Pick the free-text column the user is asking about.

    Prefers a longer column-name match in the message (so 'advisor notes
    mention X' beats a bare 'Notes' column), then any column whose normalized
    name contains a free-text pattern (Notes, Comments, ...).
    """
    normalized_text = " " + text + " "
    text_match = None
    for column in sorted(columns, key=lambda c: -len(c)):
        if f" {normalize_text(column)} " in normalized_text:
            if any(p in normalize_text(column) for p in _FREE_TEXT_NAME_PATTERNS):
                text_match = column
                break
    if text_match:
        return text_match
    # Generic mention of notes/comments -> use the first free-text column.
    # Broad words like "summary" or "reason" should not trigger a text search
    # unless the user names the actual column.
    if any(p in text for p in _GENERIC_NOTES_TERMS):
        for column in columns:
            if any(p in normalize_text(column) for p in _FREE_TEXT_NAME_PATTERNS):
                return column
    return None


def _extract_quoted_phrase(text: str) -> str | None:
    """Return the first quoted substring (any of " ' “ ” ‘ ’) or None."""
    match = re.search(r'["“”]([^"“”]+)["“”]', text)
    if match:
        return match.group(1).strip()
    match = re.search(r"['‘’]([^'‘’]+)['‘’]", text)
    if match:
        return match.group(1).strip()
    return None


def _extract_search_term(text: str, verb: str) -> str | None:
    """Extract what the user is searching FOR after the verb.

    Stops at common clause boundaries so 'notes mention attendance issues and
    GPA' yields 'attendance issues' rather than the whole tail.
    """
    pattern = rf"\b{re.escape(verb)}\b\s+(.+)$"
    match = re.search(pattern, text)
    if not match:
        return None
    tail = match.group(1).strip()
    # Stop the search term at a natural clause boundary.
    for stop in (" and gpa", " and grade", " and major", " and advisor",
                 " grouped by ", " group by ", " sorted by ", " sort by ",
                 " limit ", " then ", " also "):
        index = tail.find(stop)
        if index > 0:
            tail = tail[:index]
    tail = tail.strip().strip(".,;:!?")
    if len(tail) > 80:
        tail = tail[:80].rsplit(" ", 1)[0]
    return tail or None


def _notes_filter(text: str, columns: list[str], *, original_text: str) -> dict[str, Any] | None:
    """Detect notes-search predicates and return a filter dict, or None.

    Recognized shapes:
      "notes mention X"            → contains_text X
      "notes about X"              → contains_text X
      "notes do not mention X"     → not_contains_text X
      "no notes" / "missing notes" → is_blank
      Quoted phrases preserved verbatim from the original (un-normalized) text.
    """
    column = _find_notes_column(text, columns)
    if not column:
        return None

    # Blank checks — try these first; they're cheap and unambiguous.
    for phrase in _NOTES_BLANK_PHRASES:
        if phrase in text:
            return {"column": column, "operator": "is_blank"}

    # Quoted phrase from the ORIGINAL request — preserves case and spaces.
    quoted = _extract_quoted_phrase(original_text)

    # Decide direction (positive vs. negative). Check negatives first so
    # "do not mention" beats "mention".
    negative = any(prefix in text for prefix in _NOTES_NEGATIVE_PREFIXES)
    operator = "not_contains_text" if negative else "contains_text"

    if quoted:
        return {"column": column, "operator": operator, "value": quoted}

    # No quoted phrase — extract the bare term after the search verb.
    for verb in _NOTES_POSITIVE_VERBS:
        if f" {verb} " not in text and not text.startswith(f"{verb} ") and not text.endswith(f" {verb}"):
            continue
        term = _extract_search_term(text, verb)
        if term and len(term) >= 2:
            return {"column": column, "operator": operator, "value": term}

    return None


def _numeric_filter(text: str, columns: list[str], synonyms: dict[str, Any]) -> dict[str, Any] | None:
    # 0. Range / between: "between 2.0 and 2.5", "from 30 to 60", "30 to 60
    # credits", "gpa 2.0-2.5". Tried first so a two-number range wins over
    # the single-comparison patterns below.
    range_filter = _between_filter(text, columns, synonyms)
    if range_filter:
        return range_filter

    # 1. Word-form comparisons: "above 2.0", "less than 3", "or higher", etc.
    # Longer phrases first so "or above" (>=) beats bare "above" (>) inside the
    # same sentence; ditto for the _or_equal operators.
    ordered = sorted(
        ((operator, word) for operator, words in _COMPARISON_WORDS.items() for word in words),
        key=lambda pair: (-len(pair[1]), pair[0]),
    )
    for operator, word in ordered:
        # Optional article ("a"/"an"/"the") between the operator word and the
        # number so "above a 2.0 gpa" parses the same as "above 2.0 gpa".
        match = re.search(rf"([a-z0-9 ]*?)\s*{re.escape(word)}\s+(?:a |an |the )?([0-9]+(?:\.[0-9]+)?)([a-z0-9 ]*)", text)
        if not match:
            # Some operators appear AFTER the number with an optional column
            # name in between: "2 gpa or higher", "3 and up", "2 or above".
            match = re.search(
                rf"([a-z0-9 ]*?)([0-9]+(?:\.[0-9]+)?)\s+([a-z ]*?)\s*{re.escape(word)}([a-z0-9 ]*)",
                text,
            )
            if not match:
                continue
            before_text = match.group(1)
            number_text = match.group(2)
            # Fold the middle group (column hint) into 'after' so column
            # resolution searches both sides of the number.
            after_text = (match.group(3) + " " + match.group(4)).strip()
        else:
            before_text, number_text, after_text = match.group(1), match.group(2), match.group(3)
        resolved = _resolve_comparison(before_text, number_text, after_text, columns, synonyms, operator)
        if resolved:
            return resolved

    # 2. Symbol comparisons: "gpa >= 2", "gpa < 3".
    for symbol, operator in _SYMBOL_COMPARISONS:
        match = re.search(rf"([a-z0-9 ]*?){re.escape(symbol)}\s*([0-9]+(?:\.[0-9]+)?)([a-z0-9 ]*)", text)
        if match:
            resolved = _resolve_comparison(
                match.group(1), match.group(2), match.group(3), columns, synonyms, operator
            )
            if resolved:
                return resolved

    # 3. Suffix comparisons: "2.5+ gpa".
    for suffix, operator in _SUFFIX_COMPARISONS:
        match = re.search(rf"([a-z0-9 ]*?)([0-9]+(?:\.[0-9]+)?){re.escape(suffix)}([a-z0-9 ]*)", text)
        if match:
            resolved = _resolve_comparison(
                match.group(1), match.group(2), match.group(3), columns, synonyms, operator
            )
            if resolved:
                return resolved

    return None


_RANGE_PATTERNS: tuple[str, ...] = (
    # "between 2.0 and 2.5" with optional column hints on either side.
    r"([a-z0-9 ]*?)\bbetween\s+([0-9]+(?:\.[0-9]+)?)\s+(?:and|to|-)\s+([0-9]+(?:\.[0-9]+)?)([a-z0-9 ]*)",
    # "from 30 to 60", "from 2.0 to 2.5".
    r"([a-z0-9 ]*?)\bfrom\s+([0-9]+(?:\.[0-9]+)?)\s+to\s+([0-9]+(?:\.[0-9]+)?)([a-z0-9 ]*)",
    # "30 to 60 credits", "2.0 to 2.5 gpa". Requires whitespace around 'to'
    # so we don't match arbitrary "X to Y" prose ("show me 5 to 10 students").
    r"([a-z0-9 ]*?)\b([0-9]+(?:\.[0-9]+)?)\s+to\s+([0-9]+(?:\.[0-9]+)?)([a-z0-9 ]*)",
    # "gpa 2.0-2.5" / "credits 30-60". The hyphen form is brittle (a date or
    # a roster code can look the same), so the resolved column must score
    # comfortably above the comparison threshold.
    r"([a-z0-9 ]*?)([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)([a-z0-9 ]*)",
)


# Word-form OR connectors. We only flip filter_mode to "any" when the message
# contains one of these AND the rules produced at least two filters — both are
# needed to avoid mis-flipping bare phrasings like "on probation OR good
# standing" where only one filter resolved.
_OR_CONNECTORS: tuple[str, ...] = (
    " or ", " or, ", "either ", " and/or ",
)


def _detect_filter_mode(text: str, filters: list[dict[str, Any]]) -> str:
    """Return 'any' when the user clearly OR'd two or more filter clauses,
    otherwise 'all' (the AND default)."""
    if not text or len(filters or []) < 2:
        return "all"
    if any(token in text for token in _OR_CONNECTORS):
        return "any"
    return "all"


_PERCENT_PHRASES: tuple[str, ...] = (
    "what percent", "what percentage", "what share", "what fraction",
    "what proportion", "percentage of", "share of", "fraction of",
    "proportion of", "percent of", "what % ",
)


def _is_percent_question(text: str) -> bool:
    if not text:
        return False
    return any(phrase in text for phrase in _PERCENT_PHRASES)


def _between_filter(
    text: str, columns: list[str], synonyms: dict[str, Any]
) -> dict[str, Any] | None:
    """Detect 'between X and Y' / 'from X to Y' / 'X-Y' range filters.

    Returns a single condition with operator='between' and value=[low, high].
    Resolves the target column using the same machinery as the single-
    comparison numeric filter (concept synonyms + direct mentions).
    """
    for pattern in _RANGE_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue
        before_text, low_text, high_text, after_text = match.group(1, 2, 3, 4)
        try:
            low = float(low_text)
            high = float(high_text)
        except ValueError:
            continue
        if low > high:
            low, high = high, low
        column = _column_from_comparison_context(
            before_text, after_text, columns, synonyms,
        )
        if not column:
            continue
        low = _normalize_rate_threshold(column, low, before_text, after_text)
        high = _normalize_rate_threshold(column, high, before_text, after_text)
        return {
            "column": column,
            "operator": "between",
            "value": [low, high],
        }
    return None


_RANGE_RESOLVABLE_CONCEPTS: tuple[str, ...] = (
    *_NUMERIC_CONCEPTS,
    "credits",
)


def _column_from_comparison_context(
    before_text: str,
    after_text: str,
    columns: list[str],
    synonyms: dict[str, Any],
) -> str | None:
    """Shared column-resolution heuristic for numeric filters.

    Looks at words on either side of the number(s) for a numeric concept,
    a direct column-name mention, or any synonym token that resolves to a
    column via match_column_by_terms.
    """
    surround = f"{before_text} {after_text}".strip()
    if not surround:
        return None
    for concept in _RANGE_RESOLVABLE_CONCEPTS:
        terms = [concept.replace("_", " "), *synonyms.get(concept, [])]
        if any(normalize_text(t) in surround for t in terms):
            col, score = match_column_for_concept(concept, columns, synonyms)
            if col and score >= 0.55:
                return col
    # Direct full-name mention either side.
    for column in columns:
        if normalize_text(column) in surround:
            return column
    # Token-wise fuzzy match — handles "credits" → "Credits Completed" where
    # neither full name is a substring of the other.
    column, score = match_column_by_terms(surround.split(), columns)
    if column and score >= 0.6:
        return column
    return None


def _resolve_comparison(
    before_text: str,
    number_text: str,
    after_text: str,
    columns: list[str],
    synonyms: dict[str, Any],
    operator: str,
) -> dict[str, Any] | None:
    number = float(number_text)
    before = " ".join(before_text.strip().split()[-3:])
    after = " ".join(after_text.strip().split()[:3])
    column = _resolve_numeric_column(before, columns, synonyms) or _resolve_numeric_column(after, columns, synonyms)
    # Bare comparisons with no named column default to GPA when the number is
    # in a plausible GPA range — matches the previous behavior.
    if not column and 0.0 <= number <= 5.0:
        gpa_column, gpa_score = match_column_for_concept("gpa", columns, synonyms)
        if gpa_column and gpa_score >= 0.55:
            column = gpa_column
    if column:
        value: float | int = _normalize_rate_threshold(column, number, before_text, after_text)
        if value == int(value):
            value = int(value)
        return {"column": column, "operator": operator, "value": value}
    return None


def _normalize_rate_threshold(
    column: str,
    number: float,
    before_text: str,
    after_text: str,
) -> float:
    context = normalize_text(f"{column} {before_text} {after_text}")
    if "rate" in context and ("attendance" in context or "percent" in context or "percentage" in context):
        if number > 1:
            return number / 100.0
    return number


def _resolve_numeric_column(phrase: str, columns: list[str], synonyms: dict[str, Any]) -> str | None:
    for concept in _NUMERIC_CONCEPTS:
        terms = [concept.replace("_", " "), *synonyms.get(concept, [])]
        if any(normalize_text(t) in phrase for t in terms):
            col, s = match_column_for_concept(concept, columns, synonyms)
            if col and s >= 0.55:
                return col
    column, score = match_column_by_terms([phrase], columns)
    if column and score >= 0.6:
        return column
    return None


def _detect_sort(text: str, columns: list[str], synonyms: dict[str, Any]) -> dict[str, Any] | None:
    """Detect 'sort/order/rank by <column> [ascending|descending|lowest|highest first]'."""
    match = re.search(r"\b(?:sort(?:ed)?|order(?:ed)?|rank(?:ed)?|arrange)\b[a-z ]*?\bby\s+([a-z ]+)", text)
    if not match:
        return None
    tail = match.group(1)
    column = _resolve_phrase_to_column(tail, columns, synonyms, take=2)
    if not column:
        return None
    descending = any(
        cue in text
        for cue in ("descending", "highest first", "high to low", "largest first", "biggest first", "desc")
    )
    ascending = any(
        cue in text
        for cue in ("ascending", "lowest first", "low to high", "smallest first", "asc")
    )
    direction = "desc" if descending and not ascending else "asc"
    return {"column": column, "direction": direction}


def _detect_limit(text: str) -> int | None:
    match = re.search(r"\b(?:top|first|just the top|just|only)\s+([0-9]+)", text)
    if match:
        return int(match.group(1))
    return None


# Direction cues that pair with a row-preview top-N (NOT a min/max aggregate).
_TOP_N_DESC = ("top", "highest", "largest", "best", "greatest", "biggest", "most")
_TOP_N_ASC = ("bottom", "lowest", "smallest", "worst", "least", "weakest", "poorest")


def _detect_top_n(
    text: str, columns: list[str], synonyms: dict[str, Any]
) -> tuple[dict[str, Any], int] | None:
    """Detect "top N students by GPA" / "5 lowest GPA students" / "bottom 10".

    Returns (sort, limit) when the phrasing is asking for N rows ranked by a
    column, or None when the request is not a top-N row preview. The caller
    is responsible for switching operation to filtered_preview and clearing
    any group_by that was inferred from the sort target.
    """
    if not text:
        return None

    # Form A: "top N ... by <column>" / "bottom 5 ... by <column>".
    direction: str | None = None
    n: int | None = None
    sort_phrase: str | None = None

    for cue in _TOP_N_DESC:
        match = re.search(rf"\b{cue}\s+([0-9]+)\b", text)
        if match:
            direction = "desc"
            n = int(match.group(1))
            break
    if direction is None:
        for cue in _TOP_N_ASC:
            match = re.search(rf"\b{cue}\s+([0-9]+)\b", text)
            if match:
                direction = "asc"
                n = int(match.group(1))
                break

    # Form B: "N highest|lowest <column>" (number before the direction word).
    if direction is None:
        for cue in _TOP_N_DESC:
            match = re.search(rf"\b([0-9]+)\s+{cue}\b\s*([a-z ]+?)?(?:\s+(?:student|students|people|rows?)\b|$)", text)
            if match:
                direction = "desc"
                n = int(match.group(1))
                sort_phrase = (match.group(2) or "").strip()
                break
    if direction is None:
        for cue in _TOP_N_ASC:
            match = re.search(rf"\b([0-9]+)\s+{cue}\b\s*([a-z ]+?)?(?:\s+(?:student|students|people|rows?)\b|$)", text)
            if match:
                direction = "asc"
                n = int(match.group(1))
                sort_phrase = (match.group(2) or "").strip()
                break

    if direction is None or n is None:
        return None

    by_match = re.search(r"\bby\s+([a-z ]+?)(?:\s+(?:asc|desc|ascending|descending)\b|$)", text)
    if by_match:
        sort_phrase = by_match.group(1).strip()

    sort_column: str | None = None
    if sort_phrase:
        sort_column = _resolve_phrase_to_column(sort_phrase, columns, synonyms, take=3)

    # If we still don't have a column, fall back to the named numeric concept
    # in the message — "10 lowest GPA students" lands here with sort_phrase="gpa".
    if not sort_column:
        for concept in _NUMERIC_CONCEPTS:
            terms = [concept.replace("_", " "), *synonyms.get(concept, [])]
            if any(normalize_text(t) in text for t in terms):
                col, score = match_column_for_concept(concept, columns, synonyms)
                if col and score >= 0.55:
                    sort_column = col
                    break

    if not sort_column:
        return None

    return {"column": sort_column, "direction": direction}, int(n)


def _concept_for_phrase(phrase: str, synonyms: dict[str, Any]) -> tuple[str | None, float]:
    for concept, terms in synonyms.items():
        for term in terms:
            if normalize_text(term) == normalize_text(phrase):
                return concept, 0.9
    return None, 0.0


# K-12 grade tokens — ordinal words and numbers map to the bare grade label.
_K12_ORDINAL_WORDS: dict[str, str] = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12",
}
_K12_KINDERGARTEN_TOKENS = ("kindergarten", "pre-k", "prek", "pre k", "preschool")


def _grade_level_filter(
    user_request: str, frame, columns: list[str], synonyms: dict[str, Any]
) -> dict[str, Any] | None:
    """Recognise K-12 grade phrases ("5th grade", "grade 5", "5th graders", "K",
    "kindergarten") and emit an equals/contains filter against an actual
    Grade/Year column. Returns None when the request has no grade token, no
    Year-like column exists, or no value in the column resembles the token.
    """
    if frame is None or frame.empty:
        return None

    grade_column, score = match_column_for_concept("year", columns, synonyms)
    if not grade_column or score < 0.55 or grade_column not in frame.columns:
        return None

    raw = user_request.lower()
    norm = normalize_text(user_request)

    candidate: str | None = None

    # "K" / "kindergarten" / "pre-k".
    if any(re.search(rf"\b{re.escape(tok)}\b", norm) for tok in _K12_KINDERGARTEN_TOKENS):
        candidate = "K"
    if candidate is None:
        # Standalone "K" (school code) — guarded so it doesn't capture every "k".
        if re.search(r"\b(?:in\s+|grade\s+|grades?\s+)k\b", norm) or re.search(r"\bk\s+grade(?:rs)?\b", norm):
            candidate = "K"

    # Ordinal-word form: "fifth grade", "twelfth graders".
    if candidate is None:
        for word, digit in _K12_ORDINAL_WORDS.items():
            if re.search(rf"\b{word}\b\s+grade(?:rs)?\b", norm):
                candidate = digit
                break

    # Numeric forms: "5th grade", "5th graders", "grade 5", "grades 5".
    if candidate is None:
        match = (
            re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+grade(?:rs)?\b", norm)
            or re.search(r"\bgrade(?:s)?\s+(\d{1,2})\b", norm)
        )
        if match:
            candidate = match.group(1)

    if candidate is None:
        return None

    # Match against the actual column values (case-insensitive). Prefer exact
    # match; fall back to substring (covers "Grade 5" stored verbatim).
    values = [str(v) for v in frame[grade_column].dropna().unique().tolist()]
    target = candidate.casefold()
    exact = [v for v in values if v.casefold() == target]
    if exact:
        return {"column": grade_column, "operator": "equals", "value": exact[0]}
    contains = [v for v in values if target in v.casefold()]
    if contains:
        if len(contains) == 1:
            return {"column": grade_column, "operator": "equals", "value": contains[0]}
        return {"column": grade_column, "operator": "in", "value": contains}
    # No matching values — emit a contains filter against the raw token so the
    # query at least surfaces an empty result rather than silently dropping the
    # grade clause.
    return {"column": grade_column, "operator": "contains", "value": candidate}


# Explicit projection triggers: the message names "columns" or "fields", so
# we accept even a single resolved token as a projection.
_SELECT_TRIGGERS_EXPLICIT: tuple[str, ...] = (
    r"\bcolumns?\s+of\s+(?:just\s+|only\s+)?(.+?)$",
    r"\bonly\s+the\s+(.+?)\s+columns?\b",
    r"\bjust\s+the\s+(.+?)\s+columns?\b",
    r"\bthe\s+(.+?)\s+columns?\b",
    r"\bcolumns?\s*[:\-]\s*(.+?)$",
    r"\bfields?\s*[:\-]\s*(.+?)$",
)

# Implicit triggers: "just X and Y", "only X and Y", "show me only ...".
# Riskier because "just" / "only" can intro non-projection phrases, so we
# require 2+ resolved tokens AND no aggregation verb to commit.
_SELECT_TRIGGERS_IMPLICIT: tuple[str, ...] = (
    r"\bshow\s+(?:me\s+)?only\s+(.+?)$",
    r"\bonly\s+show\s+(?:me\s+)?(.+?)$",
    r"\bdisplay\s+only\s+(.+?)$",
    r"\bshow\s+(?:me\s+)?just\s+(.+?)$",
    r"\bjust\s+show\s+(?:me\s+)?(.+?)$",
    r"\bwith\s+only\s+(.+?)$",
    r"^\s*just\s+(.+?)$",
    r"^\s*only\s+(.+?)$",
)

# If any of these words appears, the request is asking for an aggregate or
# operation that projection should NOT override. Used to gate implicit
# triggers only — explicit "columns:" wording wins regardless.
_AGGREGATION_GUARD_WORDS: tuple[str, ...] = (
    "count", "how many", "sum", "total", "average", "avg", "mean",
    "min", "max", "minimum", "maximum", "missing", "duplicate",
    "summary", "summarize", "summarise",
)


def _detect_select_columns(
    text: str, columns: list[str], synonyms: dict[str, Any]
) -> list[str]:
    """Detect "just/only X and Y" / "columns of X and Y" projection requests.

    Returns the list of resolved column names in user order. Empty list when
    no trigger fires, the resolved set is too small for an implicit trigger
    to be safe, or the message looks like an aggregate (count/sum/avg/…).
    """
    if not text or not columns:
        return []

    chunk: str | None = None
    explicit = False
    for pattern in _SELECT_TRIGGERS_EXPLICIT:
        match = re.search(pattern, text)
        if match:
            chunk = match.group(1).strip()
            explicit = True
            break
    if chunk is None:
        for pattern in _SELECT_TRIGGERS_IMPLICIT:
            match = re.search(pattern, text)
            if match:
                chunk = match.group(1).strip()
                break
    if not chunk:
        return []

    # Strip trailing filter clauses so "just student and gpa where gpa < 2"
    # doesn't pull "where ..." into the projection list.
    chunk = re.split(
        r"\b(?:where|with\s+\w+\s*(?:[<>=]|below|above|under|over)|below|above|under|over)\b",
        chunk,
        maxsplit=1,
    )[0]
    chunk = chunk.strip().rstrip(".,;:")
    if not chunk:
        return []

    # Implicit triggers refuse to commit when the message is really an aggregate.
    if not explicit and any(word in text for word in _AGGREGATION_GUARD_WORDS):
        return []

    raw_tokens = re.split(r"\s*(?:,|\band\b|&|/)\s*", chunk)
    seen: dict[str, None] = {}
    for token in raw_tokens:
        token = token.strip(" ,.").strip()
        if not token:
            continue
        column = _resolve_phrase_to_column(token, columns, synonyms, take=3)
        if column and column not in seen:
            seen[column] = None
    resolved = list(seen.keys())
    if not resolved:
        return []
    if not explicit and len(resolved) < 2:
        # "just students" alone is too thin to safely treat as a projection.
        return []
    return resolved


def _has_missing_language(text: str) -> bool:
    return any(p in text for p in ("missing", "no ", "without", "incomplete", "not submitted", "blank", "didn", "haven"))


def _score(
    operation: str,
    group_by: str | None,
    value_column: str | None,
    filters: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    used: list[str] = []
    if group_by:
        used.append(group_by)
    if value_column:
        used.append(value_column)
    used.extend(f["column"] for f in filters if f.get("column"))

    # Operations that need a column we actually resolved.
    if operation in {"groupby_count", "groupby_sum", "groupby_average"} and not group_by:
        return LOW - 0.1, used
    if operation in {"sum_column", "average_column", "min_column", "max_column"} and not value_column:
        return LOW - 0.1, used
    if operation in {"missing_summary", "data_quality_summary"}:
        return HIGH, used
    if operation == "duplicate_check":
        return HIGH if used else MEDIUM, used
    if operation == "count_rows":
        return HIGH if filters else MEDIUM, used
    if operation == "filtered_preview":
        return HIGH if filters else MEDIUM, used
    return HIGH, used


# LLM fallback ----------------------------------------------------------------


def _llm_plan(
    *,
    user_request: str,
    selected_sheet: str,
    sheet_columns: dict[str, list[str]],
    ollama_model: str,
) -> QueryPlanResult | None:
    payload, error = plan_query_from_local_model(
        user_request=user_request,
        model_name=ollama_model,
        sheet_names=list(sheet_columns.keys()),
        sheet_columns=sheet_columns,
    )
    if error or not payload:
        return None

    sheet = payload.get("sheet") or selected_sheet
    columns = sheet_columns.get(sheet, [])
    if not _query_columns_valid(payload, columns):
        return None

    payload["sheet"] = sheet
    payload.setdefault("limit", 10)
    confidence = _safe_float(payload.get("confidence"), default=MEDIUM)
    payload["confidence"] = confidence
    used = [f.get("column") for f in payload.get("filters", []) if f.get("column")]
    if payload.get("group_by"):
        used.append(payload["group_by"])
    if payload.get("value_column"):
        used.append(payload["value_column"])
    return QueryPlanResult(query=payload, confidence=confidence, source="local_llm", columns_used=used)


def _query_columns_valid(payload: dict[str, Any], columns: list[str]) -> bool:
    referenced = [f.get("column") for f in payload.get("filters", []) if f.get("column")]
    if payload.get("group_by"):
        referenced.append(payload["group_by"])
    if payload.get("value_column"):
        referenced.append(payload["value_column"])
    return all(column in columns for column in referenced)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
