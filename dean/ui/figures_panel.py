"""Figures panel — inline chart generation for the dashboard layout.

Chart intent is detected with a small keyword rule (no LLM hops) so that
follow-ups like "create a bar chart by department" produce a chart in the
Figures panel without touching the workbook on disk. The user can still
export the chart from the Export Center.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from core.session_memory import SessionMemory
from nlp.synonym_mapper import load_json
from nlp.query_planner import _resolve_phrase_to_column


_CHART_KEYWORDS = (
    "chart", "graph", "plot", "histogram", "visualize", "visualise",
    "figure", "distribution", "pie chart", "bar chart", "bar graph",
)

_CHART_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("pie chart", "pie"),
    ("pie graph", "pie"),
    ("donut", "pie"),
    ("histogram", "histogram"),
    ("bar chart", "bar"),
    ("bar graph", "bar"),
    ("line chart", "line"),
    ("line graph", "line"),
    ("distribution", "histogram"),
)

# "average / mean X by Y" is a breakdown best shown as a bar chart, so it counts
# as a figure request even without the word "chart".
_AGG_BY_PATTERN = re.compile(r"\b(?:average|avg|mean|median)\b[a-z0-9 ]*\b(?:by|per)\b")

# "by X and Y" (two dimensions) can't be a single-axis bar chart -- it's a
# cross-tab, which belongs to the deterministic pivot planner instead
# (nlp.planner_router._parse_pivot_request). Without this, "average GPA by
# advisor and standing" would match _AGG_BY_PATTERN above and get rendered as
# a misleading one-dimension chart before the pivot planner ever sees it.
_TWO_DIMENSION_BREAKDOWN = re.compile(r"\bby\s+[a-z][a-z ]*?\s+(?:and|,)\s+[a-z][a-z ]*")

# Columns that must never be a chart's group/x-axis: grouping a count by a
# near-unique identity or contact field produces a useless (one-bar-per-row)
# chart, which is worse than asking the user to clarify.
_NON_GROUP_COLUMNS = {
    "student id", "id", "first name", "last name", "name", "full name",
    "email", "phone", "date of birth", "dob", "notes",
}

# Chart vocabulary / filler tokens that are never a real column phrase.
_FIELD_STOPWORDS = {
    "a", "an", "the", "chart", "graph", "plot", "figure", "histogram",
    "distribution", "visual", "visualization", "bar", "pie", "line", "of",
    "students", "student", "number", "how", "many", "are",
}


@dataclass
class ChartIntent:
    chart_type: str            # "bar" | "pie" | "histogram" | "line"
    field: str                 # column to group/histogram on
    metric: str = "count"      # "count" | "average"
    value_column: str = ""     # numeric column for average/sum charts


def is_chart_request(text: str) -> bool:
    """Cheap pre-check: does this look like the user wants a chart at all?

    Triggers on an explicit chart word, or an "average/mean X by/per Y"
    breakdown (which a counselor almost always wants to *see* as a bar chart).
    """
    lower = text.lower()
    if "pivot" in lower:
        return False
    if any(keyword in lower for keyword in _CHART_KEYWORDS):
        return True
    if _TWO_DIMENSION_BREAKDOWN.search(lower):
        return False
    return bool(_AGG_BY_PATTERN.search(lower))


def detect_chart_intent(text: str, columns: list[str]) -> ChartIntent | None:
    """Parse a chart request into a concrete (type, field, metric) tuple.

    Returns None when the request is too ambiguous to render confidently — the
    caller should ask the user to clarify the field.
    """
    if not is_chart_request(text):
        return None
    lower = text.lower()
    chart_type = "bar"
    for hint, kind in _CHART_TYPE_HINTS:
        if hint in lower:
            chart_type = kind
            break

    synonyms = load_json("synonyms.json")
    field = _extract_chart_field(lower, columns, synonyms)

    # Histogram defaults to GPA if no field was named — that's the most common
    # "show me the distribution" request on a student workbook.
    if chart_type == "histogram" and not field:
        gpa_match = _resolve_phrase_to_column("gpa", columns, synonyms)
        if gpa_match:
            field = gpa_match

    if not field:
        return None

    metric = "count"
    value_column = ""
    # "average GPA by major" -> bar chart of average(GPA) grouped by major.
    avg_match = re.search(r"\b(?:average|avg|mean)\s+([a-z ]+?)\s+(?:by|per)\s+([a-z ]+)", lower)
    if avg_match:
        value_phrase = avg_match.group(1).strip()
        group_phrase = avg_match.group(2).strip()
        value_resolved = _resolve_phrase_to_column(value_phrase, columns, synonyms)
        group_resolved = _resolve_phrase_to_column(group_phrase, columns, synonyms)
        if not value_resolved or not group_resolved:
            return None
        metric = "average"
        value_column = value_resolved
        field = group_resolved
    else:
        paired = _extract_numeric_category_pair(lower, columns, synonyms)
        if paired:
            value_column, field = paired
            metric = "average"

    return ChartIntent(
        chart_type=chart_type,
        field=field,
        metric=metric,
        value_column=value_column,
    )


# (display label, chart query) candidates in counselor-priority order. Each
# query is phrased so is_chart_request() fires; suggested_figure_questions keeps
# only the ones that resolve to a real chart on the uploaded sheet's columns.
_FIGURE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("GPA distribution", "GPA distribution"),
    ("Average GPA by major", "average GPA by major"),
    ("Students by department", "bar chart of students by department"),
    ("Average GPA by advisor", "average GPA by advisor"),
    ("Students by year", "bar chart of students by year"),
    ("Academic status breakdown", "pie chart of academic status"),
    ("Students by major", "bar chart of students by major"),
    ("Average GPA by department", "average GPA by department"),
    ("Students by financial aid status", "bar chart of students by financial aid status"),
    ("Students by advisor", "bar chart of students by advisor"),
    ("Credits completed distribution", "histogram of credits completed"),
)


def suggested_figure_questions(columns: list[str], limit: int = 5) -> list[tuple[str, str]]:
    """Return up to ``limit`` (label, query) figure suggestions that actually
    resolve to a chart on these columns — so every chip a counselor sees is
    guaranteed to produce a figure when clicked."""
    picks: list[tuple[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for label, query in _FIGURE_CANDIDATES:
        intent = detect_chart_intent(query, columns)
        if intent is None or intent.field not in columns:
            continue
        # An "average …" chip must actually average a real numeric column —
        # otherwise it silently degrades to a count and the label lies.
        wants_average = "average" in query.lower() or "mean" in query.lower()
        if wants_average and (intent.metric != "average" or intent.value_column not in columns):
            continue
        key = (intent.field, intent.metric, intent.chart_type)
        if key in seen:
            continue
        seen.add(key)
        picks.append((label, query))
        if len(picks) >= limit:
            break
    return picks


def _extract_chart_field(text: str, columns: list[str], synonyms: dict[str, Any]) -> str | None:
    """Pull the column the user wants to chart (the group / x-axis) from the
    request text, ignoring chart-vocabulary filler and identity columns that
    never make a useful chart axis."""

    def _resolve(phrase: str) -> str | None:
        tokens = [token for token in phrase.split() if token not in _FIELD_STOPWORDS]
        if not tokens:
            return None
        column = _resolve_phrase_to_column(" ".join(tokens), columns, synonyms, take=3, from_end=True)
        if column and column.lower() not in _NON_GROUP_COLUMNS:
            return column
        return None

    # 1. "each / every / per X" is the strongest group-by signal
    #    ("how many students does each advisor have", "in each major").
    each = re.search(
        r"\b(?:each|every|per)\s+([a-z][a-z ]*?)(?:\s+(?:have|has|had|got|is|are|in|on|with|for|by)\b|[.,?!]|$)",
        text,
    )
    if each:
        column = _resolve(each.group(1).strip())
        if column:
            return column

    # 2. "by/per/of/for X" and "show/chart/plot/visualize X" captures. Use
    #    finditer so "of students by department" can skip the junk first match.
    patterns = (
        r"\b(?:by|per|of|for)\s+([a-z ]+?)(?:\s+(?:in|on|with|for)|[.,?!]|$)",
        r"\b(?:show|chart|plot|graph|visualize|visualise)\s+(?:the\s+)?"
        r"(?:distribution\s+of\s+|number\s+of\s+)?([a-z ]+?)(?:\s+(?:by|in|on|with|for)|[.,?!]|$)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            column = _resolve(match.group(1).strip())
            if column:
                return column

    # 3. A non-identity column named literally in the text.
    for column in columns:
        normalized = column.lower()
        if normalized in _NON_GROUP_COLUMNS:
            continue
        if normalized in text:
            return column

    # 4. Final fallback: resolve the longest noun phrase to a non-identity
    #    column (covers e.g. "good vs bad academic standing" -> Academic Status).
    for phrase in sorted(re.findall(r"[a-z]+(?:\s+[a-z]+)*", text), key=len, reverse=True):
        column = _resolve(phrase)
        if column:
            return column
    return None


def _extract_numeric_category_pair(
    text: str,
    columns: list[str],
    synonyms: dict[str, Any],
) -> tuple[str, str] | None:
    """Resolve "GPA and major" / "major with GPA" as average(GPA) by major."""
    mentioned = _mentioned_columns(text, columns, synonyms)
    if len(mentioned) < 2:
        return None
    numeric = _first_numeric_named_column(mentioned)
    category = next((column for column in mentioned if column != numeric), "")
    if numeric and category:
        return numeric, category
    return None


def _mentioned_columns(text: str, columns: list[str], synonyms: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for column in columns:
        if column.lower() in text:
            found.append(column)
    for phrase in re.findall(r"[a-z][a-z ]+", text):
        column = _resolve_phrase_to_column(phrase.strip(), columns, synonyms, take=3, from_end=True)
        if column and column not in found:
            found.append(column)
    return found


def _first_numeric_named_column(columns: list[str]) -> str:
    for column in columns:
        if any(token in column.lower() for token in ("gpa", "grade", "score", "balance", "amount", "count", "rate")):
            return column
    return ""


def compute_chart(intent: ChartIntent, frame: pd.DataFrame) -> pd.DataFrame | None:
    """Build the small summary DataFrame the chart will visualize.

    Aggregation runs on pandas (deterministic, no LLM); returns None when the
    requested column is missing from the working frame.
    """
    if intent.field not in frame.columns:
        return None

    if intent.chart_type == "histogram":
        series = pd.to_numeric(frame[intent.field], errors="coerce").dropna()
        if series.empty:
            return None
        # Equal-width bins; cap at 12 for a readable axis.
        binned = pd.cut(series, bins=min(12, max(3, int(series.nunique()))), include_lowest=True)
        counts = binned.value_counts().sort_index().reset_index()
        counts.columns = [intent.field, "count"]
        counts[intent.field] = counts[intent.field].astype(str)
        return counts

    if intent.metric == "average" and intent.value_column:
        numeric = pd.to_numeric(frame[intent.value_column], errors="coerce")
        summary = numeric.groupby(frame[intent.field], dropna=False).mean().reset_index()
        summary.columns = [intent.field, f"avg_{intent.value_column}"]
        summary = summary.sort_values(by=summary.columns[1], ascending=False)
        return summary

    # Default: row count grouped by the field.
    summary = (
        frame.groupby(intent.field, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )
    return summary


def build_altair_chart(intent: ChartIntent, summary: pd.DataFrame, title: str) -> alt.Chart:
    """Build an altair chart from a summary DataFrame."""
    if summary.empty:
        return alt.Chart(pd.DataFrame({"x": [0]})).mark_text(text="No data").encode()

    label_col = summary.columns[0]
    value_col = summary.columns[1]

    if intent.chart_type == "pie":
        return (
            alt.Chart(summary)
            .mark_arc(innerRadius=40)
            .encode(
                theta=alt.Theta(field=value_col, type="quantitative"),
                color=alt.Color(field=label_col, type="nominal", legend=alt.Legend(title=label_col)),
                tooltip=[label_col, value_col],
            )
            .properties(title=title)
        )

    if intent.chart_type == "line":
        return (
            alt.Chart(summary)
            .mark_line(point=True)
            .encode(
                x=alt.X(field=label_col, type="nominal", sort=None, title=label_col),
                y=alt.Y(field=value_col, type="quantitative", title=value_col),
                tooltip=[label_col, value_col],
            )
            .properties(title=title)
        )

    # bar and histogram both render as bar charts; histogram has stringified bin labels.
    return (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X(field=label_col, type="nominal", sort="-y", title=label_col),
            y=alt.Y(field=value_col, type="quantitative", title=value_col),
            tooltip=[label_col, value_col],
        )
        .properties(title=title)
    )


def _title_for(intent: ChartIntent) -> str:
    if intent.chart_type == "histogram":
        return f"Distribution of {intent.field}"
    if intent.metric == "average" and intent.value_column:
        return f"Average {intent.value_column} by {intent.field}"
    return f"{intent.chart_type.title()} chart by {intent.field}"


def handle_chart_request(
    request: str,
    loaded,
    profile,
    selected_sheet: str,
    memory: SessionMemory,
) -> str | None:
    """If the request is a chart, build it, store in session_state, and return
    a short status string to show in chat. Otherwise return None so the caller
    falls through to the regular planner."""
    if loaded is None or profile is None or not selected_sheet:
        return None

    sheet_columns = next((s.columns for s in profile.sheets if s.name == selected_sheet), [])
    intent = _chart_followup_intent(request, list(sheet_columns))
    if intent is None:
        intent = detect_chart_intent(request, list(sheet_columns))
    if intent is None:
        if is_chart_request(request):
            return (
                "Which field should I chart? For example: department, major, advisor, "
                "year, academic status, or GPA."
            )
        return None

    dataframe = loaded.sheets[selected_sheet]
    filtered = _apply_active_filters(dataframe, memory)
    if filtered.empty:
        return "There are no rows matching the current filters, so the chart would be empty."

    summary = compute_chart(intent, filtered)
    if summary is None or summary.empty:
        return f"I couldn't compute a chart for `{intent.field}` — the column has no usable values after filters."

    title = _title_for(intent)
    st.session_state["latest_figure"] = {
        "title": title,
        "type": intent.chart_type,
        "field": intent.field,
        "metric": intent.metric,
        "value_column": intent.value_column,
        "summary_records": summary.to_dict(orient="records"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(filtered.shape[0]),
    }
    history = st.session_state.setdefault("figures_history", [])
    history.insert(
        0,
        {
            "title": title,
            "type": intent.chart_type,
            "field": intent.field,
            "metric": intent.metric,
            "value_column": intent.value_column,
            "preview": summary,
        },
    )
    # The newest chart is always at index 0 — ask the unified workbook
    # panel to switch to it on the next render so the user sees it inline.
    st.session_state["_pending_view_sheet"] = ("figure", "0")
    rows = filtered.shape[0]
    return f"Charted **{intent.field}** ({intent.chart_type}) over {rows} matching row(s). The figures panel is now open."


def _chart_followup_intent(request: str, columns: list[str]) -> ChartIntent | None:
    """Use the existing figure as context for requests like "add GPA to the chart"."""
    lower = request.lower()
    if "add" not in lower or "chart" not in lower:
        return None
    figure = st.session_state.get("latest_figure") or {}
    existing_field = str(figure.get("field") or "")
    if not existing_field:
        return None
    synonyms = load_json("synonyms.json")
    mentioned = [
        column for column in _mentioned_columns(lower, columns, synonyms)
        if column != existing_field
    ]
    added = mentioned[0] if mentioned else ""
    if not added:
        return None
    chart_type = str(figure.get("type") or "bar")
    if _first_numeric_named_column([added]):
        return ChartIntent(
            chart_type=chart_type,
            field=existing_field,
            metric="average",
            value_column=added,
        )
    if _first_numeric_named_column([existing_field]):
        return ChartIntent(
            chart_type=chart_type,
            field=added,
            metric="average",
            value_column=existing_field,
        )
    return ChartIntent(chart_type=chart_type, field=added)


def _apply_active_filters(frame: pd.DataFrame, memory: SessionMemory) -> pd.DataFrame:
    """Best-effort filter application for the chart preview.

    The query engine is the source of truth for "what the user is currently
    looking at," so we mirror the simple subset of its filter ops needed for
    chart previews (==, in, !=, >, <, >=, <=). Anything more complex falls
    back to the full dataset — the planner-driven path stays authoritative.
    """
    filtered = frame
    for condition in memory.active_filters or []:
        column = condition.get("column")
        operator = condition.get("operator")
        value = condition.get("value")
        if not column or column not in filtered.columns:
            continue
        series = filtered[column]
        try:
            if operator == "equals":
                filtered = filtered[series.astype(str).str.casefold() == str(value).casefold()]
            elif operator == "not_equals":
                filtered = filtered[series.astype(str).str.casefold() != str(value).casefold()]
            elif operator == "contains" and isinstance(value, str):
                filtered = filtered[series.astype(str).str.contains(value, case=False, na=False)]
            elif operator in {"is_missing", "is_blank"}:
                filtered = filtered[series.isna() | (series.astype(str).str.strip() == "")]
            elif operator in {"is_not_missing", "is_not_blank"}:
                filtered = filtered[~(series.isna() | (series.astype(str).str.strip() == ""))]
            elif operator in {"in", "in_list"} and isinstance(value, list):
                vals = {str(v).casefold() for v in value}
                filtered = filtered[series.astype(str).str.casefold().isin(vals)]
            elif operator in {"greater_than", "less_than", "greater_or_equal", "less_or_equal"}:
                numeric = pd.to_numeric(series, errors="coerce")
                threshold = float(value)
                if operator == "greater_than":
                    filtered = filtered[numeric > threshold]
                elif operator == "less_than":
                    filtered = filtered[numeric < threshold]
                elif operator == "greater_or_equal":
                    filtered = filtered[numeric >= threshold]
                elif operator == "less_or_equal":
                    filtered = filtered[numeric <= threshold]
        except Exception:
            continue
    return filtered


def render_figures_panel() -> None:
    """Bottom-left card content (title supplied by render_card wrapper)."""
    figure = st.session_state.get("latest_figure")
    if not figure:
        st.markdown(
            '<div class="empty-state"><strong>No figure yet.</strong> '
            'Ask the assistant for a chart, such as:'
            '<ul>'
            '<li>Create a bar chart by professor</li>'
            '<li>Make a pie chart of academic status</li>'
            '<li>Show GPA distribution</li>'
            '<li>Chart students below 2.0 by teacher</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )
        return

    summary = pd.DataFrame(figure.get("summary_records") or [])
    if summary.empty:
        st.caption("No data to display.")
        return

    intent = ChartIntent(
        chart_type=str(figure.get("type", "bar")),
        field=str(figure.get("field", "")),
        metric=str(figure.get("metric", "count")),
        value_column=str(figure.get("value_column", "")),
    )
    chart = build_altair_chart(intent, summary, str(figure.get("title", "Figure")))
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"{figure.get('title', 'Figure')} · {figure.get('row_count', 0)} row(s) · {figure.get('created_at', '')}"
    )


# Export helpers --------------------------------------------------------------


FIGURES_DIR = Path("outputs") / "figures"


def export_latest_figure_csv() -> tuple[str, bytes] | None:
    figure = st.session_state.get("latest_figure")
    if not figure:
        return None
    summary = pd.DataFrame(figure.get("summary_records") or [])
    if summary.empty:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    field_part = (figure.get("field") or "figure").replace(" ", "_")
    file_name = f"chart_{field_part}_{stamp}.csv"
    return file_name, summary.to_csv(index=False).encode("utf-8")
