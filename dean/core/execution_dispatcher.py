"""Single execution dispatcher for routed plans.

Takes a normalized routing result from nlp.planner_router and executes it with
the appropriate engine: pandas query engine for questions, confirmed_actions for
export/note/field edits. It never plans and never bypasses confirmation — the
caller only dispatches actions after the user has confirmed. Returns a uniform
response object the UI renders. No raw sensitive rows are returned unless the
caller explicitly reveals them after confirmation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core import confirmed_actions as ca
from core import privacy
from core.llm_config import from_app_settings
from core.query_engine import QueryExecutionError, run_query, _build_mask
from nlp.suggestions import suggest_next_moves
from nlp.synonym_mapper import normalize_text


# Default max result rows handed to the conversational narrator so it can name
# students. All-row local mode can disable this cap through settings.
CONVERSATION_ROW_SAMPLE_LIMIT = 40


def _response(
    *, success, response_type, message="", description="", explanation=None, result_preview=None,
    value=None, row_count=None, columns=None, removed=None, output_file=None, rows_affected=None,
    warnings=None, operation="", preview_truncated=False, narration="", conversation_llm_used=False,
    assumption_note="", alternatives=None, suggestions=None, band="high",
    sheet_outcome=None,
) -> dict[str, Any]:
    return {
        "success": success,
        "response_type": response_type,
        "message": message,
        "description": description,
        "explanation": explanation,
        "result_preview": result_preview or [],
        "value": value,
        "row_count": row_count,
        "columns": columns or [],
        "removed": removed or [],
        "output_file": output_file,
        "rows_affected": rows_affected,
        "warnings": warnings or [],
        "operation": operation,
        "preview_truncated": preview_truncated,
        "narration": narration,
        "conversation_llm_used": conversation_llm_used,
        "assumption_note": assumption_note,
        "alternatives": alternatives or [],
        "suggestions": suggestions or [],
        "band": band,
        "sheet_outcome": sheet_outcome,
    }


def execute_planned_request(
    routing: dict[str, Any],
    loaded,
    settings: dict[str, Any] | None = None,
    *,
    reveal_sensitive: bool = False,
    request_summary: str = "",
    session_workbook=None,
) -> dict[str, Any]:
    settings = settings or {}
    intent = routing.get("intent")
    plan = routing.get("plan") or {}
    narration = routing.get("narration") or ""

    if intent in {"clarify", "unavailable", "unsupported"}:
        return _response(success=True, response_type="clarification",
                         message=routing.get("confirmation_reason") or routing.get("fallback_reason") or "",
                         warnings=routing.get("warnings"), narration=narration,
                         band=routing.get("band", "low"))

    if intent == "query":
        return _run_query(plan, loaded, settings, reveal_sensitive, request_summary,
                          narration=narration, routing=routing,
                          session_workbook=session_workbook)

    sheet = plan.get("sheet") or next(iter(loaded.sheets), "")
    filters = plan.get("filters") or []
    try:
        if intent == "export":
            result = ca.execute_export_action(filters=filters, sheets=loaded.sheets, sheet=sheet,
                                              request_summary=request_summary)
        elif intent == "note_edit":
            result = ca.execute_add_note_action(filters=filters, note=plan.get("note", ""),
                                               sheets=loaded.sheets, sheet=sheet, request_summary=request_summary)
        elif intent == "field_update":
            result = ca.execute_update_field_action(filters=filters, field_name=plan.get("field", ""),
                                                   value=plan.get("value"), sheets=loaded.sheets, sheet=sheet,
                                                   request_summary=request_summary)
        elif intent == "academic_watch":
            result = ca.execute_academic_watch_action(
                filters=filters, sheets=loaded.sheets, sheet=sheet,
                value=plan.get("value", "Yes"), request_summary=request_summary,
            )
        elif intent == "attendance_watch":
            # Shares the academic_watch implementation; just writes to a
            # different column ("Attendance Watch" instead of "Academic
            # Watch") so the two flags remain independent.
            result = ca.execute_academic_watch_action(
                filters=filters, sheets=loaded.sheets, sheet=sheet,
                value=plan.get("value", "Yes"),
                column_name=plan.get("column_name", "Attendance Watch"),
                request_summary=request_summary,
            )
        elif intent == "action_chain":
            result = ca.execute_action_chain(
                actions=plan.get("actions") or [],
                filters=filters, sheets=loaded.sheets, sheet=sheet,
                request_summary=request_summary,
            )
        elif intent == "dashboard_report":
            result = ca.execute_dashboard_report_action(
                sheets=loaded.sheets,
                sheet=sheet,
                request_summary=request_summary,
            )
        else:
            return _response(success=False, response_type="error", message=f"Unsupported intent: {intent}")
    except Exception as exc:  # noqa: BLE001
        return _response(success=False, response_type="error", message=f"The action could not be completed: {exc}")

    return _response(success=result.success, response_type="confirmation" if result.success else "error",
                     message=result.message, output_file=result.output_file,
                     rows_affected=result.rows_affected, columns=result.columns)


def _run_query(plan, loaded, settings, reveal_sensitive, request_summary,
               *, narration: str = "", routing: dict[str, Any] | None = None,
               session_workbook=None) -> dict[str, Any]:
    routing = routing or {}
    band = routing.get("band", "high")
    # assumption_note is always honored when present — the router explicitly
    # decided to surface it (medium-band assumption, ambiguity interpretation
    # line, or Phase P drilldown context reminder). band only gates alternatives.
    assumption_note = routing.get("assumption_note", "") or ""
    alternatives = list(routing.get("alternatives") or []) if band == "medium" else []

    try:
        result = run_query(plan, loaded.sheets)
    except QueryExecutionError as exc:
        return _response(success=False, response_type="error",
                         message=f"I could not run that calculation: {exc}", narration=narration,
                         band=band)

    raw_table = list(result.table or [])
    table = raw_table
    removed: list[str] = []
    text_search_terms = _text_search_terms(plan)
    # Phase P: peek at the unredacted top row to record a group winner that
    # may otherwise be redacted (e.g. "Discipline" shares a keyword with
    # disciplinary status). Pure metadata — no row data is stored elsewhere.
    top_group = _capture_top_group(plan, raw_table)
    if not reveal_sensitive and table:
        original_columns = list(table[0].keys())
        table, removed = privacy.redact_table(table, original_columns)
        # Free-text notes were just redacted; surface a small "Matched Notes"
        # indicator so the user can tell those rows matched a text search
        # without seeing the actual note content.
        if table and text_search_terms:
            for note_column in [c for c in original_columns if c in removed
                                and c in text_search_terms]:
                for row in table:
                    row[f"Matched {note_column}"] = "✓"
    elif reveal_sensitive and table and text_search_terms:
        # User has confirmed reveal — replace each full note with a short
        # snippet around the matched substring (M.5). Only the note column
        # that was actually searched is shrunk; other text columns stay intact.
        for row in table:
            for column, term in text_search_terms.items():
                value = row.get(column)
                if isinstance(value, str) and value:
                    row[column] = _match_snippet(value, term)

    # Refine suggestions now that we know the row count and which columns were
    # redacted. The router-side list (when present) is the pre-execution best
    # guess; this overrides with a grounded list.
    sheet_columns = []
    if plan.get("sheet") and plan["sheet"] in loaded.sheets:
        sheet_columns = list(loaded.sheets[plan["sheet"]].columns)
    suggestions = suggest_next_moves(
        plan=plan, columns=sheet_columns, active_filters=plan.get("filters") or [],
        removed_fields=removed, row_count=result.row_count,
    )

    config = from_app_settings(settings)
    verified = {
        "operation": result.operation,
        "value": result.value,
        "row_count": result.row_count,
        "description": result.description,
        "columns": list(result.columns_used or []),
    }
    explanation = None
    conversation_text = None
    conversation_used = False

    if config["conversation_llm_enabled"]:
        from nlp.local_model import converse_about_result_with_model

        active_context = {
            "sheet": plan.get("sheet"),
            "filters": plan.get("filters") or [],
            "group_by": plan.get("group_by"),
            "sort": plan.get("sort"),
            "limit": plan.get("limit"),
        }
        allowed_actions = list(suggestions)
        if alternatives:
            allowed_actions.extend(alternatives)
        full_row_access = bool(config.get("local_llm_full_row_access"))
        row_sample_limit = None if config.get("local_llm_all_matching_rows") else CONVERSATION_ROW_SAMPLE_LIMIT
        # By default, the narrator receives the same name-safe redacted rows the
        # UI displays. When the admin enables full local row access, the
        # narrator receives a bounded sample from the matched source rows,
        # including hidden columns. The endpoint is still guarded as localhost.
        row_sample = _source_row_sample(
            plan=plan,
            loaded_sheets=loaded.sheets,
            full_access=full_row_access,
            limit=row_sample_limit,
        )
        if row_sample is None:
            row_sample = _chat_row_sample(
                raw_table if full_row_access else table,
                full_access=full_row_access,
                limit=row_sample_limit,
            )
        row_sample_policy = (
            "full_local_rows"
            if full_row_access
            else "redacted_name_safe_rows"
        )
        conversation_text, _ = converse_about_result_with_model(
            user_question=request_summary or plan.get("plain_english_question", ""),
            understood_plan=assumption_note or narration,
            verified_result=verified,
            model_name=config["explanation_model"],
            active_context=active_context,
            hidden_sensitive_fields=[] if full_row_access else removed,
            allowed_next_actions=allowed_actions,
            row_sample=row_sample,
            row_sample_policy=row_sample_policy,
        )
        if conversation_text:
            conversation_used = True
            explanation = conversation_text

    # The conversational narrator subsumes the explain pass — both generate
    # plain-English phrasing of the same verified result. Only fall through to
    # explain when the narrator was not enabled at all (so we don't fire two
    # 7B-model passes per turn just to phrase one answer).
    if (
        explanation is None
        and not config["conversation_llm_enabled"]
        and (config["llm_explanations_enabled"] or settings.get("use_local_llm"))
    ):
        from nlp.local_model import explain_result_with_model

        explanation, _ = explain_result_with_model(
            user_question=request_summary or plan.get("plain_english_question", ""),
            verified_result=verified,
            model_name=config["explanation_model"],
        )

    if explanation:
        message = explanation
    else:
        lead = assumption_note or narration
        message = f"{lead} {result.description}".strip() if lead else result.description

    # Phase O: when the routing flagged the request as a container-vs-aggregate
    # or "performing well" ambiguity, also compute the alternative in pandas
    # and join both numbers into the same response message.
    alt_result_text = _compute_ambiguity_alternative(
        routing=routing, loaded_sheets=loaded.sheets, primary_result=result,
        primary_plan=plan,
    )
    if alt_result_text:
        message = f"{message} {alt_result_text}".strip()

    sheet_outcome = _record_session_sheet(
        session_workbook=session_workbook, user_message=request_summary, plan=plan,
        result=result, loaded_sheets=loaded.sheets, removed_columns=removed,
    )

    response = _response(success=True, response_type="answer", message=message, description=result.description,
                     explanation=explanation, result_preview=table,
                     value=result.value, row_count=result.row_count, columns=result.columns_used,
                     removed=removed, operation=result.operation, preview_truncated=result.preview_truncated,
                     narration=narration, conversation_llm_used=conversation_used,
                     assumption_note=assumption_note, alternatives=alternatives,
                     suggestions=suggestions, band=band, sheet_outcome=sheet_outcome)
    # Phase P top-group passthrough (post-redaction; chat layer reads it to
    # populate session memory's winner fields).
    if top_group is not None:
        response["top_group"] = top_group
    return response


_NAME_KEYWORDS = ("first name", "last name", "full name", "name")


def _chat_row_sample(
    table,
    *,
    full_access: bool = False,
    limit: int | None = CONVERSATION_ROW_SAMPLE_LIMIT,
) -> list[dict[str, Any]] | None:
    """Build a bounded, name-safe row sample for the conversational narrator.

    In normal mode, `table` is already privacy-redacted (hidden-by-default
    columns removed). The only sensitive-but-retained columns are `identity`
    fields: student names plus IDs. The chat narrator is allowed to speak names
    but not IDs, so we drop any identity column whose name isn't a person-name
    field.

    In full local row access mode, the admin has explicitly allowed the local
    narrator to receive the full matching row sample. The localhost-only guard
    still applies at call time.
    """
    if not isinstance(table, list) or not table:
        return None
    rows = table if limit is None else table[:limit]
    if full_access:
        return [dict(row) for row in rows]
    columns = list(table[0].keys())
    drop: set[str] = set()
    for column in columns:
        sensitive, sensitivity_type = privacy.classify_sensitivity(column)
        if not sensitive or sensitivity_type not in {"identity", "identity_high"}:
            continue
        normalized = normalize_text(column)
        is_name = any(keyword in normalized for keyword in _NAME_KEYWORDS)
        if sensitivity_type == "identity_high" or not is_name:
            drop.add(column)
    sample = [
        {key: value for key, value in row.items() if key not in drop}
        for row in rows
    ]
    return sample


def _source_row_sample(
    *,
    plan: dict[str, Any],
    loaded_sheets,
    full_access: bool,
    limit: int | None = CONVERSATION_ROW_SAMPLE_LIMIT,
) -> list[dict[str, Any]] | None:
    """Sample the matched source rows, not just aggregate/count output rows."""
    sheet = plan.get("sheet")
    frame = loaded_sheets.get(sheet) if sheet else None
    if frame is None:
        return None
    operation = plan.get("operation")
    if operation not in {
        "count_rows",
        "percent_rows",
        "sum_column",
        "average_column",
        "min_column",
        "max_column",
        "groupby_count",
        "groupby_sum",
        "groupby_average",
        "filtered_preview",
    }:
        return None
    try:
        filters = plan.get("filters") or []
        filter_mode = str(plan.get("filter_mode") or "all").lower()
        if filter_mode not in {"all", "any"}:
            filter_mode = "all"
        mask = _build_mask(frame, filters, filter_mode)
        subset = frame.loc[mask]
        sort = plan.get("sort") or {}
        if isinstance(sort, dict) and sort.get("column") in subset.columns:
            ascending = str(sort.get("direction", "asc")).lower() != "desc"
            column = sort["column"]
            numeric = pd.to_numeric(subset[column], errors="coerce")
            if numeric.notna().any():
                subset = (
                    subset.assign(_sort=numeric)
                    .sort_values("_sort", ascending=ascending)
                    .drop(columns="_sort")
                )
            else:
                subset = subset.sort_values(column, ascending=ascending)
        sampled = subset if limit is None else subset.head(limit)
        rows = sampled.to_dict(orient="records")
    except Exception:  # noqa: BLE001 - narrator context is best-effort only
        return None
    if full_access:
        return _chat_row_sample(rows, full_access=True, limit=limit)
    redacted, _ = privacy.redact_table(rows, list(frame.columns))
    return _chat_row_sample(redacted, limit=limit)


def _capture_top_group(plan, table) -> dict[str, Any] | None:
    """Extract the top group's {column, value} from a groupby_* result.

    Reads the FIRST row of the unredacted result table (before privacy
    redaction) so phrases like 'in that department' can still resolve to
    the winning group name even when the column itself is privacy-redacted
    in the user-visible preview."""
    operation = plan.get("operation")
    group_by = plan.get("group_by")
    if not table or not group_by:
        return None
    if operation not in {"groupby_count", "groupby_sum", "groupby_average"}:
        return None
    first = table[0]
    value = first.get(group_by)
    if value is None or not str(value).strip():
        return None
    return {"column": group_by, "value": str(value)}


def _compute_ambiguity_alternative(
    *, routing: dict[str, Any], loaded_sheets, primary_result, primary_plan,
) -> str:
    """If the routing flags an alternative interpretation, run it in pandas
    and return a human-readable summary line. Empty string when no alternative
    was tagged or the computation can't be safely performed.
    """
    if not isinstance(routing, dict):
        return ""
    ambiguity = routing.get("ambiguity") or {}
    spec = ambiguity.get("alternative") if isinstance(ambiguity, dict) else None
    if not isinstance(spec, dict):
        return ""

    sheet = primary_plan.get("sheet")
    frame = loaded_sheets.get(sheet) if sheet else None
    if frame is None:
        return ""

    kind = spec.get("kind")
    try:
        import pandas as pd

        if kind == "aggregate_then_count_groups":
            group_column = spec["group_column"]
            metric_column = spec["metric_column"]
            operator = spec.get("operator", "less_than")
            threshold = float(spec.get("threshold", 0.0))
            numeric = pd.to_numeric(frame[metric_column], errors="coerce")
            grouped = numeric.groupby(frame[group_column]).mean()
            count = int(_compare_series(grouped, operator, threshold).sum())
            primary = primary_result.value if primary_result.value is not None else primary_result.row_count
            return (
                f"Alternative interpretation: "
                f"{ambiguity.get('alternative_phrase', 'aggregate reading')}. "
                f"Result: {count} {group_column.lower()}(s) "
                f"(vs. {primary} under the primary reading)."
            )
        if kind == "rate_above_cutoff_by_group":
            group_column = spec["group_column"]
            metric_column = spec["metric_column"]
            operator = spec.get("operator", "greater_or_equal")
            threshold = float(spec.get("threshold", 2.5))
            numeric = pd.to_numeric(frame[metric_column], errors="coerce")
            above = _compare_series(numeric, operator, threshold)
            rates = (above.groupby(frame[group_column]).mean() * 100).round(1)
            top_group = rates.sort_values(ascending=False).head(1)
            if top_group.empty:
                return ""
            top_name = str(top_group.index[0])
            top_rate = float(top_group.iloc[0])
            return (
                f"Alternative interpretation: by share of students with "
                f"{metric_column} ≥ {threshold}, the top {group_column.lower()} "
                f"is {top_name} ({top_rate:.1f}%)."
            )
    except Exception:  # noqa: BLE001 — never let the alt path bring down the answer
        return ""
    return ""


def _compare_series(series, operator: str, threshold: float):
    if operator == "less_than":
        return series < threshold
    if operator == "less_or_equal":
        return series <= threshold
    if operator == "greater_than":
        return series > threshold
    if operator == "greater_or_equal":
        return series >= threshold
    # Fall back to equality; the alternative path is best-effort.
    return series == threshold


_TEXT_SEARCH_OPS = {"contains_text", "not_contains_text"}


def _text_search_terms(plan: dict[str, Any]) -> dict[str, str]:
    """Return {column: search_term} for any text-search filter in the plan.

    Used by the dispatcher to (a) attach a 'Matched X' indicator when the
    note column is redacted, (b) render a snippet around the match when the
    user has explicitly confirmed reveal.
    """
    out: dict[str, str] = {}
    for condition in plan.get("filters") or []:
        if condition.get("operator") not in _TEXT_SEARCH_OPS:
            continue
        column = condition.get("column")
        value = condition.get("value")
        if isinstance(column, str) and isinstance(value, str) and value:
            out[column] = value
    return out


def _match_snippet(text: str, term: str, *, window: int = 40) -> str:
    """Return a short snippet around the first case-insensitive match of term.

    Format: '...prefix MATCH suffix...' with ellipsis only on the side that
    was actually truncated. Falls back to the first ``window*2`` chars when
    the term isn't found (which can happen on a not_contains_text result).
    """
    if not text:
        return text
    lower = text.casefold()
    needle = term.casefold()
    index = lower.find(needle)
    if index < 0:
        snippet = text[: window * 2]
        return snippet + ("..." if len(text) > len(snippet) else "")
    start = max(0, index - window)
    end = min(len(text), index + len(needle) + window)
    snippet = text[start:end]
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _record_session_sheet(*, session_workbook, user_message, plan, result,
                          loaded_sheets, removed_columns):
    """If a session workbook is bound, append a sheet for this turn.

    Failures are swallowed (logged but never bubble up) — a session-workbook
    write must NEVER take down a successful query reply. Returns the outcome
    dict for the response payload (or None if no workbook is bound).
    """
    if session_workbook is None:
        return None
    try:
        outcome = session_workbook.record_turn(
            user_message=user_message or "",
            plan=plan,
            result=result,
            loaded_sheets=loaded_sheets,
            removed_columns=list(removed_columns or []),
        )
    except Exception:  # noqa: BLE001 — session workbook is a side-channel
        return {"action": "skipped", "sheet_name": "", "reason": "write_failed"}
    return {"action": outcome.action, "sheet_name": outcome.sheet_name, "reason": outcome.reason}
