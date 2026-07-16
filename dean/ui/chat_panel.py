"""Unified chat panel and intent router for the assistant.

The user sees ONE chat. Internally, every message is classified into one of
three safe paths and dispatched:

    ask_question  -> query_planner -> core.query_engine (pandas) -> answer view
    edit_workbook -> edit_planner (validated plan) -> confirm-and-run controls
    clarify       -> a focused follow-up question with clickable options

Privacy: the router and planners send only the message plus sheet/column names
to the local model. The one exception is the post-execution chat narrator, which
receives a bounded, name-safe sample of the already-redacted result rows (student
names + roster fields; no IDs, contact, financial, or notes) so it can name
specific students on request; Strict Privacy Mode disables that narrator and
sends no rows at all. Exact numbers are always computed by pandas, never the model.

Phase J: assistant messages can carry an `attachment` dict that the chat
renderer uses to draw result tables, edit-plan preview cards, confirmation
buttons, or download links *inside* the chat bubble. The full conversation is
preserved across reruns so follow-ups feel like a single thread.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

import uuid

from core.command_schema import EXECUTABLE_PLANNER_ACTIONS
from core.execution_dispatcher import execute_planned_request
from core.failure_log import log_failure
from core.interaction_logger import (
    extract_corrected_request,
    is_correction_message,
    log_interaction,
    workbook_schema_hash,
)
from core.risk_settings import load_risk_settings
from core.session_memory import SessionMemory
from core.session_workbook import SessionWorkbook
from nlp.question_library import (
    lookup_by_id,
    lookup_for_message,
)
from nlp.conversation import (
    CLEAR,
    RESET,
    classify_confirmation,
    classify_context_action,
    describe_filters,
    is_bare_context_command,
)
from nlp.edit_planner import plan_edit
from nlp.followup_resolver import is_bare_reference, resolve_followup
from nlp.llm_json_parser import command_to_confirmation
from nlp.planner_router import plan_user_request
from nlp.action_chain import parse_action_chain
from nlp.request_intents import (
    is_academic_watch_request,
    is_attendance_watch_request,
    is_export_request,
    is_note_request,
    parse_field_update,
)
from nlp.synonym_mapper import normalize_text
from ui.figures_panel import handle_chart_request, is_chart_request, suggested_figure_questions


HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.60

PROMPT_SUGGESTIONS = [
    "How many students are in each department?",
    "Show me Accounting students",
    "Now only below 2.5 GPA",
    "Group them by advisor",
    "Export this list",
]

MODE_BADGE = {
    "ask_question": ("Answer only", "green"),
    "edit_workbook": ("Workbook change", "yellow"),
    "clarify": ("Needs clarification", "yellow"),
}


# Router ----------------------------------------------------------------------


def friendly_validation_error(exc_message: Any, columns: list[str], sheets: list[str]) -> str:
    msg = str(exc_message)
    if "Column does not exist:" in msg:
        col = msg.split("Column does not exist:")[-1].strip()
        if columns:
            cols_str = ", ".join([f"`{c}`" for c in columns])
            return f"I couldn't find the column '{col}' in the active sheet. Available columns are: {cols_str}."
        return f"I couldn't find the column '{col}' in the active sheet."
    if "references nonexistent sheet:" in msg:
        sheet = msg.split("references nonexistent sheet:")[-1].strip()
        if sheets:
            sheets_str = ", ".join([f"`{s}`" for s in sheets])
            return f"I couldn't find the sheet '{sheet}' in this workbook. Available sheets are: {sheets_str}."
        return f"I couldn't find the sheet '{sheet}' in this workbook."
    if "protected_field" in msg or "is a protected field" in msg:
        return "GPA and Student ID are protected fields and cannot be modified. This safeguard protects your records from accidental changes."
    if "not_editable" in msg:
        return "This field is not editable via the assistant. Only 'Notes', 'Academic Watch', and 'Attendance Watch' fields can be modified."
    if "no previous result" in msg.lower() or "no previous context" in msg.lower():
        return "You asked about 'those' or 'that', but I don't have any previous context in this session. What group of students did you want to look at?"
    return f"I couldn't turn that into a safe plan: {msg}"


# Generic read operations the free-form analyst is allowed to take. Anything
# NOT in this set (student_intervention_summary, pivot_table_summary,
# trend_summary, advisor_outcome_summary, data_quality_summary, …) is a
# purpose-built specialist and stays on the deterministic dispatcher. Allowlist,
# not denylist, so new specialized operations default to the deterministic path.
# Generic reads we let the (opt-in) LLM analyst answer. Deliberately EXCLUDES the
# groupby/ranking ops: the deterministic query engine (core.query_engine.run_query)
# computes "which group has the highest/lowest <agg>" exactly via idxmax/idxmin and
# names the winner, whereas llama3.2:3b reliably mis-ranks or blanks on these
# (dean_eval.py groupby category: 3/7 on the LLM vs 7/7 deterministic). Ranking is
# structured, not open-ended, so the tested engine wins — see docs/llm_improvement_log.md.
_ANALYST_GENERIC_OPS = frozenset({
    "filtered_preview", "count_rows", "count_unique", "list_unique",
    "average_column", "sum_column", "min_column", "max_column",
})


def route_message(
    *,
    request: str,
    selected_sheet: str,
    loaded,
    profile,
    settings: dict[str, Any],
) -> None:
    """Single entry: orchestrate, then route ALL planning through planner_router."""
    request = request.strip()
    if not request:
        return

    st.session_state["chat_messages"].append({
        "role": "user",
        "content": request,
        "timestamp": _now(),
    })
    _reset_views()
    # Drop any prior analyst trace so the debug panel never shows a stale one on
    # a turn the analyst didn't handle.
    st.session_state["analyst_trace_debug"] = None

    # Stash the workbook context so any nested failure-capture call (in the
    # query or edit branches below) has the sheet + columns without threading
    # them through three layers of helpers.
    if profile is not None and selected_sheet:
        active_columns = next(
            (s.columns for s in profile.sheets if s.name == selected_sheet),
            [],
        )
    else:
        active_columns = []
    st.session_state["_failure_capture_context"] = {
        "sheet": selected_sheet or "",
        "columns": list(active_columns),
    }

    # L.6 + L.15: capture if this turn is correcting the prior assistant
    # interpretation. We stash the prior entry id here so any branch can
    # attribute the correction, and route the *extracted* form of the message
    # through the planner so prefixes like "no, I mean" don't confuse it.
    # When an alternative chip is clicked it sets `_force_correction_target`
    # directly — chip text usually has no "no, I mean" prefix.
    forced_target = st.session_state.pop("_force_correction_target", None)
    if forced_target:
        correction_target = forced_target
    elif is_correction_message(request):
        correction_target = st.session_state.get("last_interaction_id")
    else:
        correction_target = None
    st.session_state["_pending_correction_target"] = correction_target
    planner_request = extract_corrected_request(request) if correction_target else request

    if loaded is None or profile is None or not selected_sheet:
        _set_clarify(
            request,
            "Please upload a workbook first so I know which sheet to work with.",
            reason="No workbook is loaded.",
        )
        return

    sheet_columns = {sheet.name: sheet.columns for sheet in profile.sheets}
    memory = SessionMemory.from_dict(st.session_state.get("assistant_memory"))

    # Track the most recently named individual student, independent of
    # whatever else this message does, so a later singular pronoun ("mark
    # her as academic watch") can resolve back to them even after unrelated
    # turns in between overwrite active_filters. Skipped when this message
    # is just a yes/no answer to a pending confirmation.
    if not memory.pending_action:
        from nlp.query_planner import _named_student_filter

        named_person = _named_student_filter(
            normalize_text(planner_request), loaded.sheets.get(selected_sheet), sheet_columns.get(selected_sheet, []),
        )
        if named_person:
            memory.last_named_person = named_person

    # -1. A pending confirmation takes priority. Only yes/no resolves it.
    if memory.pending_action:
        decision = classify_confirmation(request)
        prior_routing = (memory.pending_action or {}).get("routing")
        if decision == "yes":
            _mark_pending_resolved("confirmed")
            pending_workbook = ensure_session_workbook(loaded, profile)
            pending_response = _resolve_pending(
                memory, loaded, settings, session_workbook=pending_workbook,
            )
            _focus_workbook_on_new_sheet((pending_response or {}).get("sheet_outcome"))
            memory.pending_action = {}
            _save_memory(memory)
            _log_turn(request=request, routing=prior_routing, sheet_columns=sheet_columns,
                      settings=settings, confirmation_result="confirmed",
                      sheet_outcome=(pending_response or {}).get("sheet_outcome"))
            return
        if decision == "no":
            _mark_pending_resolved("cancelled")
            memory.pending_action = {}
            _say("Okay — cancelled. Nothing was changed or exported.")
            st.session_state["assistant_mode"] = None
            _save_memory(memory)
            _log_turn(request=request, routing=prior_routing, sheet_columns=sheet_columns,
                      settings=settings, confirmation_result="cancelled")
            return
        _mark_pending_resolved("expired")
        memory.pending_action = {}  # unclear -> drop it and treat as a new request

    # 0. Context commands: "start over" / "clear that".
    context_action = classify_context_action(request)
    if context_action in (RESET, CLEAR) and is_bare_context_command(request):
        if context_action == RESET:
            memory.reset_all()
        else:
            memory.clear_filters()
        _context_ack(context_action, memory)
        _save_memory(memory)
        return

    # 0b. Chart request: render in Figures panel without modifying the workbook.
    # Detected here so it short-circuits the edit/query planner.
    if is_chart_request(request):
        chart_status = handle_chart_request(request, loaded, profile, selected_sheet, memory)
        if chart_status is not None:
            _say(chart_status)
            _save_memory(memory)
            return

    # 0c. Teach a glossary term: "struggling means GPA below 2.5",
    # "when I say advisor I mean Counselor". Only when the analyst is enabled —
    # the glossary feeds the free-form analyst's prompt. Persisted durably via
    # the learning store so it survives across sessions.
    if settings.get("code_analyst_enabled") and not settings.get("strict_privacy_mode", False):
        from nlp.glossary import parse_teach, teach_term

        taught = parse_teach(request)
        if taught is not None:
            phrase, definition = taught
            teach_term(phrase, definition)
            _say(f'Got it — I\'ll remember that "{phrase}" means: {definition}.')
            _save_memory(memory)
            return

    # 1. Bare follow-up reference with nothing to refer to -> clarify. Skip this
    #    for action requests ("add a note to these students"), which carry their
    #    own confirmation gate and shouldn't be treated as dangling references.
    is_action_request = (
        is_export_request(request) or is_note_request(request) or parse_field_update(request) is not None
        or is_academic_watch_request(request)
        or is_attendance_watch_request(request)
        or parse_action_chain(request) is not None
    )
    followup = resolve_followup(request, memory)
    if (
        followup.is_followup
        and followup.needs_clarification
        and is_bare_reference(request)
        and not is_action_request
    ):
        _set_clarify(request, followup.clarification_question, reason="No previous result to refer to.")
        memory.record_clarify(request=request)
        _save_memory(memory)
        return

    # 2. THE single planning entry point (shared with tests + eval harness).
    # For correction turns we plan on the *extracted* request so prefixes like
    # "no, I mean" don't get parsed as the average verb. The user message
    # shown in chat and stored in the log keeps its original form.
    try:
        routing = plan_user_request(
            user_message=planner_request,
            sheets=loaded.sheets,
            sheet_columns=sheet_columns,
            selected_sheet=selected_sheet,
            conversation_state=memory.to_dict(),
            settings=settings,
        )
    except Exception as exc:
        friendly = friendly_validation_error(exc, active_columns, list(loaded.sheets.keys()))
        _set_clarify(request, friendly, reason="The request could not be planned.")
        memory.record_clarify(request=request)
        _save_memory(memory)
        return
    _set_routing_debug(routing)
    # Stash the rules-fallback note so renderers below can attach it to the
    # outgoing assistant message regardless of which branch handles it.
    st.session_state["_pending_source_note"] = _rules_fallback_note(routing, settings)

    intent = routing["intent"]
    if intent == "edit":
        _handle_edit(
            request=request, selected_sheet=selected_sheet, sheet_columns=sheet_columns,
            loaded=loaded, settings=settings, memory=memory, followup=followup,
        )
        _save_memory(memory)
        return

    if routing["requires_confirmation"]:
        _store_pending(routing, memory, request)
        # Update workspace view
        view = st.session_state.get("workspace_view")
        if isinstance(view, dict):
            view["mode"] = "pending_action"
            view["pending_action_summary"] = routing.get("confirmation_reason") or "Please confirm before continuing."
            update0 = routing.get("active_update") or {}
            view["pending_action_label"] = (routing.get("pending_type") or "update").replace("_", " ").title()
            view["target_column"] = (
                update0.get("column_name")
                or (routing.get("plan") or {}).get("column_name")
                or routing.get("column_name")
            )
            # The Proposed Update card lives in the Results segment.
            st.session_state["workspace_segment"] = "Results"
            try:
                from core.confirmed_actions import get_target_rows_mask
                update = routing.get("active_update") or {}
                sheet_name = update.get("sheet") or routing.get("sheet") or selected_sheet
                frame = loaded.sheets.get(sheet_name) if loaded else None
                if frame is not None:
                    filters = update.get("filters") or routing.get("filters") or []
                    mask = get_target_rows_mask(frame, filters)
                    affected_rows = int(mask.sum())
                    pending_preview_df = frame.loc[mask].head(20)
                    view["affected_rows"] = affected_rows
                    view["pending_preview_df"] = pending_preview_df
                else:
                    view["affected_rows"] = None
                    view["pending_preview_df"] = None
            except Exception:
                view["affected_rows"] = None
                view["pending_preview_df"] = None
        _set_clarify(
            request,
            routing.get("confirmation_reason") or "Please confirm before I continue.",
            reason="This needs your confirmation.",
            options=_confirm_options(routing),
        )
        _save_memory(memory)
        _log_turn(request=request, routing=routing, sheet_columns=sheet_columns, settings=settings)
        return

    if intent in {"clarify", "unavailable", "unsupported"}:
        _set_clarify(
            request,
            routing.get("confirmation_reason") or routing.get("fallback_reason")
            or "Could you tell me a bit more? For example, name the action and the column you have in mind.",
            reason="Needs clarification.",
            options=list(routing.get("clarify_options") or []),
        )
        _capture_failure(
            request=request, intent=intent,
            reason=routing.get("fallback_reason") or routing.get("confirmation_reason") or "needs_clarification",
            routing=routing,
        )
        memory.record_clarify(request=request)
        _save_memory(memory)
        _log_turn(request=request, routing=routing, sheet_columns=sheet_columns, settings=settings)
        return

    # Router: deterministic specialists vs free-form generalist.
    # The analyst (opt-in) answers OPEN-ENDED questions by writing+running pandas
    # locally — the "like Claude on a spreadsheet" path. But the planner also
    # emits SPECIALIZED operations (intervention/pivot/trend/advisor summaries,
    # dashboards) that are purpose-built, tested, and produce structured output.
    # Those must win: we only hand a turn to the analyst when the plan is a
    # GENERIC read (allowlist below). Everything else falls through to the
    # deterministic dispatcher. Edits/exports already returned above.
    if (intent == "query" and settings.get("code_analyst_enabled")
            and not settings.get("strict_privacy_mode", False)
            and (routing.get("plan") or {}).get("operation", "") in _ANALYST_GENERIC_OPS):
        if _run_code_analyst(request=request, planner_request=planner_request,
                             selected_sheet=selected_sheet, loaded=loaded,
                             settings=settings, memory=memory):
            return

    # query: execute through the dispatcher and render.
    reveal = bool(routing.get("reveal_sensitive"))
    workbook = ensure_session_workbook(loaded, profile)
    response = execute_planned_request(routing, loaded, settings, reveal_sensitive=reveal, request_summary=request,
                                       session_workbook=workbook)
    _render_query_response(request, routing, response, memory)
    _focus_workbook_on_new_sheet(response.get("sheet_outcome"))
    _save_memory(memory)
    entry_id = _log_turn(request=request, routing=routing, response=response,
                         sheet_columns=sheet_columns, settings=settings,
                         sheet_outcome=response.get("sheet_outcome"))
    _tag_latest_result_attachment(entry_id)


def _focus_workbook_on_new_sheet(sheet_outcome) -> None:
    """If the dispatcher created a session-workbook sheet for this turn, ask
    the unified workbook panel to jump to it on the next render."""
    if not isinstance(sheet_outcome, dict):
        return
    if sheet_outcome.get("action") != "created":
        return
    sheet_name = sheet_outcome.get("sheet_name")
    if not sheet_name:
        return
    st.session_state["_pending_view_sheet"] = ("generated", sheet_name)


# Routing helpers -------------------------------------------------------------


def _confirm_options(routing: dict) -> list[str]:
    if routing["intent"] == "export":
        return ["Yes, export", "No, cancel"]
    return ["Yes, do it", "No, cancel"]


def _store_pending(routing: dict, memory: SessionMemory, request: str) -> None:
    memory.pending_action = {
        "type": routing.get("pending_type") or routing["intent"],
        "reason": routing.get("confirmation_reason") or "",
        "routing": routing,
        "request": request,
    }
    update = routing.get("active_update")
    if update:  # a sensitive query already has a composed selection
        memory.set_active_filters(update.get("filters", []), update.get("sheet", ""))
        st.session_state["active_context"] = describe_filters(update.get("filters", []))


def _resolve_pending(memory: SessionMemory, loaded, settings: dict[str, Any],
                     *, session_workbook: SessionWorkbook | None = None) -> dict | None:
    action = memory.pending_action or {}
    routing = action.get("routing") or {}
    reveal = action.get("type") == "show_sensitive"
    response = execute_planned_request(
        routing, loaded, settings, reveal_sensitive=reveal, request_summary=action.get("request", ""),
        session_workbook=session_workbook,
    )
    if routing.get("intent") == "query":
        _render_query_response(action.get("request", ""), routing, response, memory, reveal=reveal)
    else:
        _render_action_response(response, memory)
    return response


def _run_code_analyst(*, request, planner_request, selected_sheet, loaded,
                      settings, memory) -> bool:
    """Answer a read question by writing+running pandas locally. Returns True
    if it handled the turn, False to fall back to the dispatcher.

    The model writes code against a copy of the sheet in a locked-down
    namespace (no imports beyond pandas/numpy, no file/network), sees the real
    output, and answers in plain English. We only show the answer when it is
    grounded in code that actually ran — otherwise we fall back rather than
    surface a fabricated number.
    """
    frame = loaded.sheets.get(selected_sheet) if loaded else None
    if frame is None or frame.empty:
        return False
    try:
        from nlp.code_analyst import analyze, default_llm_call
        from nlp.glossary import build_glossary_block

        result = analyze(
            user_message=planner_request,
            df=frame,
            llm_call=default_llm_call(
                settings.get("planner_model") or settings.get("ollama_model") or "llama3.2:3b"
            ),
            history=list(getattr(memory, "recent_turns", []) or []),
            glossary=build_glossary_block(),
        )
    except Exception:
        return False  # any failure → let the deterministic dispatcher try

    # Model outage shouldn't be a silent blank. If the analyst couldn't reach the
    # local model, tell the user plainly, then fall back to the deterministic
    # tools (which need no model) for this turn.
    if result.error and any(
        token in result.error.lower()
        for token in ("model call failed", "connection refused", "unavailable", "no response")
    ):
        st.session_state["analyst_trace_debug"] = {"question": planner_request,
                                                    "error": result.error, "grounded": False}
        _say("⚠️ The local AI model isn't reachable right now, so I answered with the built-in "
             "tools. Start Ollama (the AI analyst) to re-enable richer answers.")
        return False

    if not result.ok or not result.grounded:
        return False

    # Remember this turn so the next question can build on it ("just the
    # Biology ones", "break that down by year").
    memory.record_analyst_turn(
        question=planner_request,
        answer=result.answer,
        code=result.code_steps[-1] if result.code_steps else "",
    )

    attachment = None
    if result.code_steps:
        # Keep the executed code available for the Developer Tools view without
        # cluttering the office-facing answer.
        attachment = {"type": "analyst_trace",
                      "code_steps": list(result.code_steps),
                      "outputs": list(result.outputs),
                      "plan": result.plan,
                      "confidence": result.confidence,
                      "verified": result.verified}
    # Stash the trace for the Developer / debug panel (last analyst answer).
    st.session_state["analyst_trace_debug"] = {
        "question": planner_request,
        "plan": result.plan,
        "confidence": result.confidence,
        "verified": result.verified,
        "grounded": result.grounded,
        "iterations": result.iterations,
        "steps": [
            {"code": c, "output": o}
            for c, o in zip(result.code_steps, result.outputs)
        ],
    }
    # When the cross-check disagreed or the run was shaky, tell the user plainly
    # instead of presenting a low-confidence number as settled fact.
    answer = result.answer
    if result.verified is False:
        answer += "\n\n⚠️ I couldn't confirm this — a second cross-check disagreed, so please verify it."
    elif result.confidence == "medium":
        answer += "\n\n_(worth a quick double-check.)_"
    _say(answer, attachment=attachment)
    _save_memory(memory)
    _log_turn(request=request,
              routing={"intent": "query", "plan_source": "code_analyst",
                       "plan": {"plain_english_question": planner_request}},
              sheet_columns={selected_sheet: list(frame.columns)},
              settings=settings)
    return True


def _resolved_suggestions(request: str, response: dict, routing: dict) -> list[str]:
    """Return the post-answer suggestion list.

    Priority: (1) the curated ``follow_ups`` for the question library template
    the user clicked (or that their typed message exactly matches), (2) the
    dynamic suggester that the dispatcher already returned. A popped
    ``_clicked_template_id`` is consumed so it doesn't leak to the next turn.
    """
    template_id = st.session_state.pop("_clicked_template_id", None)
    template = lookup_by_id(template_id) if template_id else lookup_for_message(request)
    if template and template.follow_ups:
        return list(template.follow_ups)
    return list(response.get("suggestions") or routing.get("suggestions") or [])


def _render_query_response(request, routing, response, memory, *, reveal: bool = False) -> None:
    if not response["success"]:
        active_cols = st.session_state.get("_failure_capture_context", {}).get("columns") or []
        sheets = list(st.session_state.get("cached_loaded").sheets.keys()) if st.session_state.get("cached_loaded") else []
        friendly = friendly_validation_error(response["message"], active_cols, sheets)
        _set_clarify(request, friendly, reason="That query could not run.")
        _capture_failure(
            request=request, intent="query_failed",
            reason=response.get("message") or "query_execution_error",
            routing=routing,
        )
        memory.record_clarify(request=request)
        return

    plan = routing.get("plan") or {}
    table = response["result_preview"]
    removed = response["removed"]
    confidence = float(routing.get("confidence") or 0.0)
    suggestions = _resolved_suggestions(request, response, routing)

    review_note: str | None = None
    if confidence < HIGH_CONFIDENCE:
        review_note = "Medium confidence — please review the interpretation above."
    if reveal and table:
        review_note = "This view includes sensitive student-level fields. Handle and share it carefully."

    st.session_state["assistant_mode"] = "ask_question"
    st.session_state["ask_question"] = plan.get("plain_english_question") or request
    st.session_state["ask_operation"] = response["operation"]
    st.session_state["ask_description"] = response["description"]
    st.session_state["ask_explanation"] = response["explanation"]
    st.session_state["ask_value"] = response["value"]
    st.session_state["ask_row_count"] = response["row_count"]
    st.session_state["ask_table"] = table
    st.session_state["ask_columns_used"] = response["columns"]
    st.session_state["ask_confidence"] = confidence
    st.session_state["ask_source"] = "local_llm" if routing.get("plan_source") == "llm" else "rule"
    st.session_state["ask_preview_truncated"] = response["preview_truncated"]
    st.session_state["ask_redacted"] = removed
    st.session_state["ask_review_note"] = review_note
    st.session_state["ask_narration"] = response.get("narration") or routing.get("narration") or ""
    st.session_state["ask_conversation_llm_used"] = bool(response.get("conversation_llm_used"))
    st.session_state["ask_assumption_note"] = response.get("assumption_note") or routing.get("assumption_note") or ""
    st.session_state["ask_alternatives"] = list(response.get("alternatives") or routing.get("alternatives") or [])
    st.session_state["ask_suggestions"] = list(suggestions)
    st.session_state["ask_band"] = response.get("band") or routing.get("band") or "high"
    st.session_state["current_request"] = request
    st.session_state["active_context"] = describe_filters(plan.get("filters") or [])

    # Update workspace view
    import pandas as pd
    view = st.session_state.get("workspace_view")
    if isinstance(view, dict):
        view["mode"] = "result"
        view["result_df"] = pd.DataFrame(table) if table else pd.DataFrame()
        view["active_filter"] = describe_filters(plan.get("filters") or [])
        view["group_by"] = plan.get("group_by")
        view["columns_used"] = response.get("columns") or []
        view["row_count"] = response.get("row_count")
        view["title"] = response.get("description")
        view["pending_action_summary"] = response.get("description")
        # Auto-focus the right-panel Results segment after a query.
        st.session_state["workspace_segment"] = "Results"

    memory.record_ask(
        request=request,
        query_plan=plan,
        result_description=response["description"],
        row_count=response["row_count"],
        columns_used=response["columns"],
        sheet=plan.get("sheet", ""),
        summary_table=table,
        top_group=response.get("top_group"),
    )

    note = ""
    if removed:
        note = f" (Hidden by default: {', '.join(removed)}. Ask to show them if needed.)"
    attachment = {
        "type": "result",
        "table": list(table or []),
        "row_count": response["row_count"],
        "value": response["value"],
        "operation": response["operation"],
        "preview_truncated": response["preview_truncated"],
        "redacted": list(removed or []),
        "review_note": review_note,
        # Carry the medium-band offer + grounded next-moves on the attachment so
        # each historical chat message can re-render its own chips later.
        "alternatives": list(response.get("alternatives") or routing.get("alternatives") or []),
        "suggestions": list(suggestions),
        "assumption_note": response.get("assumption_note") or routing.get("assumption_note") or "",
        "band": response.get("band") or routing.get("band") or "high",
        "columns": list(response.get("columns") or routing.get("plan", {}).get("columns_used") or []),
        "filters": list(routing.get("plan", {}).get("filters") or []),
    }
    # When the conversational LLM produced the message, narration is already
    # woven in — don't double up. Otherwise prefer narration over context_note.
    if response.get("conversation_llm_used"):
        lead = ""
    elif response.get("narration"):
        # The deterministic message already starts with the narration sentence
        # (see execution_dispatcher._run_query), so no extra lead is needed.
        lead = ""
    else:
        lead = routing.get("context_note") or ""
    threshold_note = _threshold_note_for_plan(plan)
    if threshold_note:
        lead = lead + threshold_note + " "
    _say(lead + response["message"] + note, attachment=attachment)


def _threshold_note_for_plan(plan: dict[str, Any]) -> str:
    filters = plan.get("filters") or []
    if not filters:
        return ""
    settings = load_risk_settings(st.session_state)
    notes: list[str] = []
    for condition in filters:
        column = str(condition.get("column") or "").strip().lower()
        operator = str(condition.get("operator") or "").strip().lower()
        value = condition.get("value")
        if column in {"gpa", "gpa risk"} and (
            operator in {"less_than", "less_or_equal"} or value is True
        ):
            notes.append(
                f"I interpreted GPA risk as GPA below {settings.mention_gpa_risk().split('below ', 1)[-1]}."
            )
        if column in {"attendance rate", "attendance risk"} and (
            operator in {"less_than", "less_or_equal", "equals"} or value is True
        ):
            notes.append(
                f"I interpreted attendance risk as {settings.mention_attendance_risk()}."
            )
    return " ".join(dict.fromkeys(notes))


def _render_action_response(response, memory: SessionMemory) -> None:
    from pathlib import Path  # local: cheap and avoids re-importing at module load

    if not response.get("success"):
        active_cols = st.session_state.get("_failure_capture_context", {}).get("columns") or []
        sheets = list(st.session_state.get("cached_loaded").sheets.keys()) if st.session_state.get("cached_loaded") else []
        friendly = friendly_validation_error(response.get("message") or "Action failed", active_cols, sheets)
        st.session_state["assistant_mode"] = "clarify"
        st.session_state["clarify_question"] = friendly
        st.session_state["clarify_reason"] = "Action failed"
        st.session_state["clarify_options"] = []
        _say(friendly, attachment=None)
        return

    output_file = response.get("output_file")
    if output_file:
        st.session_state["latest_output_file"] = output_file
        st.session_state["latest_output_changes"] = response.get("message") or ""
        # Track every output file produced during this session so the Export
        # Center can show a list (Phase K.6).
        st.session_state.setdefault("export_history", []).insert(0, Path(output_file).name)
        
        # Update workspace view
        view = st.session_state.get("workspace_view")
        if isinstance(view, dict):
            view["mode"] = "export_ready"
            view["export_filename"] = Path(output_file).name
            view["download_path"] = output_file
            view["change_summary"] = [response.get("message") or ""]
            
    st.session_state["assistant_mode"] = None
    attachment = None
    if output_file:
        attachment = {"type": "download", "path": output_file}
    _say(response["message"], attachment=attachment)


def _log_turn(
    *,
    request: str,
    routing: dict | None = None,
    response: dict | None = None,
    sheet_columns: dict[str, list[str]] | None = None,
    settings: dict[str, Any] | None = None,
    confirmation_result: str | None = None,
    sheet_outcome: dict[str, Any] | None = None,
) -> str | None:
    """Append one sanitized record per turn (L.5 + L.6). Best-effort: never raises.

    Returns the newly written entry id so the caller can tag the latest chat
    message's attachment with it (used by alternative chips to attribute their
    click as a correction of THIS interpretation).
    """
    settings = settings or {}
    if not settings.get("interaction_logging_enabled", True):
        return None
    try:
        session_id = st.session_state.setdefault("interaction_session_id", str(uuid.uuid4()))
        target = st.session_state.pop("_pending_correction_target", None)
        correction_msg = request if target else None
        entry_id = log_interaction(
            user_message=request,
            routing=routing,
            response=response,
            session_id=session_id,
            sheet_columns=sheet_columns,
            confirmation_result=confirmation_result,
            corrects_entry_id=target,
            correction_message=correction_msg,
            sheet_outcome=sheet_outcome,
        )
        if entry_id:
            st.session_state["last_interaction_id"] = entry_id
        return entry_id
    except OSError:
        # Disk full / permission denied — do not break the chat over logging.
        return None


def ensure_session_workbook(loaded, profile) -> SessionWorkbook | None:
    """Return the SessionWorkbook bound to the currently loaded workbook.

    Creates a fresh instance on first use; calls reset_for_new_source when the
    user has uploaded a different file (per the v0.3 design: reset only on new
    upload). Returns None if there's no workbook loaded — callers can safely
    pass None through to the dispatcher.
    """
    if loaded is None or profile is None:
        return None
    schema_hash = workbook_schema_hash({sheet.name: sheet.columns for sheet in profile.sheets})
    workbook: SessionWorkbook | None = st.session_state.get("session_workbook")
    if workbook is None:
        workbook = SessionWorkbook(source_file_name=loaded.file_name, schema_hash=schema_hash)
        st.session_state["session_workbook"] = workbook
    elif not workbook.is_bound_to(source_file_name=loaded.file_name, schema_hash=schema_hash):
        workbook.reset_for_new_source(source_file_name=loaded.file_name, schema_hash=schema_hash)
    return workbook


def _tag_latest_result_attachment(entry_id: str | None) -> None:
    """Stamp the just-logged entry id onto the latest result attachment, so
    alternative chips can attribute their click as a correction of *this*
    interpretation. Safe no-op if there is no attachment to tag."""
    if not entry_id:
        return
    messages = st.session_state.get("chat_messages") or []
    if not messages:
        return
    latest = messages[-1]
    attachment = latest.get("attachment")
    if isinstance(attachment, dict) and attachment.get("type") == "result":
        attachment["entry_id"] = entry_id


def _set_routing_debug(routing: dict) -> None:
    st.session_state["routing_debug"] = {
        "plan_source": routing.get("plan_source"),
        "intent": routing.get("intent"),
        "confidence": routing.get("confidence"),
        "llm_used": routing.get("llm_used"),
        "validation_status": routing.get("validation", {}).get("status"),
        "fallback_reason": routing.get("fallback_reason"),
        "requires_confirmation": routing.get("requires_confirmation"),
        "narration": routing.get("narration"),
        "band": routing.get("band"),
        "assumption_note": routing.get("assumption_note"),
        "alternatives": routing.get("alternatives"),
        "suggestions": routing.get("suggestions"),
    }


# Edit path -------------------------------------------------------------------


def _handle_edit(
    *,
    request: str,
    selected_sheet: str,
    sheet_columns: dict[str, list[str]],
    loaded,
    settings: dict[str, Any],
    memory: SessionMemory,
    followup,
) -> None:
    synthesized = None
    if followup.is_followup and followup.resolved:
        synthesized = _synthesize_followup_command(request, followup.filters, followup.sheet or selected_sheet)

    if synthesized is not None:
        commands = [synthesized]
        plan = {
            "request_type": "edit_workbook",
            "plan_type": "single_action",
            "confidence": 0.9,
            "plain_english_summary": f"{command_to_confirmation(synthesized)} ({followup.referent_note})",
            "commands": commands,
            "assumptions": [followup.referent_note],
            "clarification_question": "",
            "source": "followup",
            "validation_error": None,
        }
    else:
        try:
            result = plan_edit(
                user_request=request,
                selected_sheet=selected_sheet,
                sheet_columns=sheet_columns,
                sheets=loaded.sheets,
                original_file_name=loaded.file_name,
                use_local_llm=bool(settings.get("use_local_llm")),
                ollama_model=settings.get("ollama_model", "llama3.2:3b"),
            )
        except Exception as exc:
            _set_clarify(
                request,
                "I couldn't turn that into a safe plan. Could you name the action and the column you have in mind?",
                reason="The request could not be planned.",
            )
            _capture_failure(
                request=request, intent="edit_planning_error",
                reason=f"{type(exc).__name__}: {exc}",
            )
            memory.record_clarify(request=request)
            return
        if result.request_type == "clarify":
            _set_clarify(
                request,
                result.clarification_question or "Which action and column should I use?",
                reason="The request needs more detail.",
            )
            _capture_failure(
                request=request, intent="edit_clarify",
                reason=result.clarification_question or "edit_needs_more_detail",
            )
            memory.record_clarify(request=request)
            return
        plan = result.to_dict()

    commands = plan.get("commands") or []
    first = commands[0] if commands else {}
    confidence = float(plan.get("confidence") or 0.0)
    level = _confidence_level(confidence)
    unsupported = [c.get("action") for c in commands if c.get("action") not in EXECUTABLE_PLANNER_ACTIONS]

    warning = None
    can_execute = bool(commands) and not plan.get("validation_error") and level != "low" and not unsupported
    if unsupported:
        warning = f"The '{unsupported[0]}' step can be planned but isn't executable yet, so this is a preview."
    elif level == "medium" and not plan.get("validation_error"):
        warning = "I think I understood this, but please review before running."
    elif len(commands) > 1 and not plan.get("validation_error"):
        warning = f"This is a {len(commands)}-step plan. Review the steps, then run."

    st.session_state["assistant_mode"] = "edit_workbook"
    st.session_state["current_request"] = request
    st.session_state["current_command"] = first
    st.session_state["current_plan"] = plan
    st.session_state["current_confirmation"] = plan.get("plain_english_summary")
    st.session_state["current_confidence"] = confidence
    st.session_state["current_confidence_level"] = level
    st.session_state["current_source"] = plan.get("source", "rule_parser")
    st.session_state["current_validation_error"] = plan.get("validation_error")
    st.session_state["current_can_execute"] = can_execute
    st.session_state["current_clarification"] = None
    st.session_state["current_warning"] = warning or (
        "I think I understood this, but please review before running." if level == "medium" else None
    )
    st.session_state["medium_review_confirmed"] = False
    st.session_state["edit_command_mode"] = False
    st.session_state["show_correction_form"] = False

    # Update workspace view
    view = st.session_state.get("workspace_view")
    if isinstance(view, dict):
        view["mode"] = "pending_action"
        view["pending_action_summary"] = plan.get("plain_english_summary") or "Apply changes to workbook."
        first_cmd0 = (plan.get("commands") or [{}])[0]
        view["pending_action_label"] = str(first_cmd0.get("action") or "Update").replace("_", " ").title()
        view["target_column"] = first_cmd0.get("column_name") or first_cmd0.get("column")
        st.session_state["workspace_segment"] = "Results"
        try:
            from core.confirmed_actions import get_target_rows_mask
            active_sheet = plan.get("commands", [{}])[0].get("sheet") if plan.get("commands") else selected_sheet
            frame = loaded.sheets.get(active_sheet) if loaded else None
            if frame is not None:
                first_cmd = plan["commands"][0] if plan.get("commands") else {}
                filters = first_cmd.get("conditions") or plan.get("filters") or []
                mask = get_target_rows_mask(frame, filters)
                affected_rows = int(mask.sum())
                pending_preview_df = frame.loc[mask].head(20)
                view["affected_rows"] = affected_rows
                view["pending_preview_df"] = pending_preview_df
            else:
                view["affected_rows"] = None
                view["pending_preview_df"] = None
        except Exception:
            view["affected_rows"] = None
            view["pending_preview_df"] = None

    memory.record_edit(request=request, edit_plan=plan, sheet=selected_sheet)
    summary = plan.get("plain_english_summary") or "Here is the planned change. Review it before running."
    attachment = {
        "type": "edit_plan",
        "plan": plan,
        "confidence": confidence,
        "level": level,
        "warning": warning,
        "validation_error": plan.get("validation_error"),
        "can_execute": can_execute,
        "resolved": None,  # "executed" | "cancelled" | None
    }
    _say(summary, attachment=attachment)


def _synthesize_followup_command(
    request: str, filters: list[dict[str, Any]], sheet: str
) -> dict[str, Any] | None:
    text = normalize_text(request)
    if not filters:
        return None
    if "highlight" in text or "mark" in text or "color" in text or "colour" in text:
        return {
            "action": "highlight_rows",
            "sheet": sheet,
            "conditions": list(filters),
            "format": {"fill_color": "yellow"},
        }
    if any(word in text for word in ("filter", "show", "list", "pull", "find")):
        return {
            "action": "filter_rows",
            "sheet": sheet,
            "conditions": list(filters),
            "output_sheet": "Selected Rows",
        }
    return None


# Rendering -------------------------------------------------------------------


def _chip(text: str) -> str:
    return f'<span class="status-badge badge-gray">{text}</span>'




def render_current_view(loaded, profile, selected_sheet: str, settings: dict[str, Any]) -> None:
    """The 'Current view' card: a plain-language summary of what the assistant is
    currently looking at (all students, a filtered slice, or a pending update),
    plus the active filter/sort/group chips."""
    memory = SessionMemory.from_dict(st.session_state.get("assistant_memory"))
    view = st.session_state.get("workspace_view") or {}

    chips: list[str] = []
    for condition in memory.active_filters:
        column = condition.get("column", "?")
        operator = str(condition.get("operator", "")).replace("_", " ")
        value = condition.get("value")
        if condition.get("operator") in {"is_missing", "is_not_missing", "is_blank", "is_not_blank"}:
            chips.append(_chip(f"{column} {operator}"))
        elif isinstance(value, list):
            chips.append(_chip(f"{column} in [{', '.join(map(str, value))}]"))
        else:
            chips.append(_chip(f"{column} {operator} {value}"))
    if memory.active_sort:
        chips.append(_chip(f"Sort: {memory.active_sort.get('column')} {memory.active_sort.get('direction', 'asc')}"))
    if memory.active_group_by:
        chips.append(_chip(f"Group by: {memory.active_group_by}"))
    if memory.active_limit:
        chips.append(_chip(f"Limit: {memory.active_limit}"))

    st.markdown("**Current view**")

    has_context = bool(memory.active_filters or memory.active_sort or memory.active_group_by)
    if not has_context:
        st.caption("All students — no active filters applied.")
        return

    total = None
    try:
        if selected_sheet and hasattr(loaded, "sheets"):
            total = len(loaded.sheets[selected_sheet])
    except Exception:
        total = None
    count = view.get("row_count") if view.get("mode") == "result" else None

    summary_bits: list[str] = []
    if count is not None and total:
        summary_bits.append(f"{count:,} of {total:,} students")
    elif total:
        summary_bits.append(f"{total:,} students")
    if memory.active_group_by:
        summary_bits.append(f"Grouped by {memory.active_group_by}")
    if summary_bits:
        st.success(" · ".join(summary_bits))
    if chips:
        st.markdown(" ".join(chips), unsafe_allow_html=True)


# Backwards-compatible alias (older callers / tests).
render_active_context = render_current_view


def render_chat_panel(loaded, profile, selected_sheet: str, settings: dict[str, Any]) -> None:
    if not loaded:
        st.markdown(
            '<div class="empty-state"><strong>Upload a workbook, then ask:</strong>'
            '<ul>'
            '<li>Top 10 students by GPA</li>'
            '<li>List every advisor</li>'
            '<li>Show me freshmen (or 5th graders)</li>'
            '<li>Show me just student name and GPA</li>'
            '<li>Which students are below 2.0 GPA?</li>'
            '</ul>'
            '<p style="margin-top:8px; opacity:0.75;">After upload, the chat panel will offer suggestions tailored to your sheet.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # Keep one chat_input mounted for the existing AppTest/e2e harnesses. CSS
    # hides it in the browser; the visible ask box is the inline form below.
    legacy_request = st.chat_input("Ask a follow-up...", key="assistant_legacy_chat_input")

    messages = st.session_state.get("chat_messages") or []
    history = st.container(height=420, border=True)
    with history:
        if not messages:
            st.caption("Ask anything about your workbook to get started.")
        for index, message in enumerate(messages):
            role = message["role"]
            with st.chat_message(role):
                st.markdown(message["content"])
                source_note = message.get("source_note")
                if source_note:
                    st.caption(source_note)
                # Result tables, confirmation cards, and edit-plan controls now
                # live in the Live Output panel; the chat bubble shows the
                # assistant's text only. We still keep `attachment` on the
                # message so the Live Output panel can read the latest one.
                attachment = message.get("attachment")
                if attachment and attachment.get("type") == "result":
                    row_count = attachment.get("row_count")
                    if isinstance(row_count, int) and row_count and (attachment.get("table") or attachment.get("value") is not None):
                        st.caption(f"Result table shown in Live Output ({row_count} row(s)).")
                elif attachment and attachment.get("type") == "edit_plan":
                    st.caption("Planned change shown in Live Output — review and Run.")
                elif attachment and attachment.get("type") == "confirmation":
                    st.caption("Confirmation needed — see Live Output.")
                elif attachment and attachment.get("type") == "download":
                    st.caption("File ready in Export Center.")

    submitted_request = ""
    with st.form("assistant_inline_ask_form", clear_on_submit=True, border=False):
        typed_request = st.text_area(
            "Ask the assistant",
            placeholder="Ask about this roster...",
            height=88,
            key="assistant_inline_request",
            label_visibility="collapsed",
        )
        ask_col, spacer_col = st.columns([0.32, 0.68])
        with ask_col:
            submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)
        with spacer_col:
            st.caption("Answers use the uploaded workbook; edits require confirmation.")
        if submitted:
            submitted_request = (typed_request or "").strip()

    request = (legacy_request or submitted_request or "").strip()
    if request:
        # Free-typed message: it may still match a library template — let the
        # post-answer suggester see if it does.
        st.session_state.pop("_clicked_template_id", None)
        route_message(
            request=request,
            selected_sheet=selected_sheet,
            loaded=loaded,
            profile=profile,
            settings=settings,
        )
        st.rerun()

    # "Current view" and "Suggested actions" cards were removed; the only
    # remaining prompt aid is data-aware figure suggestions.
    _render_figure_suggestions(loaded, profile, selected_sheet, settings)


def _render_figure_suggestions(loaded, profile, selected_sheet: str, settings: dict[str, Any]) -> None:
    """Clickable figure chips, generated from the uploaded sheet's actual
    columns. Each chip is guaranteed to produce a chart when clicked."""
    if loaded is None or profile is None or not selected_sheet:
        return
    columns = next((sheet.columns for sheet in profile.sheets if sheet.name == selected_sheet), [])
    suggestions = suggested_figure_questions(list(columns))
    if not suggestions:
        return
    st.caption("📊 Suggested figures")
    cols = st.columns(2)
    for index, (label, query) in enumerate(suggestions):
        with cols[index % 2]:
            if st.button(label, key=f"figsugg_{index}", use_container_width=True):
                route_message(
                    request=query,
                    selected_sheet=selected_sheet,
                    loaded=loaded,
                    profile=profile,
                    settings=settings,
                )
                st.rerun()


def _context_ack(action: str, memory: SessionMemory) -> None:
    message = (
        "Starting over with a clean slate. No active filters."
        if action == RESET
        else "Cleared the active filters. Showing the full dataset again."
    )
    st.session_state["assistant_mode"] = "ask_question"
    st.session_state["ask_question"] = ""
    st.session_state["ask_operation"] = "context"
    st.session_state["ask_description"] = message
    st.session_state["ask_explanation"] = None
    st.session_state["ask_value"] = None
    st.session_state["ask_row_count"] = None
    st.session_state["ask_table"] = []
    st.session_state["ask_columns_used"] = []
    st.session_state["ask_confidence"] = 1.0
    st.session_state["ask_source"] = "rule"
    st.session_state["ask_preview_truncated"] = False
    st.session_state["ask_review_note"] = None
    st.session_state["active_context"] = describe_filters(memory.active_filters)
    _say(message)


def _confidence_level(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def _reset_views() -> None:
    st.session_state["assistant_mode"] = None
    for key in (
        "ask_question",
        "ask_operation",
        "ask_description",
        "ask_explanation",
        "ask_value",
        "ask_table",
        "ask_review_note",
        "ask_redacted",
        "clarify_question",
        "clarify_reason",
        "clarify_options",
    ):
        st.session_state.pop(key, None)


def _capture_failure(
    *,
    request: str,
    intent: str,
    reason: str,
    routing: dict[str, Any] | None = None,
) -> None:
    """Append a failure record so we can triage real-world phrasings later.

    Reads the active sheet + columns from the route_message-time context
    stash, so callers in the query/edit branches don't need to thread the
    workbook profile through.
    """
    ctx = st.session_state.get("_failure_capture_context") or {}
    log_failure(
        user_message=request,
        sheet_name=ctx.get("sheet", ""),
        columns=ctx.get("columns") or [],
        intent=intent,
        reason=reason,
        routing=routing,
    )


def _set_clarify(request: str, question: str, *, reason: str = "", options: list[str] | None = None) -> None:
    st.session_state["assistant_mode"] = "clarify"
    st.session_state["current_request"] = request
    st.session_state["clarify_question"] = question
    st.session_state["clarify_reason"] = reason
    st.session_state["clarify_options"] = options or []
    attachment: dict[str, Any] | None = None
    if options:
        attachment = {
            "type": "confirmation",
            "options": list(options),
            "reason": reason,
            "resolved": None,
        }
    _say(question, attachment=attachment)


def _say(text: str, attachment: dict[str, Any] | None = None) -> None:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
        "timestamp": _now(),
    }
    if attachment:
        message["attachment"] = attachment
    source_note = st.session_state.pop("_pending_source_note", None)
    if source_note:
        message["source_note"] = source_note
    st.session_state["chat_messages"].append(message)


def _rules_fallback_note(routing: dict, settings: dict[str, Any]) -> str | None:
    """If the user has the local LLM enabled but planning answered without
    it this turn, surface a school-office-friendly one-liner. Returns None
    when there's nothing to say. Developer-flavoured detail belongs in the
    Developer Tools toggle, not here."""
    if not settings.get("use_local_llm"):
        return None
    if settings.get("strict_privacy_mode", True):
        return None
    plan_source = (routing.get("plan_source") or "").lower()
    if plan_source in {"llm", "local_llm"}:
        return None
    fallback_reason = str(routing.get("fallback_reason") or "").strip()
    if fallback_reason.lower().startswith("llm plan failed validation"):
        return None
    if fallback_reason:
        return None
    return None


def _mark_pending_resolved(state: str) -> None:
    """When a confirmation is resolved (or expired), mark the latest pending
    confirmation card so its buttons hide."""
    messages = st.session_state.get("chat_messages") or []
    for index in range(len(messages) - 1, -1, -1):
        attachment = messages[index].get("attachment") or {}
        if attachment.get("type") == "confirmation" and attachment.get("resolved") is None:
            attachment["resolved"] = state
            break


def mark_latest_edit_plan_resolved(state: str) -> None:
    """Public helper for app.py to mark an edit-plan card as executed/cancelled
    once the user clicks Run or Cancel."""
    messages = st.session_state.get("chat_messages") or []
    for index in range(len(messages) - 1, -1, -1):
        attachment = messages[index].get("attachment") or {}
        if attachment.get("type") == "edit_plan" and attachment.get("resolved") is None:
            attachment["resolved"] = state
            break


def append_assistant_message(text: str, attachment: dict[str, Any] | None = None) -> None:
    """Public helper for app.py to add an assistant message from outside the
    router (for example, after Run Action completes)."""
    _say(text, attachment=attachment)


def _save_memory(memory: SessionMemory) -> None:
    st.session_state["assistant_memory"] = memory.to_dict()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
