"""Single planning entry point: deterministic rules first, safe LLM fallback.

Both the Streamlit chat UI and the eval harness call `plan_user_request`, so
there is one planning path. The router turns a message + conversation state into
a normalized routing object. It composes follow-up filters, parses action
parameters, classifies edit/clarify/unavailable/unsupported, and (for the LLM
path) parses+repairs+safety-validates the model's JSON. It never executes
anything and the LLM never sees rows.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import pandas as pd

from core.field_policy import field_status
from core.llm_config import from_app_settings
from core.privacy import is_hidden_by_default, requested_sensitive_columns
from core.schema import canonical_map, infer_column_types
from nlp.conversation import (
    FOLLOWUP,
    classify_context_action,
    compose_filters,
    describe_filters,
    has_hard_edit_cue,
    is_additive,
)
from nlp.model_prompt import build_repair_prompt, build_safe_planner_prompt, build_validation_repair_prompt
from nlp.narration import narrate_plan
from nlp.query_planner import (
    _detect_count_unique,
    _detect_group_by,
    _detect_list_unique,
    _detect_performance_query,
    _detect_value_column,
    plan_query,
)
from nlp.action_chain import ChainedAction, parse_action_chain
from nlp.ambiguity import AmbiguityResolution, detect_ambiguity
from nlp.drilldown import resolve_drilldown
from nlp.request_intents import (
    is_academic_watch_request,
    is_attendance_watch_request,
    is_export_request,
    is_note_request,
    parse_field_update,
    parse_note,
)
from nlp.suggestions import suggest_next_moves
from nlp.synonym_mapper import load_json, match_column_for_concept, normalize_text
from nlp.uncertainty import (
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    build_assumption_note,
    classify_confidence,
    detect_vague_alternatives,
    should_assume,
)
from nlp.vague_terms import (
    VagueResolution,
    message_has_vague_risk_term,
    resolve_vague_term,
)

RULES_CONFIDENCE = 0.7
MAX_STUDENT_PREVIEW = 500

_OPERATOR_MAP = {"greater_than_or_equal": "greater_or_equal", "less_than_or_equal": "less_or_equal"}
_ALLOWED_OPERATORS = {
    "equals", "not_equals", "contains", "not_contains", "contains_any",
    "contains_text", "not_contains_text",
    "starts_with", "ends_with", "greater_than", "greater_or_equal",
    "less_than", "less_or_equal", "between", "in", "not_in", "is_blank",
    "is_not_blank", "is_missing", "is_not_missing",
}
_NUMERIC_OPS = {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "between"}
# Operators that legitimately take no value — anything else with a missing
# value is incoherent (catches the 'Advisor contains None' LLM mistake).
_VALUE_FREE_OPS = {"is_blank", "is_not_blank", "is_missing", "is_not_missing"}
# Operators that semantically require a string-shaped column.
_STRING_ONLY_OPS = {"contains", "not_contains", "starts_with", "ends_with",
                    "contains_any", "contains_text", "not_contains_text"}
# Operators that perform free-text search on a narrative column. The dispatcher
# uses this set to decide whether to hide the note text in the preview.
_TEXT_SEARCH_OPS = {"contains_text", "not_contains_text"}
# Operators whose value must be a sequence.
_LIST_VALUE_OPS = {"in", "not_in", "between", "contains_any"}
# Low-cardinality categorical columns where 'equals'/'in' with an out-of-domain
# value almost certainly means the model invented an enum value.
_CATEGORICAL_DOMAIN_MAX_UNIQUE = 25
_OPERATION_MAP = {
    "filter": "filtered_preview", "count": "count_rows", "count_unique": "count_unique",
    "average": "average_column", "aggregate": "groupby_count", "sort": "filtered_preview",
    "summarize": "data_quality_summary",
}


class OllamaUnavailable(RuntimeError):
    pass


def plan_user_request(
    *,
    user_message: str,
    sheets: dict[str, pd.DataFrame],
    sheet_columns: dict[str, list[str]],
    selected_sheet: str,
    conversation_state: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    llm_call: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    config = from_app_settings(settings)
    columns = sheet_columns.get(selected_sheet, [])
    state = conversation_state or {}
    frame = sheets.get(selected_sheet)

    # 1. Action / edit intents are deterministic.
    action = _classify_action_intent(user_message, columns, state, selected_sheet)
    if action is not None:
        return action

    # 1a. Phase P drilldown — "which students are those" / "show me students
    # in that department" / "break that down by advisor". Resolves the prior
    # result context into a complete plan; runs before the ambiguity detector
    # so a drilldown after a top-N group result doesn't get re-interpreted as
    # a fresh container/aggregate question. The drilldown plan is intentionally
    # taken AS-IS — state merging would inherit the prior turn's group_by /
    # sort / limit, which is precisely what the drilldown is overriding.
    drilldown = resolve_drilldown(user_message, state, columns=columns)
    if drilldown is not None:
        empty_state = {"active_filters": [], "active_group_by": "",
                       "active_sort": {}, "active_limit": None,
                       "active_sheet": state.get("active_sheet", "")}
        routing = _build_query_routing(
            user_message, drilldown.plan, "rules", False, empty_state, columns,
            selected_sheet, band="high",
            assumption_override=drilldown.context_reminder,
        )
        routing["drilldown_kind"] = drilldown.kind
        return routing

    # 1b. Dean-roster ambiguity (Phase O). When the question has two plausible
    # interpretations ("teachers contain students with GPA below 2.5" — at
    # least one student vs. average), we take the container reading as the
    # primary plan and stash the aggregate alternative for the dispatcher to
    # compute alongside.
    ambiguity = detect_ambiguity(user_message, sheet=selected_sheet,
                                 columns=columns)
    if ambiguity is not None:
        routing = _build_query_routing(
            user_message, ambiguity.primary_plan, "rules", False, state, columns,
            selected_sheet, band="medium",
            assumption_override=_compose_interpretation_line(ambiguity),
            alternatives_override=[ambiguity.alternative_phrase],
        )
        routing["ambiguity"] = {
            "kind": ambiguity.kind,
            "alternative": ambiguity.alternative_spec,
            "alternative_phrase": ambiguity.alternative_phrase,
            "column_mapping": [list(item) for item in ambiguity.column_mapping],
        }
        return routing

    # 1c. Academic insight layer. Broad advisor/intervention questions should
    # become transparent workbook metrics, not improvised professional
    # judgments from the model.
    workbook_tool = _workbook_tool_query(user_message, selected_sheet, columns)
    if workbook_tool is not None:
        return _build_query_routing(
            user_message,
            workbook_tool["query"],
            "rules",
            False,
            state,
            columns,
            selected_sheet,
            band="medium",
            assumption_override=workbook_tool["assumption"],
        )

    academic_insight = _academic_insight_query(user_message, selected_sheet, columns)
    if academic_insight is not None:
        return _build_query_routing(
            user_message,
            academic_insight["query"],
            "rules",
            False,
            state,
            columns,
            selected_sheet,
            band="medium",
            assumption_override=academic_insight["assumption"],
        )

    # 2. Deterministic query planner.
    rule = plan_query(user_request=user_message, selected_sheet=selected_sheet,
                      sheet_columns=sheet_columns, frame=frame)
    if (
        rule.needs_clarification
        and rule.confidence >= HIGH_CONFIDENCE
    ):
        result = _result(
            plan_source="clarification",
            intent="clarify",
            confidence=rule.confidence,
            plan=None,
            llm_used=False,
            confirmation_reason=rule.clarification_question,
            fallback_reason=None,
            band="low",
        )
        result["clarify_options"] = list(getattr(rule, "clarification_options", []) or [])
        return result
    rule_usable = not rule.needs_clarification and bool(rule.query.get("operation"))
    grounded = rule_usable and _is_grounded(rule.query, user_message)
    # A follow-up that refines an existing selection ("just the top 5",
    # "now only seniors") is grounded by the active conversation context.
    has_active = bool(state.get("active_filters") or state.get("active_group_by")
                      or state.get("active_sort") or state.get("active_limit"))
    followup_refine = classify_context_action(user_message) == FOLLOWUP and has_active

    # 2a. Vague-term resolver (L.12 + L.13 + L.14). For known vague phrases
    # ("struggling", "at risk", "overloaded advisors", "no advisor", "top
    # students", …) the resolver either produces a concrete validated plan or
    # asks the user to clarify. This runs BEFORE we accept a bare
    # filtered_preview plan that would otherwise return the whole sheet.
    vague = resolve_vague_term(
        message=user_message,
        sheet=selected_sheet,
        columns=columns,
        categorical_values=_safe_categorical_values(frame) if frame is not None else {},
    )
    if vague is not None and not _rule_plan_supersedes_vague(rule, rule_usable):
        if vague.has_plan:
            return _build_query_routing(
                user_message, vague.query, "rules", False, state, columns, selected_sheet,
                band="medium",
                assumption_override=vague.assumption,
                alternatives_override=vague.alternatives,
            )
        return _result(
            plan_source="clarification", intent="clarify", confidence=0.0,
            plan=None, llm_used=False,
            warnings=[vague.clarification],
            confirmation_reason=vague.clarification,
            fallback_reason="vague term without supporting columns",
            band="low",
        )

    rule_band = classify_confidence(rule.confidence)
    grounded = grounded or followup_refine

    # 3. LLM-first. When a local model is available it is the PRIMARY planner;
    # the deterministic rules engine stays on as a guardrail (ranking/sort/top-N
    # queries it parses exactly always win over the model) and as the offline /
    # invalid-plan fallback. There is no confidence gate and no "force" toggle —
    # the model is consulted on every question once enabled.
    if config["llm_enabled"]:
        return _route_llm_first(
            user_message, sheet_columns, selected_sheet, sheets, state, config,
            llm_call, rule=rule, rule_usable=rule_usable, grounded=grounded,
            columns=columns,
        )

    # 4. Model disabled (strict privacy / offline) -> deterministic rules.
    # - HIGH (>=0.85): clean response, no assumption note.
    # - MEDIUM [0.55, 0.85) + grounded: assume-and-offer.
    if rule_usable and (rule.confidence >= HIGH_CONFIDENCE
                        or (rule.confidence >= RULES_CONFIDENCE and grounded)):
        return _build_query_routing(user_message, rule.query, "rules", False, state, columns,
                                    selected_sheet, band=rule_band)
    if rule_usable and rule.confidence >= MEDIUM_CONFIDENCE and grounded:
        return _build_query_routing(user_message, rule.query, "rules", False, state, columns,
                                    selected_sheet, band=rule_band)
    return _fallback(rule, "local model disabled", grounded, state, user_message, columns, selected_sheet)


# Deterministic intents -------------------------------------------------------


def _classify_action_intent(message, columns, state, sheet) -> dict[str, Any] | None:
    active = list(state.get("active_filters", []))

    if _is_dashboard_report_request(message):
        return _result(
            plan_source="rules",
            intent="dashboard_report",
            confidence=0.95,
            plan={"sheet": sheet, "filters": active},
            llm_used=False,
            requires_confirmation=True,
            pending_type="dashboard_report",
            confirmation_reason=(
                "This will create a NEW local Excel dashboard workbook with summary, "
                "intervention review, advisor outcome rankings, pivot tables, trend findings, "
                "and chart data sheets. The original workbook will not be modified. Confirm?"
            ),
        )

    # Pivot/trend requests are analysis views in chat, even when the user says
    # "create". They can still be exported from the generated result table.
    if _looks_like_readonly_workbook_tool(message):
        return None

    # Phase Q: edit-then-export chains. Detected before single intents so a
    # phrase like "mark them academic watch and export" routes to one
    # confirmation, one execution, one audit entry.
    chain = parse_action_chain(message)
    if chain is not None:
        return _build_action_chain_routing(message, chain, active, sheet, columns)

    field_update = parse_field_update(message)
    if field_update is not None:
        field, value = field_update
        status = field_status(field)
        if status == "protected":
            return _result(plan_source="rules", intent="field_update", confidence=1.0,
                           plan={"field": field, "value": value}, llm_used=False,
                           validation_status="failed",
                           validation_errors=[f"{field} is a protected field and cannot be edited."],
                           confirmation_reason=f"I can't update {field} because it is a protected field. "
                           "You can export the list for manual review instead.")
        if status != "safe":
            return _result(plan_source="rules", intent="unsupported", confidence=1.0,
                           plan={"field": field}, llm_used=False, validation_status="failed",
                           validation_errors=[f"{field} is not an editable field."],
                           confirmation_reason=f"'{field}' is not an editable field for this assistant.")
        return _result(plan_source="rules", intent="field_update", confidence=1.0,
                       plan={"field": field, "value": value, "filters": active, "sheet": sheet},
                       llm_used=False, requires_confirmation=True, pending_type="field_update",
                       confirmation_reason=f"This will set {field} to '{value}' for "
                       f"{_scope(active)} and save a NEW workbook (the original is never changed). Confirm?")

    if is_export_request(message):
        sensitive = requested_sensitive_columns(message, columns)
        reason = ("This export includes sensitive student-level fields: " + ", ".join(sensitive) + ". "
                  "Please confirm before I create the export.") if sensitive else (
                  "This will export student-level records to a new file. Please confirm before I create the export.")
        return _result(plan_source="rules", intent="export", confidence=1.0,
                       plan={"filters": active, "sheet": sheet, "sensitive_fields": sensitive},
                       llm_used=False, requires_confirmation=True, pending_type="export",
                       confirmation_reason=reason,
                       warnings=([f"sensitive export: {', '.join(sensitive)}"] if sensitive else []))

    if is_attendance_watch_request(message):
        # Same workflow as academic_watch but writes to the Attendance Watch
        # column. Shares the confirm + write-new-workbook + audit pipeline.
        return _result(plan_source="rules", intent="attendance_watch", confidence=1.0,
                       plan={"filters": active, "sheet": sheet, "value": "Yes",
                             "column_name": "Attendance Watch"},
                       llm_used=False, requires_confirmation=True, pending_type="attendance_watch",
                       confirmation_reason=(
                           f"Confirmation needed. This will mark {_scope(active)} as "
                           "Attendance Watch in a NEW workbook. The original workbook will "
                           "not be modified."
                       ))

    if is_academic_watch_request(message):
        # School-roster workflow: "mark these students under Academic Watch" /
        # "flag them" / "put them on watch". Operates on the currently filtered
        # student set, requires confirmation, writes a NEW workbook.
        return _result(plan_source="rules", intent="academic_watch", confidence=1.0,
                       plan={"filters": active, "sheet": sheet, "value": "Yes"},
                       llm_used=False, requires_confirmation=True, pending_type="academic_watch",
                       confirmation_reason=(
                           f"Confirmation needed. This will mark {_scope(active)} as "
                           "Academic Watch in a NEW workbook. The original workbook will "
                           "not be modified."
                       ))

    if is_note_request(message):
        note = parse_note(message)
        note_part = f' "{note}"' if note else ""
        return _result(plan_source="rules", intent="note_edit", confidence=1.0,
                       plan={"note": note, "filters": active, "sheet": sheet},
                       llm_used=False, requires_confirmation=True, pending_type="note_edit",
                       confirmation_reason=f"This will add a note{note_part} to {_scope(active)} and save a "
                       "NEW workbook (the original is never changed). Confirm?")

    if has_hard_edit_cue(message):
        # Structural in-workbook edits (highlight/chart/report/...) keep their
        # own planner; the chat delegates this intent to the edit path.
        return _result(plan_source="rules", intent="edit", confidence=0.9,
                       plan={"message": message}, llm_used=False)
    return None


_COUNT_QUESTION_RE = re.compile(r"\b(?:how many|number of|count of|how much|total number of)\b")


def _academic_insight_query(message: str, sheet: str, columns: list[str]) -> dict[str, Any] | None:
    text = normalize_text(message or "")
    if not text:
        return None

    # Count questions ("how many students need attendance support") ask for a
    # COUNT of a column, not a ranked intervention/advisor LIST. Without this
    # guard, "need ... support" / "attention" cues hijack such questions into the
    # specialist summaries and answer the wrong number. Let them fall through to
    # the generic count planner (and on to the analyst).
    if _COUNT_QUESTION_RE.search(text):
        return None

    if _asks_for_advisor_outcomes(text):
        advisor_column = _find_column_by_names(columns, ("advisor", "teacher", "professor"))
        if not advisor_column:
            return {
                "query": {},
                "assumption": "Advisor outcome ranking needs an Advisor or Teacher column in the workbook.",
            }
        direction = "asc" if _asks_for_low_outcome_advisors(text) else "desc"
        return {
            "query": {
                "request_type": "ask_question",
                "operation": "advisor_outcome_summary",
                "sheet": sheet,
                "filters": [],
                "group_by": advisor_column,
                "sort": {"column": "Outcome Score", "direction": direction},
                "limit": 20,
                "plain_english_question": message,
                "confidence": 0.85,
            },
            "assumption": (
                "I interpreted advisor performance as workbook outcome indicators by advisor: "
                "average GPA, attendance, low-GPA share, attendance-support share, and standing risk. "
                "This is a student-outcome summary, not a personnel judgment."
            ),
        }
    if _asks_for_student_intervention(text):
        available = _academic_signal_columns(columns)
        if not available:
            return {
                "query": {
                    "request_type": "ask_question",
                    "operation": "data_quality_summary",
                    "sheet": sheet,
                    "filters": [],
                    "plain_english_question": message,
                    "confidence": 0.65,
                },
                "assumption": (
                    "I looked for intervention indicators, but this workbook does not expose the usual "
                    "GPA, attendance, standing, or absence columns."
                ),
            }
        return {
            "query": {
                "request_type": "ask_question",
                "operation": "student_intervention_summary",
                "sheet": sheet,
                "filters": [],
                "sort": {"column": "Intervention Signals", "direction": "desc"},
                "limit": 50,
                "plain_english_question": message,
                "confidence": 0.85,
            },
            "assumption": (
                "I interpreted intervention as a workbook review list using available indicators: "
                f"{', '.join(available)}. This is evidence for staff review, not a final decision."
            ),
        }
    return None


def _looks_like_readonly_workbook_tool(message: str) -> bool:
    text = normalize_text(message or "")
    return "pivot" in text or bool(re.search(r"\b(?:find|show|identify|summarize|summarise).{0,20}\btrends?\b", text))


def _is_dashboard_report_request(message: str) -> bool:
    text = normalize_text(message or "")
    if not re.search(r"\b(?:dashboard|report|workbook|packet)\b", text):
        return False
    if not re.search(r"\b(?:build|create|make|generate|compile|prepare)\b", text):
        return False
    return bool(
        re.search(r"\b(?:advisor|intervention|support|risk|trend|trends|pivot|pivots|figure|figures|chart|charts|dashboard)\b", text)
    )


def _workbook_tool_query(message: str, sheet: str, columns: list[str]) -> dict[str, Any] | None:
    text = normalize_text(message or "")
    if not text:
        return None
    if "pivot" in text:
        pivot = _parse_pivot_request(text, sheet, columns)
        if pivot is not None:
            return pivot
    if re.search(r"\b(?:trend|trends|pattern|patterns|outlier|outliers)\b", text):
        return {
            "query": {
                "request_type": "ask_question",
                "operation": "trend_summary",
                "sheet": sheet,
                "filters": [],
                "plain_english_question": message,
                "confidence": 0.85,
            },
            "assumption": (
                "I interpreted this as a trend scan across available numeric metrics and category columns. "
                "The result ranks the largest group differences for review."
            ),
        }
    return None


def _parse_pivot_request(text: str, sheet: str, columns: list[str]) -> dict[str, Any] | None:
    value_column = _metric_column_from_text(text, columns)
    metric = "average" if re.search(r"\b(?:average|avg|mean)\b", text) and value_column else "count"
    if re.search(r"\b(?:sum|total)\b", text) and value_column:
        metric = "sum"

    row_column = ""
    column_column = ""

    # "rows = Advisor, columns = Year" style.
    row_match = re.search(r"\brows?\s*(?:=|as|by)?\s*([a-z ]+?)(?:\s+columns?\b|,|$)", text)
    col_match = re.search(r"\bcolumns?\s*(?:=|as|by)?\s*([a-z ]+?)(?:,|$)", text)
    if row_match:
        row_column = _resolve_column_phrase(row_match.group(1), columns) or ""
    if col_match:
        column_column = _resolve_column_phrase(col_match.group(1), columns) or ""

    # "average GPA by advisor and year" / "pivot GPA by advisor and year".
    by_two = re.search(r"\bby\s+([a-z ]+?)\s+(?:and|,)\s+([a-z ]+?)(?:,|$)", text)
    if by_two and value_column:
        row_column = _resolve_column_phrase(by_two.group(1), columns) or row_column
        column_column = _resolve_column_phrase(by_two.group(2), columns) or column_column

    # "pivot table of standing by advisor" -> rows advisor, columns standing.
    if not row_column or not column_column:
        by_match = re.search(r"\bpivot(?:\s+table)?(?:\s+of)?\s+([a-z ]+?)\s+by\s+([a-z ]+?)(?:,|$)", text)
        if by_match:
            first = _resolve_column_phrase(by_match.group(1), columns)
            second = _resolve_column_phrase(by_match.group(2), columns)
            if second and not row_column:
                row_column = second
            if first and first != row_column and not column_column:
                column_column = first

    if not row_column:
        if by_two:
            row_column = _resolve_column_phrase(by_two.group(1), columns) or ""
            column_column = _resolve_column_phrase(by_two.group(2), columns) or ""
    if not row_column:
        by_one = re.search(r"\bby\s+([a-z ]+?)(?:,|$)", text)
        if by_one:
            row_column = _resolve_column_phrase(by_one.group(1), columns) or ""

    if not row_column:
        row_column = _find_column_by_names(columns, ("advisor", "major", "discipline", "year", "standing", "location")) or ""
    if not row_column:
        return None

    metric_label = f"{metric} {value_column}".strip() if value_column else "count"
    cross = f" with columns from {column_column}" if column_column else ""
    return {
        "query": {
            "request_type": "ask_question",
            "operation": "pivot_table_summary",
            "sheet": sheet,
            "filters": [],
            "group_by": row_column,
            "pivot_rows": row_column,
            "pivot_columns": column_column,
            "value_column": value_column or "",
            "metric": metric,
            "plain_english_question": text,
            "confidence": 0.85,
        },
        "assumption": (
            f"I interpreted this as a pivot-style summary using {row_column} as rows{cross} "
            f"and {metric_label} as the value."
        ),
    }


def _metric_column_from_text(text: str, columns: list[str]) -> str:
    for column in columns:
        normalized = normalize_text(column)
        if normalized and re.search(rf"\b{re.escape(normalized)}\b", text):
            if any(token in normalized for token in ("gpa", "attendance", "absent", "present", "sat", "psat", "score", "rate", "days", "credit", "balance")):
                return column
    for phrase in ("gpa", "attendance rate", "days absent", "sat total", "sat math", "psat total", "psat math"):
        column = _resolve_column_phrase(phrase, columns)
        if column and phrase in text:
            return column
    return ""


def _resolve_column_phrase(phrase: str, columns: list[str]) -> str | None:
    phrase = normalize_text(phrase)
    phrase = re.sub(r"\b(?:students|student|count|counts|number|total|average|avg|mean|pivot|table|of|the|a|an)\b", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if not phrase:
        return None
    for column in columns:
        if normalize_text(column) == phrase:
            return column
    alias_match = _find_column_by_names(columns, (phrase,))
    if alias_match:
        return alias_match
    for column in columns:
        normalized = normalize_text(column)
        if normalized in phrase or phrase in normalized:
            return column
    return None


def _asks_for_student_intervention(text: str) -> bool:
    if re.search(r"\badvisor\s+attention\b", text):
        return True
    if re.search(r"\b(?:advisor|advisors|teacher|teachers|professor|professors)\b", text):
        return False
    return bool(
        re.search(r"\b(?:who|which students|students).{0,40}\b(?:intervention|support|attention|outreach|review|help)\b", text)
        or re.search(r"\b(?:need|needs|needing|required|requiring).{0,30}\b(?:intervention|support|attention|outreach|review|help)\b", text)
        or re.search(r"\b(?:highest|most).{0,20}\b(?:risk|at risk)\b", text)
        or re.search(r"\bintervention\s+review\s+list\b", text)
        or re.search(r"\badvisor\s+attention\b", text)
    )


def _asks_for_advisor_outcomes(text: str) -> bool:
    if not any(token in text for token in ("advisor", "teacher", "professor", "counselor", "counsellor")):
        return False
    return bool(
        re.search(r"\b(?:good job|doing well|best|strongest|highest performing|performance|outcome|outcomes)\b", text)
        or re.search(r"\b(?:struggling|lowest performing|need support|needs support|most risk|worst)\b", text)
    )


def _asks_for_low_outcome_advisors(text: str) -> bool:
    return bool(re.search(r"\b(?:struggling|lowest|need support|needs support|most risk|worst|weakest)\b", text))


def _academic_signal_columns(columns: list[str]) -> list[str]:
    labels = []
    for names, label in (
        (("gpa", "grade point average"), "GPA"),
        (("attendance rate", "attendance %", "attendance"), "attendance"),
        (("standing", "academic standing", "academic status"), "standing"),
        (("attendance category",), "attendance category"),
        (("days absent", "absences", "total absences"), "days absent"),
    ):
        if _find_column_by_names(columns, names):
            labels.append(label)
    return labels


def _find_column_by_names(columns: list[str], names: tuple[str, ...]) -> str | None:
    expanded = set(names)
    for name in names:
        expanded.update(_column_aliases_for_planner(name))
    targets = {normalize_text(name) for name in expanded}
    for column in columns:
        if normalize_text(column) in targets:
            return column
    for column in columns:
        normalized = normalize_text(column)
        if any(target and (target in normalized or normalized in target) for target in targets):
            return column
    return None


def _column_aliases_for_planner(name: str) -> tuple[str, ...]:
    normalized = normalize_text(name)
    aliases = {
        "advisor": ("advisor counselor", "advisor/counselor", "counselor", "school counselor", "teacher", "professor"),
        "teacher": ("advisor", "counselor", "professor"),
        "professor": ("advisor", "teacher", "counselor"),
        "major": ("major program", "major/program", "program", "department", "discipline"),
        "program": ("major", "major program", "major/program", "department", "discipline"),
        "department": ("major", "program", "discipline"),
        "discipline": ("major", "program", "department"),
        "year": ("grade", "grade level", "yr grade", "yr/grade", "class year"),
        "grade": ("year", "grade level", "yr grade", "yr/grade"),
        "standing": ("academic standing", "academic status", "status"),
        "academic standing": ("standing", "academic status"),
        "academic status": ("standing", "academic standing"),
        "gpa": ("current gpa", "cumulative gpa", "grade point average"),
        "attendance rate": ("attendance", "attendance percent", "attendance percentage", "attendance %"),
        "attendance category": ("attendance status", "attendance support", "follow up needed", "needs counselor follow up"),
        "days absent": ("absences", "total absences"),
        "sat total": ("sat", "sat score"),
        "sat math": ("sat math", "sat-math"),
        "sat ebrw": ("sat reading writing", "sat-reading/writing", "sat english"),
        "psat total": ("psat", "psat score"),
        "psat math": ("psat m", "psat_m", "psat-math"),
        "psat ebrw": ("psat english score", "psat english", "psat reading writing"),
        "location": ("campus", "site"),
    }
    return aliases.get(normalized, ())


# Query routing (composition) -------------------------------------------------


def _build_query_routing(message, base_query, plan_source, llm_used, state, columns, sheet,
                         *, band: str = "high",
                         assumption_override: str = "",
                         alternatives_override: list[str] | None = None) -> dict[str, Any]:
    context_action = classify_context_action(message)
    active = list(state.get("active_filters", []))
    new_filters = list(base_query.get("filters") or [])
    effective = compose_filters(active, new_filters, context_action, is_additive(message))
    context_note = _phrase_context_change(context_action, active, new_filters)

    operation = base_query.get("operation") or "count_rows"
    new_group = base_query.get("group_by") or ""
    new_sort = base_query.get("sort") or None
    new_limit = base_query.get("limit")
    if context_action == FOLLOWUP:
        group_by = new_group or state.get("active_group_by", "")
        sort = new_sort or (state.get("active_sort") or None)
        limit = new_limit if new_limit is not None else state.get("active_limit")
    else:
        group_by, sort, limit = new_group, new_sort, new_limit

    if group_by:
        if operation == "average_column":
            operation = "groupby_average"
        elif operation == "sum_column":
            operation = "groupby_sum"
        elif operation in ("count_rows", "filtered_preview"):
            operation = "groupby_count"
    elif (context_action == FOLLOWUP and operation == "count_rows"
          and state.get("last_operation") == "filtered_preview" and not _has_count_word(message)):
        operation = "filtered_preview"

    query = {
        "request_type": "ask_question", "operation": operation, "sheet": base_query.get("sheet") or sheet,
        "filters": effective, "group_by": group_by, "value_column": base_query.get("value_column", ""),
        "sort": sort, "limit": limit,
        "select_columns": list(base_query.get("select_columns") or []),
        "filter_mode": base_query.get("filter_mode") or "all",
        "plain_english_question": base_query.get("plain_english_question", message),
        "confidence": base_query.get("confidence", 0.7),
    }
    for extra_key in ("pivot_rows", "pivot_columns", "metric"):
        if extra_key in base_query:
            query[extra_key] = base_query.get(extra_key)
    active_update = {"filters": effective, "sort": sort or {}, "group_by": group_by or "",
                     "limit": limit, "operation": operation, "sheet": query["sheet"]}

    narration = narrate_plan(
        plan=query,
        context_action=context_action,
        prior_filters=active,
        new_filters=new_filters,
        additive=is_additive(message),
    )

    # When the user explicitly names sensitive columns (e.g. emails) in a
    # preview, show them directly — there is no confirmation gate. We reveal
    # exactly the requested fields by skipping redaction for this result.
    requested = requested_sensitive_columns(message, columns) if operation == "filtered_preview" else []
    reveal_requested = bool(requested)

    # Medium-confidence reads get an explicit assumption note + alternative
    # interpretations the user can click. The vague-term resolver may supply a
    # tailored assumption and alternatives that take priority over the
    # generic narration-based ones; otherwise we fall back to the generic
    # detection from `nlp.uncertainty`.
    if assumption_override:
        assumption_note = assumption_override
    else:
        assumption_note = build_assumption_note(narration) if band == "medium" else ""
    if alternatives_override:
        alternatives = list(alternatives_override)
    else:
        alternatives = detect_vague_alternatives(message) if band == "medium" else []

    # Grounded next-move suggestions (L.3). Row count is unknown here — the
    # dispatcher refines this once it has the result, but having a baseline
    # available pre-execution keeps tests / debug deterministic.
    suggestions = suggest_next_moves(
        plan=query, columns=columns, active_filters=effective,
        removed_fields=None, row_count=None,
    )

    return _result(plan_source=plan_source, intent="query", confidence=float(query["confidence"]),
                   plan=query, llm_used=llm_used, active_update=active_update,
                   context_note=context_note, narration=narration, band=band,
                   assumption_note=assumption_note, alternatives=alternatives,
                   suggestions=suggestions, reveal_sensitive=reveal_requested)


# JSON repair + LLM plan ------------------------------------------------------


def clean_json_text(raw: str) -> str | None:
    if not raw:
        return None
    text = re.sub(r"```$", "", re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start: index + 1]
    return None


def _parse_plan_text(raw: str) -> dict[str, Any] | None:
    cleaned = clean_json_text(raw)
    if cleaned is None:
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) and payload.get("intent") else None


def _parse_with_repair(raw, call, config) -> dict[str, Any] | None:
    plan = _parse_plan_text(raw)
    if plan is not None:
        return plan
    if int(config.get("max_retries", 1)) < 1:
        return None
    try:
        repaired = call(build_repair_prompt(raw or ""))
    except OllamaUnavailable:
        return None
    return _parse_plan_text(repaired)


def _llm_plan(message, sheet_columns, selected_sheet, sheets, state, config, llm_call):
    columns = sheet_columns.get(selected_sheet, [])
    frame = sheets.get(selected_sheet)
    prompt = build_safe_planner_prompt(
        user_request=message, sheet_names=list(sheet_columns), sheet_columns=sheet_columns,
        active_filters=state.get("active_filters", []),
        canonical_map=canonical_map(columns) if columns else {},
        safe_values=_safe_categorical_values(frame) if frame is not None else {},
        row_context=_planner_row_context(sheets) if config.get("planner_full_row_access") else None,
    )
    call = llm_call or _default_llm_call(config)
    return _parse_with_repair(call(prompt), call, config)


def _planner_row_context(sheets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Return local-only workbook rows for the planner prompt.

    This is intentionally gated by settings before it is called. The model still
    has to return a JSON plan that passes validation before pandas executes it.
    """
    context: dict[str, Any] = {
        "policy": "full_local_workbook_rows",
        "sheets": {},
    }
    for sheet_name, frame in sheets.items():
        context["sheets"][sheet_name] = {
            "row_count": int(len(frame.index)),
            "columns": [str(column) for column in frame.columns],
            "rows": _dataframe_records_for_prompt(frame),
        }
    return context


def _dataframe_records_for_prompt(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")


# Ranking / sort / top-N cues. These are deterministic patterns the rules
# engine parses exactly ("top 10 by GPA" = sort GPA desc, limit 10), so when one
# appears AND the rules planner independently produced a sort, the rules plan
# wins over the model — preventing the model from mangling a ranking into a
# made-up threshold filter. Two independent signals must agree, so a loose cue
# here can't force the wrong plan on a non-ranking question.
_RANKING_CUE_RE = re.compile(
    r"\b(top|bottom|highest|lowest|best|worst|largest|smallest|most|least|"
    r"first|last|sort(?:ed)?\s+by|order(?:ed)?\s+by|rank(?:ed)?)\b"
)


def _is_ranking_query(message: str) -> bool:
    return bool(_RANKING_CUE_RE.search(normalize_text(message or "")))


def _route_llm_first(
    user_message: str,
    sheet_columns: dict[str, list[str]],
    selected_sheet: str,
    sheets: dict[str, pd.DataFrame],
    state: dict[str, Any],
    config: dict[str, Any],
    llm_call: Callable[[str], str] | None,
    *,
    rule,
    rule_usable: bool,
    grounded: bool,
    columns: list[str],
) -> dict[str, Any]:
    """LLM is the primary planner; the rules plan is the deterministic guardrail.

    Guardrail 1 (ranking): a top-N/sort/superlative question that the rules
    engine resolved into a concrete sort is answered by the rules plan, not the
    model. Guardrail 2 (fallback): if the model is unavailable, unusable, or its
    plan fails validation, fall back to the rules plan when it is grounded rather
    than bouncing the user to a clarify prompt.
    """
    rule_band = classify_confidence(rule.confidence)

    # Guardrail 1 — ranking queries are deterministic; never let the model
    # reinterpret "top 10 by GPA" as a filter.
    if rule_usable and _is_ranking_query(user_message) and rule.query.get("sort"):
        return _build_query_routing(user_message, rule.query, "rules", False, state, columns,
                                    selected_sheet, band=rule_band)

    try:
        plan = _llm_plan(user_message, sheet_columns, selected_sheet, sheets, state, config, llm_call)
    except OllamaUnavailable as exc:
        return _fallback(rule, f"local model unavailable ({exc})", grounded, state,
                         user_message, columns, selected_sheet)
    if plan is None:
        return _fallback(rule, "LLM plan could not be used", grounded, state,
                         user_message, columns, selected_sheet)

    intent = plan.get("intent", "query")
    if intent in {"clarify", "unavailable", "unsupported"}:
        return _result(
            plan_source="llm", intent=intent,
            confidence=float(plan.get("confidence", 0.6)), plan=None,
            llm_used=True,
            confirmation_reason=plan.get("clarification_question"),
            fallback_reason=plan.get("clarification_question"),
        )
    query = _map_llm_to_query(plan, selected_sheet)
    query = _repair_llm_query(user_message, query, columns, sheets=sheets)
    verdict = validate_llm_plan(plan, query, sheets, selected_sheet)
    if not verdict["ok"]:
        repaired_plan = _repair_plan_after_validation(
            user_message=user_message,
            bad_plan=plan,
            validation_errors=verdict["errors"],
            sheet_columns=sheet_columns,
            selected_sheet=selected_sheet,
            sheets=sheets,
            state=state,
            config=config,
            llm_call=llm_call,
            columns=columns,
        )
        if repaired_plan is not None:
            repaired_query = _map_llm_to_query(repaired_plan, selected_sheet)
            repaired_query = _repair_llm_query(user_message, repaired_query, columns, sheets=sheets)
            repaired_verdict = validate_llm_plan(repaired_plan, repaired_query, sheets, selected_sheet)
            if repaired_verdict["ok"]:
                if _should_semantic_repair_with_rules(rule, rule_usable, grounded, repaired_query):
                    routing = _build_query_routing(
                        user_message, rule.query, "llm", True, state, columns, selected_sheet,
                        band=rule_band,
                    )
                    routing["semantic_repaired"] = True
                    routing["validation_repaired"] = True
                    routing["validation_errors_repaired"] = list(verdict["errors"])
                    routing["semantic_repair_reason"] = (
                        "Validation-repaired LLM plan differed from a high-confidence workbook parse."
                    )
                    routing["llm_original_query"] = repaired_query
                    return routing
                llm_band = classify_confidence(float(repaired_query.get("confidence", 0.0)))
                routing = _build_query_routing(
                    user_message, repaired_query, "llm", True, state, columns, selected_sheet,
                    band=llm_band,
                )
                routing["validation_repaired"] = True
                routing["validation_errors_repaired"] = list(verdict["errors"])
                routing["requires_confirmation"] = (
                    routing["requires_confirmation"] or repaired_verdict["requires_confirmation"]
                )
                routing["reveal_sensitive"] = (
                    routing["reveal_sensitive"] or repaired_verdict.get("reveal_sensitive", False)
                )
                return routing
        # Guardrail 2 — prefer a correct deterministic answer over a clarify.
        if rule_usable and grounded:
            routing = _build_query_routing(user_message, rule.query, "rules", False, state,
                                           columns, selected_sheet, band=rule_band)
            routing["fallback_reason"] = "LLM plan failed validation; used rules plan"
            routing["llm_validation_errors"] = list(verdict["errors"])
            return routing
        return _result(
            plan_source="clarification", intent="clarify", confidence=0.0,
            plan=None, llm_used=True, warnings=verdict["errors"],
            validation_status="failed", validation_errors=verdict["errors"],
            confirmation_reason="Ollama returned a plan that failed safety validation.",
            fallback_reason="LLM plan failed validation",
            band="low",
        )
    if _should_semantic_repair_with_rules(rule, rule_usable, grounded, query):
        routing = _build_query_routing(
            user_message, rule.query, "llm", True, state, columns, selected_sheet,
            band=rule_band,
        )
        routing["semantic_repaired"] = True
        routing["semantic_repair_reason"] = "LLM plan differed from a high-confidence workbook parse."
        routing["llm_original_query"] = query
        return routing
    llm_band = classify_confidence(float(query.get("confidence", 0.0)))
    routing = _build_query_routing(
        user_message, query, "llm", True, state, columns, selected_sheet,
        band=llm_band,
    )
    routing["requires_confirmation"] = routing["requires_confirmation"] or verdict["requires_confirmation"]
    routing["reveal_sensitive"] = routing["reveal_sensitive"] or verdict.get("reveal_sensitive", False)
    return routing


def _should_semantic_repair_with_rules(
    rule,
    rule_usable: bool,
    grounded: bool,
    query: dict[str, Any],
) -> bool:
    if not rule_usable or rule.confidence < RULES_CONFIDENCE:
        return False
    rule_query = rule.query or {}
    if not rule_query.get("operation"):
        return False
    if _llm_query_is_more_specific_group_metric_rank(rule_query, query):
        return False
    return _query_signature(rule_query) != _query_signature(query)


def _llm_query_is_more_specific_group_metric_rank(
    rule_query: dict[str, Any],
    query: dict[str, Any],
) -> bool:
    """Keep a validated LLM grouped-average rank when rules found the same
    aggregate but missed the rank sort/limit.

    Example: rules may parse "which major has the highest average SAT Total"
    as average SAT Total by Major. The repaired model plan can add
    sort=SAT Total desc and limit=1, which is more faithful to "which ... has
    the highest".
    """
    core_keys = ("operation", "group_by", "value_column")
    for key in core_keys:
        if (rule_query.get(key) or "") != (query.get(key) or ""):
            return False
    rule_filters = _query_signature({**rule_query, "sort": None, "limit": None})["filters"]
    query_filters = _query_signature({**query, "sort": None, "limit": None})["filters"]
    if rule_filters != query_filters:
        return False
    sort = query.get("sort") or {}
    return (
        query.get("operation") == "groupby_average"
        and sort.get("column") == query.get("value_column")
        and sort.get("direction") in {"asc", "desc"}
        and query.get("limit") == 1
        and not rule_query.get("sort")
        and rule_query.get("limit") in (None, "")
    )


def _query_signature(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation": query.get("operation") or "",
        "filters": sorted(
            (
                str(f.get("column") or ""),
                str(f.get("operator") or ""),
                json.dumps(f.get("value", f.get("values")), sort_keys=True, default=str),
            )
            for f in (query.get("filters") or [])
        ),
        "group_by": query.get("group_by") or "",
        "value_column": query.get("value_column") or "",
        "sort": query.get("sort") or None,
        "limit": query.get("limit"),
        "select_columns": tuple(query.get("select_columns") or []),
        "filter_mode": query.get("filter_mode") or "all",
    }


def _repair_plan_after_validation(
    *,
    user_message: str,
    bad_plan: dict[str, Any],
    validation_errors: list[str],
    sheet_columns: dict[str, list[str]],
    selected_sheet: str,
    sheets: dict[str, pd.DataFrame],
    state: dict[str, Any],
    config: dict[str, Any],
    llm_call: Callable[[str], str] | None,
    columns: list[str],
) -> dict[str, Any] | None:
    if int(config.get("max_retries", 1)) < 1:
        return None
    frame = sheets.get(selected_sheet)
    prompt = build_validation_repair_prompt(
        user_request=user_message,
        previous_plan=bad_plan,
        validation_errors=list(validation_errors),
        sheet_names=list(sheet_columns),
        sheet_columns=sheet_columns,
        active_filters=state.get("active_filters", []),
        canonical_map=canonical_map(columns) if columns else {},
        safe_values=_safe_categorical_values(frame) if frame is not None else {},
    )
    call = llm_call or _default_llm_call(config)
    try:
        repaired_raw = call(prompt)
    except OllamaUnavailable:
        return None
    return _parse_plan_text(repaired_raw)


def _default_llm_call(config) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        from nlp.local_model import _call_ollama

        raw, error = _call_ollama(prompt, config["planner_model"], timeout=config["planner_timeout_seconds"])
        if error or raw is None:
            raise OllamaUnavailable(error or "no response")
        return raw

    return _call


def _map_llm_to_query(plan, default_sheet) -> dict[str, Any]:
    operation = _OPERATION_MAP.get(plan.get("operation"), "filtered_preview")
    group_by = plan.get("group_by") or ""
    if group_by and operation == "average_column":
        operation = "groupby_average"
    if group_by and plan.get("operation") == "aggregate":
        operation = "groupby_count"
    filters = []
    for condition in plan.get("filters") or []:
        operator = _OPERATOR_MAP.get(condition.get("operator"), condition.get("operator"))
        new = {"column": condition.get("column"), "operator": operator}
        if "value" in condition and operator not in {"is_blank", "is_not_blank", "is_missing", "is_not_missing"}:
            new["value"] = condition.get("value")
        filters.append(new)
    return {
        "request_type": "ask_question", "operation": operation, "sheet": plan.get("target_sheet") or default_sheet,
        "filters": filters, "group_by": group_by, "value_column": plan.get("value_column") or "",
        "sort": plan.get("sort") or None, "limit": plan.get("limit"),
        "plain_english_question": plan.get("plain_english_question", ""),
        "confidence": float(plan.get("confidence", 0.6)),
    }


def _repair_llm_query(
    user_message: str,
    query: dict[str, Any],
    columns: list[str],
    *,
    sheets: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Normalize common malformed LLM plans before safety validation.

    Example: llama3.2 often reads "how many majors are there?" as
    ``filter Major equals <missing>``. The safe intent is a distinct count of
    Major, so convert only that narrow shape to count_unique.
    """
    if not user_message or not columns:
        return query
    text = normalize_text(user_message)
    synonyms = load_json("synonyms.json")
    performance = _detect_performance_query(text, columns, synonyms)
    if performance:
        group_column, gpa_column, direction = performance
        if group_column and gpa_column and _has_malformed_filter_for_column(query, group_column):
            repaired = dict(query)
            repaired["operation"] = "groupby_average"
            repaired["value_column"] = gpa_column
            repaired["group_by"] = group_column
            repaired["filters"] = [
                f for f in (query.get("filters") or [])
                if not _same_column_missing_value_filter(f, group_column)
            ]
            repaired["sort"] = {"column": gpa_column, "direction": direction}
            repaired["limit"] = 1
            repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
            return repaired

    aggregate_metric = _detect_value_column(text, columns, synonyms, "groupby_average")
    if aggregate_metric and _asks_for_average_metric(text):
        group_column = _detect_group_by(text, columns, synonyms, exclude_column=aggregate_metric)
        if query.get("operation") in {"filtered_preview", "count_rows", "groupby_count", "average_column"} or (
            query.get("operation") == "groupby_average"
            and (query.get("value_column") != aggregate_metric or not query.get("group_by"))
        ):
            repaired = dict(query)
            repaired["operation"] = "groupby_average" if group_column or query.get("group_by") else "average_column"
            repaired["group_by"] = group_column or query.get("group_by") or ""
            repaired["value_column"] = aggregate_metric
            repaired["filters"] = [
                f for f in (query.get("filters") or [])
                if not (
                    _same_column_missing_value_filter(f, aggregate_metric)
                    or (repaired["group_by"] and _same_column_missing_value_filter(f, repaired["group_by"]))
                )
            ]
            direction = _average_rank_direction(text)
            if direction and repaired["group_by"]:
                repaired["sort"] = {"column": aggregate_metric, "direction": direction}
                repaired["limit"] = query.get("limit") or 1
            else:
                repaired["sort"] = None
                repaired["limit"] = None
            repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
            return repaired

    group_column = _detect_group_by(text, columns, synonyms)
    gpa_column = next((column for column in columns if normalize_text(column) == "gpa"), None)
    if (
        gpa_column
        and query.get("operation") in {"filtered_preview", "count_rows", "groupby_count"}
        and re.search(r"\b(?:average|avg|mean)\b", text)
    ):
        repaired = dict(query)
        repaired["operation"] = "groupby_average" if group_column else "average_column"
        repaired["group_by"] = group_column or ""
        repaired["value_column"] = gpa_column
        repaired["sort"] = None
        repaired["limit"] = None
        repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
        return repaired

    support_repaired = _repair_attendance_support_filter(user_message, query, sheets or {})
    if support_repaired is not None:
        query = support_repaired

    attendance_rate_repaired = _repair_attendance_rate_filter(user_message, query, columns, synonyms)
    if attendance_rate_repaired is not None:
        query = attendance_rate_repaired

    preview_repaired = _repair_show_filtered_unique_count(user_message, query)
    if preview_repaired is not None:
        query = preview_repaired

    if (
        group_column
        and query.get("operation") in {"filtered_preview", "count_rows"}
        and (
            re.search(r"\b(?:group|break down|count|counts|how many|number of)\b", text)
            or re.search(r"\bwhich\s+[a-z ]+\s+has\s+the\s+(?:most|fewest|least)\b", text)
        )
    ):
        repaired = dict(query)
        repaired["operation"] = "groupby_count"
        repaired["group_by"] = group_column
        repaired["value_column"] = ""
        repaired["filters"] = [
            f for f in (query.get("filters") or [])
            if f.get("column") != group_column
        ]
        repaired["sort"] = query.get("sort")
        repaired["limit"] = query.get("limit")
        repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
        return repaired

    if (
        query.get("operation") == "filtered_preview"
        and re.search(r"\b(?:how many|number of|count of)\b", text)
        and not _detect_count_unique(text, columns, synonyms)
    ):
        repaired = dict(query)
        repaired["operation"] = "count_rows"
        repaired["group_by"] = ""
        repaired["value_column"] = ""
        repaired["sort"] = None
        repaired["limit"] = None
        repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
        return repaired

    value_repaired = _repair_filter_value_to_matching_column(query, sheets or {})
    if value_repaired is not None:
        query = value_repaired

    list_column = _detect_list_unique(text, columns, synonyms)
    unique_column = list_column or _detect_count_unique(text, columns, synonyms)
    if not unique_column:
        return query

    operation = query.get("operation")
    bad_filters = []
    for condition in query.get("filters") or []:
        if condition.get("column") != unique_column:
            continue
        if condition.get("operator") not in {"equals", "contains", "in"}:
            continue
        if _is_meaningful_value(condition.get("value", condition.get("values"))):
            continue
        bad_filters.append(condition)
    target_operation = "list_unique" if list_column else "count_unique"
    if not bad_filters and operation != target_operation:
        return query

    repaired = dict(query)
    repaired["operation"] = target_operation
    repaired["value_column"] = unique_column
    repaired["group_by"] = ""
    repaired["filters"] = [
        f for f in (query.get("filters") or [])
        if f not in bad_filters
    ]
    repaired["sort"] = None
    repaired["limit"] = None
    repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
    return repaired


def _repair_attendance_support_filter(
    user_message: str,
    query: dict[str, Any],
    sheets: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:
    text = normalize_text(user_message or "")
    if not re.search(r"\b(?:need|needs|needing|required|requiring)\s+attendance\s+support\b", text):
        return None
    sheet = query.get("sheet")
    frame = sheets.get(sheet) if sheet else None
    if frame is None:
        return None
    category_column = next(
        (column for column in frame.columns if normalize_text(column) == "attendance category"),
        None,
    )
    if not category_column:
        return None
    filters = list(query.get("filters") or [])
    if any(f.get("column") == category_column for f in filters):
        return None
    values = {
        str(value).strip()
        for value in frame[category_column].dropna().astype(str).unique()
        if str(value).strip()
    }
    value = next(
        (candidate for candidate in values if normalize_text(candidate) == "needs attendance support"),
        "Needs Attendance Support",
    )
    repaired = dict(query)
    repaired["filters"] = filters + [{"column": category_column, "operator": "equals", "value": value}]
    repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
    return repaired


def _repair_attendance_rate_filter(
    user_message: str,
    query: dict[str, Any],
    columns: list[str],
    synonyms: dict[str, Any],
) -> dict[str, Any] | None:
    text = normalize_text(user_message or "")
    if "attendance" not in text:
        return None
    rate_column, score = match_column_for_concept("attendance_rate", columns, synonyms)
    if not rate_column or score < 0.55:
        return None
    filters = list(query.get("filters") or [])
    changed = False
    repaired_filters = []
    for condition in filters:
        column = condition.get("column")
        operator = condition.get("operator")
        if (
            column
            and column != rate_column
            and operator in {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "between"}
            and "attendance" in normalize_text(column)
            and ("rate" not in normalize_text(column))
            and ("percent" not in normalize_text(column))
        ):
            repaired = dict(condition)
            repaired["column"] = rate_column
            repaired_filters.append(repaired)
            changed = True
        else:
            repaired_filters.append(condition)
    if not changed:
        return None
    repaired_query = dict(query)
    repaired_query["filters"] = repaired_filters
    repaired_query["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
    return repaired_query


def _repair_show_filtered_unique_count(
    user_message: str,
    query: dict[str, Any],
) -> dict[str, Any] | None:
    text = normalize_text(user_message or "")
    if query.get("operation") != "count_unique":
        return None
    if not query.get("filters"):
        return None
    if not re.search(r"\b(?:show|list|find|pull)\b", text):
        return None
    if re.search(r"\b(?:how many|number of|count|different|distinct|unique|represented)\b", text):
        return None
    repaired = dict(query)
    repaired["operation"] = "filtered_preview"
    repaired["value_column"] = ""
    repaired["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
    return repaired


def _asks_for_average_metric(text: str) -> bool:
    return bool(
        re.search(r"\b(?:average|avg|mean)\b", text)
        or re.search(r"\b(?:highest|lowest|best|worst|top|bottom)\s+average\b", text)
    )


def _average_rank_direction(text: str) -> str | None:
    if re.search(r"\b(?:lowest|worst|bottom|least|smallest)\b", text):
        return "asc"
    if re.search(r"\b(?:highest|best|top|most|largest|greatest)\b", text):
        return "desc"
    return None


def _repair_filter_value_to_matching_column(
    query: dict[str, Any],
    sheets: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:
    """Move an enum filter to the one categorical column that contains it.

    Example: a small LLM may plan ``Standing = Senior`` because both are
    academic words. If ``Senior`` appears in exactly one other categorical
    column, ``Year``, repair the filter before validation. If the value appears
    in zero or several columns, do nothing.
    """
    sheet = query.get("sheet")
    frame = sheets.get(sheet) if sheet else None
    if frame is None:
        return None
    domains = _safe_categorical_values(frame, max_unique=_CATEGORICAL_DOMAIN_MAX_UNIQUE)
    if not domains:
        return None

    filters = list(query.get("filters") or [])
    repaired_filters: list[dict[str, Any]] = []
    changed = False
    for condition in filters:
        column = condition.get("column")
        operator = condition.get("operator")
        if column not in domains or operator not in {"equals", "in"}:
            repaired_filters.append(condition)
            continue
        raw_value = condition.get("value", condition.get("values"))
        requested = raw_value if isinstance(raw_value, (list, tuple)) else [raw_value]
        requested_norm = [str(v).strip().casefold() for v in requested if str(v).strip()]
        if not requested_norm:
            repaired_filters.append(condition)
            continue
        current_domain = {v.casefold() for v in domains.get(column, [])}
        if all(value in current_domain for value in requested_norm):
            repaired_filters.append(condition)
            continue

        matches = [
            candidate
            for candidate, domain in domains.items()
            if candidate != column
            and all(value in {v.casefold() for v in domain} for value in requested_norm)
        ]
        if len(matches) != 1:
            repaired_filters.append(condition)
            continue

        repaired = dict(condition)
        repaired["column"] = matches[0]
        repaired_filters.append(repaired)
        changed = True

    if not changed:
        return None
    repaired_query = dict(query)
    repaired_query["filters"] = repaired_filters
    repaired_query["confidence"] = max(float(query.get("confidence", 0.0) or 0.0), 0.85)
    return repaired_query


def _has_malformed_filter_for_column(query: dict[str, Any], column: str) -> bool:
    return any(
        _same_column_missing_value_filter(condition, column)
        for condition in (query.get("filters") or [])
    )


def _same_column_missing_value_filter(condition: dict[str, Any], column: str) -> bool:
    return (
        condition.get("column") == column
        and condition.get("operator") in {"equals", "contains", "in"}
        and not _is_meaningful_value(condition.get("value", condition.get("values")))
    )


def validate_llm_plan(plan, query, sheets, selected_sheet) -> dict[str, Any]:
    errors: list[str] = []
    requires_confirmation = False
    sheet = query.get("sheet") or selected_sheet
    frame = sheets.get(sheet)
    if frame is None:
        return {"ok": False, "errors": [f"Unknown sheet: {sheet}"], "requires_confirmation": False}
    columns = list(frame.columns)
    types = infer_column_types(frame)

    referenced = [f.get("column") for f in query.get("filters", []) if f.get("column")]
    if query.get("group_by"):
        referenced.append(query["group_by"])
    if query.get("value_column"):
        referenced.append(query["value_column"])
    if query.get("sort") and query["sort"].get("column"):
        referenced.append(query["sort"]["column"])
    for column in referenced:
        if column not in columns:
            errors.append(f"Plan references nonexistent column: {column}")

    # Cache categorical domains once — used by the invented-enum check.
    categorical_domains = _safe_categorical_values(frame, max_unique=_CATEGORICAL_DOMAIN_MAX_UNIQUE)

    for condition in query.get("filters", []):
        operator = condition.get("operator")
        column = condition.get("column")
        value = condition.get("value", condition.get("values"))  # llama3.2:3b sometimes uses 'values'
        if operator not in _ALLOWED_OPERATORS:
            errors.append(f"Invalid operator: {operator}")
            continue
        if operator in _NUMERIC_OPS and column in columns and types.get(column, {}).get("analysis_dtype") != "numeric":
            errors.append(f"Operator {operator} needs a numeric column: {column}")
            continue
        # --- semantic coherence checks ----------------------------------------
        # (1) Operators that need a value must actually have one. None/empty
        # string/empty list means the model produced an incoherent filter.
        if operator not in _VALUE_FREE_OPS and not _is_meaningful_value(value):
            errors.append(f"Filter '{column} {operator} ...' is missing a value.")
            continue
        # (2) List-valued operators must receive a sequence.
        if operator in _LIST_VALUE_OPS and not isinstance(value, (list, tuple)):
            errors.append(f"Operator {operator} on {column} needs a list of values.")
            continue
        # (3) 'between' is the special case: exactly 2 numeric bounds.
        if operator == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                errors.append(f"Operator between on {column} needs exactly 2 values.")
                continue
            if not all(_is_numeric_literal(v) for v in value):
                errors.append(f"Operator between on {column} needs numeric bounds.")
                continue
        # (4) Numeric comparison operators need a numeric value (not a string
        # like 'high' or 'low').
        if operator in (_NUMERIC_OPS - {"between"}) and not _is_numeric_literal(value):
            errors.append(f"Operator {operator} on {column} needs a numeric value, got {value!r}.")
            continue
        # (5) String-only operators don't make sense on numeric columns.
        if (operator in _STRING_ONLY_OPS
                and column in columns
                and types.get(column, {}).get("analysis_dtype") == "numeric"):
            errors.append(f"Operator {operator} doesn't apply to numeric column {column}.")
            continue
        # (6) Invented-enum check: equals/in/not_equals/not_in on a low-cardinality
        # categorical column with a value that doesn't appear in the column's
        # actual domain almost always means the model fabricated a value
        # (e.g. 'Academic Status = "incomplete"' when the real enum is
        # Good Standing / Warning / Probation / At Risk).
        if column in categorical_domains and operator in {"equals", "not_equals", "in", "not_in"}:
            domain_lower = {v.lower() for v in categorical_domains[column]}
            requested = value if isinstance(value, (list, tuple)) else [value]
            missing = [v for v in requested
                       if v is not None and str(v).strip()
                       and str(v).strip().lower() not in domain_lower]
            if missing:
                pretty = ", ".join(repr(m) for m in missing)
                domain_pretty = ", ".join(repr(v) for v in categorical_domains[column])
                errors.append(
                    f"Filter on {column}: {pretty} is not a known value. "
                    f"Known values: {domain_pretty}."
                )

    limit = query.get("limit")
    if query.get("operation") == "filtered_preview" and isinstance(limit, int) and limit > MAX_STUDENT_PREVIEW:
        errors.append(f"Requested row limit {limit} is too large for student-level data.")

    hidden = [c for c in (plan.get("display") or {}).get("columns", []) if is_hidden_by_default(c)]
    reveal_sensitive = bool(hidden)

    if plan.get("intent") in {"export", "note_edit", "field_update"}:
        requires_confirmation = True
        if plan.get("intent") == "field_update" and field_status(str(plan.get("field", ""))) == "protected":
            errors.append("Protected field cannot be updated.")

    return {"ok": not errors, "errors": errors, "requires_confirmation": requires_confirmation, "reveal_sensitive": reveal_sensitive}


# Helpers ---------------------------------------------------------------------


def _build_action_chain_routing(
    message: str, chain: list[ChainedAction], active: list[dict[str, Any]],
    sheet: str, columns: list[str],
) -> dict[str, Any]:
    """Route a Phase Q action chain through pending_type=action_chain.

    The plan carries the ordered list of {type, payload} so the dispatcher
    can run them in sequence after the user confirms. Confirmation reason
    enumerates the steps in plain English.
    """
    # Serialize each step for the plan + audit. Strip the raw_phrase from
    # the audit-bound payload (already implicit in the action sequence).
    actions = [
        {"type": step.type, **step.payload}
        for step in chain
    ]
    edit_steps = [s for s in chain if s.type != "export"]
    has_export = any(s.type == "export" for s in chain)

    # Build a step-by-step confirmation message.
    steps_text: list[str] = []
    step_number = 1
    for step in edit_steps:
        if step.type == "academic_watch":
            column = step.payload.get("column_hint", "Academic Watch")
            value = step.payload.get("value", "Yes")
            steps_text.append(f"{step_number}. Set {column} = '{value}' for "
                              f"{_scope(active)}.")
        elif step.type == "note_edit":
            note = step.payload.get("note") or ""
            note_part = f' "{note}"' if note else ""
            steps_text.append(f"{step_number}. Add a note{note_part} to "
                              f"{_scope(active)}.")
        elif step.type == "field_update":
            field = step.payload.get("field", "")
            value = step.payload.get("value", "")
            steps_text.append(f"{step_number}. Set {field} to '{value}' for "
                              f"{_scope(active)}.")
        step_number += 1
        # Always save the modified workbook after the edit.
        steps_text.append(f"{step_number}. Save a new workbook.")
        step_number += 1
    if has_export:
        steps_text.append(f"{step_number}. Export the updated workbook.")
        step_number += 1

    confirmation = (
        "Confirmation needed. This will:\n"
        + "\n".join(steps_text)
        + "\n\nThe original workbook will not be modified."
    )

    return _result(
        plan_source="rules",
        intent="action_chain",
        confidence=1.0,
        plan={
            "actions": actions,
            "filters": active,
            "sheet": sheet,
            "has_export": has_export,
        },
        llm_used=False,
        requires_confirmation=True,
        pending_type="action_chain",
        confirmation_reason=confirmation,
    )


def _compose_interpretation_line(ambiguity: AmbiguityResolution) -> str:
    """Build a one-line interpretation summary the narrator can show, e.g.
    'I interpreted teacher as Advisor, department as Discipline, and
    performance as GPA.' followed by the kind-specific assumption."""
    mapping_phrase = ""
    if ambiguity.column_mapping:
        bits = [f"{src} as {dst}" for src, dst in ambiguity.column_mapping]
        mapping_phrase = "I interpreted " + ", ".join(bits) + ". "
    return f"{mapping_phrase}{ambiguity.assumption_note}".strip()


def _is_meaningful_value(value) -> bool:
    """A filter value is meaningful if it is present and non-empty."""
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, tuple)) and not value:
        return False
    return True


def _is_numeric_literal(value) -> bool:
    """True if value can be interpreted as a number (int/float or numeric str)."""
    if isinstance(value, bool):
        return False  # bools are an int subclass but never a sensible filter value
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _scope(active: list[dict]) -> str:
    return f"the current selection ({describe_filters(active)})" if active else "ALL rows"


def _phrase_context_change(action, prior, new_filters) -> str:
    if action != FOLLOWUP or not prior or not new_filters:
        return ""
    prior_columns = {f.get("column") for f in prior}
    replaced = [f["column"] for f in new_filters if f.get("column") in prior_columns]
    added = [f for f in new_filters if f.get("column") not in prior_columns]
    parts = []
    if replaced:
        parts.append(f"I replaced the {', '.join(dict.fromkeys(replaced))} filter")
    if added:
        parts.append((" and added " if parts else "Keeping the current filters, I added ") + describe_filters(added))
    return ("".join(parts) + ". ") if parts else ""


def _safe_categorical_values(frame, max_unique=25) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for column in frame.columns:
        if is_hidden_by_default(column) or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        uniques = frame[column].dropna().astype(str).unique()
        if 0 < len(uniques) <= max_unique:
            values[str(column)] = [str(u) for u in uniques]
    return values


def _rule_plan_supersedes_vague(rule, rule_usable: bool) -> bool:
    """Return True if the rule planner already built something concrete.

    A bare `filtered_preview` with no filters / group / sort never supersedes
    the vague-term resolver — letting it through is exactly what produced the
    600-row whole-sheet bug. A high-confidence plan with real filters
    (e.g. "show me struggling Accounting students" → Dept=Accounting) does.
    """
    if not rule_usable:
        return False
    query = rule.query
    if query.get("filters") or query.get("group_by") or query.get("value_column"):
        return True
    if query.get("operation") not in {"filtered_preview", "count_rows"}:
        return True
    return rule.confidence >= HIGH_CONFIDENCE


def _is_grounded(query, message) -> bool:
    if query.get("filters") or query.get("group_by") or query.get("value_column") or query.get("sort"):
        return True
    if query.get("operation") not in {"count_rows", "filtered_preview"}:
        return True
    return _has_count_word(message)


def _has_count_word(message) -> bool:
    text = normalize_text(message)
    return any(w in text for w in ("how many", "count", "number of", "are there", "how much", "list", "show"))


def _result(
    *, plan_source, intent, confidence, plan, llm_used,
    requires_confirmation=False, confirmation_reason=None, pending_type=None,
    warnings=None, validation_status="passed", validation_errors=None,
    fallback_reason=None, active_update=None, context_note="", reveal_sensitive=None,
    narration="", band=None, assumption_note="", alternatives=None, suggestions=None,
    clarify_options=None,
) -> dict[str, Any]:
    if validation_errors:
        validation_status = "failed"
    if band is None:
        # Derive band from the confidence value when the caller did not specify
        # one. Clarify/unavailable/unsupported paths land here.
        band = classify_confidence(float(confidence))
    return {
        "plan_source": plan_source,
        "intent": intent,
        "confidence": round(float(confidence), 3),
        "plan": plan,
        "requires_confirmation": requires_confirmation,
        "confirmation_reason": confirmation_reason,
        "pending_type": pending_type,
        "warnings": warnings or [],
        "llm_used": llm_used,
        "validation": {"status": validation_status, "errors": validation_errors or []},
        "fallback_reason": fallback_reason,
        "active_update": active_update,
        "context_note": context_note,
        "reveal_sensitive": reveal_sensitive,
        "narration": narration,
        "band": band,
        "assumption_note": assumption_note,
        "alternatives": alternatives or [],
        "suggestions": suggestions or [],
        "clarify_options": clarify_options or [],
    }


def _fallback(rule, reason, grounded, state, message, columns, sheet) -> dict[str, Any]:
    if grounded:
        routing = _build_query_routing(message, rule.query, "rules", False, state, columns, sheet)
        routing["fallback_reason"] = reason
        routing["warnings"] = [reason]
        return routing
    return _result(plan_source="clarification", intent="clarify", confidence=0.0, plan=None, llm_used=False,
                   warnings=[rule.clarification_question or reason],
                   confirmation_reason=rule.clarification_question, fallback_reason=reason)
