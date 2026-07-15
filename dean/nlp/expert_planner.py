"""Expert planner orchestrator.

This module is the single entry point for turning a user request into a
validated, plain-English-summarized plan that the user can review and confirm
before any spreadsheet change happens.

Flow (matches the architecture rule):
    User request
    -> workbook context is collected (sheet names + column names only)
    -> column concepts are mapped (rule-based, local)
    -> rule parser tries first
    -> if confidence is low or the request looks complex,
       the local Ollama planner is called
    -> Ollama returns JSON only
    -> JSON is validated against the workbook
    -> the plan is returned for the UI to display:
       plain-English summary, confidence, assumptions, commands
    -> the user reviews and confirms
    -> the action engine (separate module) executes the spreadsheet changes

The planner itself never writes to Excel, never calls cloud APIs, never sends
spreadsheet rows or student records to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import re

from core.command_schema import (
    PLAN_TYPES,
    SUPPORTED_PLANNER_ACTIONS,
)
from core.correction_manager import (
    add_learned_column_mapping,
    add_learned_synonym,
    sync_learning_files,
)
from core.validator import (
    PlanValidationError,
    validate_plan,
)
from nlp.llm_json_parser import (
    LLMCommandError,
    ParsedPlan,
    command_to_confirmation,
    plan_to_summary,
)
from nlp.local_model import (
    LocalPlannerResult,
    OllamaStatus,
    get_ollama_status,
    plan_from_local_model,
)
from nlp.rule_parser import ParseResult, parse_request
from nlp.synonym_mapper import load_json, load_synonyms_with_learned, match_column_for_concept


RULE_CONFIDENCE_THRESHOLD = 0.75
COMPLEX_KEYWORDS = (
    "report",
    "summary",
    "chart",
    "graph",
    "plot",
    "and then",
    "then",
    "by program",
    "by major",
    "by advisor",
    "by department",
    "follow up",
    "outreach",
    "needs follow",
    "at risk",
    "clean it up",
    "clean this up",
)
CLARIFY_KEYWORDS = (
    "clean it up",
    "clean this up",
    "fix it",
    "do something",
    "help me",
    "make it nice",
)


@dataclass(frozen=True)
class PlanResult:
    """Final, UI-ready plan returned to the chat layer."""

    plan_type: str
    confidence: float
    plain_english_summary: str
    commands: list[dict[str, Any]] = field(default_factory=list)
    clarification_question: str = ""
    assumptions: list[str] = field(default_factory=list)
    requires_confirmation: bool = True
    source: str = "rule_parser"
    validation_error: str | None = None
    can_execute: bool = False
    confidence_level: str = "medium"
    mapped_columns: dict[str, str] = field(default_factory=dict)


def plan_request(
    *,
    user_request: str,
    selected_sheet: str,
    sheet_columns: dict[str, list[str]],
    sheets: dict[str, Any],
    original_file_name: str,
    use_local_llm: bool,
    ollama_model: str,
) -> PlanResult:
    """Plan a user request and return a validated PlanResult.

    sheets: the dict of in-memory pandas DataFrames keyed by sheet name. Used
    ONLY for validation (column types, projected output schemas); rows are
    never sent to the LLM.
    """
    sheet_names = list(sheet_columns.keys())
    mapped_columns = _resolve_concepts(sheet_columns.get(selected_sheet, []))

    rule_result = parse_request(
        user_request,
        selected_sheet,
        sheet_columns.get(selected_sheet, []),
        sheet_columns,
    )

    rule_plan = _rule_result_to_plan(rule_result, selected_sheet)
    needs_llm = _should_call_llm(
        user_request=user_request,
        rule_plan=rule_plan,
        rule_result=rule_result,
        use_local_llm=use_local_llm,
    )

    if not needs_llm:
        return _finalize(rule_plan, sheets, original_file_name, mapped_columns)

    status: OllamaStatus = get_ollama_status(ollama_model)
    if not status.available:
        # Ollama unavailable — fall back to rule plan but lower confidence.
        rule_plan = _with_fallback_note(rule_plan, status.user_message)
        return _finalize(rule_plan, sheets, original_file_name, mapped_columns)

    llm_result: LocalPlannerResult = plan_from_local_model(
        user_request=user_request,
        model_name=ollama_model,
        sheet_names=sheet_names,
        sheet_columns=sheet_columns,
        mapped_columns=mapped_columns,
    )

    if llm_result.error or llm_result.plan is None:
        rule_plan = _with_fallback_note(
            rule_plan,
            llm_result.error or "Local model did not return a usable plan.",
        )
        return _finalize(rule_plan, sheets, original_file_name, mapped_columns)

    llm_plan = _llm_plan_to_planresult(llm_result.plan)

    # Auto-learn from the LLM's stated assumptions ("FASFA was interpreted as
    # FAFSA Status."). Only saves mappings whose target resolves to a real
    # column in this workbook, so the model cannot pollute the synonym table.
    try:
        record_llm_assumptions(
            assumptions=llm_plan.assumptions,
            sheet_columns=sheet_columns.get(selected_sheet, []),
        )
    except Exception:
        pass

    return _finalize(llm_plan, sheets, original_file_name, mapped_columns)


def _resolve_concepts(columns: list[str]) -> dict[str, str]:
    """Map well-known university concepts to real workbook columns.

    Returns ONLY a {concept: column_name} dictionary — no spreadsheet values.
    """
    synonyms = load_synonyms_with_learned()
    mapped: dict[str, str] = {}
    for concept in [
        "balance_due",
        "fafsa_status",
        "enrollment_status",
        "advisor",
        "program",
        "major",
        "student_id",
        "semester",
        "gpa",
    ]:
        column, score = match_column_for_concept(concept, columns, synonyms)
        if column and score >= 0.55:
            mapped[concept] = column
    return mapped


def _rule_result_to_plan(parse_result: ParseResult, selected_sheet: str) -> PlanResult:
    command = parse_result.command or {}
    confidence = float(parse_result.confidence or 0.0)
    summary = parse_result.confirmation or parse_result.clarification or ""

    if parse_result.clarification:
        return PlanResult(
            plan_type="clarify",
            confidence=confidence,
            plain_english_summary=parse_result.clarification,
            commands=[],
            clarification_question=parse_result.clarification,
            assumptions=[],
            requires_confirmation=False,
            source="rule_parser",
            confidence_level=_confidence_level(confidence),
        )

    if command.get("action") == "clarify":
        question = command.get("question", "Please clarify your request.")
        return PlanResult(
            plan_type="clarify",
            confidence=confidence,
            plain_english_summary=question,
            commands=[],
            clarification_question=question,
            assumptions=[],
            requires_confirmation=False,
            source="rule_parser",
            confidence_level=_confidence_level(confidence),
        )

    plain_english = summary or command_to_confirmation(command)
    return PlanResult(
        plan_type="single_action",
        confidence=confidence,
        plain_english_summary=plain_english,
        commands=[command] if command else [],
        clarification_question="",
        assumptions=[],
        requires_confirmation=True,
        source="rule_parser",
        confidence_level=_confidence_level(confidence),
    )


def _llm_plan_to_planresult(plan: ParsedPlan) -> PlanResult:
    summary = plan.plain_english_summary or plan_to_summary(plan)
    return PlanResult(
        plan_type=plan.plan_type,
        confidence=plan.confidence,
        plain_english_summary=summary,
        commands=list(plan.commands),
        clarification_question=plan.clarification_question,
        assumptions=list(plan.assumptions),
        requires_confirmation=plan.requires_confirmation,
        source="local_llm",
        confidence_level=_confidence_level(plan.confidence),
    )


def _should_call_llm(
    *,
    user_request: str,
    rule_plan: PlanResult,
    rule_result: ParseResult,
    use_local_llm: bool,
) -> bool:
    if not use_local_llm:
        return False
    if rule_plan.plan_type == "clarify":
        return True
    if rule_plan.confidence < RULE_CONFIDENCE_THRESHOLD:
        return True
    normalized = " ".join(user_request.lower().split())
    if any(keyword in normalized for keyword in COMPLEX_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in CLARIFY_KEYWORDS):
        return True
    return False


def _finalize(
    plan: PlanResult,
    sheets: dict[str, Any],
    original_file_name: str,
    mapped_columns: dict[str, str],
) -> PlanResult:
    plan_envelope = {
        "plan_type": plan.plan_type,
        "commands": plan.commands,
        "clarification_question": plan.clarification_question,
    }

    if plan.plan_type not in PLAN_TYPES:
        return _replace(
            plan,
            mapped_columns=mapped_columns,
            validation_error=f"Unknown plan_type: {plan.plan_type}",
            can_execute=False,
        )

    if plan.plan_type == "clarify":
        return _replace(plan, mapped_columns=mapped_columns, can_execute=False)

    for command in plan.commands:
        action = command.get("action")
        if action not in SUPPORTED_PLANNER_ACTIONS:
            return _replace(
                plan,
                mapped_columns=mapped_columns,
                validation_error=f"Unsupported action: {action}",
                can_execute=False,
            )

    try:
        validate_plan(plan_envelope, sheets, original_file_name)
    except PlanValidationError as exc:
        return _replace(
            plan,
            mapped_columns=mapped_columns,
            validation_error=str(exc),
            can_execute=False,
        )

    can_execute = plan.confidence_level != "low"
    return _replace(plan, mapped_columns=mapped_columns, can_execute=can_execute)


def _with_fallback_note(plan: PlanResult, note: str) -> PlanResult:
    assumptions = list(plan.assumptions)
    if note and note not in assumptions:
        assumptions.append(f"Local model fallback unavailable: {note}")
    return PlanResult(
        plan_type=plan.plan_type,
        confidence=min(plan.confidence, 0.6),
        plain_english_summary=plan.plain_english_summary,
        commands=plan.commands,
        clarification_question=plan.clarification_question,
        assumptions=assumptions,
        requires_confirmation=plan.requires_confirmation,
        source=plan.source,
        validation_error=plan.validation_error,
        can_execute=plan.can_execute,
        confidence_level=_confidence_level(min(plan.confidence, 0.6)),
        mapped_columns=plan.mapped_columns,
    )


_UNSET = object()


def _replace(
    plan: PlanResult,
    *,
    mapped_columns: dict[str, str] | None = None,
    validation_error: Any = _UNSET,
    can_execute: bool | None = None,
) -> PlanResult:
    return PlanResult(
        plan_type=plan.plan_type,
        confidence=plan.confidence,
        plain_english_summary=plan.plain_english_summary,
        commands=plan.commands,
        clarification_question=plan.clarification_question,
        assumptions=plan.assumptions,
        requires_confirmation=plan.requires_confirmation,
        source=plan.source,
        validation_error=plan.validation_error if validation_error is _UNSET else validation_error,
        can_execute=plan.can_execute if can_execute is None else can_execute,
        confidence_level=plan.confidence_level,
        mapped_columns=mapped_columns if mapped_columns is not None else plan.mapped_columns,
    )


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


# Logging helper --------------------------------------------------------------


def describe_source(plan: PlanResult) -> str:
    """Return a privacy-safe label for the source of the plan.

    Used by the audit logger so we record whether the rule parser or the local
    LLM produced the plan, without ever logging raw rows or values.
    """
    return f"source={plan.source};plan_type={plan.plan_type};confidence={plan.confidence:.2f}"


# Learning loop ---------------------------------------------------------------


def record_clarification_answer(
    *,
    user_phrase: str,
    chosen_column: str,
    concept: str | None = None,
    source: str = "user_clarification",
) -> bool:
    """Persist a user's clarification answer so future requests don't need to ask.

    Call this when the UI shows a clarification (e.g., a column_mapping_request
    or any plan_type "clarify") and the user picks a column. The next time the
    user uses the same phrase, the rule parser will resolve it without help.

    Returns True if something was saved. Never raises on storage errors so the
    chat flow always proceeds; the worst case is the user has to clarify again.
    """
    phrase = (user_phrase or "").strip()
    column = (chosen_column or "").strip()
    if not phrase or not column:
        return False

    try:
        # Map the phrase the user typed (e.g., "gpa", "fasfa") to the chosen
        # column or concept, so future rule-parser passes pick it up.
        target_concept = (concept or column).strip()
        add_learned_synonym(phrase=phrase, mapped_concept=target_concept, source=source)
        add_learned_column_mapping(
            raw_column_name=column,
            standard_concept=target_concept,
            confidence=0.95,
            source=source,
        )
        sync_learning_files()
        return True
    except Exception:
        # Learning is best-effort. Never block a request because the DB hiccupped.
        return False


# Match assumption strings like "FASFA was interpreted as FAFSA." or
# "Major was mapped to Program." that the LLM produces in its assumptions list.
_ASSUMPTION_PATTERNS = (
    re.compile(
        r"^\s*(?P<phrase>[^.]+?)\s+(?:was\s+)?(?:interpreted|mapped|treated)\s+as\s+(?P<target>[^.]+?)\s*\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?P<phrase>[^.]+?)\s+(?:was\s+)?mapped\s+to\s+(?P<target>[^.]+?)\s*\.?\s*$",
        re.IGNORECASE,
    ),
)


def record_llm_assumptions(
    *,
    assumptions: list[str],
    sheet_columns: list[str],
    source: str = "llm_assumption",
) -> int:
    """Persist mappings the LLM stated in its assumptions list.

    Only saves an assumption when the target side resolves to an actual column
    in the current workbook. This prevents the LLM from poisoning the synonym
    table with hallucinated concepts.
    """
    if not assumptions:
        return 0

    saved = 0
    column_lookup = {column.lower(): column for column in sheet_columns}
    for assumption in assumptions:
        for pattern in _ASSUMPTION_PATTERNS:
            match = pattern.match(assumption)
            if not match:
                continue
            phrase = match.group("phrase").strip()
            target = match.group("target").strip().rstrip(".")
            actual_column = column_lookup.get(target.lower())
            if not phrase or not actual_column:
                break
            if record_clarification_answer(
                user_phrase=phrase,
                chosen_column=actual_column,
                source=source,
            ):
                saved += 1
            break
    return saved
