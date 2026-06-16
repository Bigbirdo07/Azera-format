"""Edit Mode planner.

Thin adapter over the existing expert_planner.plan_request so the unified
assistant has a single edit entry point that returns the edit_workbook schema.
All the real work — rule parsing, local LLM fallback, validation against the
workbook, and the learning loop — already lives in expert_planner and validator;
this module only reshapes the result and tags it as edit_workbook.

The LLM never edits the workbook. It only proposes a plan, which is validated
here and must be confirmed by the user before core.action_engine executes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nlp.expert_planner import PlanResult, plan_request


@dataclass
class EditPlanResult:
    request_type: str  # "edit_workbook" or "clarify"
    plan_type: str
    confidence: float
    plain_english_summary: str
    commands: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    clarification_question: str = ""
    requires_confirmation: bool = True
    can_execute: bool = False
    validation_error: str | None = None
    source: str = "rule_parser"
    mapped_columns: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_type": self.request_type,
            "plan_type": self.plan_type,
            "confidence": self.confidence,
            "plain_english_summary": self.plain_english_summary,
            "commands": self.commands,
            "assumptions": self.assumptions,
            "clarification_question": self.clarification_question,
            "requires_confirmation": self.requires_confirmation,
            "can_execute": self.can_execute,
            "validation_error": self.validation_error,
            "source": self.source,
        }


def plan_edit(
    *,
    user_request: str,
    selected_sheet: str,
    sheet_columns: dict[str, list[str]],
    sheets: dict[str, Any],
    original_file_name: str,
    use_local_llm: bool,
    ollama_model: str,
) -> EditPlanResult:
    result: PlanResult = plan_request(
        user_request=user_request,
        selected_sheet=selected_sheet,
        sheet_columns=sheet_columns,
        sheets=sheets,
        original_file_name=original_file_name,
        use_local_llm=use_local_llm,
        ollama_model=ollama_model,
    )

    request_type = "clarify" if result.plan_type == "clarify" else "edit_workbook"
    return EditPlanResult(
        request_type=request_type,
        plan_type=result.plan_type,
        confidence=result.confidence,
        plain_english_summary=result.plain_english_summary,
        commands=list(result.commands),
        assumptions=list(result.assumptions),
        clarification_question=result.clarification_question,
        requires_confirmation=result.requires_confirmation,
        can_execute=result.can_execute,
        validation_error=result.validation_error,
        source=result.source,
        mapped_columns=dict(result.mapped_columns),
    )
