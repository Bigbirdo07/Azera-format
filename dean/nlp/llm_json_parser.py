from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from core.command_schema import SUPPORTED_ACTIONS, VALID_OPERATORS


PLANNER_ACTIONS = {
    "filter_rows",
    "highlight_rows",
    "sort_rows",
    "sum_column",
    "average_column",
    "count_rows",
    "count_by_group",
    "sum_by_group",
    "create_formula",
    "create_chart",
    "create_report",
    "format_report",
    "detect_missing_values",
    "remove_duplicates",
    "move_rows_to_sheet",
    "create_summary_sheet",
    "column_mapping_request",
    "create_data_quality_report",
}

PLANNER_OPERATORS = set(VALID_OPERATORS) | {"in"}

PLAN_TYPES = {"single_action", "multi_step_plan", "clarify"}


class LLMCommandError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedPlan:
    plan_type: str
    confidence: float
    plain_english_summary: str
    commands: list[dict[str, Any]] = field(default_factory=list)
    clarification_question: str = ""
    assumptions: list[str] = field(default_factory=list)
    requires_confirmation: bool = True
    raw: dict[str, Any] = field(default_factory=dict)


def parse_llm_response(
    response_text: str,
    sheet_columns: dict[str, list[str]],
) -> dict[str, Any]:
    """Legacy single-command parser. Returns one validated command dict.

    Kept for the existing single-action chat flow. New code should call
    parse_llm_plan() instead.
    """
    payload = _extract_json_object(response_text)

    if "plan_type" in payload:
        plan = _normalize_plan(payload)
        _validate_plan(plan, sheet_columns)
        if plan.plan_type == "clarify":
            return {
                "action": "clarify",
                "question": plan.clarification_question
                or "Please clarify your request.",
            }
        if not plan.commands:
            raise LLMCommandError("Plan must contain at least one command.")
        return plan.commands[0]

    command = payload
    if command.get("action") == "clarify":
        question = command.get("question")
        if not question:
            raise LLMCommandError("Clarification response must include a question.")
        return command

    command = _normalize_command(command)
    _validate_basic_command(command, sheet_columns)
    return command


def parse_llm_plan(
    response_text: str,
    sheet_columns: dict[str, list[str]],
) -> ParsedPlan:
    """Parse the expert-planner envelope and validate every command in it.

    Raises LLMCommandError on bad JSON, unsupported actions, nonexistent sheets
    or columns, invalid operators, or schema mismatches.
    """
    payload = _extract_json_object(response_text)

    if "plan_type" not in payload:
        # The model returned a bare single command. Wrap it in a single_action plan.
        if payload.get("action") == "clarify":
            return ParsedPlan(
                plan_type="clarify",
                confidence=float(payload.get("confidence", 0.5) or 0.5),
                plain_english_summary=payload.get("question", ""),
                clarification_question=payload.get("question", ""),
                requires_confirmation=False,
                raw=payload,
            )
        command = _normalize_command(payload)
        _validate_basic_command(command, sheet_columns)
        return ParsedPlan(
            plan_type="single_action",
            confidence=float(payload.get("confidence", 0.7) or 0.7),
            plain_english_summary=payload.get("plain_english_summary", ""),
            commands=[command],
            assumptions=list(payload.get("assumptions", []) or []),
            requires_confirmation=True,
            raw=payload,
        )

    plan = _normalize_plan(payload)
    _validate_plan(plan, sheet_columns)
    return plan


def command_to_confirmation(command: dict[str, Any]) -> str:
    action = command.get("action")

    if command.get("conditions") and action in {"filter_rows", "highlight_rows", "move_rows_to_sheet"}:
        condition_text = " and ".join(_condition_text(item) for item in command["conditions"])
        verb = {
            "filter_rows": "filter rows",
            "highlight_rows": "highlight rows",
            "move_rows_to_sheet": "copy rows to another sheet",
        }.get(action, "use rows")
        return f"I am going to {verb} where {condition_text}. Continue?"

    if action == "sort_rows":
        sort_by = command.get("sort_by") or []
        if sort_by:
            columns = ", ".join(item.get("column", "") for item in sort_by if isinstance(item, dict))
            return f"I am going to sort rows by {columns}. Continue?"
        return "I am going to sort the rows. Continue?"

    if action in {"sum_column", "average_column"}:
        column = command.get("column", "the selected column")
        group_by = command.get("group_by")
        verb = "average" if action == "average_column" else "sum"
        if group_by:
            return f"I am going to {verb} {column} by {group_by}. Continue?"
        return f"I am going to {verb} {column}. Continue?"

    if action == "sum_by_group":
        return (
            f"I am going to sum {command.get('sum_column', 'the selected column')} "
            f"by {command.get('group_by', 'the selected group')}. Continue?"
        )

    if action == "count_by_group":
        return f"I am going to count rows by {command.get('group_by', 'the selected column')}. Continue?"

    if action == "count_rows":
        return f"I am going to count rows on {command.get('sheet', 'the selected sheet')}. Continue?"

    if action == "create_chart":
        return (
            f"I am going to create a {command.get('chart_type', 'bar')} chart on "
            f"{command.get('output_sheet', 'a new sheet')}. Continue?"
        )

    if action == "create_formula":
        return f"I am going to create a {command.get('formula_type', 'formula')} formula. Continue?"

    if action == "create_report":
        return f"I am going to create {command.get('output_sheet', 'a report sheet')}. Continue?"

    if action == "create_summary_sheet":
        return f"I am going to build a summary sheet at {command.get('output_sheet', 'a new sheet')}. Continue?"

    if action == "format_report":
        sheets = command.get("sheets") or [command.get("sheet")]
        sheets = [sheet for sheet in sheets if sheet]
        return f"I am going to format {', '.join(sheets) if sheets else 'the selected sheets'}. Continue?"

    if action == "detect_missing_values":
        scope = command.get("column") or "all columns"
        return f"I am going to scan {scope} for missing values. Continue?"

    if action == "remove_duplicates":
        return f"I am going to remove duplicates on {command.get('sheet', 'the selected sheet')}. Continue?"

    if action == "create_data_quality_report":
        return (
            f"I am going to scan {command.get('sheet', 'the selected sheet')} for data quality "
            f"issues and create {command.get('output_sheet', 'Data Quality Report')}. Continue?"
        )

    if action == "column_mapping_request":
        candidates = command.get("candidates") or []
        joined = ", ".join(str(item) for item in candidates) or "the listed candidates"
        return f"I need you to choose which column to use for '{command.get('user_phrase', 'that term')}': {joined}."

    return f"I interpreted your request as action `{action}`. Continue?"


def plan_to_summary(plan: ParsedPlan) -> str:
    if plan.plan_type == "clarify":
        return plan.clarification_question or plan.plain_english_summary
    if plan.plain_english_summary:
        return plan.plain_english_summary
    if plan.commands:
        return command_to_confirmation(plan.commands[0])
    return "I am ready to run the plan once you confirm."


def _extract_json_object(response_text: str) -> dict[str, Any]:
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise LLMCommandError("Local model response did not contain valid JSON.")


def _normalize_plan(payload: dict[str, Any]) -> ParsedPlan:
    plan_type = str(payload.get("plan_type", "")).strip()
    if plan_type not in PLAN_TYPES:
        raise LLMCommandError(f"Unsupported plan_type from local model: {plan_type!r}")

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        raise LLMCommandError("Plan confidence must be a number between 0 and 1.")
    confidence = max(0.0, min(confidence, 1.0))

    raw_commands = payload.get("commands") or []
    if not isinstance(raw_commands, list):
        raise LLMCommandError("Plan commands must be a list.")
    commands = [_normalize_command(item) for item in raw_commands if isinstance(item, dict)]

    assumptions = payload.get("assumptions") or []
    if not isinstance(assumptions, list):
        raise LLMCommandError("Plan assumptions must be a list of strings.")

    return ParsedPlan(
        plan_type=plan_type,
        confidence=confidence,
        plain_english_summary=str(payload.get("plain_english_summary", "") or ""),
        commands=commands,
        clarification_question=str(payload.get("clarification_question", "") or ""),
        assumptions=[str(item) for item in assumptions],
        requires_confirmation=bool(payload.get("requires_confirmation", plan_type != "clarify")),
        raw=payload,
    )


def _validate_plan(plan: ParsedPlan, sheet_columns: dict[str, list[str]]) -> None:
    if plan.plan_type == "clarify":
        if not plan.clarification_question:
            raise LLMCommandError("Clarify plans must include a clarification_question.")
        return

    if not plan.commands:
        raise LLMCommandError(f"{plan.plan_type} plans must have at least one command.")

    if plan.plan_type == "single_action" and len(plan.commands) != 1:
        raise LLMCommandError("single_action plans must contain exactly one command.")

    cumulative_sheets = dict(sheet_columns)

    for command in plan.commands:
        action = command.get("action")
        if action == "column_mapping_request":
            _validate_column_mapping_request(command)
            continue
        if action == "format_report":
            _validate_format_report(command, cumulative_sheets)
            continue

        _validate_basic_command(command, cumulative_sheets)

        # Register output_sheet so later steps can reference it.
        output_sheet = command.get("output_sheet")
        if output_sheet and output_sheet not in cumulative_sheets:
            cumulative_sheets[output_sheet] = _projected_output_columns(command, cumulative_sheets)


def _validate_column_mapping_request(command: dict[str, Any]) -> None:
    if not command.get("concept"):
        raise LLMCommandError("column_mapping_request requires a concept.")
    candidates = command.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMCommandError("column_mapping_request requires a non-empty candidates list.")


def _validate_format_report(command: dict[str, Any], sheet_columns: dict[str, list[str]]) -> None:
    sheets = command.get("sheets")
    if isinstance(sheets, list) and sheets:
        for sheet in sheets:
            if sheet not in sheet_columns:
                raise LLMCommandError(f"Local model used a nonexistent sheet: {sheet}")
        return
    sheet = command.get("sheet")
    if not sheet or sheet not in sheet_columns:
        raise LLMCommandError(f"Local model used a nonexistent sheet for format_report: {sheet}")


def _projected_output_columns(
    command: dict[str, Any],
    sheet_columns: dict[str, list[str]],
) -> list[str]:
    action = command.get("action")
    source_sheet = command.get("sheet")
    source_columns = sheet_columns.get(source_sheet, [])

    if action == "filter_rows" or action == "move_rows_to_sheet" or action == "highlight_rows" or action == "sort_rows":
        return list(source_columns)

    if action == "count_by_group":
        group_by = command.get("group_by")
        return [column for column in [group_by, "count"] if column]

    if action == "sum_by_group":
        group_by = command.get("group_by")
        sum_column = command.get("sum_column")
        return [column for column in [group_by, sum_column] if column]

    if action == "average_column":
        group_by = command.get("group_by")
        value_column = command.get("column")
        if group_by:
            return [column for column in [group_by, value_column] if column]
        return [value_column] if value_column else []

    if action == "sum_column":
        group_by = command.get("group_by")
        value_column = command.get("column")
        if group_by:
            return [column for column in [group_by, value_column] if column]
        return [value_column] if value_column else []

    if action == "create_summary_sheet":
        metrics = command.get("metrics") or []
        labels = [str(metric.get("label", "metric")) for metric in metrics if isinstance(metric, dict)]
        return labels or ["metric", "value"]

    return list(source_columns)


def _normalize_command(command: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(command)

    # Accept "category_column" as an alias for "group_by" in chart commands.
    if normalized.get("action") == "create_chart":
        if "category_column" in normalized and "group_by" not in normalized:
            normalized["group_by"] = normalized["category_column"]
        if "metric" not in normalized:
            if normalized.get("value_column"):
                normalized["metric"] = "sum"
            else:
                normalized["metric"] = "count_rows"

    if normalized.get("action") == "sum_by_group":
        # Keep the planner action name but also expose a compatible
        # representation for the legacy single-command engine path.
        if "sum_column" in normalized and "column" not in normalized:
            normalized.setdefault("column", normalized["sum_column"])

    return normalized


def _validate_basic_command(command: dict[str, Any], sheet_columns: dict[str, list[str]]) -> None:
    action = command.get("action")
    if action not in PLANNER_ACTIONS and action not in SUPPORTED_ACTIONS:
        raise LLMCommandError(f"Unsupported action from local model: {action}")

    if action in {"format_report", "column_mapping_request"}:
        return

    sheet = command.get("sheet")
    if sheet not in sheet_columns:
        raise LLMCommandError(f"Local model used a nonexistent sheet: {sheet}")

    columns = sheet_columns[sheet]
    for column in _referenced_columns(command):
        if column not in columns:
            raise LLMCommandError(f"Local model used a nonexistent column: {column}")

    lookup = command.get("lookup")
    if isinstance(lookup, dict):
        lookup_sheet = lookup.get("lookup_sheet")
        if lookup_sheet not in sheet_columns:
            raise LLMCommandError(f"Local model used a nonexistent lookup sheet: {lookup_sheet}")
        lookup_columns = sheet_columns[lookup_sheet]
        for key in ["lookup_key_column", "return_column"]:
            if lookup.get(key) not in lookup_columns:
                raise LLMCommandError(f"Local model used a nonexistent lookup column: {lookup.get(key)}")
        if lookup.get("lookup_value_column") not in columns:
            raise LLMCommandError(
                f"Local model used a nonexistent lookup value column: {lookup.get('lookup_value_column')}"
            )

    for condition in command.get("conditions", []):
        operator = condition.get("operator")
        if operator not in PLANNER_OPERATORS:
            raise LLMCommandError(f"Local model used an invalid operator: {operator}")
        if operator == "in" and not isinstance(condition.get("value"), list):
            raise LLMCommandError("Operator 'in' requires a list value.")


def _referenced_columns(command: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key in [
        "column",
        "group_by",
        "value_column",
        "category_column",
        "sum_column",
        "chart_group_by",
    ]:
        if command.get(key):
            columns.append(command[key])

    for condition in command.get("conditions", []):
        if condition.get("column"):
            columns.append(condition["column"])

    logic = command.get("logic")
    if isinstance(logic, dict):
        if logic.get("condition_column"):
            columns.append(logic["condition_column"])
        for rule in logic.get("rules", []):
            if rule.get("condition_column"):
                columns.append(rule["condition_column"])

    if isinstance(command.get("columns"), list):
        columns.extend(command["columns"])

    for sort_item in command.get("sort_by") or []:
        if isinstance(sort_item, dict) and sort_item.get("column"):
            columns.append(sort_item["column"])

    metrics = command.get("metrics")
    if isinstance(metrics, list):
        for metric in metrics:
            if isinstance(metric, dict) and metric.get("column"):
                columns.append(metric["column"])

    lookup = command.get("lookup")
    if isinstance(lookup, dict):
        if lookup.get("lookup_value_column"):
            columns.append(lookup["lookup_value_column"])

    return columns


def _condition_text(condition: dict[str, Any]) -> str:
    column = condition.get("column", "the selected column")
    operator = condition.get("operator", "")
    value = condition.get("value")
    if operator == "greater_than":
        return f"{column} is greater than {value}"
    if operator == "greater_or_equal":
        return f"{column} is at least {value}"
    if operator == "less_than":
        return f"{column} is less than {value}"
    if operator == "less_or_equal":
        return f"{column} is at most {value}"
    if operator == "equals":
        return f"{column} equals {value}"
    if operator == "not_equals":
        return f"{column} does not equal {value}"
    if operator == "is_missing":
        return f"{column} is blank"
    if operator == "is_not_missing":
        return f"{column} is not blank"
    if operator == "contains":
        return f"{column} contains {value}"
    if operator == "not_contains":
        return f"{column} does not contain {value}"
    if operator == "contains_any" and isinstance(value, list):
        joined = ", ".join(str(item) for item in value)
        return f"{column} contains any of {joined}"
    if operator == "in" and isinstance(value, list):
        joined = ", ".join(str(item) for item in value)
        return f"{column} is one of {joined}"
    if value is not None:
        return f"{column} {operator.replace('_', ' ')} {value}"
    return f"{column} {operator.replace('_', ' ')}"
