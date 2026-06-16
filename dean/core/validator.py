from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.command_schema import (
    NUMERIC_OPERATORS,
    OPERATORS_WITHOUT_VALUE,
    PLANNER_OPERATORS,
    SUPPORTED_CHART_METRICS,
    SUPPORTED_CHART_TYPES,
    SUPPORTED_FORMULAS,
    SUPPORTED_PLANNER_ACTIONS,
    SUPPORTED_REPORT_TYPES,
    SUPPORTED_ACTIONS,
    VALID_OPERATORS,
)
from core.exporter import OUTPUTS_DIR


class CommandValidationError(ValueError):
    pass


def validate_command(
    command: dict[str, Any],
    sheets: dict[str, pd.DataFrame],
    original_file_name: str,
) -> None:
    if not isinstance(command, dict):
        raise CommandValidationError("Command must be a JSON object.")

    action = command.get("action")
    if action not in SUPPORTED_ACTIONS:
        raise CommandValidationError(f"Unsupported action: {action}")

    sheet_name = command.get("sheet")
    if not sheet_name or sheet_name not in sheets:
        raise CommandValidationError(f"Sheet does not exist: {sheet_name}")

    dataframe = sheets[sheet_name]

    if action == "create_chart":
        _validate_chart_command(command, dataframe)

    if action == "create_formula":
        _validate_formula_command(command, sheets)

    if action == "create_report":
        _validate_report_command(command, dataframe)

    if action == "create_data_quality_report":
        _validate_data_quality_command(command)

    if action in {"filter_rows", "highlight_rows"}:
        _validate_conditions(command.get("conditions", []), dataframe)

    if action == "count_rows" and command.get("conditions"):
        _validate_conditions(command.get("conditions", []), dataframe)

    if action == "sum_column":
        column = command.get("column")
        _validate_column(column, dataframe)
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise CommandValidationError(f"Column must be numeric for sum_column: {column}")
        if command.get("group_by"):
            _validate_column(command.get("group_by"), dataframe)

    if action == "detect_missing_values" and command.get("column"):
        _validate_column(command.get("column"), dataframe)

    if action == "count_by_group":
        column = command.get("group_by") or command.get("column")
        _validate_column(column, dataframe)
        if command.get("conditions"):
            _validate_conditions(command.get("conditions", []), dataframe)

    if action == "remove_duplicates":
        columns = command.get("columns")
        if columns:
            if not isinstance(columns, list):
                raise CommandValidationError("remove_duplicates columns must be a list.")
            for column in columns:
                _validate_column(column, dataframe)

    _validate_export_target(original_file_name)


def _validate_conditions(conditions: Any, dataframe: pd.DataFrame) -> None:
    if not isinstance(conditions, list) or not conditions:
        raise CommandValidationError("This action requires at least one condition.")

    for condition in conditions:
        if not isinstance(condition, dict):
            raise CommandValidationError("Each condition must be a JSON object.")

        column = condition.get("column")
        operator = condition.get("operator")

        _validate_column(column, dataframe)

        if operator not in VALID_OPERATORS:
            raise CommandValidationError(f"Invalid operator: {operator}")

        if operator not in OPERATORS_WITHOUT_VALUE and "value" not in condition:
            raise CommandValidationError(f"Operator requires a value: {operator}")

        if operator not in OPERATORS_WITHOUT_VALUE:
            _validate_condition_value_compatible(condition, dataframe)

        if operator in NUMERIC_OPERATORS:
            _validate_numeric_condition_value(condition, dataframe)


def _validate_column(column: Any, dataframe: pd.DataFrame) -> None:
    if not column or column not in dataframe.columns:
        raise CommandValidationError(f"Column does not exist: {column}")


def _validate_formula_command(command: dict[str, Any], sheets: dict[str, pd.DataFrame]) -> None:
    sheet_name = command["sheet"]
    dataframe = sheets[sheet_name]
    formula_type = str(command.get("formula_type", "")).upper()

    if formula_type not in SUPPORTED_FORMULAS:
        raise CommandValidationError(f"Unsupported formula type: {formula_type}")

    row_formula_types = {"IF", "IFS", "XLOOKUP", "VLOOKUP", "CONCAT", "TEXT"}
    if formula_type in row_formula_types and not str(command.get("new_column", "")).strip():
        raise CommandValidationError(f"{formula_type} formulas require a new_column.")

    if command.get("new_column") is not None and not str(command.get("new_column")).strip():
        raise CommandValidationError("Formula column name cannot be blank.")

    if formula_type in {"SUM", "AVERAGE", "COUNT", "COUNTA", "UNIQUE", "SORT", "TEXT"}:
        _validate_column(command.get("column"), dataframe)

    if formula_type in {"COUNTIF", "SUMIF", "FILTER"}:
        criteria = _validate_criteria(command.get("criteria"), dataframe)
        if len(criteria) != 1:
            raise CommandValidationError(f"{formula_type} requires exactly one criterion.")

    if formula_type in {"COUNTIFS", "SUMIFS"}:
        _validate_criteria(command.get("criteria"), dataframe)

    if formula_type in {"SUMIF", "SUMIFS"}:
        _validate_column(command.get("sum_column"), dataframe)
        if not pd.api.types.is_numeric_dtype(dataframe[command["sum_column"]]):
            raise CommandValidationError(f"sum_column must be numeric: {command['sum_column']}")

    if formula_type == "IF":
        logic = command.get("logic")
        if not isinstance(logic, dict):
            raise CommandValidationError("IF formulas require a logic object.")
        _validate_formula_condition(logic, dataframe)

    if formula_type == "IFS":
        logic = command.get("logic")
        rules = logic.get("rules") if isinstance(logic, dict) else None
        if not isinstance(rules, list) or not rules:
            raise CommandValidationError("IFS formulas require logic.rules.")
        for rule in rules:
            _validate_formula_condition(rule, dataframe)

    if formula_type in {"XLOOKUP", "VLOOKUP"}:
        lookup = command.get("lookup")
        if not isinstance(lookup, dict):
            raise CommandValidationError(f"{formula_type} formulas require a lookup object.")
        _validate_column(lookup.get("lookup_value_column"), dataframe)
        lookup_sheet = lookup.get("lookup_sheet")
        if lookup_sheet not in sheets:
            raise CommandValidationError(f"Lookup sheet does not exist: {lookup_sheet}")
        lookup_dataframe = sheets[lookup_sheet]
        _validate_column(lookup.get("lookup_key_column"), lookup_dataframe)
        _validate_column(lookup.get("return_column"), lookup_dataframe)

    if formula_type == "CONCAT":
        columns = command.get("columns")
        if not isinstance(columns, list) or not columns:
            raise CommandValidationError("CONCAT formulas require a columns list.")
        for column in columns:
            _validate_column(column, dataframe)


def _validate_chart_command(command: dict[str, Any], dataframe: pd.DataFrame) -> None:
    chart_type = command.get("chart_type")
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise CommandValidationError(f"Unsupported chart type: {chart_type}")

    metric = command.get("metric")
    if metric not in SUPPORTED_CHART_METRICS:
        raise CommandValidationError(f"Unsupported chart metric: {metric}")

    _validate_column(command.get("group_by"), dataframe)

    if metric in {"sum", "count_missing"}:
        value_column = command.get("value_column")
        _validate_column(value_column, dataframe)
        if metric == "sum" and not pd.api.types.is_numeric_dtype(dataframe[value_column]):
            raise CommandValidationError(f"value_column must be numeric for sum charts: {value_column}")

    output_sheet = command.get("output_sheet")
    if output_sheet is not None and not str(output_sheet).strip():
        raise CommandValidationError("Chart output_sheet cannot be blank.")


def _validate_report_command(command: dict[str, Any], dataframe: pd.DataFrame) -> None:
    report_type = command.get("report_type")
    if report_type not in SUPPORTED_REPORT_TYPES:
        raise CommandValidationError(f"Unsupported report type: {report_type}")

    if command.get("conditions"):
        _validate_conditions(command.get("conditions", []), dataframe)

    for column in command.get("required_columns", []):
        _validate_column(column, dataframe)

    if command.get("chart_group_by"):
        _validate_column(command.get("chart_group_by"), dataframe)

    output_sheet = command.get("output_sheet")
    if output_sheet is not None and not str(output_sheet).strip():
        raise CommandValidationError("Report output_sheet cannot be blank.")


def _validate_data_quality_command(command: dict[str, Any]) -> None:
    output_sheet = command.get("output_sheet")
    if output_sheet is not None and not str(output_sheet).strip():
        raise CommandValidationError("Data quality output_sheet cannot be blank.")


def _validate_criteria(criteria: Any, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(criteria, list) or not criteria:
        raise CommandValidationError("Formula criteria must be a non-empty list.")
    for item in criteria:
        _validate_formula_condition(item, dataframe)
    return criteria


def _validate_formula_condition(condition: Any, dataframe: pd.DataFrame) -> None:
    if not isinstance(condition, dict):
        raise CommandValidationError("Formula conditions must be JSON objects.")

    column = condition.get("condition_column") or condition.get("column")
    operator = condition.get("operator")
    normalized_condition = {
        "column": column,
        "operator": operator,
    }
    if "value" in condition:
        normalized_condition["value"] = condition["value"]
    _validate_conditions([normalized_condition], dataframe)


def _validate_numeric_condition_value(condition: dict[str, Any], dataframe: pd.DataFrame) -> None:
    column = condition["column"]
    value = condition.get("value")

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise CommandValidationError(f"Column must be numeric for {condition['operator']}: {column}")

    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise CommandValidationError(
            f"Value must be numeric for column {column}: {value}"
        ) from exc


def _validate_condition_value_compatible(
    condition: dict[str, Any],
    dataframe: pd.DataFrame,
) -> None:
    column = condition["column"]
    operator = condition["operator"]
    value = condition.get("value")
    series = dataframe[column]

    if operator in {"contains", "not_contains", "starts_with", "ends_with"}:
        return

    if operator in {"in", "not_in", "contains_any", "between"}:
        # List-valued operators are validated by their own handlers.
        return

    if pd.api.types.is_numeric_dtype(series):
        try:
            float(value)
        except (TypeError, ValueError) as exc:
            raise CommandValidationError(
                f"Value must be numeric for column {column}: {value}"
            ) from exc

    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            pd.to_datetime(value)
        except (TypeError, ValueError) as exc:
            raise CommandValidationError(
                f"Value must be date-like for column {column}: {value}"
            ) from exc

    if pd.api.types.is_bool_dtype(series) and not isinstance(value, bool):
        raise CommandValidationError(f"Value must be true or false for column {column}.")


def _validate_export_target(original_file_name: str) -> None:
    original_path = Path(original_file_name)
    if original_path.parent == OUTPUTS_DIR:
        raise CommandValidationError("Original file cannot already point inside the outputs folder.")


# Plan-level validation -------------------------------------------------------


class PlanValidationError(CommandValidationError):
    """Raised when a multi-step plan from the local Ollama planner fails validation."""


def validate_plan(
    plan: dict[str, Any],
    sheets: dict[str, pd.DataFrame],
    original_file_name: str,
) -> None:
    """Validate every command in a planner envelope.

    The plan may produce intermediate output_sheets that later steps reference.
    Each new output_sheet is registered with a projected column schema so the
    next step's column references can be validated without executing anything.
    """
    if not isinstance(plan, dict):
        raise PlanValidationError("Plan must be a JSON object.")

    plan_type = plan.get("plan_type")
    if plan_type not in {"single_action", "multi_step_plan", "clarify"}:
        raise PlanValidationError(f"Unsupported plan_type: {plan_type}")

    if plan_type == "clarify":
        if not str(plan.get("clarification_question", "")).strip():
            raise PlanValidationError("Clarify plans must include a clarification_question.")
        return

    commands = plan.get("commands")
    if not isinstance(commands, list) or not commands:
        raise PlanValidationError(f"{plan_type} plans must contain at least one command.")

    if plan_type == "single_action" and len(commands) != 1:
        raise PlanValidationError("single_action plans must contain exactly one command.")

    _validate_export_target(original_file_name)

    projected_sheets: dict[str, pd.DataFrame] = dict(sheets)
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            raise PlanValidationError(f"Plan step {index + 1} must be a JSON object.")

        action = command.get("action")
        if action not in SUPPORTED_PLANNER_ACTIONS:
            raise PlanValidationError(
                f"Plan step {index + 1} uses unsupported action: {action}"
            )

        if action == "column_mapping_request":
            _validate_column_mapping_step(command)
            continue

        if action == "format_report":
            _validate_format_report_step(command, projected_sheets)
            continue

        _validate_planner_command(command, projected_sheets)

        output_sheet = command.get("output_sheet")
        if output_sheet:
            projected_sheets[output_sheet] = _projected_dataframe(command, projected_sheets)


def _validate_planner_command(
    command: dict[str, Any],
    sheets: dict[str, pd.DataFrame],
) -> None:
    action = command["action"]
    sheet_name = command.get("sheet")
    if not sheet_name or sheet_name not in sheets:
        raise PlanValidationError(f"Plan references nonexistent sheet: {sheet_name}")

    dataframe = sheets[sheet_name]

    if action in {"filter_rows", "highlight_rows", "move_rows_to_sheet"}:
        _validate_planner_conditions(command.get("conditions", []), dataframe)
        return

    if action == "sort_rows":
        sort_by = command.get("sort_by")
        if not isinstance(sort_by, list) or not sort_by:
            raise PlanValidationError("sort_rows requires a non-empty sort_by list.")
        for item in sort_by:
            if not isinstance(item, dict):
                raise PlanValidationError("Each sort_by entry must be an object.")
            _validate_column(item.get("column"), dataframe)
            direction = str(item.get("direction", "asc")).lower()
            if direction not in {"asc", "desc"}:
                raise PlanValidationError(f"Invalid sort direction: {direction}")
        return

    if action == "sum_column":
        column = command.get("column")
        _validate_column(column, dataframe)
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise PlanValidationError(f"Column must be numeric for sum_column: {column}")
        if command.get("group_by"):
            _validate_column(command["group_by"], dataframe)
        return

    if action == "average_column":
        column = command.get("column")
        _validate_column(column, dataframe)
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise PlanValidationError(f"Column must be numeric for average_column: {column}")
        if command.get("group_by"):
            _validate_column(command["group_by"], dataframe)
        return

    if action == "sum_by_group":
        sum_column = command.get("sum_column") or command.get("column")
        group_by = command.get("group_by")
        if not sum_column or not group_by:
            raise PlanValidationError("sum_by_group requires both sum_column and group_by.")
        _validate_column(sum_column, dataframe)
        _validate_column(group_by, dataframe)
        if not pd.api.types.is_numeric_dtype(dataframe[sum_column]):
            raise PlanValidationError(f"Column must be numeric for sum_by_group: {sum_column}")
        return

    if action == "count_rows":
        if command.get("conditions"):
            _validate_planner_conditions(command["conditions"], dataframe)
        return

    if action == "count_by_group":
        column = command.get("group_by") or command.get("column")
        _validate_column(column, dataframe)
        if command.get("conditions"):
            _validate_planner_conditions(command["conditions"], dataframe)
        return

    if action == "detect_missing_values":
        if command.get("column"):
            _validate_column(command["column"], dataframe)
        return

    if action == "remove_duplicates":
        columns = command.get("columns")
        if columns:
            if not isinstance(columns, list):
                raise PlanValidationError("remove_duplicates columns must be a list.")
            for column in columns:
                _validate_column(column, dataframe)
        return

    if action == "create_chart":
        _validate_planner_chart(command, dataframe)
        return

    if action == "create_formula":
        _validate_formula_command(command, sheets)
        return

    if action == "create_report":
        _validate_report_command(command, dataframe)
        return

    if action == "create_summary_sheet":
        metrics = command.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise PlanValidationError("create_summary_sheet requires a non-empty metrics list.")
        for metric in metrics:
            if not isinstance(metric, dict):
                raise PlanValidationError("Each metric must be an object.")
            kind = metric.get("kind")
            if kind not in {"count", "sum", "count_missing", "average"}:
                raise PlanValidationError(f"Unsupported metric kind: {kind}")
            if kind in {"sum", "count_missing", "average"}:
                _validate_column(metric.get("column"), dataframe)
            if kind in {"sum", "average"} and not pd.api.types.is_numeric_dtype(
                dataframe[metric["column"]]
            ):
                raise PlanValidationError(
                    f"Metric column must be numeric for {kind}: {metric['column']}"
                )
        return

    if action in {"freeze_header", "autofit_columns"}:
        # The sheet was already validated above; nothing else is required.
        return

    if action == "apply_conditional_formatting":
        _validate_planner_conditions(command.get("conditions", []), dataframe)
        if command.get("column"):
            _validate_column(command["column"], dataframe)
        return


def _validate_planner_conditions(conditions: Any, dataframe: pd.DataFrame) -> None:
    if not isinstance(conditions, list) or not conditions:
        raise PlanValidationError("This action requires at least one condition.")

    for condition in conditions:
        if not isinstance(condition, dict):
            raise PlanValidationError("Each condition must be a JSON object.")

        column = condition.get("column")
        operator = condition.get("operator")

        try:
            _validate_column(column, dataframe)
        except CommandValidationError as exc:
            raise PlanValidationError(str(exc)) from exc

        if operator not in PLANNER_OPERATORS:
            raise PlanValidationError(f"Invalid operator: {operator}")

        # Value-less operators need no further checks.
        if operator in OPERATORS_WITHOUT_VALUE:
            continue

        if operator in {"in", "not_in", "contains_any"}:
            value = condition.get("value")
            if not isinstance(value, list) or not value:
                raise PlanValidationError(f"Operator '{operator}' requires a non-empty list value.")
            continue

        if operator == "between":
            value = condition.get("value")
            if not isinstance(value, list) or len(value) != 2:
                raise PlanValidationError("Operator 'between' requires a [low, high] list.")
            try:
                float(value[0]); float(value[1])
            except (TypeError, ValueError):
                raise PlanValidationError("'between' bounds must be numeric.")
            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                raise PlanValidationError(f"'between' requires a numeric column: {column}")
            continue

        if "value" not in condition:
            raise PlanValidationError(f"Operator requires a value: {operator}")

        if operator in {"starts_with", "ends_with"} and pd.api.types.is_numeric_dtype(dataframe[column]):
            raise PlanValidationError(f"'{operator}' requires a text column: {column}")

        try:
            _validate_condition_value_compatible(condition, dataframe)
        except CommandValidationError as exc:
            raise PlanValidationError(str(exc)) from exc

        if operator in NUMERIC_OPERATORS:
            try:
                _validate_numeric_condition_value(condition, dataframe)
            except CommandValidationError as exc:
                raise PlanValidationError(str(exc)) from exc


def _validate_planner_chart(command: dict[str, Any], dataframe: pd.DataFrame) -> None:
    chart_type = command.get("chart_type")
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise PlanValidationError(f"Unsupported chart type: {chart_type}")

    category = command.get("category_column") or command.get("group_by")
    _validate_column(category, dataframe)

    metric = command.get("metric")
    if metric is None:
        if command.get("value_column"):
            metric = "sum"
        else:
            metric = "count_rows"

    if metric not in SUPPORTED_CHART_METRICS:
        raise PlanValidationError(f"Unsupported chart metric: {metric}")

    if metric in {"sum", "count_missing"}:
        value_column = command.get("value_column")
        _validate_column(value_column, dataframe)
        if metric == "sum" and not pd.api.types.is_numeric_dtype(dataframe[value_column]):
            raise PlanValidationError(f"value_column must be numeric for sum charts: {value_column}")

    output_sheet = command.get("output_sheet")
    if output_sheet is not None and not str(output_sheet).strip():
        raise PlanValidationError("Chart output_sheet cannot be blank.")


def _validate_column_mapping_step(command: dict[str, Any]) -> None:
    if not str(command.get("concept", "")).strip():
        raise PlanValidationError("column_mapping_request requires a concept.")
    candidates = command.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise PlanValidationError("column_mapping_request requires a non-empty candidates list.")


def _validate_format_report_step(
    command: dict[str, Any],
    sheets: dict[str, pd.DataFrame],
) -> None:
    sheets_target = command.get("sheets")
    if isinstance(sheets_target, list) and sheets_target:
        for sheet in sheets_target:
            if sheet not in sheets:
                raise PlanValidationError(f"format_report references nonexistent sheet: {sheet}")
        return
    sheet = command.get("sheet")
    if not sheet or sheet not in sheets:
        raise PlanValidationError(f"format_report references nonexistent sheet: {sheet}")


def _projected_dataframe(
    command: dict[str, Any],
    sheets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return a stub DataFrame so later steps can validate column references.

    No real data is materialized here — only column names matter for validation.
    """
    action = command.get("action")
    source = command.get("sheet")
    source_frame = sheets.get(source, pd.DataFrame())
    source_columns = list(source_frame.columns)

    def _empty_like(columns: list[str]) -> pd.DataFrame:
        frame = pd.DataFrame({column: pd.Series(dtype=source_frame[column].dtype) for column in columns if column in source_frame.columns})
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame[columns]

    if action in {"filter_rows", "highlight_rows", "move_rows_to_sheet", "sort_rows"}:
        return _empty_like(source_columns)

    if action == "count_by_group":
        group_by = command.get("group_by")
        frame = pd.DataFrame()
        if group_by and group_by in source_frame.columns:
            frame[group_by] = pd.Series(dtype=source_frame[group_by].dtype)
        elif group_by:
            frame[group_by] = pd.Series(dtype="object")
        frame["count"] = pd.Series(dtype="int64")
        return frame

    if action == "sum_by_group":
        group_by = command.get("group_by")
        sum_column = command.get("sum_column") or command.get("column")
        frame = pd.DataFrame()
        if group_by:
            dtype = source_frame[group_by].dtype if group_by in source_frame.columns else "object"
            frame[group_by] = pd.Series(dtype=dtype)
        if sum_column:
            dtype = source_frame[sum_column].dtype if sum_column in source_frame.columns else "float64"
            frame[sum_column] = pd.Series(dtype=dtype)
        return frame

    if action == "average_column":
        group_by = command.get("group_by")
        value_column = command.get("column")
        frame = pd.DataFrame()
        if group_by:
            dtype = source_frame[group_by].dtype if group_by in source_frame.columns else "object"
            frame[group_by] = pd.Series(dtype=dtype)
        if value_column:
            dtype = source_frame[value_column].dtype if value_column in source_frame.columns else "float64"
            frame[value_column] = pd.Series(dtype=dtype)
        return frame

    if action == "sum_column":
        group_by = command.get("group_by")
        value_column = command.get("column")
        frame = pd.DataFrame()
        if group_by:
            dtype = source_frame[group_by].dtype if group_by in source_frame.columns else "object"
            frame[group_by] = pd.Series(dtype=dtype)
        if value_column:
            dtype = source_frame[value_column].dtype if value_column in source_frame.columns else "float64"
            frame[value_column] = pd.Series(dtype=dtype)
        return frame

    if action == "create_summary_sheet":
        metrics = command.get("metrics") or []
        labels = [str(metric.get("label", "metric")) for metric in metrics if isinstance(metric, dict)]
        return pd.DataFrame(columns=labels or ["metric", "value"])

    if action == "create_chart":
        return _empty_like(source_columns)

    return _empty_like(source_columns)
