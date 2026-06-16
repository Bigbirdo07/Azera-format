from __future__ import annotations

from typing import Any

import pandas as pd


DELETE_ROW_ACTIONS = {"remove_duplicates"}


def is_delete_row_action(command: dict[str, Any]) -> bool:
    return command.get("action") in DELETE_ROW_ACTIONS


def affected_columns(command: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for key in ["column", "group_by", "value_column", "sum_column", "chart_group_by"]:
        if command.get(key):
            columns.append(str(command[key]))

    for condition in command.get("conditions", []):
        if condition.get("column"):
            columns.append(str(condition["column"]))

    logic = command.get("logic")
    if isinstance(logic, dict):
        if logic.get("condition_column"):
            columns.append(str(logic["condition_column"]))
        for rule in logic.get("rules", []):
            if rule.get("condition_column"):
                columns.append(str(rule["condition_column"]))

    lookup = command.get("lookup")
    if isinstance(lookup, dict):
        for key in ["lookup_value_column", "lookup_key_column", "return_column"]:
            if lookup.get(key):
                columns.append(str(lookup[key]))

    if isinstance(command.get("columns"), list):
        columns.extend(str(column) for column in command["columns"])

    if isinstance(command.get("required_columns"), list):
        columns.extend(str(column) for column in command["required_columns"])

    return sorted(set(columns))


def affected_row_count(command: dict[str, Any], sheets: dict[str, pd.DataFrame]) -> int:
    sheet = command.get("sheet")
    if sheet not in sheets:
        return 0
    dataframe = sheets[sheet]

    if command.get("action") in {"sum_column", "count_by_group", "detect_missing_values", "create_chart", "create_report", "create_data_quality_report", "format_report"}:
        return len(dataframe.index)

    if command.get("action") == "remove_duplicates":
        columns = command.get("columns") or None
        return len(dataframe.index) - len(dataframe.drop_duplicates(subset=columns).index)

    if command.get("conditions"):
        return _condition_mask(dataframe, command["conditions"])

    return len(dataframe.index)


def change_summary(command: dict[str, Any], sheets: dict[str, pd.DataFrame]) -> str:
    action = command.get("action", "unknown")
    columns = affected_columns(command)
    rows = affected_row_count(command, sheets)
    return (
        f"Action: {action}\n"
        f"Sheet: {command.get('sheet', '')}\n"
        f"Columns affected: {', '.join(columns) if columns else 'none detected'}\n"
        f"Estimated rows affected: {rows}"
    )


def _condition_mask(dataframe: pd.DataFrame, conditions: list[dict[str, Any]]) -> int:
    mask = pd.Series(True, index=dataframe.index)
    for condition in conditions:
        column = condition["column"]
        operator = condition["operator"]
        value = condition.get("value")
        series = dataframe[column]
        if operator == "equals":
            condition_mask = series.astype(str).str.casefold() == str(value).casefold()
        elif operator == "greater_than":
            condition_mask = pd.to_numeric(series, errors="coerce") > float(value)
        elif operator == "is_missing":
            condition_mask = series.isna() | (series.astype(str).str.strip() == "")
        elif operator == "is_not_missing":
            condition_mask = series.notna() & (series.astype(str).str.strip() != "")
        elif operator == "contains":
            condition_mask = series.astype(str).str.contains(str(value), case=False, na=False)
        else:
            condition_mask = pd.Series(True, index=dataframe.index)
        mask &= condition_mask.fillna(False)
    return int(mask.sum())
