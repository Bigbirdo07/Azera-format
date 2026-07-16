from __future__ import annotations


SUPPORTED_ACTIONS = {
    "create_chart",
    "create_data_quality_report",
    "create_formula",
    "create_report",
    "filter_rows",
    "highlight_rows",
    "sum_column",
    "count_rows",
    "count_by_group",
    "detect_missing_values",
    "remove_duplicates",
    "format_report",
}

# Actions the local Ollama planner is allowed to emit. The planner is only a
# planner — execution still routes through SUPPORTED_ACTIONS plus per-step
# expansion (e.g., sum_by_group is rewritten to sum_column at execution time).
SUPPORTED_PLANNER_ACTIONS = SUPPORTED_ACTIONS | {
    "sort_rows",
    "average_column",
    "sum_by_group",
    "move_rows_to_sheet",
    "create_summary_sheet",
    "freeze_header",
    "autofit_columns",
    "apply_conditional_formatting",
    "column_mapping_request",
}

# Actions the plan executor (core.action_engine.execute_plan) can actually run.
# column_mapping_request is a clarification step, not an executable change.
EXECUTABLE_PLANNER_ACTIONS = SUPPORTED_PLANNER_ACTIONS - {"column_mapping_request"}

PLANNER_OPERATORS = set()  # populated below

PLAN_TYPES = {"single_action", "multi_step_plan", "clarify"}

SUPPORTED_CHART_TYPES = {"bar", "column", "line", "pie", "stacked_bar"}
SUPPORTED_CHART_METRICS = {"count_rows", "sum", "count_missing"}
SUPPORTED_REPORT_TYPES = {
    "enrollment_summary",
    "missing_fafsa",
    "outstanding_balance",
    "inactive_withdrawn",
    "program_enrollment",
    "advisor_caseload",
    "registration_status",
    "missing_fafsa_and_balance",
}

VALID_OPERATORS = {
    "equals",
    "not_equals",
    "greater_than",
    "greater_or_equal",
    "less_than",
    "less_or_equal",
    "contains",
    "contains_any",
    "not_contains",
    "contains_text",
    "not_contains_text",
    "starts_with",
    "ends_with",
    "between",
    "not_in",
    "is_missing",
    "is_not_missing",
    "is_blank",
    "is_not_blank",
}

PLANNER_OPERATORS = VALID_OPERATORS | {"in"}

OPERATORS_WITHOUT_VALUE = {"is_missing", "is_not_missing", "is_blank", "is_not_blank"}
# Operators whose value is a list.
LIST_OPERATORS = {"in", "not_in", "contains_any", "between"}
NUMERIC_OPERATORS = {
    "greater_than",
    "greater_or_equal",
    "less_than",
    "less_or_equal",
}

SUPPORTED_FORMULAS = {
    "SUM",
    "AVERAGE",
    "COUNT",
    "COUNTA",
    "COUNTIF",
    "COUNTIFS",
    "SUMIF",
    "SUMIFS",
    "IF",
    "IFS",
    "XLOOKUP",
    "VLOOKUP",
    "FILTER",
    "SORT",
    "UNIQUE",
    "CONCAT",
    "TEXT",
    "TODAY",
}


