from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from nlp.synonym_mapper import (
    concept_from_text,
    load_json,
    load_json_list,
    load_synonyms_with_learned,
    match_column_by_terms,
    match_column_for_concept,
    normalize_text,
)


LOW_CONFIDENCE_THRESHOLD = 0.65
STANDARD_CONCEPT_KEYS = {
    "balance due": "balance_due",
    "fafsa status": "fafsa_status",
    "financial aid status": "fafsa_status",
    "enrollment status": "enrollment_status",
    "registration status": "enrollment_status",
    "program": "program",
    "major": "major",
    "advisor": "advisor",
    "semester": "semester",
}


@dataclass(frozen=True)
class ParseResult:
    command: dict[str, Any] | None
    confidence: float
    confirmation: str | None = None
    clarification: str | None = None


def parse_request(
    user_request: str,
    sheet_name: str,
    columns: list[str],
    all_sheet_columns: dict[str, list[str]] | None = None,
) -> ParseResult:
    actions = load_json("actions.json")
    charts = load_json("charts.json")
    report_templates = load_json("report_templates.json")
    synonyms = load_synonyms_with_learned()
    synonyms = _with_learned_synonyms(synonyms)
    university_terms = load_json("university_terms.json")
    templates = load_json("response_templates.json")

    text = normalize_text(user_request)
    action, action_score = _detect_action(text, actions)
    concept, concept_score = _detect_concept(text, synonyms, university_terms)

    command: dict[str, Any] = {
        "action": action,
        "sheet": sheet_name,
    }

    column: str | None = None
    column_score = 0.0

    if action == "format_report":
        confidence = _confidence(action_score, 0.85, 0.85)
        command["confidence"] = confidence
        return _finalize(command, confidence, templates["format_report"].format(sheet=sheet_name), templates)

    if action == "create_data_quality_report":
        confidence = _confidence(action_score, 0.9, 0.9)
        command["output_sheet"] = "Data Quality Report"
        command["confidence"] = confidence
        return _finalize(
            command,
            confidence,
            templates["create_data_quality_report"].format(
                sheet=sheet_name,
                output_sheet=command["output_sheet"],
            ),
            templates,
        )

    if action == "remove_duplicates":
        confidence = _confidence(action_score, 0.7, 0.7)
        command["confidence"] = confidence
        return _finalize(command, confidence, templates["remove_duplicates"].format(sheet=sheet_name), templates)

    explicit_column, explicit_column_score = _detect_explicit_column(text, columns)

    if concept:
        column, column_score = match_column_for_concept(_base_concept(concept), columns, synonyms)

    if explicit_column and (
        not column
        or explicit_column_score > column_score
        or _base_concept(concept or "") == "student"
    ):
        column = explicit_column
        column_score = explicit_column_score

    if action == "create_report":
        command, column_score, missing_concepts = _report_command(
            command,
            text,
            columns,
            synonyms,
            report_templates,
        )
        if missing_concepts:
            confidence = 0.45
            command["confidence"] = confidence
            missing = ", ".join(missing_concepts)
            return ParseResult(
                command=command,
                confidence=confidence,
                clarification=f"I need you to choose which column to use for: {missing}.",
            )

    elif action == "create_chart":
        command, column_score = _chart_command(
            command,
            text,
            columns,
            synonyms,
            charts,
        )

    elif action == "create_formula":
        command, column_score = _formula_command(
            command,
            text,
            concept,
            columns,
            synonyms,
            university_terms,
            all_sheet_columns or {sheet_name: columns},
        )

    elif action == "count_by_group":
        group_column = _detect_group_column(text, columns, synonyms)
        if group_column:
            column = group_column[0]
            column_score = max(column_score, group_column[1])
        if column:
            command["group_by"] = column
        if concept and concept.startswith("enrollment_status:"):
            status_column, status_score = match_column_for_concept("enrollment_status", columns, synonyms)
            if status_column:
                command["conditions"] = [_condition_for_text(text, concept, status_column, university_terms)]
                column_score = max(column_score, status_score)

    elif action == "sum_column":
        sum_column = column
        if concept in {"program", "major", "enrollment_status", "advisor"}:
            balance_column, balance_score = match_column_for_concept("balance_due", columns, synonyms)
            if balance_column:
                sum_column = balance_column
                column_score = max(column_score, balance_score)
        if sum_column:
            command["column"] = sum_column
        group_column = _detect_group_column(text, columns, synonyms)
        if group_column and group_column[0] != sum_column:
            command["group_by"] = group_column[0]

    elif action == "detect_missing_values":
        if column:
            command["column"] = column

    elif action in {"highlight_rows", "filter_rows", "count_rows"}:
        if column:
            command["conditions"] = [_condition_for_text(text, concept, column, university_terms)]
        if action == "highlight_rows":
            command["format"] = actions["highlight_rows"]["default_format"]

    confidence = _confidence(action_score, concept_score or 0.55, column_score)
    command["confidence"] = confidence
    confirmation = _confirmation(command, templates)
    return _finalize(command, confidence, confirmation, templates)


def _detect_action(text: str, actions: dict[str, Any]) -> tuple[str, float]:
    if any(phrase in text for phrase in actions["create_data_quality_report"]["phrases"]):
        return "create_data_quality_report", 0.94
    if "duplicate id" in text or "duplicate student id" in text:
        return "create_data_quality_report", 0.92
    if ("missing student" in text or "student data" in text) and any(term in text for term in ["find", "check", "audit"]):
        return "create_data_quality_report", 0.9
    if "problem" in text and any(term in text for term in ["file", "spreadsheet", "workbook", "enrollment"]):
        return "create_data_quality_report", 0.9
    if any(phrase in text for phrase in actions["create_report"]["phrases"]):
        return "create_report", 0.92
    if any(phrase in text for phrase in actions["create_chart"]["phrases"]):
        return "create_chart", 0.92
    if any(phrase in text for phrase in actions["create_formula"]["phrases"]):
        return "create_formula", 0.92
    if any(phrase in text for phrase in actions["format_report"]["phrases"]):
        return "format_report", 0.95
    if any(phrase in text for phrase in actions["remove_duplicates"]["phrases"]):
        return "remove_duplicates", 0.9
    if "sum" in text or "total" in text or "add up" in text:
        return "sum_column", 0.9
    if "missing counts by" in text or "missing count by" in text:
        return "create_chart", 0.86
    if ("count" in text or "how many" in text or "number of" in text) and " by " in f" {text} ":
        return "count_by_group", 0.9
    if "count" in text or "how many" in text or "number of" in text:
        return "count_rows", 0.85
    if any(term in text for term in ["missing", "blank", "blanks", "empty", "null"]):
        if text.startswith("show") or text.startswith("find") or text.startswith("list"):
            return "filter_rows", 0.82
        return "detect_missing_values", 0.86
    if any(phrase in text for phrase in actions["highlight_rows"]["phrases"]):
        return "highlight_rows", 0.92
    if any(phrase in text for phrase in actions["filter_rows"]["phrases"]):
        return "filter_rows", 0.82
    return "filter_rows", 0.45


def _report_command(
    command: dict[str, Any],
    text: str,
    columns: list[str],
    synonyms: dict[str, list[str]],
    report_templates: dict[str, Any],
) -> tuple[dict[str, Any], float, list[str]]:
    report_type = _detect_report_type(text)
    template = report_templates[report_type]
    concept_columns: dict[str, str] = {}
    scores: list[float] = []
    missing_concepts: list[str] = []

    required_concepts = set(template.get("required_concepts", []))
    for item in template.get("conditions", []):
        required_concepts.add(item["concept"])
    if template.get("chart_group_concept"):
        required_concepts.add(template["chart_group_concept"])

    for concept in required_concepts:
        column, score = match_column_for_concept(concept, columns, synonyms)
        if column:
            concept_columns[concept] = column
            scores.append(score)
        elif concept in template.get("required_concepts", []):
            missing_concepts.append(concept.replace("_", " "))

    conditions = []
    for condition_template in template.get("conditions", []):
        concept = condition_template["concept"]
        if concept not in concept_columns:
            missing_concepts.append(concept.replace("_", " "))
            continue
        condition = {
            "column": concept_columns[concept],
            "operator": condition_template["operator"],
        }
        if "value" in condition_template:
            condition["value"] = condition_template["value"]
        conditions.append(condition)

    command.update(
        {
            "report_type": report_type,
            "title": template["title"],
            "output_sheet": template["output_sheet"],
            "conditions": conditions,
            "include_summary": template.get("include_summary", True),
            "include_chart": template.get("include_chart", True),
            "required_columns": [concept_columns[concept] for concept in template.get("required_concepts", []) if concept in concept_columns],
        }
    )

    chart_concept = template.get("chart_group_concept")
    if chart_concept and chart_concept in concept_columns:
        command["chart_group_by"] = concept_columns[chart_concept]
        command["chart_title"] = f"{template['title']} by {concept_columns[chart_concept]}"

    return command, max(scores) if scores else 0.7, sorted(set(missing_concepts))


def _detect_report_type(text: str) -> str:
    if "missing" in text and "fafsa" in text and ("owe" in text or "balance" in text):
        return "missing_fafsa_and_balance"
    if "fafsa" in text:
        return "missing_fafsa"
    if "balance" in text or "owe" in text or "owed" in text:
        return "outstanding_balance"
    if "inactive" in text or "withdrawn" in text or "withdraw" in text:
        return "inactive_withdrawn"
    if "advisor" in text or "caseload" in text:
        return "advisor_caseload"
    if "registration" in text or "registered" in text:
        return "registration_status"
    if "program" in text or "major" in text:
        return "program_enrollment"
    return "enrollment_summary"


def _chart_command(
    command: dict[str, Any],
    text: str,
    columns: list[str],
    synonyms: dict[str, list[str]],
    charts: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    chart_type = _detect_chart_type(text, charts)
    metric = _detect_chart_metric(text, charts)
    group_column, group_score = _detect_group_column(text, columns, synonyms) or (None, 0.0)

    if not group_column:
        concept, concept_score = _detect_concept(text, synonyms, {"status_terms": {}})
        if concept:
            group_column, group_score = match_column_for_concept(_base_concept(concept), columns, synonyms)
            group_score = max(group_score, concept_score)

    value_column = None
    value_score = 0.0
    if metric == "sum":
        value_column, value_score = match_column_for_concept("balance_due", columns, synonyms)
        if not value_column:
            value_column, value_score = _detect_explicit_column(text, columns)
    elif metric == "count_missing":
        value_column, value_score = match_column_for_concept("fafsa_status", columns, synonyms)
        if not value_column:
            value_column, value_score = _detect_explicit_column(text, columns)

    if group_column:
        command.update(
            {
                "chart_type": chart_type,
                "group_by": group_column,
                "metric": metric,
                "output_sheet": _chart_output_sheet(text, chart_type, group_column, value_column, metric),
            }
        )
        if value_column:
            command["value_column"] = value_column
        command["title"] = _chart_title_text(chart_type, group_column, value_column, metric)

    return command, max(group_score, value_score)


def _detect_chart_type(text: str, charts: dict[str, Any]) -> str:
    if "stacked bar" in text:
        return "stacked_bar"
    if "pie" in text:
        return "pie"
    if "line" in text or "trend" in text:
        return "line"
    if "column" in text:
        return "column"
    if "bar" in text:
        return "bar"
    return charts.get("default_chart_type", "bar")


def _detect_chart_metric(text: str, charts: dict[str, Any]) -> str:
    if "missing count" in text or "missing counts" in text or "blank count" in text or "blank counts" in text:
        return "count_missing"
    if "total" in text or "sum" in text or "balance" in text:
        return "sum"
    for metric, phrases in charts["metric_triggers"].items():
        if any(phrase in text for phrase in phrases):
            return metric
    if "missing" in text or "blank" in text:
        return "count_missing"
    return "count_rows"


def _chart_output_sheet(
    text: str,
    chart_type: str,
    group_column: str,
    value_column: str | None,
    metric: str,
) -> str:
    if metric == "sum" and value_column:
        return f"{value_column} by {group_column} Chart"
    if metric == "count_missing" and value_column:
        return f"Missing {value_column} Chart"
    return f"{group_column} Chart"


def _chart_title_text(
    chart_type: str,
    group_column: str,
    value_column: str | None,
    metric: str,
) -> str:
    if metric == "sum" and value_column:
        return f"{value_column} by {group_column}"
    if metric == "count_missing" and value_column:
        return f"Missing {value_column} by {group_column}"
    return f"Students by {group_column}"


def _formula_command(
    command: dict[str, Any],
    text: str,
    concept: str | None,
    columns: list[str],
    synonyms: dict[str, list[str]],
    university_terms: dict[str, Any],
    all_sheet_columns: dict[str, list[str]],
) -> tuple[dict[str, Any], float]:
    column_score = 0.0

    if "lookup" in text or "look up" in text or "pull from" in text or "bring in" in text:
        lookup_command, lookup_score = _lookup_formula_command(command, text, columns, synonyms, all_sheet_columns)
        if lookup_command:
            return lookup_command, lookup_score

    if "sum" in text or "total" in text:
        column, column_score = match_column_for_concept("balance_due" if concept in {"balance_due", "program", "major"} else _base_concept(concept or ""), columns, synonyms)
        if column:
            command.update({"formula_type": "SUM", "column": column})
        return command, column_score

    column = None
    if concept:
        column, column_score = match_column_for_concept(_base_concept(concept), columns, synonyms)
    if not column:
        column, column_score = _detect_explicit_column(text, columns)

    if concept == "balance_due" or "owe" in text or "balance" in text:
        column, column_score = match_column_for_concept("balance_due", columns, synonyms)
        if column:
            command.update(
                {
                    "new_column": "Owes Balance Flag",
                    "formula_type": "IF",
                    "logic": {
                        "condition_column": column,
                        "operator": "greater_than",
                        "value": 0,
                        "true_value": "Yes",
                        "false_value": "No",
                    },
                }
            )
            return command, column_score

    if column:
        command.update(
            {
                "new_column": f"{column} Flag",
                "formula_type": "IF",
                "logic": {
                    "condition_column": column,
                    "operator": "is_not_missing",
                    "true_value": "Yes",
                    "false_value": "No",
                },
            }
        )
    return command, column_score


def _lookup_formula_command(
    command: dict[str, Any],
    text: str,
    columns: list[str],
    synonyms: dict[str, list[str]],
    all_sheet_columns: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, float]:
    current_sheet = command["sheet"]
    other_sheets = {
        sheet: sheet_columns
        for sheet, sheet_columns in all_sheet_columns.items()
        if sheet != current_sheet
    }
    if not other_sheets:
        return None, 0.0

    lookup_sheet, lookup_columns = next(iter(other_sheets.items()))
    common_columns = [column for column in columns if column in lookup_columns]
    if not common_columns:
        return None, 0.2

    return_column, return_score = _detect_explicit_column(text, lookup_columns)
    if not return_column or return_column in common_columns:
        return_column = next((column for column in lookup_columns if column not in common_columns), lookup_columns[-1])
        return_score = max(return_score, 0.45)

    key_column = common_columns[0]
    command.update(
        {
            "new_column": f"{return_column} Lookup",
            "formula_type": "XLOOKUP",
            "lookup": {
                "lookup_sheet": lookup_sheet,
                "lookup_value_column": key_column,
                "lookup_key_column": key_column,
                "return_column": return_column,
            },
        }
    )
    return command, max(return_score, 0.7)


def _detect_concept(
    text: str,
    synonyms: dict[str, list[str]],
    university_terms: dict[str, Any],
) -> tuple[str | None, float]:
    for status, phrases in university_terms["status_terms"].items():
        if any(phrase in text for phrase in phrases):
            return f"enrollment_status:{status}", 0.93

    priority_concepts = [
        "balance_due",
        "fafsa_status",
        "gpa",
        "advisor",
        "enrollment_status",
        "semester",
        "major",
        "program",
        "student_id",
        "student",
    ]
    for concept in priority_concepts:
        for phrase in synonyms.get(concept, []):
            normalized_phrase = normalize_text(phrase)
            if normalized_phrase and normalized_phrase in text:
                return concept, min(0.95, 0.65 + (len(normalized_phrase) / max(len(text), 1)))

    concept, score = concept_from_text(text, synonyms)
    return concept, score


def _detect_explicit_column(text: str, columns: list[str]) -> tuple[str | None, float]:
    best_column: str | None = None
    best_score = 0.0
    text_tokens = set(text.split())

    for column in columns:
        normalized_column = normalize_text(column)
        column_tokens = set(normalized_column.split())
        if not normalized_column:
            continue

        if normalized_column in text:
            score = 0.98
        elif len(column_tokens) == 1 and next(iter(column_tokens)) in text_tokens:
            score = 0.94
        else:
            overlap = len(text_tokens & column_tokens)
            score = overlap / max(len(column_tokens), 1)
            if score == 1.0 and len(column_tokens) > 1:
                score = 0.9

        if score > best_score:
            best_column = column
            best_score = score

    if best_score >= 0.55:
        return best_column, best_score
    return match_column_by_terms([text], columns)


def _base_concept(concept: str) -> str:
    return concept.split(":", 1)[0]


def _detect_group_column(
    text: str,
    columns: list[str],
    synonyms: dict[str, list[str]],
) -> tuple[str, float] | None:
    if " by " not in f" {text} ":
        return None
    group_text = text.split(" by ", 1)[1]
    column, column_score = match_column_by_terms([group_text], columns)
    if column and column_score >= 0.85:
        return column, column_score

    concept, concept_score = concept_from_text(group_text, synonyms)
    if concept:
        column, column_score = match_column_for_concept(concept, columns, synonyms)
        if column:
            return column, max(concept_score, column_score)
    if column:
        return column, column_score
    return None


def _condition_for_text(
    text: str,
    concept: str | None,
    column: str,
    university_terms: dict[str, Any],
) -> dict[str, Any]:
    defaults = university_terms["concept_defaults"]

    numeric_condition = _numeric_condition_from_text(text, column)
    if numeric_condition:
        return numeric_condition

    if concept and concept in defaults:
        condition = {"column": column, **defaults[concept]}
    elif concept == "fafsa_status" and _has_missing_language(text, university_terms):
        condition = {"column": column, "operator": "is_missing"}
    elif _has_missing_language(text, university_terms):
        condition = {"column": column, "operator": "is_missing"}
    else:
        condition = {"column": column, "operator": "is_not_missing"}

    if concept and concept.startswith("enrollment_status:"):
        condition = {"column": column, **defaults.get(concept, {"operator": "contains", "value": concept.split(":", 1)[1]})}

    return condition


def _numeric_condition_from_text(text: str, column: str) -> dict[str, Any] | None:
    normalized_column = normalize_text(column)
    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not number_match:
        return None

    number_value = float(number_match.group(0))
    if number_value.is_integer():
        number_value = int(number_value)

    numeric_words = {
        "above": "greater_than",
        "over": "greater_than",
        "greater than": "greater_than",
        "more than": "greater_than",
        "higher than": "greater_than",
        "below": "less_than",
        "under": "less_than",
        "less than": "less_than",
        "lower than": "less_than",
        "at least": "greater_or_equal",
        "minimum": "greater_or_equal",
        "no less than": "greater_or_equal",
        "at most": "less_or_equal",
        "maximum": "less_or_equal",
        "no more than": "less_or_equal",
        "equal to": "equals",
        "equals": "equals",
    }
    for phrase, operator in numeric_words.items():
        if phrase in text:
            return {"column": column, "operator": operator, "value": number_value}

    if normalized_column in text:
        return {"column": column, "operator": "equals", "value": number_value}
    return None


def _has_missing_language(text: str, university_terms: dict[str, Any]) -> bool:
    return any(term.strip() and term in text for term in university_terms["missing_terms"])


def _confidence(action_score: float, concept_score: float, column_score: float) -> float:
    score = (action_score * 0.35) + (concept_score * 0.25) + (column_score * 0.40)
    return round(min(max(score, 0.0), 0.99), 2)


def _confirmation(command: dict[str, Any], templates: dict[str, str]) -> str:
    action = command["action"]
    sheet = command["sheet"]

    if action == "sum_column":
        group_text = f" by {command['group_by']}" if command.get("group_by") else ""
        return templates[action].format(
            sheet=sheet,
            column=command.get("column", "the selected column"),
            group_text=group_text,
        )

    if action == "count_by_group":
        condition_text = ""
        if command.get("conditions"):
            condition_text = f" where {_condition_text(command['conditions'][0])}"
        return templates[action].format(
            sheet=sheet,
            group_by=command.get("group_by", "the selected column"),
            condition_text=condition_text,
        )

    if action == "create_chart":
        return templates[action].format(
            sheet=sheet,
            group_by=command.get("group_by", "the selected column"),
            chart_type=command.get("chart_type", "bar").replace("_", " "),
            output_sheet=command.get("output_sheet", "the chart sheet"),
        )

    if action == "create_report":
        return templates[action].format(
            sheet=sheet,
            output_sheet=command.get("output_sheet", "the report"),
        )

    if action == "create_data_quality_report":
        return templates[action].format(
            sheet=sheet,
            output_sheet=command.get("output_sheet", "Data Quality Report"),
        )

    if action == "create_formula":
        referenced_column = "the selected column"
        if command.get("logic"):
            referenced_column = command["logic"].get("condition_column", referenced_column)
        elif command.get("column"):
            referenced_column = command["column"]
        elif command.get("lookup"):
            referenced_column = command["lookup"].get("lookup_value_column", referenced_column)
        return templates[action].format(
            sheet=sheet,
            column=referenced_column,
            new_column=command.get("new_column", "the new column"),
            formula_type=command.get("formula_type", "formula"),
        )

    if command.get("conditions"):
        condition = command["conditions"][0]
        condition_text = _condition_text(condition)
        column = condition["column"]
        return templates[action].format(
            sheet=sheet,
            column=column,
            condition=condition_text,
            condition_text=f" where {condition_text}",
        )

    if action == "detect_missing_values":
        column = command.get("column", "all columns")
        return templates[action].format(sheet=sheet, column=column)

    if action == "count_rows":
        return templates[action].format(sheet=sheet, condition_text="")

    return templates[action].format(sheet=sheet)


def _condition_text(condition: dict[str, Any]) -> str:
    column = condition["column"]
    operator = condition["operator"]
    if operator == "greater_than":
        return f"{column} is greater than {condition['value']}"
    if operator == "contains":
        return f"{column} contains {condition['value']}"
    if operator == "contains_any":
        return f"{column} contains any of {', '.join(map(str, condition['value']))}"
    if operator == "is_missing":
        return f"{column} is blank"
    if operator == "is_not_missing":
        return f"{column} is not blank"
    return f"{column} {operator.replace('_', ' ')} {condition.get('value', '')}".strip()


def _finalize(
    command: dict[str, Any],
    confidence: float,
    confirmation: str,
    templates: dict[str, str],
) -> ParseResult:
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return ParseResult(
            command=command,
            confidence=confidence,
            clarification=templates["clarification"],
        )
    return ParseResult(command=command, confidence=confidence, confirmation=confirmation)


def _with_learned_synonyms(synonyms: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {concept: list(phrases) for concept, phrases in synonyms.items()}

    for item in load_json_list("learned_synonyms.json"):
        phrase = str(item.get("phrase", "")).strip()
        mapped_concept = str(item.get("mapped_concept", "")).strip()
        if phrase and mapped_concept:
            for concept_key in _learned_concept_keys(mapped_concept):
                merged.setdefault(concept_key, [])
                if phrase not in merged[concept_key]:
                    merged[concept_key].append(phrase)

    for item in load_json_list("learned_column_mappings.json"):
        raw_column_name = str(item.get("raw_column_name", "")).strip()
        standard_concept = str(item.get("standard_concept", "")).strip()
        if raw_column_name and standard_concept:
            for concept_key in _learned_concept_keys(standard_concept):
                merged.setdefault(concept_key, [])
                if raw_column_name not in merged[concept_key]:
                    merged[concept_key].append(raw_column_name)

    return merged


def _learned_concept_keys(mapped_concept: str) -> list[str]:
    normalized = normalize_text(mapped_concept)
    canonical = STANDARD_CONCEPT_KEYS.get(normalized)
    if canonical and canonical != mapped_concept:
        return [mapped_concept, canonical]
    return [mapped_concept]
