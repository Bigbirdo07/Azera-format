import json
from pathlib import Path
import os
import sys
import socket
from typing import Any

# Auto-detect if any compiled binaries exist under dean/bin/
_base_dir = Path(__file__).resolve().parents[1]
_bin_dir = _base_dir / "bin"
_has_bundled_binaries = False

if _bin_dir.exists():
    _expected_binaries = [
        "ollama-darwin-arm64",
        "ollama-darwin-amd64",
        "ollama-windows-amd64.exe",
        "ollama-linux-amd64"
    ]
    _has_bundled_binaries = any((_bin_dir / name).exists() for name in _expected_binaries)


def _resolve_ollama_port() -> int:
    # 1. Allow environment variable override
    env_port = os.environ.get("DEAN_OLLAMA_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass

    # 2. Check if development fallback to system-wide Ollama is active
    allow_fallback = os.environ.get("DEAN_ALLOW_SYSTEM_OLLAMA_FALLBACK", "").lower() == "true"
    is_packaged = hasattr(sys, "_MEIPASS")
    
    if not _has_bundled_binaries:
        if not is_packaged or allow_fallback:
            return 11434
        return 11438  # Default fallback if packaged but binaries are missing

    # 3. Bundled mode: scan 11438-11442 to select first open or reusable port
    for port in range(11438, 11443):
        # Test if port is free to bind
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            # Port is occupied. Test if it responds to TCP connections (could be already running)
            s.close()
            try:
                con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                con.settimeout(0.1)
                con.connect(("127.0.0.1", port))
                con.close()
                return port  # Port is occupied but responsive; manager will query /api/tags
            except OSError:
                pass
    return 11438


OLLAMA_PORT = _resolve_ollama_port()
OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}"

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge"

ALLOWED_PLANNER_ACTIONS = [
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
    "freeze_header",
    "autofit_columns",
    "apply_conditional_formatting",
    "column_mapping_request",
]

ALLOWED_OPERATORS = [
    "equals",
    "not_equals",
    "greater_than",
    "greater_or_equal",
    "less_than",
    "less_or_equal",
    "contains",
    "contains_any",
    "in",
    "not_contains",
    "is_missing",
    "is_not_missing",
]

CONCEPT_SYNONYMS = {
    "balance_due": [
        "balance due",
        "amount due",
        "tuition balance",
        "unpaid balance",
        "owes money",
        "still owes",
        "money owed",
        "outstanding balance",
    ],
    "fafsa_status": [
        "fafsa",
        "fasfa",
        "financial aid",
        "aid status",
        "missing aid",
        "aid documents",
        "financial aid incomplete",
    ],
    "enrollment_status": [
        "active",
        "inactive",
        "enrolled",
        "not enrolled",
        "withdrawn",
        "registered",
        "not registered",
    ],
    "program": [
        "major",
        "department",
        "degree program",
        "college",
    ],
    "student_id": [
        "student number",
        "banner id",
        "university id",
        "id number",
    ],
}

EXPERT_SYSTEM_PROMPT = """You are an offline, expert Excel analyst working at a university registrar's office.

Your only job is to convert one user request into ONE JSON object that follows the response schema exactly.

You DO NOT execute spreadsheet changes. You DO NOT see spreadsheet rows. You ONLY see column names and sheet names.

Hard rules:
1. Return JSON only. No prose, no markdown, no commentary.
2. Use only sheet names from available_sheet_names.
3. Use only column names from available_column_names_by_sheet for the sheet you reference.
4. Use only actions from allowed_actions.
5. Use only operators from allowed_operators.
6. Never delete data. Never overwrite the original workbook. Use a new output_sheet name for any result.
7. Always set requires_confirmation to true, unless plan_type is "clarify".
8. If the request is too vague, return plan_type "clarify" with a specific clarification_question.
9. Pick exactly one plan_type:
   - "single_action": ONE operation answers the request.
   - "multi_step_plan": several operations are needed (e.g., filter then summarize then chart then format).
   - "clarify": the action, the target column, the comparison value, or the output type is missing or ambiguous.
10. Map user wording to actual workbook columns using concept_synonyms. Record any mapping in "assumptions".
11. Never invent a column. If no column matches, return plan_type "clarify" or use action column_mapping_request.
12. Prefer filter_rows for "show / find / pull / list".
13. Prefer highlight_rows for "highlight / mark / make stand out".
14. For "by <category>", prefer sum_by_group when summing, count_by_group when counting, average_column with group_by when averaging.
15. For charts, FIRST build the summary table on its own sheet, THEN create the chart from that sheet.
16. For reports, build a multi-step plan: filter -> summarize -> optional chart -> format_report.
17. For "match / pull from another sheet / lookup", use create_formula with formula_type XLOOKUP when lookup columns exist; otherwise ask to clarify.
18. For "flag", use create_formula with formula_type IF.
19. For "move rows" or "send to another sheet", use move_rows_to_sheet (which copies, not deletes).
19a. For "freeze the header / top row", use freeze_header. For "autofit / auto-size columns", use autofit_columns. For "color/shade cells that meet a condition", use apply_conditional_formatting with conditions and an optional target column.
20. For "clean it up", return plan_type "clarify" and ask whether the user means formatting, duplicates, missing values, or a summary report.
21. Set confidence between 0.0 and 1.0 based on how confidently you mapped the wording to columns and actions.
22. Provide a one-sentence plain_english_summary written for a non-technical reader.

Output JSON schema (return exactly these keys):
{
  "plan_type": "single_action" | "multi_step_plan" | "clarify",
  "confidence": 0.0,
  "plain_english_summary": "",
  "commands": [],
  "clarification_question": "",
  "assumptions": [],
  "requires_confirmation": true
}
"""


def _load_knowledge(file_name: str) -> dict[str, Any] | list[Any] | None:
    path = KNOWLEDGE_DIR / file_name
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _playbook_summaries() -> list[dict[str, Any]]:
    data = _load_knowledge("expert_playbooks.json") or {}
    playbooks = data.get("playbooks", {}) if isinstance(data, dict) else {}
    summaries: list[dict[str, Any]] = []
    for key, playbook in playbooks.items():
        summaries.append(
            {
                "id": key,
                "title": playbook.get("title", key),
                "summary": playbook.get("summary", ""),
                "triggers": playbook.get("triggers", []),
                "required_concepts": playbook.get("required_concepts", []),
                "step_count": len(playbook.get("steps", [])),
            }
        )
    return summaries


def _decision_rules_summary() -> dict[str, Any]:
    data = _load_knowledge("excel_decision_rules.json")
    if not isinstance(data, dict):
        return {}
    return {
        "verb_to_action": data.get("verb_to_action", []),
        "by_phrase_routing": data.get("by_phrase_routing", {}).get("rules", []),
        "chart_rules": data.get("chart_rules", {}).get("rules", []),
        "report_default_plan": data.get("report_rules", {}).get("default_plan", []),
        "safety_rules": data.get("safety_rules", {}).get("rules", []),
        "clean_up_options": data.get("clean_up_rule", {}).get("options", []),
    }


def _clarification_summary() -> dict[str, Any]:
    data = _load_knowledge("clarification_rules.json")
    if not isinstance(data, dict):
        return {}
    return {
        "default_questions": data.get("default_questions", {}),
        "trigger_rules": data.get("trigger_rules", []),
    }


def _few_shot_examples() -> list[dict[str, Any]]:
    return [
        {
            "user_request": "show ppl who didnt do fasfa and owe money",
            "available_sheet_names": ["Enrollment"],
            "available_column_names_by_sheet": {
                "Enrollment": ["Student ID", "Name", "Program", "FAFSA Status", "Balance Due"]
            },
            "expected_output": {
                "plan_type": "single_action",
                "confidence": 0.92,
                "plain_english_summary": "Filter students where FAFSA Status is missing or incomplete and Balance Due is greater than 0.",
                "commands": [
                    {
                        "action": "filter_rows",
                        "sheet": "Enrollment",
                        "conditions": [
                            {
                                "column": "FAFSA Status",
                                "operator": "in",
                                "value": ["Missing", "Incomplete", "Not Submitted"],
                            },
                            {
                                "column": "Balance Due",
                                "operator": "greater_than",
                                "value": 0,
                            },
                        ],
                        "output_sheet": "Missing FAFSA With Balance",
                    }
                ],
                "clarification_question": "",
                "assumptions": [
                    "FASFA was interpreted as FAFSA.",
                    "Owe money was interpreted as Balance Due greater than 0.",
                ],
                "requires_confirmation": True,
            },
        },
        {
            "user_request": "make a report showing who owes money by major",
            "available_sheet_names": ["Enrollment"],
            "available_column_names_by_sheet": {
                "Enrollment": ["Student ID", "Name", "Program", "FAFSA Status", "Balance Due"]
            },
            "expected_output": {
                "plan_type": "multi_step_plan",
                "confidence": 0.94,
                "plain_english_summary": "Create an outstanding balance report, summarize total balance by Program, add a chart, and format the report.",
                "commands": [
                    {
                        "action": "filter_rows",
                        "sheet": "Enrollment",
                        "conditions": [
                            {"column": "Balance Due", "operator": "greater_than", "value": 0}
                        ],
                        "output_sheet": "Students With Balance",
                    },
                    {
                        "action": "sum_by_group",
                        "sheet": "Students With Balance",
                        "group_by": "Program",
                        "sum_column": "Balance Due",
                        "output_sheet": "Balance by Program",
                    },
                    {
                        "action": "create_chart",
                        "sheet": "Balance by Program",
                        "chart_type": "bar",
                        "category_column": "Program",
                        "value_column": "Balance Due",
                        "title": "Outstanding Balance by Program",
                    },
                    {
                        "action": "format_report",
                        "sheets": ["Students With Balance", "Balance by Program"],
                    },
                ],
                "clarification_question": "",
                "assumptions": ["Major was mapped to Program."],
                "requires_confirmation": True,
            },
        },
        {
            "user_request": "clean this up",
            "available_sheet_names": ["Enrollment"],
            "available_column_names_by_sheet": {
                "Enrollment": ["Student ID", "Name", "Program", "FAFSA Status", "Balance Due"]
            },
            "expected_output": {
                "plan_type": "clarify",
                "confidence": 0.45,
                "plain_english_summary": "The request is too broad because 'clean this up' could mean formatting, removing duplicates, checking missing values, or creating a summary report.",
                "commands": [],
                "clarification_question": "What would you like me to clean up: formatting, duplicates, missing values, or create a summary report?",
                "assumptions": [],
                "requires_confirmation": False,
            },
        },
    ]


def build_expert_planner_prompt(
    *,
    user_request: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
    mapped_columns: dict[str, str] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "privacy_boundary": {
            "sent_to_model": [
                "typed user request",
                "sheet names",
                "column names",
                "mapped column concepts",
                "allowed actions",
                "allowed operators",
                "concept synonyms",
                "few-shot examples",
            ],
            "not_sent_to_model": [
                "spreadsheet rows",
                "student names",
                "student IDs",
                "emails",
                "individual balances",
                "grades",
                "financial aid details",
                "workbook contents",
                "file paths",
                "logs",
            ],
        },
        "user_request": user_request,
        "available_sheet_names": sheet_names,
        "available_column_names_by_sheet": sheet_columns,
        "mapped_column_concepts": mapped_columns or {},
        "allowed_actions": ALLOWED_PLANNER_ACTIONS,
        "allowed_operators": ALLOWED_OPERATORS,
        "concept_synonyms": CONCEPT_SYNONYMS,
        "expert_playbooks": _playbook_summaries(),
        "excel_decision_rules": _decision_rules_summary(),
        "clarification_rules": _clarification_summary(),
        "few_shot_examples": _few_shot_examples(),
        "output_schema": {
            "plan_type": "single_action | multi_step_plan | clarify",
            "confidence": "0.0 to 1.0",
            "plain_english_summary": "one sentence summary",
            "commands": "list of validated action objects",
            "clarification_question": "non-empty only when plan_type is clarify",
            "assumptions": "list of mapping notes such as 'Major was mapped to Program.'",
            "requires_confirmation": "true unless plan_type is clarify",
        },
    }
    return f"{EXPERT_SYSTEM_PROMPT}\n\nContext JSON:\n{json.dumps(payload, indent=2)}"


INTENT_SYSTEM_PROMPT = """You are the intent router for an offline university spreadsheet assistant.

Classify ONE user message into exactly one request_type:
- "ask_question": the user wants to KNOW something about the data and does not ask to change the file (how many, which, who, what, are there, summarize, top N, totals as a question).
- "edit_workbook": the user asks the app to create, modify, highlight, format, move, sort, add, freeze, export, or build something in the workbook.
- "clarify": the request is too vague, missing required detail, or could mean several things (e.g. "clean this up", "fix the bad ones", "do the report"), OR it refers to a previous result that does not exist.

Hard rules:
1. Return JSON only. No prose, no markdown.
2. You only see the message plus sheet and column names. You never see rows.
3. If a verb implies a file change (highlight/move/create/format/add/sort/freeze/export/make a report/make a chart), choose edit_workbook.
4. If the message is a question about the data with no change requested, choose ask_question.
5. If you cannot tell, or the wording is vague, choose clarify.
6. Set confidence 0.0-1.0 for how sure you are.

Output JSON schema (exactly these keys):
{
  "request_type": "ask_question | edit_workbook | clarify",
  "confidence": 0.0,
  "reason": ""
}
"""

QUERY_SYSTEM_PROMPT = """You are the read-only query planner for an offline university spreadsheet assistant.

Convert ONE user QUESTION into a single JSON query plan. You DO NOT compute the answer; pandas does. You only describe WHICH calculation to run.

Hard rules:
1. Return JSON only. No prose, no markdown.
2. Use only sheet names from available_sheet_names and column names from available_column_names_by_sheet.
3. Never invent a column. Map wording to real columns using concept_synonyms; if nothing matches, lower confidence.
4. Choose exactly one operation:
   count_rows, count_unique, sum_column, average_column, min_column, max_column,
   groupby_count, groupby_sum, groupby_average, missing_summary,
   duplicate_check, filtered_preview, data_quality_summary.
5. "how many / number of" -> count_rows (or groupby_count if "by/per <category>").
5a. "how many <category> are there", "how many distinct/different <category>" -> count_unique with value_column set to that category column (distinct values, not rows). Entity nouns like "students" stay count_rows.
6. "total / sum" -> sum_column (or groupby_sum by category). "average / mean" -> average_column (or groupby_average).
7. "which X has the most / top / largest" -> groupby_count or groupby_sum sorted descending with a limit.
8. "which columns are missing / blank" -> missing_summary. "duplicate" -> duplicate_check. "summarize / what looks wrong" -> data_quality_summary.
9. "show / list / who / which students ..." -> filtered_preview.
10. For missing/incomplete status filters use operator "in" with the likely status values plus "".
11. Set confidence 0.0-1.0.

Output JSON schema (exactly these keys):
{
  "request_type": "ask_question",
  "operation": "",
  "sheet": "",
  "filters": [],
  "group_by": "",
  "value_column": "",
  "sort_by": "",
  "limit": 10,
  "plain_english_question": "",
  "confidence": 0.0
}
"""

EXPLAIN_SYSTEM_PROMPT = """You are an offline university spreadsheet assistant explaining a result to a non-technical staff member.

You are given the user's question and a VERIFIED result that pandas already computed. Restate the result in one or two clear, friendly sentences.

Hard rules:
1. Use ONLY the numbers and facts in verified_result. Never invent or recompute numbers.
2. Do not add caveats about data you were not given.
3. Plain text only. No markdown, no JSON, no bullet lists.
4. Be concise.
"""


SAFE_PLANNER_SYSTEM_PROMPT = """You are an offline dean-office planning assistant.

You do not answer the user directly. You only produce ONE valid JSON plan that
the application will validate and execute. You never see student rows.

You may use: workbook schema, column names, data types, active conversation
state, allowed operations, allowed operators, and workbook rows when row access
is explicitly enabled in local-only mode.

You must not:
- invent student facts
- calculate results yourself
- assume columns that are not in the schema
- expose sensitive data
- modify records directly
- output prose or markdown
- ignore active filters from conversation state
- use columns not present in available_columns

Ranking rule: "top N", "bottom N", "highest", "lowest", "best", "worst", or
"sort by <column>" means SORT by that column plus a limit — never a threshold
filter. "Top 10 students by GPA" is {operation: filter, filters: [], sort:
{column: "GPA", direction: "desc"}, limit: 10}. Do NOT invent a cutoff such as
GPA >= 3.0, and do NOT change the limit the user asked for. "Highest"/"top"/
"most"/"best" sort descending; "lowest"/"bottom"/"least"/"worst" sort ascending.

If the request is ambiguous, return intent "clarify" with a clarification_question.
If the requested data is not in the schema, return intent "unavailable".
If you cannot map it to a supported operation, return intent "unsupported".

Supported intents: query, clarify, unavailable, export, note_edit, field_update, summarize, unsupported
Supported operations: filter, aggregate, sort, count, count_unique, average, summarize, export, add_note, update_field
Supported operators: equals, not_equals, contains, starts_with, ends_with, greater_than, greater_than_or_equal, less_than, less_than_or_equal, between, in, not_in, is_blank, is_not_blank

Grouped metric rule: if the user asks for "average/mean <metric> by <group>"
or "which <group> has the highest/lowest average <metric>", use operation
"average", set "value_column" to the metric column, and set "group_by" to the
group column. Do not use "aggregate" alone, do not count rows, and do not add a
filter with a missing value. This applies to GPA, Attendance Rate, Days Absent,
SAT/PSAT Math, SAT/PSAT English/EBRW, and SAT/PSAT Total.

Each filter must include "source": "conversation_state" or "current_user_message".

Return ONLY this JSON shape (no markdown, no commentary):
{
  "intent": "query",
  "operation": "filter",
  "target_sheet": "",
  "filters": [{"column": "", "operator": "", "value": null, "source": "current_user_message"}],
  "group_by": null,
  "sort": null,
  "limit": 50,
  "clarification_question": null,
  "confidence": 0.0
}
"""


def build_safe_planner_prompt(
    *,
    user_request: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
    active_filters: list[dict[str, Any]] | None = None,
    canonical_map: dict[str, str] | None = None,
    safe_values: dict[str, list[str]] | None = None,
    row_context: dict[str, Any] | None = None,
    clarification_hint: dict[str, Any] | None = None,
    conversation_hint: dict[str, Any] | None = None,
) -> str:
    detected_capabilities = _detected_capabilities(canonical_map or {})
    payload = {
        "user_request": user_request,
        "available_sheet_names": sheet_names,
        "available_columns": sheet_columns,
        "canonical_columns": canonical_map or {},
        "detected_capabilities": detected_capabilities,
        "safe_categorical_values": safe_values or {},
        "workbook_row_context": row_context or {},
        "active_filters": active_filters or [],
        "conversation_turn_hint": (
            {
                "note": "The app will always merge your filters with the active ones below "
                "according to this turn type — 'followup' means the new question refines/adds "
                "to what's already active, 'fresh' means it replaces it, 'reset'/'clear' means "
                "the active filters are being dropped. Plan your own filters to already agree "
                "with this instead of treating every turn as independent.",
                "turn_type": conversation_hint.get("turn_type"),
                "additive": conversation_hint.get("additive"),
                "active_filters_description": conversation_hint.get("active_filters_description"),
            }
            if conversation_hint
            else None
        ),
        "rules_engine_clarification_hint": (
            {
                "note": "The deterministic parser found this request ambiguous and would have "
                "asked the question below. Use conversation context (active_filters, prior "
                "turns) to resolve it yourself if you can; otherwise return intent=\"clarify\" "
                "with your own clarification_question.",
                "question": clarification_hint.get("question"),
                "options": clarification_hint.get("options", []),
            }
            if clarification_hint
            else None
        ),
        "allowed_intents": ["query", "clarify", "unavailable", "export", "note_edit", "field_update", "summarize", "unsupported"],
        "allowed_operations": ["filter", "aggregate", "sort", "count", "count_unique", "average", "summarize", "export", "add_note", "update_field"],
        "allowed_operators": [
            "equals", "not_equals", "contains", "starts_with", "ends_with",
            "greater_than", "greater_than_or_equal", "less_than", "less_than_or_equal",
            "between", "in", "not_in", "is_blank", "is_not_blank",
        ],
        "few_shot_examples": [
            {"user": "show me Accounting students", "plan": {"intent": "query", "operation": "filter", "filters": [{"column": "Department", "operator": "equals", "value": "Accounting", "source": "current_user_message"}], "confidence": 0.95}},
            {"user": "who needs advisor attention?", "plan": {"intent": "query", "operation": "filter", "filters": [{"column": "GPA", "operator": "less_than", "value": 2.5, "source": "current_user_message"}], "confidence": 0.6}},
            {"user": "give me a dean summary by department", "plan": {"intent": "aggregate", "operation": "aggregate", "group_by": "Department", "confidence": 0.7}},
            {"user": "average Attendance Rate by advisor", "plan": {"intent": "query", "operation": "average", "filters": [], "group_by": "Advisor", "value_column": "Attendance Rate", "confidence": 0.9}},
            {"user": "which major has the highest average SAT Total", "plan": {"intent": "query", "operation": "average", "filters": [], "group_by": "Major", "value_column": "SAT Total", "sort": {"column": "SAT Total", "direction": "desc"}, "limit": 1, "confidence": 0.9}},
            {"user": "what is their housing status", "plan": {"intent": "unavailable", "operation": None, "clarification_question": "The workbook does not include housing status.", "confidence": 0.9}},
        ],
    }
    return f"{SAFE_PLANNER_SYSTEM_PROMPT}\n\nContext JSON:\n{json.dumps(payload, indent=2, default=str)}"


def _detected_capabilities(canonical_map: dict[str, str]) -> list[str]:
    canonicals = set(canonical_map.keys())
    capabilities: list[str] = []
    if "advisor" in canonicals or "teacher" in canonicals:
        capabilities.append("advisor caseload grouping")
    if "gpa" in canonicals:
        capabilities.append("GPA summaries and academic-risk filters")
    if "academic_status" in canonicals:
        capabilities.append("standing/status filters")
    if {"attendance_rate", "attendance_category", "days_absent", "attendance_risk"}.intersection(canonicals):
        capabilities.append("attendance-risk filters and attendance averages")
    if {"sat_total", "sat_math", "sat_ebrw", "psat_total", "psat_math", "psat_reading_writing"}.intersection(canonicals):
        capabilities.append("SAT/PSAT assessment summaries")
    if "major" in canonicals or "discipline" in canonicals:
        capabilities.append("program/discipline breakdowns")
    if "location" in canonicals:
        capabilities.append("campus/location breakdowns")
    return capabilities


def build_repair_prompt(bad_output: str) -> str:
    return (
        SAFE_PLANNER_SYSTEM_PROMPT
        + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object, "
        "no markdown fences, no prose.\n\nPrevious reply:\n"
        + bad_output[:2000]
    )


def build_validation_repair_prompt(
    *,
    user_request: str,
    previous_plan: dict[str, Any],
    validation_errors: list[str],
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
    active_filters: list[dict[str, Any]] | None = None,
    canonical_map: dict[str, str] | None = None,
    safe_values: dict[str, list[str]] | None = None,
) -> str:
    """Prompt the local planner to repair a syntactically valid but unsafe plan."""
    payload = {
        "user_request": user_request,
        "previous_plan": previous_plan,
        "validation_errors": validation_errors,
        "available_sheet_names": sheet_names,
        "available_columns": sheet_columns,
        "canonical_columns": canonical_map or {},
        "safe_categorical_values": safe_values or {},
        "active_filters": active_filters or [],
        "repair_rules": [
            "Return ONLY one JSON object.",
            "Use only the listed sheets and columns.",
            "For equals/in filters on categorical fields, use exact values from safe_categorical_values.",
            "If a requested field is absent, return intent='unavailable' or intent='clarify'.",
            "Do not invent rows, counts, columns, or category values.",
        ],
    }
    return (
        f"{SAFE_PLANNER_SYSTEM_PROMPT}\n\n"
        "Your previous JSON was parsed, but failed workbook safety validation. "
        "Repair the plan using the validation errors and available workbook context.\n\n"
        f"Context JSON:\n{json.dumps(payload, indent=2, default=str)}"
    )


def build_intent_prompt(
    *,
    user_request: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
) -> str:
    payload = {
        "user_request": user_request,
        "available_sheet_names": sheet_names,
        "available_column_names_by_sheet": sheet_columns,
    }
    return f"{INTENT_SYSTEM_PROMPT}\n\nContext JSON:\n{json.dumps(payload, indent=2)}"


def build_query_planner_prompt(
    *,
    user_request: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
    mapped_columns: dict[str, str] | None = None,
) -> str:
    payload = {
        "user_request": user_request,
        "available_sheet_names": sheet_names,
        "available_column_names_by_sheet": sheet_columns,
        "mapped_column_concepts": mapped_columns or {},
        "concept_synonyms": CONCEPT_SYNONYMS,
        "allowed_operators": ALLOWED_OPERATORS,
    }
    return f"{QUERY_SYSTEM_PROMPT}\n\nContext JSON:\n{json.dumps(payload, indent=2)}"


def build_explain_prompt(
    *,
    user_question: str,
    verified_result: dict[str, Any],
) -> str:
    payload = {
        "user_question": user_question,
        "verified_result": verified_result,
    }
    return f"{EXPLAIN_SYSTEM_PROMPT}\n\nContext JSON:\n{json.dumps(payload, indent=2)}"


CONVERSATIONAL_SYSTEM_PROMPT = """You are an offline dean-office assistant talking with a non-technical staff member.

You did NOT compute the result. Pandas already computed it and the validator already checked it. Your job is to phrase a short, helpful reply from verified_result plus result_rows.

You receive:
- the user's question
- the interpretation the app already decided on (understood_plan)
- a verified result summary (numbers, counts, columns)
- row_sample_policy: whether result_rows is a redacted/name-safe sample or a full local-only matching-row sample
- result_rows: a bounded sample of actual rows, including student names when available
- the active conversation context (filters, sheet)
- a list of sensitive fields that were hidden by default, if any
- a list of allowed next actions the app supports

You MUST:
1. Reply in 1 to 3 short sentences. Plain text only. No markdown, no bullet lists, no JSON.
2. Use ONLY the numbers, names, and facts present in verified_result and result_rows. Never invent or infer rows, names, IDs, totals, averages, or fields that are not given to you.
3. Start by reflecting the understood interpretation, then state the result, then optionally point to allowed next actions.
4. If hidden_sensitive_fields is non-empty, say briefly that those fields stayed hidden from the visible answer.
5. Do not promise actions outside allowed_next_actions. Do not claim to have changed or exported anything.
6. When result_rows is provided you MAY name the specific students it contains (for example, who is underperforming). Name only students that appear in result_rows. If result_rows holds more students than you can list in 1-3 sentences, name a few and say how many more there are.
7. If verified_result indicates the calculation could not run, say so plainly without inventing causes.
8. NEVER describe, restate, or mention the input itself. Do not talk about JSON, an API, a payload, arrays, fields, schema, metadata, or the names of any keys you were given (such as verified_result, active_context, result_rows, or allowed_next_actions). Just answer the user's question in plain language as if you were speaking to them.

Example. If the question is "Top 10 students by GPA" and result_rows lists students, a good reply is: "Here are the top students by GPA: Ada Lovelace (3.98), Alan Turing (3.95), and Grace Hopper (3.94), among the 10 shown on the right." A BAD reply describes the data ("The provided JSON contains an array of student objects...") — never do that.
"""


def build_conversational_prompt(
    *,
    user_question: str,
    understood_plan: str,
    verified_result: dict[str, Any],
    active_context: dict[str, Any] | None = None,
    hidden_sensitive_fields: list[str] | None = None,
    allowed_next_actions: list[str] | None = None,
    row_sample: list[dict[str, Any]] | None = None,
    row_sample_policy: str = "redacted_name_safe_rows",
) -> str:
    full_local = row_sample_policy == "full_local_rows"
    payload = {
        "privacy_boundary": {
            "sent_to_model": [
                "user question",
                "interpretation summary",
                "verified result summary (counts, columns, value)",
                (
                    "a bounded full local-only sample of matching source rows, including hidden columns"
                    if full_local
                    else "a bounded redacted/name-safe sample of matching rows"
                ),
                "active conversation context (filters, sheet)",
                "names of hidden sensitive fields",
                "list of allowed next actions",
            ],
            "not_sent_to_model": [
                "remote or cloud services",
                "file paths or full workbook contents",
                (
                    "rows beyond the bounded result_rows sample"
                    if full_local
                    else "student emails, phone numbers, individual balances, notes, addresses, DOB, conduct, medical or accommodation values"
                ),
            ],
        },
        "user_question": user_question,
        "understood_plan": understood_plan,
        "verified_result": verified_result,
        "row_sample_policy": row_sample_policy,
        "result_rows": row_sample or [],
        "active_context": active_context or {},
        "hidden_sensitive_fields": hidden_sensitive_fields or [],
        "allowed_next_actions": allowed_next_actions or [],
    }
    return (
        f"{CONVERSATIONAL_SYSTEM_PROMPT}\n\nContext JSON:\n"
        f"{json.dumps(payload, indent=2, default=str)}"
    )


# Backwards-compatibility -----------------------------------------------------

ALLOWED_LLM_ACTIONS = ALLOWED_PLANNER_ACTIONS

SYSTEM_PROMPT = EXPERT_SYSTEM_PROMPT


def build_prompt(
    *,
    user_request: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
) -> str:
    """Legacy entry point retained so existing callers continue to work.

    The legacy single-command path now wraps the expert planner prompt so the
    LLM always receives the same instructions and schema.
    """
    return build_expert_planner_prompt(
        user_request=user_request,
        sheet_names=sheet_names,
        sheet_columns=sheet_columns,
    )
