# Architecture

## Purpose

The app is an offline academic roster assistant for Excel workbooks. It helps
users ask questions, review risk signals, prepare watch lists, and export new
workbooks while keeping the uploaded workbook as the source of truth.

## High-Level Flow

1. A user uploads one academic `.xlsx` workbook.
2. `core.excel_loader` loads workbook sheets into pandas DataFrames.
3. `core.workbook_profiler` profiles sheets and columns.
4. `core.schema` maps messy column names to canonical academic concepts.
5. `core.data_sources.DataSourceRegistry` builds an enriched roster view.
6. `core.workbook_capabilities` detects supported workflows and readiness.
7. The Streamlit UI renders the chat, workbook, working sheet, figures, export
   center, settings, readiness checks, and workflow templates.
8. User messages are routed through `ui.chat_panel.route_message`.
9. `nlp.planner_router` chooses query, edit, export, watch, chart, or clarify.
10. Queries run through deterministic pandas execution.
11. Confirmed actions run through validated local action handlers.
12. Exports and edits write new files; the uploaded workbook is not modified.

## Context Layer

The school-context layer is display and workflow guidance only. It adapts UI
language and suggestions for:

- PK-12
- College / University
- Generic Academic

It also adapts suggested workflows by role:

- Administrator / Dean
- Counselor / Advisor
- Teacher / Professor
- Registrar / Data Staff

The context layer does not change backend planner semantics, row matching,
privacy policy, confirmation gates, or audit behavior.

## Risk Layer

`core.risk_settings.RiskSettings` stores configurable thresholds for:

- GPA risk
- Attendance risk
- Severe attendance risk
- Unexcused absence concern
- Tardy concern
- High risk signal count
- Moderate risk signal count

`core.data_sources` passes these settings into attendance metrics and combined
risk scoring. `core.combined_risk` adds risk signals, risk level, and Risk
Reason. Watch-list exports add watch reason and Date Flagged when relevant.

## Privacy And Safety

The app keeps these constraints:

- No cloud APIs.
- No PDF or DOCX ingestion.
- No general document AI.
- No student rows sent to the LLM.
- LLM output never executes actions.
- Sensitive fields are hidden unless confirmed.
- Protected fields remain non-editable.
- The original workbook is read-only.
- Confirmed edits create new workbook files.
- Audit logs store metadata, not raw student rows.

## Main Modules

- `app.py`: Streamlit shell and panel composition.
- `ui/chat_panel.py`: chat routing, pending confirmations, working sheet panel.
- `ui/settings_panel.py`: privacy, local LLM, institution mode, role view, and
  risk settings.
- `ui/figures_panel.py`: chart intent handling and figures panel.
- `core/schema.py`: canonical field detection.
- `core/workbook_capabilities.py`: detected capabilities, missing-field notes,
  readiness checks, and upload greeting.
- `core/institution_context.py`: institution mode labels, role snippets, and
  workflow templates.
- `core/risk_settings.py`: configurable risk thresholds.
- `core/combined_risk.py`: combined risk signals, risk level, and reasons.
- `core/confirmed_actions.py`: local confirmed workbook actions and provenance.
- `core/data_sources.py`: roster, attendance, assessment, and enriched views.
- `nlp/planner_router.py`: rules-first planning route.
- `nlp/dynamic_suggestions.py`: workbook-aware suggested questions.

## Test Strategy

The suite covers deterministic planning, query execution, privacy rules,
confirmed actions, UI layout, conversation state, attendance handling, context
wording, readiness checks, risk settings, and provenance exports.
