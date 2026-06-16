# Release Notes

## School Context Checkpoint - June 2, 2026

This checkpoint captures the current offline academic roster assistant after
the PK-12 and College Readiness Layer was added. The app remains a workbook
assistant for uploaded academic Excel files; it was not rebuilt and no general
document AI, cloud API, PDF ingestion, or DOCX ingestion was added.

### Current State

- Supports one uploaded academic workbook as the core workflow.
- Detects academic roster fields including student identity, teacher /
  professor / instructor, department, GPA, major, grade or class year,
  academic standing, attendance metrics, watch fields, and export workflows.
- Provides institution modes for PK-12, College / University, and Generic
  Academic. The mode changes visible labels, capability wording, workflow
  prompts, and suggestions.
- Provides role views for Administrator / Dean, Counselor / Advisor,
  Teacher / Professor, and Registrar / Data Staff. Role view changes suggested
  workflow language only; it is not an authentication or permission system.
- Adds workbook readiness checks for required and recommended fields, missing
  IDs, blank IDs, duplicate IDs, missing GPA, missing teacher/professor,
  attendance availability, and creatable watch columns.
- Adds configurable risk settings for GPA, attendance, severe attendance,
  unexcused absences, tardies, and combined risk signal counts.
- Adds human-readable risk and watch provenance, including Risk Reason, Watch
  Reason columns, and Date Flagged where relevant.
- Adds workflow templates for Academic Watch Review, Attendance Watch Review,
  and Combined Risk Review.
- Preserves the existing privacy, validation, confirmation, audit, and
  local-only safeguards.

### Safeguards

- No cloud APIs are used.
- Spreadsheet rows are not sent to the LLM.
- The LLM never executes workbook actions.
- Sensitive fields remain protected by the existing privacy and confirmation
  layers.
- Confirmed workbook changes write new output files; the uploaded source
  workbook is not modified in place.
- Audit logging remains metadata-only.

### Verification

- Full pytest suite: 559 tests passing.
- Latest command run: `.venv/bin/python -m pytest -q`.
- Existing warnings are from openpyxl about generated sheet names longer than
  31 characters.

### Known Limitations

- Role view is contextual UI language, not access control.
- Workflow templates are suggested workflows, not one-click automations.
- Risk thresholds are configurable in session state; persistent site-wide
  policy management is not implemented.
- The app remains scoped to academic workbook workflows, not general document
  understanding.
