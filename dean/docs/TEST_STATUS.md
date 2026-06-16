# Test Status

## Checkpoint

Date: June 2, 2026

Command:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
559 tests passing
```

## Coverage Areas

- Academic workflow routing
- Attendance ingestion and risk review
- Combined risk scoring
- Confirmed workbook actions
- Conversation state and follow-ups
- Dashboard layout and UI panels
- Data source enrichment
- Dynamic suggestions
- Failure logging
- Interaction logging
- Notes search
- Numeric phrasing
- Pending actions
- Planner router
- Privacy safeguards
- Query engine
- Schema mapping
- Session workbook behavior
- Uncertainty and vague-term handling
- Validator behavior
- Workbook attendance detection
- Workbook capabilities and readiness checks

## Warnings

The latest full test run emitted openpyxl warnings for generated sheet names
longer than 31 characters. No test failures were reported.

## Safeguard Status

- No cloud API requirement introduced.
- No PDF or DOCX ingestion introduced.
- No student rows sent to the LLM.
- Confirmation gates remain in place for sensitive actions and workbook edits.
- Audit logging remains metadata-only.
