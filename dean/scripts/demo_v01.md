# v0.1 Demo Walkthrough

A ~5-minute scripted demo of the Offline Dean Assistant. Everything runs locally;
with the local model off, all of this is handled by the rules engine + pandas.

## 0. Setup

```bash
cd /path/to/dean
.venv/bin/python scripts/make_synthetic_workbook.py     # writes tests/fixtures/synthetic_students.xlsx
.venv/bin/streamlit run app.py                          # opens in your browser
```

Sign in (first run creates a local admin), then upload
`tests/fixtures/synthetic_students.xlsx` in the center panel.

## 1. Conversation memory + filter composition

| Type in chat | What it shows |
|---|---|
| `Show me Accounting students` | count + redacted preview; sensitive columns hidden by default |
| `now only below 2.5 GPA` | keeps Accounting, **adds** GPA < 2.5 (additive follow-up) |
| `now only seniors` | **adds** Year = Senior (multi-step composition) |
| `what is their average GPA` | average over the **current selection** only (pandas, not the model) |
| `what about Biology` | **replaces** Department = Accounting with Biology (not Accounting AND Biology) |

The **Current context** box (left) updates each turn; **Clear filters** / **Start over** reset it.

## 2. Privacy confirmation

| Type in chat | What it shows |
|---|---|
| `show all student emails and GPAs` | the assistant **asks to confirm** (Email is sensitive) |
| click **No, keep them hidden** | nothing is revealed | 
| (or) click **Yes, show them** | the sensitive columns are shown, with a handle-with-care note |

## 3. Confirmed export (writes a NEW file)

| Type in chat | What it shows |
|---|---|
| `Show me Accounting students below 2.5 GPA` | sets the selection |
| `export this list` | **asks to confirm** before creating a file |
| `yes, export` | writes `outputs/student_export_<timestamp>.xlsx`; a path is shown |

## 4. Confirmed note edit (new modified workbook)

| Type in chat | What it shows |
|---|---|
| `add note: Advisor follow-up needed` | **asks to confirm** |
| `yes, do it` | writes `outputs/student_records_modified_<timestamp>.xlsx` with the note added to matching rows |

## 5. Protected-field refusal

| Type in chat | What it shows |
|---|---|
| `change their GPA to 4.0` | **refused** — "GPA is a protected field…"; no pending, no file written |

## 6. Verify the safety properties

```bash
# Audit log — one JSON line per confirmed action (metadata only, no rows)
cat logs/audit_log.jsonl

# Outputs folder — only NEW timestamped files
ls -la outputs/

# Original workbook is byte-for-byte unchanged
shasum tests/fixtures/synthetic_students.xlsx     # same before and after the demo
```

## 7. (Optional) enable the local LLM fallback

Admin → Settings → uncheck **Strict Privacy Mode** → check **Enable local LLM
fallback** (after `ollama serve` + `ollama pull mistral:7b`). Use **Test Ollama
connection**. Then try a vague prompt like *"who seems like they need advisor
attention?"* — with the model off it clarifies; with it on, the model proposes a
plan that the validator still checks before anything runs.

## One-shot health check (no browser)

```bash
.venv/bin/python scripts/health_check.py        # pytest + eval + e2e summary
```
