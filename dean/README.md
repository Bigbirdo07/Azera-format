# Offline Dean-Office Student-Record Assistant

A fully offline assistant that lets authorized school staff upload a student
Excel workbook and **talk to it** — asking questions and requesting changes in
plain language, with conversation memory and privacy protection.

## V1 Readiness Checklist
- [x] **Upload Workbook:** Verify that a student roster `.xlsx` file can be successfully uploaded and parsed.
- [x] **Ask Roster Question:** Ask natural-language roster questions (e.g., identify GPA-risk students) and get correct results.
- [x] **View Assumptions:** Verify that column interpretations, filters, and logical assumptions are clearly visible.
- [x] **Drill Into Result:** Support follow-up questions to drill into previous queries (e.g., "Which students are those?").
- [x] **Confirm Safe Edit:** Propose a change, verify the pending confirmation card appears, and confirm the action.
- [x] **Export New Workbook:** Download/export the modified copy and check that changes were applied correctly.
- [x] **Verify Original Workbook Unchanged:** Assert that the original uploaded `.xlsx` file remains unmodified.
- [x] **Verify Local-Only Privacy Status:** Verify that no row-level data is sent to external APIs and loopback checks pass.
- [x] **Verify Audit Log Entry:** Ensure that the action is recorded in the local database audit log.

## v0.2 status (conversational prototype checkpoint)

This is a frozen checkpoint — safe to demo or restore. Backend behavior, chip
routing, validator gates, and confirmed-action engines are unchanged from v0.1.
See `RELEASE_NOTES.md` for the full list (including the v0.2 demo script) and
`docs/ARCHITECTURE.md` for the design.

**What works:** everything from v0.1 (Excel upload + messy-workbook cleanup,
canonical schema mapping, type inference, sensitive-column detection;
rules-first planner with optional local-LLM fallback; pandas-grounded answers;
conversation memory with additive/replacement/include follow-ups,
sort/group/limit/average/count; privacy redaction + sensitive-display
confirmation; confirmed export, note-edit, and safe-field update with a local
audit log; protected-field refusal; one unified UI/test/eval planning path; a
debug panel) **plus** the conversational layer: conversational LLM mode,
assumption narration, alternative-definition chips, next-move chips, correction
logging, sanitized interaction learning logs, and the rule-mining workflow.

**Verification at this checkpoint:** pytest **254 passed**; live drive
(`scripts/e2e_live_drive_phaseL.py`) **passing**.

**What does not work yet:** structural workbook edits (charts/highlights/reports)
are a separate path; live-LLM quality depends on the local model (only validated
plans run); numeric *in-place edits* can be blocked when columns load as text;
rich phrasing may clarify instead of guessing when the model is off; the
login/role feature is **not** a hardened security boundary.

**Quick commands**

| Goal | Command |
|---|---|
| Run the app | `.venv/bin/streamlit run app.py` |
| Generate synthetic data | `.venv/bin/python scripts/make_synthetic_workbook.py` |
| Run tests | `.venv/bin/python -m pytest` |
| Run planner eval | `.venv/bin/python scripts/eval_planner.py` (add `--dispatch`) |
| Run e2e smoke / health check | `.venv/bin/python scripts/health_check.py` |
| Guided demo | see `scripts/demo_v01.md` |

**Enable local LLM fallback:** `ollama serve` + `ollama pull llama3.2:3b`, then
Admin → Settings → uncheck Strict Privacy → check "Enable local LLM fallback"
(+ optional explanations or conversational mode); use **Test Ollama connection**.

**Analyze interaction patterns:**
`.venv/bin/python scripts/analyze_interaction_logs.py` (writes
`outputs/interaction_learning_report.md`).

**Confirm no cloud APIs:** the only model endpoint is `http://localhost:11434`
(loopback-checked in `core/privacy_guard`); there are no OpenAI/Anthropic/Google
or other network calls anywhere in `core/`, `nlp/`, or `ui/`
(`grep -ri "openai\|anthropic\|api.openai\|googleapis" core nlp ui` returns nothing).

**Verify the original workbook is never modified:** confirmed actions write only
new timestamped files under `outputs/`. Hash the upload before and after a
confirmed edit — it is unchanged:
`shasum tests/fixtures/synthetic_students.xlsx`.

## What this project is

The uploaded Excel workbook is the **source of truth**. A local LLM
(Mistral via Ollama) is used only as a **natural-language planner and
explainer** — it never stores the data, never does the arithmetic, and never
edits the file directly.

| Layer | Role |
|-------|------|
| **Excel workbook** | Source of truth |
| **pandas** | Exact calculation / query engine |
| **Ollama / Mistral** | Local natural-language planner + explanation layer (optional) |
| **validator** | Safety / correctness gate |
| **session memory** | Conversation continuity (composable follow-ups) |
| **privacy layer** | Sensitive-data protection + confirmation gates |

## What this project is **not**

- It is **not** training or fine-tuning a new LLM.
- The LLM is **not** the database. It does not memorize rows or compute answers.
- Nothing is sent to the cloud. There are no external API calls.

## Detected fields and available workflows

When you upload an academic workbook, the assistant inspects the columns
(and any sibling sheets) and surfaces what it found in **plain school-office
language** — never raw schema.

- Fields are bucketed into five categories: **Roster**, **Performance**,
  **Attendance**, **Actions**, **Export**.
- The Academic Workbook panel shows a checklist of *workflows* the workbook
  supports — not the underlying column names.
- **Attendance is part of the workbook when present.** It can live as
  inline columns on the roster sheet (Attendance Rate, Days Absent, …) or
  as a sibling "Attendance" sheet inside the same .xlsx. No separate
  upload is required.
- Missing fields don't block anything — the panel explains what's still
  possible and which questions to skip.
- **Academic Watch** and **Attendance Watch** can be set on students even
  if the workbook doesn't already include those columns; the action creates
  the column in the exported workbook (the original .xlsx is never
  modified, and every edit is confirmation-gated and audit-logged).

### Example: after upload

```
This workbook supports:
✓ Teacher and department questions
✓ GPA performance review
✓ Major-based grouping
✓ Academic standing review
✓ Attendance-risk review
✓ Academic Watch updates
✓ Attendance Watch updates
✓ Export updated workbook

Try asking:
- Show me all teachers that teach Biology.
- Which students are below a 2.0 GPA?
- Show students with attendance below 90%.
- Mark these students Academic Watch and export.
```

If attendance fields aren't detected, the assistant says so explicitly and
keeps the GPA / standing / teacher / department / Academic Watch workflows
available. Technical detail (canonical mappings, planner routing,
confidence bands, raw JSON) is hidden from the normal-mode panels and
lives only behind the **Show developer/debug info** toggle.

## Architecture

```
User uploads Excel
        ↓
Workbook ingestion / profile        (core/excel_loader.py, core/workbook_profiler.py)
        ↓
Schema + sensitive-column detection (core/privacy.py)
        ↓
User asks a question
        ↓
Conversation state + planner        (core/session_memory.py, nlp/conversation.py,
                                     nlp/query_planner.py, nlp/intent_router.py,
                                     nlp/edit_planner.py, nlp/local_model.py)
        ↓
JSON plan
        ↓
Validator / privacy gate            (core/validator.py, core/privacy.py)
        ↓
pandas query / action engine        (core/query_engine.py, core/action_engine.py)
        ↓
Redacted result preview OR confirmation prompt
        ↓
Natural-language explanation        (nlp/local_model.py — optional)
```

The LLM only ever receives: the workbook **schema** (sheet/column names),
allowed values for low-cardinality categorical columns, the **conversation
state**, and **validated result summaries / small redacted previews**. It never
receives the full workbook, all rows, sensitive notes, or broad PII.

## Conversational LLM mode

The local model can act as an always-on conversational layer that runs *after*
validated execution. When enabled, every turn produces a 1–3 sentence reply
that reflects how the assistant interpreted the request, states the result, and
points to grounded next moves.

**The conversational layer never executes anything.** Pandas computes, the
validator gates, and the privacy layer redacts — before the model is ever
called. The model only sees:

- the user's typed question
- a short plain-English summary of the interpretation
- a verified result summary (operation, value, row count, column names)
- the active conversation state (filters, sheet, sort, group, limit)
- the names of any sensitive fields that stayed hidden
- a list of allowed next actions the app can actually honor

It does **not** see student rows, names, IDs, emails, phone numbers, grades,
financial values, notes, addresses, DOB, conduct details, or any redacted
content. The prompt template makes this explicit and the dispatcher's payload
is verified by tests (`tests/test_conversation_narration.py`).

**Confidence bands.** The planner classifies every plan as high / medium / low:

- **≥0.85** — execute cleanly with a brief "I understood this as…" lead.
- **0.55–0.85** — *assume-and-offer*: execute the most likely safe read-only
  interpretation, surface it as an explicit assumption note, and present
  alternative interpretations the user can click. Sensitive-field requests,
  edits, exports, and field updates **never** enter this path — they continue
  to the confirmation gate.
- **<0.55** — ask a focused clarification question.

**Next-move suggestions.** After every answer, up to three grounded follow-ups
(e.g. "Group these by Advisor", "Export this filtered list") are surfaced as
buttons. They are generated from the validated plan + available columns; no
suggestion ever references a missing column or proposes exposing a hidden
sensitive field.

**Settings:** Admin → Settings → uncheck Strict Privacy → enable any of
"Enable local LLM fallback" / "Enable local LLM explanations" / "Enable
conversational local LLM". Defaults: strict privacy on, all LLM toggles off,
deterministic narration and suggestions on for free.

### Example turn

User:

> Who is struggling?

Assistant (conversational mode on):

> I interpreted "struggling" as students with GPA below 2.5. I found 32 matching
> students with Email and Phone kept hidden. You could group these by Advisor,
> export the filtered list, or use a different definition.

Logged record (sanitized):

```json
{
  "user_message": "Who is struggling?",
  "normalized_message": "who is struggling",
  "plan_source": "rules",
  "band": "medium",
  "assumption_used": "I interpreted this as: ...",
  "validated_plan": {"operation": "filtered_preview",
                     "filters": [{"column": "GPA", "operator": "less_than", "value": 2.5}]},
  "result_shape": {"rows": 32, "columns": 7},
  "safe_for_rule_mining": true
}
```

Note that no student names, emails, IDs, or row values appear in the record.

## How vague questions are handled

Not every request is precise. Dean classifies each turn into one of three
confidence bands and behaves differently for each.

- **High confidence (≥0.85)** — execute directly with a one-sentence
  interpretation ("I understood this as: list the matching rows where
  Department = Accounting.").
- **Medium confidence (0.55–0.85)** — *assume-and-offer*. Execute the most
  likely safe read-only interpretation, surface the assumption explicitly, and
  show 2–3 alternative interpretations the user can click.
- **Low confidence (<0.55)** — ask a focused clarification question rather
  than guessing.

For known vague phrases like *"struggling students"*, *"at-risk students"*,
*"overloaded advisors"*, *"students with no advisor"*, or *"top students"*,
a dedicated resolver (`nlp/vague_terms.py`) maps the phrase to a concrete,
validated plan when the supporting columns exist (Academic Status, GPA,
Advisor). When the workbook does **not** have a supporting column, the
resolver asks for a definition instead of running a broad query.

**Broad whole-sheet results are forbidden on vague-risk terms.** If you ask
"show me struggling students" and the rule planner alone would have returned
every row, the vague-term resolver intervenes — either with a concrete filter
or with a clarification. Specific queries that legitimately list everyone
(e.g. "list all students") are not affected.

**Sensitive, export, edit, and field-update requests still require explicit
confirmation** regardless of band. The medium-band assume path is read-only.

**Example.** With the synthetic workbook:

> User: show me struggling students
>
> A: I interpreted 'struggling' as students with Academic Status in
> At Risk, Probation, Warning. I found 307 students. If that's not the
> definition you wanted, try "Now use GPA below 2.5 instead",
> "Now use GPA below 2.0 instead", or "Use Probation only".

The end-to-end behavior is verified by `scripts/e2e_live_drive_phaseL.py`
(repeatable in CI):

```bash
.venv/bin/python scripts/e2e_live_drive_phaseL.py
```

## Interaction learning log

A separate, **append-only** local log captures how requests are phrased and how
the assistant resolved them so we can later promote repeated patterns into
deterministic rules. It is **not** the audit log:

| Log | Purpose | File |
|-----|---------|------|
| Audit log | Accountability for confirmed actions | SQLite (`database/`) |
| Interaction learning log | Rule-mining signal | `logs/interaction_learning.jsonl` |

**Sanitization is enforced at write time** (see `core/interaction_logger`):

- PII patterns in the user message — emails, phone numbers, SSN-like, long
  ID-like numbers, currency amounts, DOB wording, named-person patterns — are
  replaced with `[REDACTED:<kind>]` tokens. Any redaction flips
  `safe_for_rule_mining = false`.
- Filter values targeting sensitive columns (contact, financial, health,
  disciplinary, notes, identity_high) are replaced with `[REDACTED]`. Values on
  non-sensitive categorical columns (Department, Year, Academic Status) are
  preserved — that's the signal we want for rule mining.
- The result shape carries **counts only** (rows, columns). No row preview,
  ever.

**Strict privacy mode keeps logging on** (Option A in the privacy spec):
maximum privacy disables every LLM call but the sanitized log remains
available because no row data is ever in the record.

**Rule mining:**

```bash
.venv/bin/python scripts/analyze_interaction_logs.py
```

writes `outputs/interaction_learning_report.md` with the most common phrasings,
medium-confidence prompts, repeated assumptions, validation failures, and
**candidate deterministic rules** — phrasings that appear ≥2 times with a
single resolved operation, ranked by frequency.

## Offline-only design

- Local model runs through **Ollama on `localhost`** (enforced by a loopback check).
- No OpenAI / Anthropic / Google / cloud calls. No cloud vector DBs.
- Correction examples and audit logs are stored locally in SQLite.
- Original uploads are never overwritten — edits export a **new copy** to `outputs/`.

## Requirements

- Python 3.11+ with the project virtualenv (`.venv`)
- Dependencies: `pip install -r requirements.txt`
- (Optional) [Ollama](https://ollama.com) for the natural-language planner/explainer

## Running Ollama (optional)

The app is fully usable with its rule-based parser and **no model**. To enable
the conversational LLM layer:

```bash
ollama serve            # start the local server
ollama pull llama3.2:3b  # one-time model download (~2 GB; mistral:7b also supported)
```

Then in the app: sign in as an Admin, open **Settings**, turn **off** Strict
Privacy Mode, and turn **on** "Use local LLM". Note: on a typical Mac, a model
call takes ~1–2 minutes, so the model is off by default and only used when the
rules are unsure.

## Running the app

```bash
cd /path/to/dean
.venv/bin/streamlit run app.py
```

(Or use the `dean` shell alias / double-click `start.command`.) Then sign in,
upload a `.xlsx`, and start chatting in the left panel.

## Generating the synthetic test workbook

```bash
.venv/bin/python scripts/make_synthetic_workbook.py
# writes tests/fixtures/synthetic_students.xlsx (deterministic, fake data)
```

## Running the tests

```bash
.venv/bin/python -m pytest          # full suite
.venv/bin/python -m pytest tests/test_followup_context.py   # one file
```

The suite uses the deterministic synthetic workbook and compares the
assistant's answers against pandas ground truth. Runnable smoke scripts also
remain under `scripts/e2e_*.py`.

## Dashboard layout

The main screen is a five-zone workspace dashboard:

```
┌──────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Chat Assistant       │ Original Workbook            │ Live Output                  │
│ (left, full height)  │ (top middle)                 │ (top right)                  │
│                      │ Upload .xlsx, sheet picker,  │ Latest result table /        │
│ Conversation history │ row/col counts, badges       │ confirmation card /          │
│ Current context      │                              │ edit-plan card               │
│ Chat input           ├──────────────────────────────┼──────────────────────────────┤
│                      │ Figures                      │ Export Center                │
│                      │ (bottom middle)              │ (bottom right)               │
│                      │ Latest chart (bar/pie/       │ Download edited workbook,    │
│                      │ histogram), context-aware    │ result CSV, figure CSV,      │
│                      │                              │ session export history       │
└──────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

**Workflow:**
1. **Upload** the workbook in the Original Workbook panel. The file stays unmodified — exports and edits are written as new files.
2. **Ask** questions in the Chat Assistant. The chat shows the conversation as text; the actual result table appears in **Live Output**.
3. **Refine** with follow-ups — filters/sort/group accumulate in the active context (shown as chips in the chat panel).
4. **Visualize** by asking "create a bar chart by department" or "show GPA distribution" — the chart renders in the **Figures** panel without modifying the workbook.
5. **Export** when ready ("export this list") — the confirmation card appears in Live Output, and once confirmed the file lands in the **Export Center**.

Charts are generated locally via altair (bundled with Streamlit). No external APIs.

Sidebar contents: user bar, Clear filters / Start over, advanced settings, and a Developer Tools toggle (debug routing JSON stays hidden by default).

The end-to-end script `scripts/e2e_dashboard_layout.py` walks this full workflow headlessly.

## Conversational workflow

The assistant behaves like a persistent chat for the uploaded workbook.

- The uploaded workbook stays active across every turn.
- Each follow-up question is planned with the **current filters/sort/group/limit
  context**, so you can build up a narrow selection through a sequence of short
  prompts instead of one giant query.
- The full chat history (user prompts, assistant replies, result tables,
  confirmation cards, downloads) stays visible inside the chat container.
- The assistant **never** triggers an export, edit, or note action until you
  explicitly ask for one — analytical questions just filter and summarize.
- When an action *is* requested, a confirmation card appears in chat. Click
  Confirm/Cancel or just type `yes` / `no`. Confirming appends a success
  message (with a download card if a file was written).
- **Clear filters** (sidebar) drops the active context but keeps the
  conversation. **Start over** (sidebar) wipes the conversation and resets the
  pending action while keeping the uploaded workbook loaded.
- Uploading a different workbook automatically starts a fresh chat thread.

Example session:

```
User:      Show me Accounting students
Assistant: I found 105 Accounting students. [result table]

User:      Now only below 2.5 GPA
Assistant: Keeping Department = Accounting, I found 50 students below 2.5 GPA.

User:      Now only seniors
Assistant: Keeping Department = Accounting and GPA < 2.5, 12 seniors match.

User:      Export this list
Assistant: Confirmation needed. This export includes sensitive student-level
           information. Please confirm before I create the export.
           [Confirm]  [Cancel]

User:      yes
Assistant: Export created. Original workbook was not modified.
           [Download edited workbook (.xlsx)]
```

The end-to-end script `scripts/e2e_conversation_loop.py` exercises this exact
flow headlessly and asserts that every prior turn stays visible.

## How conversation memory works

Each turn updates an active query context (`core/session_memory.py`). Follow-ups
**compose** on the previous selection instead of starting over:

- **Additive**: a new filter on a different column is ANDed on.
- **Replacement**: a new value for the *same* column replaces the old one.
- **Include**: "include X too" merges values into an `in` filter.
- **Sort / limit / group**: refine the current view without losing filters.
- **Clear / reset**: "clear that" drops filters; "start over" resets everything.

Example 1:
```
User: Show me Accounting students.
User: Now only below 2.5 GPA.
```
The assistant preserves `Department = Accounting` and adds `GPA < 2.5`.

Example 2:
```
User: What about Biology?
```
The assistant **replaces** `Department = Accounting` with `Department = Biology`
(it does not produce Accounting AND Biology).

## How privacy / sensitive fields work

Columns are classified by sensitivity (`core/privacy.py`): contact, financial,
disciplinary, health, notes, identity. Sensitive columns
(email, phone, DOB, notes, financial aid, conduct, …) are **hidden by default**
in student-level previews. Aggregate questions ("how many per department") never
expose individuals and need no confirmation.

Example 3:
```
User: Show me all emails and GPAs.
```
The privacy layer requires confirmation because Email is sensitive — it answers
with a confirmation prompt instead of revealing the column.

## Academic roster workflows

The assistant is optimized for structured school-roster questions involving
**teachers / professors**, **departments**, **students**, **majors**, **GPA**,
**academic standing**, and **academic watch / follow-up** columns. The same
canonical concept covers `teacher` / `professor` / `instructor` / `faculty` so
the user can phrase the question however the column on their workbook is
labeled.

The canonical four-step workflow:

```
1.  Show me all teachers that teach Biology.
        → count_unique on Teacher with Department = Biology

2.  Based on all teachers that have Biology, how many of their students
    have above a 2.00 GPA?
        → keeps the Biology filter, adds GPA > 2.00, counts students

3.  Based on this, which students under which professor are not performing
    well based on GPA?
        → keeps Biology, adds GPA < 2.00 (the default "not performing well"
          threshold), groups by Teacher

4.  Mark these students under Academic Watch.
        → confirmation gate, then sets Academic Watch = "Yes" on the
          underlying student rows in a NEW workbook. Original unchanged.

5.  Export me a new Excel sheet.
        → returns the workbook produced in step 4.
```

**Performance phrases the planner recognizes** (default → `GPA < 2.00`):
*"not performing well"*, *"low performing"*, *"performing poorly"*,
*"academically struggling"*, *"low gpa"*, *"low grades"*, *"needs academic
watch"*. Phrasings the existing vague-term resolver already handled
(*"struggling"*, *"at risk"*, *"needs attention"*) still go through that path
with assumption-note + alternative chips.

**Follow-up references the planner preserves scope across:** *"their students"*,
*"these teachers"*, *"under those professors"*, *"based on this"*, *"from
that group"*, *"in that department"*, *"under each professor"*.

**Academic Watch action.** *"Mark these students under Academic Watch"* /
*"flag these students"* / *"put them on watch"* / *"set Academic Watch to
Yes"* / *"mark as follow up needed"* all route to a confirmed-action that:

- uses the current filter set (resolved from an aggregate result when needed),
- creates the `Academic Watch` column if it doesn't exist,
- writes a NEW workbook (the upload is never modified),
- audits the action (filter shape, row count, output path — never raw rows).

**Test workbook + E2E driver:** the spec fixture lives at
`tests/fixtures/academic_roster.xlsx` (regenerate with
`.venv/bin/python scripts/make_academic_workbook.py`) and the full six-turn
workflow is verified by `.venv/bin/python scripts/e2e_academic_watch_workflow.py`.

### Multi-action commands

You can chain a single edit with an export in one message:

| You say | The assistant does |
|---|---|
| *"Mark these students Academic Watch and export me a new Excel sheet"* | one confirmation, one new workbook, one audit entry of type `action_chain` |
| *"Add note: Advisor follow-up needed and export"* | adds the note + writes & surfaces the file |
| *"Set Follow Up Needed to Yes and export"* | same shape — the edited workbook IS the export |
| *"Flag them for follow-up and download this as Excel"* | follow-up flag + download path |

The chain still requires explicit confirmation. *"No, cancel"* clears the
pending action without writing anything. The exported workbook is the same
file the edit step wrote — there is no second copy. The audit log records
the chain as one `action_chain` entry that lists `actions: [edit_type,
export]`, the row count, the changed column, and `original_modified: false`.

Verified end-to-end by `.venv/bin/python scripts/e2e_action_chain_workflow.py`.

## Free-text notes search

The assistant can search free-text columns like **Notes**, **Advisor Notes**,
**Comments**, **Follow Up Notes**, **Internal Notes**, **Counselor Notes**,
**Case Notes**, **Description**, or **Reason** — the schema detector tags any
column whose name matches those patterns (or whose values look like prose) as
`semantic_role: free_text`.

Two new operators are available:

- `contains_text` — case-insensitive substring search, NaN-safe, never
  interpreted as a regex.
- `not_contains_text` — the negation.

Natural phrasings the rule planner recognizes:

| You ask | Planner produces |
|---|---|
| Show students whose notes mention attendance | `Notes contains_text "attendance"` |
| Students with notes about "mom called" | `Notes contains_text "mom called"` (quoted phrase preserved) |
| Find students whose advisor notes mention session | `Advisor Notes contains_text "session"` |
| Show students whose notes do not mention follow-up | `Notes not_contains_text "follow-up"` |
| Which students have no notes? | `Notes is_blank` |

**Privacy contract.** Free-text note columns are sensitive (`sensitivity_type:
notes`) and **hidden by default** in result previews. After a notes search you
see the matching students' regular roster fields plus a small **Matched Notes ✓**
indicator — *never the note content itself*.

To see the full text, ask explicitly:

```
You: show me the full notes for these students
Assistant: This includes sensitive student-level information (Notes).
           Please confirm that you want to show these fields.
You: yes
```

Only after confirmation does the result include short **snippets** of each note
around the matched substring (e.g. *"…mom called about attendance issues…"*) —
not the entire note.

**Logging.** The interaction learning log records the *search term* the user
typed (so we can mine recurring patterns into rules — see
`scripts/promote_learned_patterns.py`) but never the actual note row values.
The defensive PII scrubber still redacts obvious emails or long ID-like
numbers from the search term itself.

## How confirmation works

When an action would expose sensitive fields, export a file, or edit the
workbook, a `pending_action` is stored and the assistant asks you to confirm:

- **"yes" / "confirm" / "export it"** → executes the pending action, then clears it.
- **"no" / "cancel"** → discards the pending action; nothing happens.
- A brand-new request abandons the pending action rather than auto-approving it.

Exports never run before confirmation, and the original workbook is never
overwritten.

## Confirmed actions (export & edits)

The assistant can perform local-only actions, but **only after you confirm**,
and **never on the original file**. Every confirmed action writes a brand-new,
timestamped workbook under `outputs/` and appends a privacy-safe line to
`logs/audit_log.jsonl`.

**Why the original is never modified:** edits run on copies of the in-memory
sheet data and are written out as a new `.xlsx`. The uploaded file's bytes and
the working copy are left untouched, so you can always re-ask questions against
the original.

**Where outputs go:**
- Exports → `outputs/student_export_YYYYMMDD_HHMMSS.xlsx`
- Edits → `outputs/student_records_modified_YYYYMMDD_HHMMSS.xlsx`

Filenames are sanitized and never contain student values.

**Audit log** (`logs/audit_log.jsonl`, one JSON object per line): timestamp,
action type, target sheet, filters applied, rows affected, columns
changed/exported, output file, sensitive fields involved, and confirmation
status. It records metadata only — never raw student rows.

**Editable fields** (operational flags): Notes, Follow Up Needed,
Advisor Follow Up, Internal Flag, Outreach Status, Review Status.

**Protected fields** (never editable by this assistant): Student ID, First/Last
Name, Email, Phone, Date of Birth, GPA, Academic Status, Conduct Status,
Financial Aid Status. Editing these requires a manual, higher-privilege process
that is intentionally not implemented.

Example — export:
```
User: Show me Accounting students below 2.5 GPA.
User: Export this list.
Assistant: (asks to confirm if sensitive fields are involved, then)
           Exported N record(s) to a new file: outputs/student_export_...xlsx
```

Example — add note:
```
User: Show me Accounting students below 2.5 GPA.
User: Add note: Advisor follow-up needed.
Assistant: This will add a note to the current selection ... Confirm?
User: Yes
Assistant: Added the note to N matching record(s). A new workbook was saved: ...
```

Example — protected field:
```
User: Change their GPA to 4.0.
Assistant: I can't update GPA because it is a protected field. You can export
           the list for manual review instead.
```

## Local LLM planner (optional fallback)

The deterministic rules handle most requests. For rich or novel phrasing the app
can fall back to a **local** model (Mistral via Ollama) — but only as a
*planner*, never as the database.

**Why it's optional:** rules answer the common cases instantly with no model.
The LLM is off by default and is only consulted when rules aren't confident.
If Ollama isn't running, the app says so and stays on the rules planner — it
never crashes.

**Why the LLM never sees student rows:** it receives only the workbook schema,
column names, canonical names, low-cardinality non-sensitive categorical values,
the conversation state, and the allowed operations/operators. It returns a
**JSON plan** — it does not compute answers and cannot execute anything.

**Rules planner vs LLM fallback:**
- *Rules* — fast, deterministic, used when confident (e.g. "Show me Accounting
  students", "now only below 2.5 GPA", "sort by GPA lowest first").
- *LLM fallback* — used only when rules are unsure (e.g. "Who seems like they
  need advisor attention?"). The model proposes a plan; the app does the rest.

**How validation protects against bad LLM output:** every LLM plan is parsed
(with one repair retry; markdown is stripped, prose is rejected) and then
safety-validated. Plans are refused if they reference nonexistent columns, use
invalid/incompatible operators, request protected-field edits, ask for sensitive
exports without confirmation, request hidden sensitive columns, or use excessive
row limits. A rejected plan becomes a clarification — it is never executed.

**Enabling it:** start Ollama (`ollama serve` + `ollama pull llama3.2:3b`), then
as an Admin in Settings turn **off** Strict Privacy Mode and turn **on** "Enable
local LLM fallback" (and optionally "Enable local LLM explanations"). Use the
**Test Ollama connection** button to confirm it's reachable.

**Evaluating the planner:**
```
.venv/bin/python scripts/eval_planner.py            # rules-only
.venv/bin/python scripts/eval_planner.py --with-llm # live local model
```
The eval set lives at `tests/fixtures/planner_eval_cases.json`.

**Troubleshooting Ollama:**
- "local model is not available" → run `ollama serve`; check
  `http://localhost:11434`.
- Model missing → `ollama pull llama3.2:3b` (or whatever model is selected in Settings).
- Slow first response → the model is loading; subsequent calls are faster.
- Strict Privacy Mode forces the model off regardless of the toggle.

## Using the interface

The UI is a clean internal assistant, not a developer tool.

**Main area**
- Header: **Dean Assistant** + "Ask questions about an uploaded student workbook. All processing stays local."
- Three status badges: `[Workbook loaded]` `[Privacy protected]` `[Local LLM off]`.
- **Current context** chips (e.g. `[Department = Accounting] [GPA < 2.5] [Sort: GPA asc]`) with **Clear filters** / **Start over** buttons — or "No active filters."
- A simple chat box (`Ask about the student workbook...`) with a few clickable suggestions.
- A clean **Result** card (count + table preview), plus **Download visible result**, **Export full filtered list**, and **Add note** actions.
- **Confirmation needed** cards for sensitive display / export / edits (Confirm / Cancel). Protected-field requests get a clear refusal.
- An **Export created** / download card after a file is produced ("Original workbook was not modified.").

**Sidebar**
- **Workbook**: upload, name, sheet · rows · columns, and a collapsed "Workbook notes" expander.
- Signed-in chip + sign out.
- Collapsed advanced expanders (admin): Settings (local LLM toggles, model, test connection, strict privacy), Privacy controls, Learning admin, Users — plus a **Show developer/debug info** checkbox.

### Simple mode vs developer mode

- **Simple mode** (default): for normal dean-office users — upload, ask, see the answer, confirm sensitive/export/edit actions, download outputs. No technical jargon (no "routing", "plan source", "validation object", etc.).
- **Developer mode** (sidebar → *Show developer/debug info*): for debugging — shows plan source, intent, confidence, LLM used, validation status, fallback reason, active-context JSON, pending-action JSON, and the schema summary. It never shows raw sensitive rows.

## Demo walkthrough (≈5 minutes)

A scripted demo using the synthetic workbook. It shows context memory, filter
composition, replacement filters, pandas-grounded answers, privacy
confirmation, and offline operation.

1. Generate the data: `.venv/bin/python scripts/make_synthetic_workbook.py`
2. Launch: `.venv/bin/streamlit run app.py`, sign in, upload
   `tests/fixtures/synthetic_students.xlsx`.
3. **"Show me Accounting students"** → a count + a redacted preview.
   *Demonstrates: pandas-grounded filtering; sensitive columns hidden by default.*
4. **"now only below 2.5 GPA"** → keeps the Accounting filter and adds `GPA < 2.5`.
   *Demonstrates: additive follow-up / context memory.*
5. **"now only seniors"** → adds `Year = Senior` on top.
   *Demonstrates: multi-step composition.*
6. **"what is their average GPA"** → average computed over the current selection only.
   *Demonstrates: aggregates respect active filters; the model never does the math.*
7. **"what about Biology"** → replaces `Department = Accounting` with `Biology`
   (not Accounting AND Biology). *Demonstrates: replacement filters.*
8. **"show me all student emails and GPAs"** → the assistant asks you to confirm
   before revealing Email. *Demonstrates: privacy confirmation gate.*
9. Click **"Yes, show them"** to reveal, or **"No, keep them hidden"** to decline.
   *Demonstrates: pending-action confirm/cancel.*
10. **"start over"** → clears all filters, sort, grouping, and pending actions.
    *Demonstrates: reset.*

Everything runs locally; with the local model off, all of the above is handled
by the rule-based parser and pandas — no network calls at any point.

The "Current context" box and the **Clear filters / Start over** buttons (left
panel) show what's active at each step. Tick **"Show developer/debug info"** to
inspect the normalized schema, sensitivity flags, and live state.

## One planning path (UI = tests = eval)

There is a **single** planning path, used by the Streamlit chat UI, the test
suite, and the eval harness:

```
User message
→ nlp/planner_router.plan_user_request   (rules-first, safe LLM fallback)
→ validator / privacy gate               (inside the router)
→ core/execution_dispatcher.execute_planned_request
→ pandas query engine OR confirmed_actions
→ response renderer (ui/chat_panel)
→ session memory update
→ debug state update
```

Why this matters: the UI cannot behave differently from the tested
planner/eval path, because they call the same `plan_user_request` and the same
`execute_planned_request`. The router returns one normalized object
(`plan_source`, `intent`, `confidence`, `plan`, `requires_confirmation`,
`warnings`, `llm_used`, `validation`, `fallback_reason`, …) that the UI consumes
directly. Run the eval through the full dispatcher with:

```
.venv/bin/python scripts/eval_planner.py --dispatch
```

(Structural in-workbook edits — highlight/chart/report — are a distinct
capability with their own planner; the router classifies them as the `edit`
intent and the chat delegates to that path.)

## Current limitations

- Follow-up understanding is rule-based; genuinely novel phrasings use the
  optional local LLM planner (off by default, slow on laptops). With it off,
  such prompts get a clarification rather than a guess.
- "Clear that" clears **all** active filters (not just the most recent).
- Confirmed export / add-note / safe-field-update now execute and save a new
  workbook; in-workbook edits like highlighting and charts run through the
  separate plan-execution path. Only the safe fields above are editable.
- The workbook loader types all columns as text; numeric *edits* (e.g. highlight
  GPA < 3) can be blocked by the validator even though numeric *questions* work
  (the query engine coerces). Fixing this is a loader change.

## Roadmap

- Wire note/field edit execution behind the existing confirmation gate.
- Numeric-aware workbook loading so numeric edits aren't blocked.
- Richer multi-sheet / cross-sheet questions.
- Optional faster local model / streaming explanations.
```
## PSAT/SAT assessment support

The assistant can detect optional PSAT/SAT assessment data inside the uploaded
academic workbook. Assessment data can be inline on the roster sheet or in a
sibling sheet such as `Assessments`, `PSAT`, or `SAT`. Sibling assessment
sheets are matched to the roster by `Student ID`; name-based guessing is not
used by default.

Assessment fields are protected and cannot be edited by the assistant. This
includes SAT/PSAT scores, benchmark status, college readiness, and assessment
risk fields. Staff can still mark operational action fields such as `Academic
Watch`, `Attendance Watch`, `Follow Up Needed`, and `Review Status` after the
normal confirmation step.

Benchmark-risk workflows require either benchmark fields in the workbook or
configured benchmark thresholds in Risk Settings. If the workbook has scores
but no benchmark fields or configured thresholds, the assistant can still sort
or filter raw scores when the user gives a threshold, such as `show SAT Math
below 500`.

Assessment risk can contribute to combined academic risk alongside GPA,
attendance, and academic standing. The original uploaded workbook is never
modified in place; confirmed actions write a new workbook and record metadata
in the audit log.

Example capability set:

- Teacher and department questions
- GPA performance review
- Attendance-risk review
- PSAT/SAT assessment review
- Benchmark-risk review
- Combined academic risk review
- Academic Watch updates
- Export updated workbook

## Packaging and Deployment (Local LLM Bundling)

To package and deploy the assistant with the local LLM fully bundled (enabling a self-contained, zero-install, 100% offline experience for end-users), follow this checklist:

### 1. Populate Compiled Binaries (`dean/bin/`)
Download and copy the compiled CLI binaries matching each platform/architecture:
* **macOS Apple Silicon:** Extract `ollama` from the App package and rename to `ollama-darwin-arm64`.
* **macOS Intel:** Rename `ollama` to `ollama-darwin-amd64`.
* **Windows x64:** Copy `ollama.exe` and rename to `ollama-windows-amd64.exe`.
* **Linux x64:** Download the CLI and rename to `ollama-linux-amd64`.

Ensure executable permissions are set: `chmod +x dean/bin/ollama-darwin-*`

### 2. Preload Model Weights (`dean/models/`)
Rather than shipping a single GGUF file, Ollama requires its standard manifest/blobs registry database directory structure. 

To populate the `dean/models/` folder:
1. Initialize the environment variable on your developer machine pointing to your project path:
   `export OLLAMA_MODELS=/path/to/azera-formatting/dean/models`
2. Start the local server:
   `ollama serve`
3. Download the model:
   `ollama pull llama3.2:3b`
4. Shut down Ollama. The `dean/models/` folder is now populated with `blobs/` and `manifests/` directories containing the Llama 3.2 weights. Include these files inside your application distribution package.

### 3. Verification & Build Checklist
Before building the installer:
1. **Clear Development Fallback:** Set `DEAN_ALLOW_SYSTEM_OLLAMA_FALLBACK=false` to ensure the app doesn't fall back to system-wide Ollama during testing.
2. **Disconnect Internet:** Turn off Wi-Fi/Ethernet to simulate a strict offline classroom/office environment.
3. **Verify Badges:** Start the app (`streamlit run app.py`). Go to settings, check "Enable local LLM fallback", and verify the LLM status badge says `Local LLM: Bundled (llama3.2:3b)`.
4. **Test Query:** Send a natural language query and verify that the plan parses correctly.
5. **Clean Up Processes:** Stop the app, and run `ps aux | grep ollama` to verify no orphan background processes are left running on the machine.

---

## What V1 Does / Does Not Do

### What V1 Does
* **Zero-Cloud Local Ingestion & Profiling**: Safely ingests Excel student rosters (`.xlsx`) via Pandas without transmitting any data over the internet.
* **Messy Roster Parsing & Canonical Mapping**: Automatically profiles the sheet columns and maps messy column names (e.g. `Grade Point Average`, `G.P.A`, `counselor`) to canonical database schema concepts (`GPA`, `Advisor`).
* **Local-First Execution Engine**: Formulates execution plans and processes them using deterministic Pandas calculations locally, completely separating raw data from the LLM.
* **Rule-First Planner with Local LLM Fallback**: Leverages quick regular-expression matching for standard queries, falling back to a locally-hosted Ollama Llama 3.2 model on the loopback interface for advanced language parsing.
* **Interactive Session Profiles**: Displays the workbook's state, active sheet, dimensions, mapped key columns, and LLM connection status dynamically at the top of the interface.
* **Detailed Query Response Cards**: Displays answer details, columns used, active filter conditions, and separates the plain-English answers from assumptions.
* **High-Contrast Confirmation Gates**: Requires users to review and manually confirm any roster update or note addition before writing changes.
* **Isolated Workbook Exports**: Saves changes into a brand-new, timestamped `.xlsx` file under `outputs/` leaving the original uploaded roster completely untouched.
* **Privacy Redaction & Safe Audit Logs**: Redacts sensitive student columns (names, grades) by default and records operational metadata (without row-level data) to a local JSONL log file.

### What V1 Does NOT Do
* **No Cloud Processing**: Does not make external network requests to OpenAI, Anthropic, Google, or any cloud API.
* **No Direct Mutation of Original File**: Does not modify the uploaded workbook file in-place; all changes are saved in new copies.
* **No Row-Level Data to LLM**: Never transmits raw row-level student data to any LLM. The LLM only receives schema columns, filters, and planning requests.
* **No Arbitrary Code Execution**: Does not use dangerous execution environments (`exec()` or `eval()`) to run model-generated code on the workbook.
* **No Hard Security Boundary**: The application is an office tool and does not provide built-in database-level encryption or enterprise user authorization (the sidebar role view is for testing/presentation purposes).
* **No Modification of Protected Columns**: Restricts changes to sensitive identity and grade columns (`Student ID`, `GPA`, `Discipline`) to prevent accidental bulk modifications.

