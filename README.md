# Offline Dean Student Assistant

This is a local desktop app for tracking enrolled students by college, major,
status, and campus location. It does not use an online AI service or a local
language model. The assistant is rule-based software that responds to common
student-record questions and commands.

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

## Run the Streamlit MVP

This is the main offline enrollment Excel assistant MVP:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py
```

The app opens in your browser and runs locally.

## Run the Older Desktop Prototype

Use Python 3:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python student_dean_bot.py
```

The app creates a local SQLite database file named `students.db` in this folder.
It starts with sample student records the first time it runs.

The Streamlit MVP creates `assistant_logs.db` for request logs and correction
records.

## Example Commands

Type these into the assistant box:

```text
help
How many enrolled students are in Engineering?
Show inactive students in Nursing
Where is student 1001?
Generate a report for Business
Update student 1003 to enrolled
Show students at Main Campus
Summarize workbook
Show workbook headers
Extract all BES students by name and year
Highlight BES in workbook
Highlight students who are missing FAFSA documents
Show me students enrolled but not registered
Sum tuition balances by program
Move inactive students to a new sheet
Format this report
```

## What It Can Do

- Run fully offline in a desktop window
- Store student records locally in SQLite
- Search, count, and list students through chat-like commands
- Find a student by ID
- Update a student's enrollment status
- Generate simple college summaries
- Open an `.xlsx` workbook and view its headers and rows inside the app
- Summarize, search, extract, and highlight rows from the loaded workbook
- Convert common university spreadsheet requests into validated actions
- Ask for confirmation before saving edited workbook copies

## Excel Workbook Viewer

Use `Open XLSX` to load an Excel workbook into the workbook pane. The app reads
the first worksheet, treats the first non-empty row as headers, and shows the
remaining rows in a table.

The assistant and workbook are separated by a draggable divider. Drag it left or
right to give more room to either side. You can also use `Wider Sheet` and
`Reset Layout` for quick resizing.

Use `Open Original` to launch the same workbook in Excel, Numbers, LibreOffice,
or whatever spreadsheet app is installed on the computer. This lets a dean or
advisor see the original file while still using the assistant inside this app.

If the optional `tkinterdnd2` package is installed, the workbook area also
accepts dragged `.xlsx` files. Without that package, use `Open XLSX`.

After loading a workbook, ask the assistant:

```text
Summarize workbook
Show workbook headers
Extract all BES students by name and year
Highlight Nursing in workbook
```

Matching workbook rows are highlighted in yellow.

## Controlled Spreadsheet Actions

The assistant is not a full general chatbot. It translates university workflow
requests into allowed spreadsheet actions, validates the workbook columns, then
asks for confirmation before making a new edited file.

Supported starter actions:

```text
highlight rows
filter rows to a new sheet
move matching rows to a new sheet
sum a numeric column
count rows by group
format report
```

Example:

```text
Highlight students who are missing FAFSA documents
```

The app previews the action and match count. Type:

```text
confirm
```

to save an edited workbook such as:

```text
Fall_Enrollment_Report_edited.xlsx
```

## Data Fields

Each student record includes:

```text
Student ID
Name
College
Major
Enrollment status
Campus location
Advisor
```

## Privacy Note

This starter app keeps data local, but it is not yet production-ready for real
student records. Before using it with real college data, add login accounts,
role-based access, database encryption, backups, and audit logs.

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

