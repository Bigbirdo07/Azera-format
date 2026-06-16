# V1 Manual Acceptance Testing Protocol

This document provides a comprehensive suite of manual validation checks to run before certifying a release. Each check should be executed on a clean run of the application.

---

## 1. Roster Loading & Schema Mapping
**Goal**: Verify that raw student roster files are loaded safely and mapped to canonical concepts.

* [ ] **Test Case 1.1: File Upload**
  * **Steps**:
    1. Open the application.
    2. Upload [synthetic_students.xlsx](file:///Users/albertopaz/azera-formatting/dean/tests/fixtures/synthetic_students.xlsx).
  * **Expected Outcome**:
    * The workbook is successfully loaded.
    * The **Session Profile** dashboard card displays:
      * Active Workbook: `synthetic_students.xlsx`
      * Active Sheet: `Students`
      * Dimensions: `5 rows x 5 columns`
      * Key Columns Auto-Mapped: `Student ID`, `Advisor`, `GPA`, `Academic Watch`
  * **Status**: `[ PASS / FAIL ]`

---

## 2. Natural-Language Queries & Interactive Feedback
**Goal**: Verify that queries run correctly and return clear metadata.

* [ ] **Test Case 2.1: Roster Query**
  * **Steps**:
    1. Run query: `"Which advisors have students under 2.5?"`
  * **Expected Outcome**:
    * Answer lists `Dr. Alpha` and `Dr. Bravo`.
    * A metadata banner or section displays:
      * **Columns Used**: `Advisor`, `GPA`
      * **Active Filters**: `GPA below 2.5`
    * The system lists the exact assumptions made to execute the calculation.
  * **Status**: `[ PASS / FAIL ]`

* [ ] **Test Case 2.2: Contextual Drilldown**
  * **Steps**:
    1. Follow up immediately with: `"Which students are those?"`
  * **Expected Outcome**:
    * Assistant displays a table containing exactly two rows: `S001` (GPA: 1.8) and `S003` (GPA: 2.2).
    * Columns used matches the active columns list.
  * **Status**: `[ PASS / FAIL ]`

---

## 3. Edit Confirmations & Safety Boundaries
**Goal**: Ensure no data is mutated without explicit confirmation, and protected fields are locked.

* [ ] **Test Case 3.1: Confirmation Card Display**
  * **Steps**:
    1. Run request: `"Mark them Academic Watch and export"`
  * **Expected Outcome**:
    * A high-contrast bordered container appears labeled `Confirmation Required` or `Planned Action`.
    * The card details the exact scope of the edit: `Set Academic Watch = Yes for 2 matching records`.
    * A **Confirm Action** button is present.
    * No change is written to disk yet.
  * **Status**: `[ PASS / FAIL ]`

* [ ] **Test Case 3.2: Export Isolation & Safe Writes**
  * **Steps**:
    1. Click **Confirm Action**.
  * **Expected Outcome**:
    * An export file is generated in the `outputs/` directory.
    * The **Export Center** lists the generated path and details the modified columns.
    * The original uploaded file [synthetic_students.xlsx](file:///Users/albertopaz/azera-formatting/dean/tests/fixtures/synthetic_students.xlsx) has **NOT** been modified (verify its timestamp or file contents).
  * **Status**: `[ PASS / FAIL ]`

---

## 4. Privacy Guards & Audit Logs
**Goal**: Verify that no row-level data leaks to model APIs, and audit trails remain privacy-safe.

* [ ] **Test Case 4.1: Sensitive Column Redaction**
  * **Steps**:
    1. Ask: `"Show all students"`
  * **Expected Outcome**:
    * Student list renders, but sensitive columns (e.g., student names, grades, disciplinary indicators) are redacted or show placeholders unless explicitly permitted.
  * **Status**: `[ PASS / FAIL ]`

* [ ] **Test Case 4.2: Privacy-Safe Audit Trail**
  * **Steps**:
    1. Open the local audit log file: `logs/audit_log.jsonl` (or database log records).
  * **Expected Outcome**:
    * A record for the `action_chain` from Test Case 3.2 is appended.
    * The log entry contains: `action_type`, `rows_affected`, `columns_changed`, `filters_applied`, and the user query summary.
    * **CRITICAL**: The entry does **NOT** contain any row-level student names, IDs, GPAs, or notes.
  * **Status**: `[ PASS / FAIL ]`
