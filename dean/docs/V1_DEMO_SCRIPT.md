# V1 Demo Script: Dean-Office Student Roster Walkthrough

This guide walks you through a complete end-to-end demonstration of the **Offline Dean-Office Student-Record Assistant**. It is designed for school administrators, deans, advisors, and registrars to show how the system can securely analyze and modify student roster spreadsheets using natural language, entirely offline.

---

## Prerequisites & Setup

1. **Launch the Application**:
   Make sure you are in the `dean` subdirectory and run the Streamlit app:
   ```bash
   ./.venv/bin/streamlit run app.py
   ```
2. **Retrieve the Demo Data**:
   Use the synthetic student roster located at:
   [synthetic_students.xlsx](file:///Users/albertopaz/azera-formatting/dean/tests/fixtures/synthetic_students.xlsx) (or generate a new one using `python scripts/make_synthetic_workbook.py`).
3. **Configure Local LLM (Optional but Recommended)**:
   * Click **Admin Settings** in the left sidebar.
   * Uncheck **Strict Privacy** (this allows local loopback port connections to your offline model).
   * Check **Enable local LLM fallback** and **Enable LLM Explanations**.
   * Run **Test Ollama connection** to confirm the offline `llama3.2:3b` model is running and reachable.

---

## The Demo Prompt Library (Actual App Chips)

Under the suggested questions panel in the chat area, you will find the **Realistic Demo Walkthrough** tab. These chips correspond to the exact sequence of this walkthrough:

1. `Which advisors have students under 2.5?`
2. `Which students are those?`
3. `Break that down by department.`
4. `Mark them Academic Watch and export.`
5. `Which department has the best average GPA?`
6. `Show low-performing students in that department.`
7. `Export that list.`

---

## Step-by-Step Walkthrough

### Step 1: Upload the Student Roster
1. Locate the file upload area in the left sidebar.
2. Upload `synthetic_students.xlsx`.
3. **Observe the Session Profile Card**:
   * A clean, multi-column dashboard card immediately appears at the top of the interface.
   * It shows the active workbook name, active sheet name, row/column counts, detected key columns (e.g. `Student ID`, `Advisor`, `GPA`, `Discipline`), and the offline LLM / Privacy status.

### Step 2: Identify Advisors at Risk (Turn 1)
1. Click the first suggestion chip: **"Which advisors have students under 2.5?"** (or type it in).
2. **Observe the Response Card**:
   * The response lists the advisors matching the criteria.
   * Beneath the answer, a metadata section shows:
     * **Columns Used**: `Advisor`, `GPA`
     * **Active Filters**: `GPA below 2.5`
   * Assumptions and caveats are clearly separated from the final answer text.
   * No row-level student data was sent to any LLM.

### Step 3: Drill Down into Specific Students (Turn 2)
1. Click the second suggestion chip: **"Which students are those?"**
2. **Observe the Follow-up Resolution**:
   * The assistant retains the context from the previous turn (`GPA < 2.5`).
   * A table of student records appears showing the specific students who have GPAs under 2.5 (e.g., S001, S003).
   * Notice that sensitive columns are automatically redacted or require user permission to show.

### Step 4: Breakdown by Department/Discipline (Turn 3)
1. Click the third suggestion chip: **"Break that down by department."**
2. **Observe the Grouping Result**:
   * The assistant aggregates the filtered subset of students (`GPA < 2.5`) by their department/discipline.
   * You see a breakdown count of how many students are at risk in each department (e.g., Biology: 1, Chemistry: 1).

### Step 5: Mark the Academic Watch list (Turn 4)
1. Click the fourth suggestion chip: **"Mark them Academic Watch and export."**
2. **Observe the High-Contrast Confirmation Card**:
   * A warning card titled `Confirmation Required` or `Planned Action` appears in a high-contrast border.
   * It outlines the proposed action: setting the `Academic Watch` column to `Yes` for the 2 students identified in Turn 2.
   * Click **Confirm Action**.

### Step 6: Verify and Download the Exported Workbook
1. Once confirmed, look at the **Export Center** in the sidebar or main panel.
2. **Observe the Export Details**:
   * It displays the output filename and a list of changes applied (e.g., `Marked 2 student(s) in Academic Watch, Academic Watch Reason, Date Flagged`).
   * A **Download Exported XLSX** button becomes active.
   * A prominent warning is shown: *"Data Safety Notice: The original uploaded workbook remains completely untouched. Changes are only saved in this downloaded copy."*
3. Click the download button and open the new spreadsheet to verify:
   * S001 and S003 have `Academic Watch = Yes`.
   * A new `Academic Watch Reason` column is populated showing provenance (`GPA below 2.5`).
   * A `Date Flagged` column is added.
   * S002 and other students are untouched.

### Step 7: Analyze Group Performance (Turn 5)
1. Click the fifth suggestion chip: **"Which department has the best average GPA?"**
2. **Observe the Average Calculation**:
   * The assistant aggregates all students by department/discipline, averages their GPAs, and reports the highest department.

### Step 8: Drill Down to Low Performers in the Top Department (Turn 6)
1. Click the sixth suggestion chip: **"Show low-performing students in that department."**
2. **Observe the Smart Filter chaining**:
   * The assistant combines the department resolved in Turn 5 with a low GPA threshold (< 2.5 or < 3.0 depending on rules) and displays the matching records.

### Step 9: Export the final subset (Turn 7)
1. Click the seventh suggestion chip: **"Export that list."**
2. Confirm the export and download the final clean worksheet from the Export Center.
