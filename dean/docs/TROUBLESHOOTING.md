# V1 Troubleshooting Guide

This guide describes how to resolve common issues, warnings, and errors in the **Offline Dean-Office Student-Record Assistant**.

---

## 1. No Workbook Uploaded
* **Symptom**: The user enters a question or command, but the app shows a warning instead of a result.
* **Friendly Error Shown**:
  > **No Workbook Loaded**: I couldn't run that calculation because there is no workbook active. Please upload a student roster file using the sidebar.
* **Resolution**:
  1. Look at the left sidebar.
  2. Locate the file upload block under the **Roster File** section.
  3. Upload a valid Excel file (`.xlsx`).
  4. Confirm that the **Session Profile** dashboard card displays active dimensions.

---

## 2. Missing GPA Column
* **Symptom**: The user asks a question about academic watch or student performance, but the schema doesn't contain GPA metrics.
* **Friendly Error Shown**:
  > **Missing Key Column**: I couldn't find the column 'GPA' in the active sheet. The available columns are: [ColumnNames].
* **Resolution**:
  1. Inspect the columns of the sheet in the right-hand workbook viewer pane.
  2. If the GPA column has a messy name (e.g., `G.P.A.`, `Grade Point Average`), ensure the system auto-mapped it under **Session Profile**.
  3. If missing, open the original file, rename the column to `GPA` or `Grade Point Average`, save it, and re-upload.

---

## 3. Local LLM Unavailable
* **Symptom**: The Advanced Settings panel shows an LLM status of `Offline` or `Failed`, and a banner explains that rule fallback is active.
* **Friendly Errors Shown**:
  * `Ollama connection failed.`
  * `Bundled Ollama started, but model llama3.2:3b was not found in dean/models.`
* **Resolution**:
  1. Ensure Ollama is installed on the host machine, or that you have downloaded the weights into the `models/` directory.
  2. Open a terminal and run `ollama serve` to start the background service.
  3. Pull the required model: `ollama pull llama3.2:3b`.
  4. In the app, go to **Admin Settings**, check **Enable local LLM fallback**, and click **Test Ollama connection**.

---

## 4. Export Failed
* **Symptom**: The user confirms an edit, but the Export Center shows an export failure warning.
* **Friendly Error Shown**:
  > **Export Failed**: The action could not be completed: [ErrorDetails].
* **Resolution**:
  1. Ensure that the target output folder (`outputs/` or your system temp directory) has read/write permissions.
  2. Verify that there is enough disk space.
  3. Check `dean/logs/ollama.log` or your terminal logs for any underlying permission errors.

---

## 5. No Previous Context for "those"
* **Symptom**: The user asks `"Which students are those?"` or `"Export those"`, but the assistant clarifications card says it doesn't know who "those" refers to.
* **Friendly Error Shown**:
  > **No Previous Context**: You asked to follow up or drill down, but I don't have any previous context in this session. Please ask a specific roster question first.
* **Resolution**:
  1. Ask an initial roster question first (e.g., `"Which students have GPAs under 2.5?"`).
  2. Once the results are displayed, ask `"Which students are those?"` to drill down.

---

## 6. Protected-Field Edit Blocked
* **Symptom**: The user asks to edit a protected column (e.g., `"Change Student ID to S999"` or `"Update GPA to 4.0"`), but the assistant blocks the edit.
* **Friendly Error Shown**:
  > **Protected Field**: I can't update [FieldName] because it is a protected field. You can export the list for manual review instead.
* **Resolution**:
  1. The system locks critical academic identity fields (e.g., `Student ID`, `GPA`, `Discipline`) to prevent accidental bulk modifications.
  2. To modify these values, export the roster using `"Export that list"` and make the changes manually in Excel.
