# User Workflows

## PK-12 Examples

### Attendance Watch Review

1. Set Institution mode to `PK-12`.
2. Set Role view to `Counselor / Advisor`.
3. Upload an academic workbook with student, teacher, grade, GPA, and
   attendance fields.
4. Review the Workbook Readiness panel.
5. Use the Attendance Watch Review template.
6. Ask: `Show students with attendance below 90%`.
7. Review the result and threshold narration.
8. Ask: `Group them by teacher`.
9. Ask: `Mark these students Attendance Watch and export`.
10. Confirm the action.
11. Download the new workbook from Export Center.

Expected output includes Attendance Watch, Attendance Watch Reason, and Date
Flagged where applicable.

### Academic Intervention List

1. Set Institution mode to `PK-12`.
2. Set Role view to `Counselor / Advisor`.
3. Upload a roster with Student ID, student name, teacher, grade, GPA, and
   Academic Standing.
4. Ask: `Show students with GPA below 2.0`.
5. Ask: `Group them by teacher`.
6. Ask: `Mark these students Academic Watch and export`.
7. Confirm the action.

The exported workbook includes a reason such as `GPA below 2.0`.

### Registrar Data Review

1. Set Role view to `Registrar / Data Staff`.
2. Upload the workbook.
3. Review Workbook Readiness for missing, blank, or duplicate Student IDs.
4. Ask: `Show me the data quality summary`.
5. Export validation results when needed.

## College / University Examples

### Retention Risk Review

1. Set Institution mode to `College / University`.
2. Set Role view to `Administrator / Dean`.
3. Upload a roster with Student ID, student name, professor or instructor,
   department, major, GPA, class year, and Academic Standing.
4. Review Detected Capabilities and Workbook Readiness.
5. Use the Combined Risk Review template.
6. Ask: `Show students with GPA below 2.0 and attendance below 90%`.
7. Ask: `Group them by department`.
8. Ask: `Mark these students Follow Up Needed and export`.
9. Confirm the action.

The assistant should use college wording such as professor, advisor, class
year, advisor outreach, and retention risk.

### Advisor Outreach

1. Set Institution mode to `College / University`.
2. Set Role view to `Counselor / Advisor`.
3. Upload a workbook with Advisor, Major, GPA, and Academic Standing.
4. Ask: `How many students are at academic risk?`.
5. Ask: `Group them by advisor`.
6. Export the filtered list after confirmation.

### Professor Group Review

1. Set Role view to `Teacher / Professor`.
2. Ask: `Show students under my classes with low GPA or attendance risk`.
3. Review results in the Working Sheet panel.
4. Export only after confirming the intended filtered list.

## Generic Academic Examples

### Academic Watch Review

1. Set Institution mode to `Generic Academic`.
2. Upload an academic workbook.
3. Use the Academic Watch Review workflow template.
4. Ask: `Find students below GPA threshold`.
5. Ask: `Mark these students Academic Watch and export`.
6. Confirm before the workbook is written.

## Notes For All Workflows

- The uploaded workbook is never modified in place.
- Attendance can be detected from workbook fields or workbook sheets.
- Separate attendance documents are not required.
- Sensitive fields remain hidden unless explicitly confirmed.
- Spreadsheet rows are not sent to the LLM.
- Exports are written as new workbook files.
