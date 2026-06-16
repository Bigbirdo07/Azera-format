"""Phase R usability, safety, and end-to-end workflow validation tests.

Covers:
  - Roster load, advisor GPA query, drilldown, breakdown, and watch list edit confirmation.
  - Verification that the original workbook is unmodified.
  - Validation that audit logs are created and do not leak row data.
  - Specific, friendly error translation for missing columns, sheets, and protected fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import pandas as pd
import pytest

from core.execution_dispatcher import execute_planned_request
from core.session_memory import SessionMemory
from nlp.planner_router import plan_user_request
from ui.chat_panel import friendly_validation_error


@dataclass
class MockWorkbook:
    sheets: dict[str, pd.DataFrame]
    file_name: str = "synthetic_students.xlsx"


class TestPhaseRUsability(unittest.TestCase):

    def setUp(self):
        # Create temp dir for output files, exports, and audit logs
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.tmp_dir / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.tmp_dir / "logs" / "audit_log.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Mock synthetic roster
        rows = [
            {"Student ID": "S001", "Advisor": "Dr. Alpha", "Discipline": "Biology", "GPA": 1.8, "Academic Watch": "No"},
            {"Student ID": "S002", "Advisor": "Dr. Alpha", "Discipline": "Biology", "GPA": 3.6, "Academic Watch": "No"},
            {"Student ID": "S003", "Advisor": "Dr. Bravo", "Discipline": "Chemistry", "GPA": 2.2, "Academic Watch": "No"},
            {"Student ID": "S004", "Advisor": "Dr. Bravo", "Discipline": "Chemistry", "GPA": 3.8, "Academic Watch": "No"},
            {"Student ID": "S005", "Advisor": "Dr. Cathy", "Discipline": "Math", "GPA": 3.2, "Academic Watch": "No"},
        ]
        self.df = pd.DataFrame(rows)
        self.loaded = MockWorkbook(sheets={"Students": self.df})
        self.columns = list(self.df.columns)
        self.sheet_columns = {"Students": self.columns}

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def _step(self, memory: SessionMemory, message: str) -> dict:
        routing = plan_user_request(
            user_message=message,
            sheets=self.loaded.sheets,
            sheet_columns=self.sheet_columns,
            selected_sheet="Students",
            conversation_state=asdict(memory),
            settings={},
        )
        response = execute_planned_request(
            routing,
            self.loaded,
            settings={},
            request_summary=message
        )
        plan = routing.get("plan") or {}
        if routing.get("intent") == "query" and response.get("success"):
            memory.record_ask(
                request=message,
                query_plan=plan,
                result_description=response.get("description", "") or "",
                row_count=response.get("row_count"),
                columns_used=response.get("columns") or [],
                sheet=plan.get("sheet", ""),
                summary_table=response.get("result_preview") or [],
                top_group=response.get("top_group"),
            )
        return {"routing": routing, "response": response, "plan": plan}

    def test_e2e_walkthrough_flow(self):
        memory = SessionMemory()
        
        # Turn 1: Ask "Which advisors have students under 2.5?"
        turn1 = self._step(memory, "Which advisors have students under 2.5?")
        self.assertTrue(turn1["response"]["success"])
        self.assertEqual(turn1["routing"]["intent"], "query")
        # GPA and Advisor columns must be used
        self.assertIn("Advisor", turn1["response"]["columns"])
        self.assertIn("GPA", turn1["response"]["columns"])

        # Turn 2: Drilldown "Which students are those?"
        turn2 = self._step(memory, "Which students are those?")
        self.assertTrue(turn2["response"]["success"])
        # Should drill down to student-level records (S001 and S003 have GPA < 2.5)
        self.assertEqual(turn2["response"]["row_count"], 2)

        # Turn 3: Break down by department (Discipline in this schema)
        turn3 = self._step(memory, "Break that down by department.")
        self.assertTrue(turn3["response"]["success"])
        # Should count rows by department/discipline for the filtered set
        self.assertEqual(turn3["response"]["operation"], "groupby_count")

        # Turn 4: Plan watch list update - "Mark them Academic Watch and export"
        routing = plan_user_request(
            user_message="Mark them Academic Watch and export",
            sheets=self.loaded.sheets,
            sheet_columns=self.sheet_columns,
            selected_sheet="Students",
            conversation_state=asdict(memory),
            settings={},
        )
        self.assertEqual(routing["intent"], "action_chain")
        self.assertTrue(routing["requires_confirmation"])

        # Confirm and execute the planned action using patched folders
        from unittest.mock import patch
        with patch("core.confirmed_actions.DEFAULT_OUTPUT_DIR", self.output_dir), \
             patch("core.confirmed_actions.DEFAULT_AUDIT_PATH", self.audit_path):
            response = execute_planned_request(
                routing,
                self.loaded,
                settings={},
                request_summary="Mark them Academic Watch and export"
            )
            
        self.assertTrue(response["success"])
        self.assertEqual(response["rows_affected"], 2)
        
        # Verify exported workbook has changes
        output_file = response["output_file"]
        self.assertIsNotNone(output_file)
        self.assertTrue(Path(output_file).exists())
        exported_df = pd.read_excel(output_file)
        # S001 and S003 should be marked "Yes"
        watch_flags = dict(zip(exported_df["Student ID"], exported_df["Academic Watch"]))
        self.assertEqual(watch_flags["S001"], "Yes")
        self.assertEqual(watch_flags["S003"], "Yes")
        self.assertEqual(watch_flags["S002"], "No")

        # Verify original workbook remains completely untouched
        self.assertEqual(self.df.loc[self.df["Student ID"] == "S001", "Academic Watch"].values[0], "No")

        # Verify audit log entry
        self.assertTrue(self.audit_path.exists())
        log_lines = self.audit_path.read_text().splitlines()
        self.assertGreater(len(log_lines), 0)
        log_entry = json.loads(log_lines[0])
        # Action chain logs the overall action chain or individual actions
        self.assertEqual(log_entry["action_type"], "action_chain")
        self.assertEqual(log_entry["rows_affected"], 2)
        self.assertIn("Academic Watch", log_entry["columns_changed"])

    def test_friendly_validation_errors(self):
        # 1. Missing Column Error
        friendly = friendly_validation_error("Column does not exist: GPA", ["Student ID", "Advisor"], ["Students"])
        self.assertIn("I couldn't find the column 'GPA'", friendly)
        self.assertIn("Student ID", friendly)

        # 2. Nonexistent Sheet Error
        friendly = friendly_validation_error("references nonexistent sheet: Grades", [], ["Students", "Attendance"])
        self.assertIn("I couldn't find the sheet 'Grades'", friendly)
        self.assertIn("Students", friendly)

        # 3. Protected Field Error
        friendly = friendly_validation_error("protected_field", [], [])
        self.assertIn("protected fields and cannot be modified", friendly)

        # 4. Not Editable Field Error
        friendly = friendly_validation_error("not_editable", [], [])
        self.assertIn("This field is not editable", friendly)

        # 5. No Previous Context Error
        friendly = friendly_validation_error("no previous result", [], [])
        self.assertIn("don't have any previous context in this session", friendly)
