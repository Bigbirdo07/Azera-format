"""Phase E: messy-workbook ingestion, canonical mapping, type inference, debug."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from core.excel_loader import load_excel_workbook
from core.privacy import detect_sensitive_columns
from core.schema import (
    build_debug_state,
    build_workbook_schema,
    canonical_for,
    canonical_map,
    infer_column_types,
)


class _Upload:
    def __init__(self, path: Path):
        self._bytes = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self) -> bytes:
        return self._bytes


def _messy_workbook(path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Students"
    # Duplicate "GPA", a header with surrounding spaces, an unnamed blank column,
    # GPA stored as text, and a fully-blank row in the middle.
    ws.append(["Student ID", " Dept ", "GPA", "GPA", "Notes", None])
    ws.append(["S1", "Bio", "3.1", "3.2", "", None])
    ws.append(["S2", "Acc", "2.0", "2.1", "", None])
    ws.append([None, None, None, None, None, None])
    ws.append(["S3", "Bio", "1.5", "1.6", "", None])
    wb.create_sheet("Empty")  # entirely empty sheet
    wb.save(path)


def test_loader_warns_on_messy_input(tmp_path):
    path = tmp_path / "messy.xlsx"
    _messy_workbook(path)
    loaded = load_excel_workbook(_Upload(path))
    blob = " ".join(loaded.warnings).lower()
    assert "duplicate column" in blob
    assert "blank row" in blob
    assert "is empty and was ignored" in blob
    assert "interpreted numerically" in blob  # GPA stored as text
    assert "Empty" not in loaded.sheets  # empty sheet skipped


def test_loader_cleans_headers(tmp_path):
    path = tmp_path / "messy.xlsx"
    _messy_workbook(path)
    cols = list(load_excel_workbook(_Upload(path)).sheets["Students"].columns)
    assert "Dept" in cols  # trimmed
    assert "GPA" in cols and any(c.startswith("GPA_") for c in cols)  # deduped
    assert not any(str(c).startswith("Column ") for c in cols)  # unnamed blank dropped


def test_canonical_for_messy_names():
    assert canonical_for(" G.P.A. ") == "gpa"
    assert canonical_for("Cum GPA") == "cumulative_gpa"
    assert canonical_for("Dept") == "department"
    assert canonical_for("Academic Program") in {"major", "program"}
    assert canonical_for("Class Year") == "year"
    assert canonical_for("Student Level") in {"class_level", "year"}
    assert canonical_for("Advisor Name") == "advisor"
    assert canonical_for("Totally Unknown Column") is None


def test_canonical_map_first_match(columns):
    mapping = canonical_map(columns)
    assert mapping["department"] == "Department"
    assert mapping["gpa"] == "GPA"
    assert mapping["advisor"] == "Advisor"
    assert mapping["email"] == "Email"


def test_numeric_as_text_gpa(tmp_path):
    path = tmp_path / "messy.xlsx"
    _messy_workbook(path)
    types = infer_column_types(load_excel_workbook(_Upload(path)).sheets["Students"])
    assert types["GPA"]["analysis_dtype"] == "numeric"
    assert types["GPA"]["coercion_success_rate"] >= 0.9
    assert types["GPA"]["coercion_warning"]


def test_type_inference_on_synthetic(sheets):
    types = infer_column_types(sheets["Students"])
    assert types["GPA"]["analysis_dtype"] == "numeric"
    assert types["Date of Birth"]["analysis_dtype"] == "date"
    assert types["Department"]["analysis_dtype"] in {"category", "text"}


def test_sensitive_detection_after_normalization():
    detected = detect_sensitive_columns([" Email ", "Phone Number", "Date of Birth", "GPA", "Dept"])
    assert detected.get(" Email ") == "contact"
    assert detected.get("Phone Number") == "contact"
    assert detected.get("Date of Birth") == "identity_high"
    assert "GPA" not in detected


def test_build_workbook_schema(sheets):
    schema = build_workbook_schema(sheets)["Students"]
    assert schema["row_count"] == len(sheets["Students"])
    assert "GPA" in schema["columns"]
    assert "Email" in schema["sensitive"]
    assert "Email" not in schema["default_visible"]


def test_build_debug_state_sanitizes_pending():
    schema = {"Students": {"columns": ["GPA"], "canonical_map": {}, "column_types": {},
                           "sensitive": {}, "default_visible": ["GPA"]}}
    memory = {"active_filters": [{"column": "Department"}], "active_limit": 5,
              "pending_action": {"type": "export", "query": {"secret": 1}}}
    state = build_debug_state(memory, schema, "Students")
    assert state["active_sheet"] == "Students"
    assert state["active_limit"] == 5
    assert state["pending_action"] == {"type": "export"}  # query stripped
    assert "GPA" in state["normalized_schema"]["columns"]
