"""Phase C: plan validation, including the new operators and clean failures."""

from __future__ import annotations

import pandas as pd
import pytest

from core.validator import PlanValidationError, validate_plan


@pytest.fixture()
def sheets():
    df = pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Department": ["Bio", "Acc"],
        "GPA": [3.1, 2.2],
    })
    return {"Students": df}


def _plan(condition):
    return {
        "plan_type": "single_action",
        "commands": [{"action": "filter_rows", "sheet": "Students", "conditions": [condition],
                      "output_sheet": "Out"}],
    }


def test_accepts_between_numeric(sheets):
    validate_plan(_plan({"column": "GPA", "operator": "between", "value": [2.0, 3.0]}), sheets, "f.xlsx")


def test_accepts_not_in_list(sheets):
    validate_plan(_plan({"column": "Department", "operator": "not_in", "value": ["Bio"]}), sheets, "f.xlsx")


def test_accepts_is_blank(sheets):
    validate_plan(_plan({"column": "Name", "operator": "is_blank"}), sheets, "f.xlsx")


def test_accepts_starts_with_on_text(sheets):
    validate_plan(_plan({"column": "Name", "operator": "starts_with", "value": "A"}), sheets, "f.xlsx")


def test_between_on_text_fails_cleanly(sheets):
    with pytest.raises(PlanValidationError):
        validate_plan(_plan({"column": "Name", "operator": "between", "value": [1, 2]}), sheets, "f.xlsx")


def test_starts_with_on_numeric_fails_cleanly(sheets):
    with pytest.raises(PlanValidationError):
        validate_plan(_plan({"column": "GPA", "operator": "starts_with", "value": "3"}), sheets, "f.xlsx")


def test_unknown_operator_fails_cleanly(sheets):
    with pytest.raises(PlanValidationError):
        validate_plan(_plan({"column": "GPA", "operator": "frob", "value": 1}), sheets, "f.xlsx")


def test_unknown_column_fails_cleanly(sheets):
    with pytest.raises(PlanValidationError):
        validate_plan(_plan({"column": "Nope", "operator": "equals", "value": 1}), sheets, "f.xlsx")


def test_between_requires_two_bounds(sheets):
    with pytest.raises(PlanValidationError):
        validate_plan(_plan({"column": "GPA", "operator": "between", "value": [2.0]}), sheets, "f.xlsx")
