"""Planner evaluation harness.

Runs the planner-router over a set of prompts and reports the plan source,
intent, confidence, and validation for each. Defaults to rules-only mode (no
Ollama). Use --with-llm to exercise a live local model if available.

    .venv/bin/python scripts/eval_planner.py            # rules-only
    .venv/bin/python scripts/eval_planner.py --with-llm # live local model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.excel_loader import load_excel_workbook  # noqa: E402
from nlp.planner_router import plan_user_request  # noqa: E402
from scripts.make_synthetic_workbook import write_workbook  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_students.xlsx"
CASES = REPO_ROOT / "tests" / "fixtures" / "planner_eval_cases.json"


class _Upload:
    def __init__(self, path: Path):
        self._bytes = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self) -> bytes:
        return self._bytes


def load_cases() -> list[dict]:
    return json.loads(CASES.read_text())["cases"]


def run_eval(*, llm_enabled: bool = False, llm_call=None, dispatch: bool = False) -> list[dict]:
    if not FIXTURE.exists():
        write_workbook(FIXTURE)
    loaded = load_excel_workbook(_Upload(FIXTURE))
    sheet_columns = {n: list(d.columns) for n, d in loaded.sheets.items()}
    settings = {"llm_enabled": llm_enabled}

    rows = []
    for case in load_cases():
        result = plan_user_request(
            user_message=case["prompt"],
            sheets=loaded.sheets,
            sheet_columns=sheet_columns,
            selected_sheet="Students",
            settings=settings,
            llm_call=llm_call,
        )
        validation_status = result["validation"]["status"]
        passed = result["intent"] == case["expected_intent"]
        if "expected_validation" in case:
            passed = passed and validation_status == case["expected_validation"]
        if "expected_requires_confirmation" in case:
            passed = passed and result["requires_confirmation"] == case["expected_requires_confirmation"]

        dispatched = None
        if dispatch and result["intent"] == "query" and not result["requires_confirmation"]:
            # Same path as the UI: router -> validator -> execution dispatcher.
            from core.execution_dispatcher import execute_planned_request

            response = execute_planned_request(result, loaded, settings, request_summary=case["prompt"])
            dispatched = response["success"]
            passed = passed and dispatched

        rows.append({
            "prompt": case["prompt"],
            "category": case.get("category", ""),
            "expected_intent": case["expected_intent"],
            "actual_intent": result["intent"],
            "plan_source": result["plan_source"],
            "confidence": result["confidence"],
            "validation": result["validation"]["status"],
            "llm_used": result["llm_used"],
            "dispatched": dispatched,
            "pass": passed,
        })
    return rows


def main() -> int:
    rows = run_eval(llm_enabled="--with-llm" in sys.argv, dispatch="--dispatch" in sys.argv)
    width = max(len(r["prompt"]) for r in rows)
    for r in rows:
        flag = "PASS" if r["pass"] else "FAIL"
        print(f"[{flag}] {r['prompt']:<{width}}  expect={r['expected_intent']:<11} "
              f"got={r['actual_intent']:<11} src={r['plan_source']:<13} val={r['validation']}")
    passed = sum(1 for r in rows if r["pass"])
    print(f"\n{passed}/{len(rows)} cases passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
