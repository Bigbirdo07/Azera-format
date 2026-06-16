from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.excel_loader import load_excel_workbook
from core.execution_dispatcher import execute_planned_request
from nlp.planner_router import plan_user_request
from ui.settings_panel import load_settings


DEFAULT_WORKBOOK = Path(
    "/Users/albertopaz/azera-formatting/"
    "mock_dean_student_roster_with_assessments_attendance_cleaned.xlsx"
)
DEFAULT_REPORT = Path("outputs/qa_dean_advisor_live_questions_report.json")


@dataclass(frozen=True)
class Scenario:
    question: str
    expected_operations: set[str] = field(default_factory=set)
    expected_value_column: str = ""
    expected_group_by: str = ""
    expected_filter_columns: set[str] = field(default_factory=set)
    allow_clarify: bool = False


@dataclass
class Upload:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


SCENARIOS = [
    Scenario("which advisors have the most students needing attendance support", {"groupby_count"}, expected_group_by="Advisor", expected_filter_columns={"Attendance Category"}),
    Scenario("average attendance rate for bad standing students by advisor", {"groupby_average"}, expected_value_column="Attendance Rate", expected_group_by="Advisor", expected_filter_columns={"Standing"}),
    Scenario("which major has the lowest average days absent", {"groupby_average"}, expected_value_column="Days Absent", expected_group_by="Major"),
    Scenario("show seniors with attendance below 90% and GPA below 2.5", {"filtered_preview"}, expected_filter_columns={"Year", "Attendance Rate", "GPA"}),
    Scenario("how many health administration students need attendance support", {"count_rows", "filtered_preview"}, expected_filter_columns={"Major", "Attendance Category"}),
    Scenario("compare Dr. Nadia Pierce and Dr. Victor Ford students", {"cohort_comparison"}, expected_group_by="Advisor", expected_filter_columns={"Advisor"}),
    Scenario("what summary can you give for Prof. Omar Sloan students", {"cohort_summary"}, expected_filter_columns={"Advisor"}),
    Scenario("which advisor has the lowest average SAT Total", {"groupby_average"}, expected_value_column="SAT Total", expected_group_by="Advisor"),
    Scenario("average PSAT Math by year for Engineering students", {"groupby_average"}, expected_value_column="PSAT Math", expected_group_by="Year", expected_filter_columns={"Discipline"}),
    Scenario("show students with SAT Total below 1000 and attendance below 90%", {"filtered_preview"}, expected_filter_columns={"SAT Total", "Attendance Rate"}),
    Scenario("count bad standing students by location", {"groupby_count"}, expected_group_by="Location", expected_filter_columns={"Standing"}),
    Scenario("which discipline has the most students with GPA below 2.5", {"groupby_count"}, expected_group_by="Discipline", expected_filter_columns={"GPA"}),
    Scenario("list students at online campus with no second major", {"filtered_preview"}, expected_filter_columns={"Location", "Second Major"}),
    Scenario("how many students have days absent over 10 by year", {"groupby_count"}, expected_group_by="Year", expected_filter_columns={"Days Absent"}),
    Scenario("average GPA by location for students needing attendance support", {"groupby_average"}, expected_value_column="GPA", expected_group_by="Location", expected_filter_columns={"Attendance Category"}),
    Scenario("show top 5 students by SAT Math", {"filtered_preview"}),
    Scenario("show bottom 5 students by Attendance Rate", {"filtered_preview"}),
    Scenario("what majors are represented among bad standing students", {"list_unique"}, expected_value_column="Major", expected_filter_columns={"Standing"}),
    Scenario("how many students are in good standing but attendance below 90%", {"count_rows"}, expected_filter_columns={"Standing", "Attendance Rate"}),
    Scenario("show nursing majors with bad standing", {"filtered_preview"}, expected_filter_columns={"Major", "Standing"}),
]


def _selected_sheet(loaded) -> str:
    for name in loaded.sheets:
        if name.lower() in {"student roster", "students"}:
            return name
    return next(iter(loaded.sheets))


def _settings() -> dict[str, Any]:
    settings = load_settings()
    settings.update(
        {
            "strict_privacy_mode": False,
            "use_local_llm": True,
            "conversation_llm_enabled": False,
            "llm_explanations_enabled": False,
            "planner_model": "llama3.2:3b",
        }
    )
    return settings


def _problems(scenario: Scenario, routing: dict[str, Any], response: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    plan = routing.get("plan") or {}
    intent = routing.get("intent")
    if scenario.allow_clarify:
        if intent != "clarify":
            problems.append(f"expected clarify, got {intent}")
        return problems
    if intent in {"clarify", "unsupported", "unavailable"}:
        problems.append(f"unexpected {intent}: {routing.get('confirmation_reason')}")
        return problems
    if not response.get("success"):
        problems.append(f"execution failed: {response.get('message')}")
    operation = plan.get("operation") or response.get("operation")
    if scenario.expected_operations and operation not in scenario.expected_operations:
        problems.append(f"operation {operation!r} not in {sorted(scenario.expected_operations)}")
    if scenario.expected_value_column and plan.get("value_column") != scenario.expected_value_column:
        problems.append(f"value_column {plan.get('value_column')!r} != {scenario.expected_value_column!r}")
    if scenario.expected_group_by and plan.get("group_by") != scenario.expected_group_by:
        problems.append(f"group_by {plan.get('group_by')!r} != {scenario.expected_group_by!r}")
    actual_filters = {f.get("column") for f in (plan.get("filters") or [])}
    missing = scenario.expected_filter_columns - actual_filters
    if missing:
        problems.append(f"missing filter columns {sorted(missing)}; got {sorted(actual_filters)}")
    return problems


def run(workbook: Path) -> dict[str, Any]:
    loaded = load_excel_workbook(Upload(workbook))
    selected = _selected_sheet(loaded)
    sheet_columns = {name: list(frame.columns) for name, frame in loaded.sheets.items()}
    settings = _settings()

    results = []
    failures = []
    fallback_count = 0
    llm_count = 0
    for index, scenario in enumerate(SCENARIOS, start=1):
        started = time.monotonic()
        routing = plan_user_request(
            user_message=scenario.question,
            sheets=loaded.sheets,
            sheet_columns=sheet_columns,
            selected_sheet=selected,
            conversation_state={},
            settings=settings,
        )
        response = execute_planned_request(
            routing,
            loaded,
            settings,
            request_summary=scenario.question,
        )
        problems = _problems(scenario, routing, response)
        item = {
            "index": index,
            "question": scenario.question,
            "ok": not problems,
            "problems": problems,
            "intent": routing.get("intent"),
            "plan_source": routing.get("plan_source"),
            "llm_used": routing.get("llm_used"),
            "fallback_reason": routing.get("fallback_reason"),
            "validation": routing.get("validation"),
            "plan": routing.get("plan"),
            "operation": response.get("operation"),
            "row_count": response.get("row_count"),
            "value": response.get("value"),
            "message": response.get("message"),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        if item["fallback_reason"]:
            fallback_count += 1
        if item["llm_used"]:
            llm_count += 1
        if problems:
            failures.append(item)
        results.append(item)
        print(
            f"{index:02d}. {'OK' if item['ok'] else 'FAIL'} {scenario.question} -> "
            f"{item['plan_source']} {item['operation']} rows={item['row_count']} "
            f"value={item['value']} fallback={item['fallback_reason'] or '-'} "
            f"{'; '.join(problems)}",
            flush=True,
        )

    return {
        "workbook": str(workbook),
        "selected_sheet": selected,
        "question_count": len(SCENARIOS),
        "llm_count": llm_count,
        "fallback_count": fallback_count,
        "failure_count": len(failures),
        "failures": failures,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run(args.workbook)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Wrote {args.out} | questions={report['question_count']} "
        f"llm={report['llm_count']} fallbacks={report['fallback_count']} "
        f"failures={report['failure_count']}"
    )


if __name__ == "__main__":
    main()
