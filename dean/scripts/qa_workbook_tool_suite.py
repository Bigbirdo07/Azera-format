from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.data_sources import DataSourceRegistry
from core.excel_loader import LoadedWorkbook, load_excel_workbook
from core.execution_dispatcher import execute_planned_request
from nlp.planner_router import plan_user_request
from ui.figures_panel import detect_chart_intent, is_chart_request


DEFAULT_WORKBOOK = Path(
    "/Users/albertopaz/azera-formatting/"
    "mock_dean_student_roster_with_assessments_attendance_cleaned.xlsx"
)
DEFAULT_REPORT = Path("outputs/qa_workbook_tool_suite_report.json")


@dataclass(frozen=True)
class Scenario:
    category: str
    question: str
    expected_kind: str
    expected_operation: str = ""
    expected_intent: str = "query"
    expected_chart_metric: str = ""
    expected_chart_field: str = ""
    expected_chart_type: str = ""
    required_preview_columns: set[str] = field(default_factory=set)


@dataclass
class Upload:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


SCENARIOS = [
    # Intervention/advisor insights
    Scenario("insight", "Who should receive intervention?", "query", "student_intervention_summary", required_preview_columns={"Intervention Signals", "Review Reason"}),
    Scenario("insight", "Which students need advisor attention?", "query", "student_intervention_summary", required_preview_columns={"Intervention Signals"}),
    Scenario("insight", "Show students who need support based on GPA and attendance", "query", "student_intervention_summary", required_preview_columns={"Review Reason"}),
    Scenario("insight", "Which advisors are doing a good job?", "query", "advisor_outcome_summary", required_preview_columns={"Outcome Score"}),
    Scenario("insight", "Which advisors need support?", "query", "advisor_outcome_summary", required_preview_columns={"Outcome Score"}),
    Scenario("insight", "Rank advisors by student outcomes", "query", "advisor_outcome_summary", required_preview_columns={"Outcome Score"}),
    Scenario("insight", "Which advisor groups have the strongest student performance?", "query", "advisor_outcome_summary", required_preview_columns={"Outcome Score"}),
    Scenario("insight", "Which advisor groups have the most risk?", "query", "advisor_outcome_summary", required_preview_columns={"Outcome Score"}),
    Scenario("insight", "Find the highest risk students for review", "query", "student_intervention_summary", required_preview_columns={"Intervention Signals"}),
    Scenario("insight", "Give me an intervention review list", "query", "student_intervention_summary", required_preview_columns={"Review Reason"}),

    # Pivot tables
    Scenario("pivot", "Create a pivot table of standing by advisor", "query", "pivot_table_summary"),
    Scenario("pivot", "Pivot average GPA by advisor and year", "query", "pivot_table_summary"),
    Scenario("pivot", "Create a pivot table of year by major", "query", "pivot_table_summary"),
    Scenario("pivot", "Pivot attendance category by advisor", "query", "pivot_table_summary"),
    Scenario("pivot", "Pivot average SAT Total by major and year", "query", "pivot_table_summary"),
    Scenario("pivot", "Make a pivot table of location by standing", "query", "pivot_table_summary"),
    Scenario("pivot", "Pivot average Attendance Rate by advisor and standing", "query", "pivot_table_summary"),
    Scenario("pivot", "Create pivot of discipline by location", "query", "pivot_table_summary"),
    Scenario("pivot", "Pivot average Days Absent by major and year", "query", "pivot_table_summary"),
    Scenario("pivot", "Create a pivot table by advisor", "query", "pivot_table_summary"),

    # Trends
    Scenario("trend", "Find trends in this workbook", "query", "trend_summary"),
    Scenario("trend", "Show me the strongest trends", "query", "trend_summary"),
    Scenario("trend", "Identify patterns in student outcomes", "query", "trend_summary"),
    Scenario("trend", "Find GPA and attendance trends", "query", "trend_summary"),
    Scenario("trend", "Show trends by major and location", "query", "trend_summary"),
    Scenario("trend", "Find outliers in this workbook", "query", "trend_summary"),
    Scenario("trend", "Summarize the trends for advisor review", "query", "trend_summary"),
    Scenario("trend", "What patterns stand out in the data?", "query", "trend_summary"),
    Scenario("trend", "Find academic performance trends", "query", "trend_summary"),
    Scenario("trend", "Identify attendance and GPA patterns", "query", "trend_summary"),

    # Figures
    Scenario("figure", "Create a figure showing average GPA by major", "figure", expected_chart_metric="average", expected_chart_field="Major", expected_chart_type="bar"),
    Scenario("figure", "Make a bar chart of students by standing", "figure", expected_chart_metric="count", expected_chart_field="Standing", expected_chart_type="bar"),
    Scenario("figure", "Show GPA distribution", "figure", expected_chart_field="GPA", expected_chart_type="histogram"),
    Scenario("figure", "Create a chart of students by advisor", "figure", expected_chart_metric="count", expected_chart_field="Advisor", expected_chart_type="bar"),
    Scenario("figure", "Make a pie chart of standing", "figure", expected_chart_metric="count", expected_chart_field="Standing", expected_chart_type="pie"),
    Scenario("figure", "Create a figure of average Attendance Rate by advisor", "figure", expected_chart_metric="average", expected_chart_field="Advisor", expected_chart_type="bar"),
    Scenario("figure", "Plot average SAT Total by major", "figure", expected_chart_metric="average", expected_chart_field="Major", expected_chart_type="bar"),
    Scenario("figure", "Graph students by location", "figure", expected_chart_metric="count", expected_chart_field="Location", expected_chart_type="bar"),
    Scenario("figure", "Visualize average Days Absent by year", "figure", expected_chart_metric="average", expected_chart_field="Year", expected_chart_type="bar"),
    Scenario("figure", "Create a bar graph of attendance category", "figure", expected_chart_metric="count", expected_chart_field="Attendance Category", expected_chart_type="bar"),

    # Dashboard/report/export
    Scenario("dashboard", "Build me a dashboard report for advisor intervention review", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Create an advisor intervention dashboard", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Generate a support dashboard report", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Compile a risk and trend dashboard workbook", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Prepare an advisor outcome report workbook", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Build a report with intervention review and trends", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Make a dashboard with pivots and figures", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Create a workbook packet for advisor support review", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Generate an intervention dashboard with charts", "dashboard", expected_intent="dashboard_report"),
    Scenario("dashboard", "Prepare the dean dashboard report", "dashboard", expected_intent="dashboard_report"),
]


def _selected_sheet(loaded) -> str:
    for name in loaded.sheets:
        if name.lower() in {"student roster", "students", "roster"}:
            return name
    return next(iter(loaded.sheets))


def _enriched_workbook(loaded: LoadedWorkbook) -> tuple[LoadedWorkbook, str]:
    registry = DataSourceRegistry()
    registry.set_roster(loaded)
    enriched_sheets = registry.enriched_sheets() or dict(loaded.sheets)
    selected = registry.enriched_roster_sheet or _selected_sheet(loaded)
    enriched = LoadedWorkbook(
        file_name=loaded.file_name,
        workbook=loaded.workbook,
        sheets=enriched_sheets,
        warnings=list(loaded.warnings or []),
    )
    return enriched, selected


def _settings(use_llm: bool = False) -> dict[str, Any]:
    return {
        "strict_privacy_mode": not use_llm,
        "use_local_llm": use_llm,
        "llm_enabled": use_llm,
        "conversation_llm_enabled": False,
        "llm_explanations_enabled": False,
        "planner_full_row_access": False,
        "local_llm_full_row_access": False,
        "local_llm_all_matching_rows": False,
        "planner_model": "llama3.2:3b",
        "planner_timeout_seconds": 300,
    }


def _preview_columns(response: dict[str, Any]) -> set[str]:
    preview = response.get("result_preview") or []
    if not preview:
        return set()
    return set(preview[0].keys())


def _evaluate_chart(scenario: Scenario, columns: list[str]) -> tuple[dict[str, Any], list[str]]:
    item: dict[str, Any] = {"kind": "figure"}
    problems: list[str] = []
    missing_average_value = _missing_average_value_concept(scenario.question, columns)
    if missing_average_value:
        item["not_applicable"] = True
        item["not_applicable_reason"] = (
            f"workbook has no column matching average value {missing_average_value!r}"
        )
        return item, problems
    if scenario.expected_chart_field and not _has_concept(columns, scenario.expected_chart_field):
        item["not_applicable"] = True
        item["not_applicable_reason"] = (
            f"workbook has no column matching {scenario.expected_chart_field!r}"
        )
        return item, problems
    if not is_chart_request(scenario.question):
        problems.append("not detected as a figure request")
        return item, problems
    intent = detect_chart_intent(scenario.question, columns)
    item["chart_intent"] = intent.__dict__ if intent else None
    if intent is None:
        problems.append("figure intent did not resolve")
        return item, problems
    if scenario.expected_chart_field and not _same_concept(intent.field, scenario.expected_chart_field):
        problems.append(f"chart field {intent.field!r} != {scenario.expected_chart_field!r}")
    if scenario.expected_chart_metric and intent.metric != scenario.expected_chart_metric:
        problems.append(f"chart metric {intent.metric!r} != {scenario.expected_chart_metric!r}")
    if scenario.expected_chart_type and intent.chart_type != scenario.expected_chart_type:
        problems.append(f"chart type {intent.chart_type!r} != {scenario.expected_chart_type!r}")
    return item, problems


def _same_concept(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    actual_norm = _norm(actual)
    expected_norm = _norm(expected)
    aliases = {
        "advisor": {"advisor", "advisor counselor", "counselor", "teacher", "professor"},
        "major": {"major", "major program", "program", "department", "discipline"},
        "standing": {"standing", "academic standing", "academic status", "status"},
        "year": {"year", "grade", "grade level", "yr grade", "class year"},
        "gpa": {"gpa", "current gpa", "cumulative gpa", "grade point average"},
        "attendance rate": {"attendance rate", "attendance", "attendance percent", "attendance percentage"},
        "attendance category": {"attendance category", "attendance status", "attendance categories", "attendance statuses"},
        "location": {"location", "campus", "site"},
        "days absent": {"days absent", "absences", "total absences"},
    }
    for values in aliases.values():
        if expected_norm in values and (actual_norm in values or any(v in actual_norm for v in values)):
            return True
    return expected_norm in actual_norm or actual_norm in expected_norm


def _has_concept(columns: list[str], expected: str) -> bool:
    return any(_same_concept(column, expected) for column in columns)


def _missing_average_value_concept(question: str, columns: list[str]) -> str:
    match = re.search(
        r"\b(?:average|avg|mean)\s+([a-z ]+?)\s+(?:by|per)\s+([a-z ]+)",
        question.lower(),
    )
    if not match:
        return ""
    value_phrase = match.group(1).strip()
    return "" if _has_concept(columns, value_phrase) else value_phrase


def _norm(value: str) -> str:
    return " ".join(str(value).replace("/", " ").replace("-", " ").replace("_", " ").lower().split())


def run_suite(workbook: Path, *, use_llm: bool = False) -> dict[str, Any]:
    loaded, selected = _enriched_workbook(load_excel_workbook(Upload(workbook)))
    sheet_columns = {name: list(frame.columns) for name, frame in loaded.sheets.items()}
    settings = _settings(use_llm)
    results = []
    failures = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for index, scenario in enumerate(SCENARIOS, start=1):
            started = time.monotonic()
            item: dict[str, Any] = {
                "index": index,
                "category": scenario.category,
                "question": scenario.question,
            }
            problems: list[str] = []
            try:
                if scenario.expected_kind == "figure":
                    extra, problems = _evaluate_chart(scenario, sheet_columns[selected])
                    item.update(extra)
                else:
                    routing = plan_user_request(
                        user_message=scenario.question,
                        sheets=loaded.sheets,
                        sheet_columns=sheet_columns,
                        selected_sheet=selected,
                        conversation_state={},
                        settings=settings,
                    )
                    item.update({
                        "intent": routing.get("intent"),
                        "plan_source": routing.get("plan_source"),
                        "llm_used": bool(routing.get("llm_used")),
                        "pending_type": routing.get("pending_type"),
                        "requires_confirmation": bool(routing.get("requires_confirmation")),
                        "plan": routing.get("plan"),
                    })
                    if routing.get("intent") != scenario.expected_intent:
                        problems.append(f"intent {routing.get('intent')!r} != {scenario.expected_intent!r}")
                    if scenario.expected_kind == "dashboard":
                        if not routing.get("requires_confirmation"):
                            problems.append("dashboard did not require confirmation")
                        # Exercise the actual writer through a temp outputs dir by
                        # calling the action directly after route validation.
                        from core.confirmed_actions import execute_dashboard_report_action

                        result = execute_dashboard_report_action(
                            sheets=loaded.sheets,
                            sheet=selected,
                            request_summary=scenario.question,
                            output_dir=tmp_path,
                            audit_path=tmp_path / "audit.jsonl",
                        )
                        item["action_success"] = result.success
                        item["output_file"] = result.output_file
                        item["rows_affected"] = result.rows_affected
                        if not result.success or not result.output_file:
                            problems.append("dashboard workbook was not created")
                    else:
                        response = execute_planned_request(
                            routing,
                            loaded,
                            settings,
                            request_summary=scenario.question,
                        )
                        item.update({
                            "success": bool(response.get("success")),
                            "operation": response.get("operation"),
                            "row_count": response.get("row_count"),
                            "value": response.get("value"),
                            "message": response.get("message"),
                            "preview_columns": sorted(_preview_columns(response)),
                        })
                        plan = routing.get("plan") or {}
                        operation = plan.get("operation") or response.get("operation")
                        if scenario.expected_operation and operation != scenario.expected_operation:
                            problems.append(f"operation {operation!r} != {scenario.expected_operation!r}")
                        if not response.get("success"):
                            problems.append(f"execution failed: {response.get('message')}")
                        missing = scenario.required_preview_columns - _preview_columns(response)
                        if missing:
                            problems.append(f"missing preview columns {sorted(missing)}")
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{type(exc).__name__}: {exc}")
            item["elapsed_seconds"] = round(time.monotonic() - started, 3)
            item["ok"] = not problems
            item["problems"] = problems
            if problems:
                failures.append(item)
            results.append(item)
            print(
                f"{index:02d}. {'OK' if item['ok'] else 'FAIL'} "
                f"[{scenario.category}] {scenario.question} "
                f"op={item.get('operation') or item.get('intent') or item.get('kind')} "
                f"{'; '.join(problems)}",
                flush=True,
            )

    by_category: dict[str, dict[str, int]] = {}
    not_applicable_count = 0
    for item in results:
        bucket = by_category.setdefault(item["category"], {"total": 0, "failures": 0})
        bucket["total"] += 1
        if item.get("not_applicable"):
            not_applicable_count += 1
        if not item["ok"]:
            bucket["failures"] += 1
    return {
        "workbook": str(workbook),
        "selected_sheet": selected,
        "use_llm": use_llm,
        "question_count": len(results),
        "failure_count": len(failures),
        "not_applicable_count": not_applicable_count,
        "by_category": by_category,
        "failures": failures,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()
    report = run_suite(args.workbook, use_llm=args.use_llm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Wrote {args.out} | questions={report['question_count']} "
        f"failures={report['failure_count']} by_category={report['by_category']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
