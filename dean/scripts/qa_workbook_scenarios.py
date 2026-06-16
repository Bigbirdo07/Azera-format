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
DEFAULT_REPORT = Path("outputs/qa_workbook_scenarios_report.json")


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


def _settings(use_llm: bool) -> dict[str, Any]:
    settings = load_settings()
    settings.update(
        {
            "strict_privacy_mode": False,
            "use_local_llm": bool(use_llm),
            "conversation_llm_enabled": False,
            "llm_explanations_enabled": False,
        }
    )
    return settings


def _selected_sheet(loaded) -> str:
    for name in loaded.sheets:
        if name.lower() in {"student roster", "students"}:
            return name
    return next(iter(loaded.sheets))


def scenarios() -> list[Scenario]:
    s: list[Scenario] = []

    def add(question: str, ops=(), value="", group="", filters=(), clarify=False):
        s.append(
            Scenario(
                question=question,
                expected_operations=set(ops),
                expected_value_column=value,
                expected_group_by=group,
                expected_filter_columns=set(filters),
                allow_clarify=clarify,
            )
        )

    # Whole-roster counts and lists.
    add("how many students are there", {"count_rows"})
    add("show me all students", {"filtered_preview"})
    add("what majors are listed", {"list_unique"}, value="Major")
    add("list every major", {"list_unique"}, value="Major")
    add("what advisors do we have", {"list_unique"}, value="Advisor")
    add("how many different advisors are there", {"count_unique"}, value="Advisor")
    add("what years are listed", {"list_unique"}, value="Year")
    add("what locations are listed", {"list_unique"}, value="Location")
    add("what attendance categories are listed", {"list_unique"}, value="Attendance Category")

    # Basic categorical filters.
    for year in ("Freshman", "Sophomore", "Junior", "Senior"):
        add(f"how many {year.lower()}s are there", {"count_rows"}, filters={"Year"})
        add(f"show me all {year.lower()}s", {"filtered_preview"}, filters={"Year"})
        add(f"average GPA for {year.lower()}s", {"average_column"}, value="GPA", filters={"Year"})
    for standing in ("Good Standing", "Bad Standing"):
        add(f"how many students are in {standing}", {"count_rows"}, filters={"Standing"})
        add(f"show {standing} students", {"filtered_preview"}, filters={"Standing"})
        add(f"group {standing} students by advisor", {"groupby_count"}, group="Advisor", filters={"Standing"})
    for location in ("North Campus", "Health Campus", "Main Campus", "Online", "Downtown Campus"):
        add(f"show students at {location}", {"filtered_preview"}, filters={"Location"})
        add(f"how many students are at {location}", {"count_rows"}, filters={"Location"})

    # Major, discipline, advisor cohorts.
    for major in (
        "Health Administration",
        "Computer Engineering",
        "Data Analytics",
        "Biological and Environmental Sciences",
    ):
        add(f"show me all students in {major}", {"filtered_preview"}, filters={"Major"})
        add(f"how many {major} students are there", {"count_rows"}, filters={"Major"})
        add(f"group {major} students by year", {"groupby_count"}, group="Year", filters={"Major"})
    add("show me all students in Nursing", clarify=True)
    add("how many Nursing students are there", clarify=True)
    add("group Nursing students by year", clarify=True)
    add("show students majoring in Nursing", {"filtered_preview"}, filters={"Major"})
    add("how many students majoring in Nursing", {"count_rows"}, filters={"Major"})
    for discipline in ("Health Sciences", "Business", "Engineering", "Education", "BES"):
        add(f"show {discipline} students", {"filtered_preview"}, filters={"Discipline"})
        add(f"average GPA for {discipline}", {"average_column"}, value="GPA", filters={"Discipline"})
    for advisor in ("Dr. Nadia Pierce", "Prof. Omar Sloan", "Dr. Priya Shah", "Dr. Victor Ford"):
        add(f"show students advised by {advisor}", {"filtered_preview"}, filters={"Advisor"})
        add(f"what summary can you say about {advisor}'s students", {"cohort_summary"}, filters={"Advisor"})
        add(f"count {advisor} students by major", {"groupby_count"}, group="Major", filters={"Advisor"})

    # Numeric GPA.
    add("top 10 students by GPA", {"filtered_preview"}, value="", filters=set())
    add("bottom 10 students by GPA", {"filtered_preview"})
    add("who has GPA below 2.0", {"filtered_preview"}, filters={"GPA"})
    add("how many students have GPA below 2.5", {"count_rows"}, filters={"GPA"})
    add("show students with GPA between 2.0 and 3.0", {"filtered_preview"}, filters={"GPA"})
    add("average GPA by major", {"groupby_average"}, value="GPA", group="Major")
    add("average GPA by advisor", {"groupby_average"}, value="GPA", group="Advisor")
    add("which major has the worst average GPA", {"groupby_average"}, value="GPA", group="Major")
    add("which advisor has the lowest average GPA", {"groupby_average"}, value="GPA", group="Advisor")

    # Assessment questions.
    for col in ("PSAT Math", "PSAT English", "PSAT Total", "SAT Math", "SAT English", "SAT Total"):
        add(f"average {col} by major", {"groupby_average"}, value=col, group="Major")
        add(f"show students with {col} below 500", {"filtered_preview"}, filters={col})
        add(f"top 10 students by {col}", {"filtered_preview"})
    add("which major has the highest average SAT Total", {"groupby_average"}, value="SAT Total", group="Major")
    add("which advisor has the lowest average PSAT Total", {"groupby_average"}, value="PSAT Total", group="Advisor")
    add("show students with SAT Math above 700 and GPA below 3.0", {"filtered_preview"}, filters={"SAT Math", "GPA"})
    add("show students with PSAT Total below 900", {"filtered_preview"}, filters={"PSAT Total"})

    # Attendance questions.
    add("show students with Needs Attendance Support", {"filtered_preview"}, filters={"Attendance Category"})
    add("how many students need attendance support", {"count_rows", "filtered_preview"}, filters={"Attendance Category"})
    add("average Attendance Rate by advisor", {"groupby_average"}, value="Attendance Rate", group="Advisor")
    add("which major has the lowest average Attendance Rate", {"groupby_average"}, value="Attendance Rate", group="Major")
    add("show students with Attendance Rate below 90%", {"filtered_preview"}, filters={"Attendance Rate"})
    add("show students with more than 10 days absent", {"filtered_preview"}, filters={"Days Absent"})
    add("average Days Absent by year", {"groupby_average"}, value="Days Absent", group="Year")
    add("top 10 students by Days Absent", {"filtered_preview"})

    # Combined / multi-filter.
    add("show seniors in Engineering", {"filtered_preview"}, filters={"Year", "Discipline"})
    add("show bad standing seniors with GPA below 2.5", {"filtered_preview"}, filters={"Standing", "Year", "GPA"})
    add("show Health Campus juniors", {"filtered_preview"}, filters={"Location", "Year"})
    add("show Nursing students on Main Campus", clarify=True)
    add("show students majoring in Nursing on Main Campus", {"filtered_preview"}, filters={"Major", "Location"})
    add("compare Dr. Nadia Pierce and Prof. Omar Sloan's students", {"cohort_comparison"}, group="Advisor", filters={"Advisor"})
    add("show students in Health Administration with Needs Attendance Support", {"filtered_preview"}, filters={"Major", "Attendance Category"})
    add("group students with GPA below 2.5 by advisor", {"groupby_count"}, group="Advisor", filters={"GPA"})
    add("group students with Attendance Rate below 90% by major", {"groupby_count"}, group="Major", filters={"Attendance Rate"})

    # Missing / second major / data quality.
    add("show students with no second major", {"filtered_preview"}, filters={"Second Major"})
    add("how many students have a second major", {"count_rows"}, filters={"Second Major"})
    add("which columns have missing values", {"missing_summary"})
    add("give me a data quality summary", {"data_quality_summary"})
    add("show duplicate students", {"duplicate_check", "filtered_preview"})

    # Expected clarifications / unsupported fields.
    add("how many students are on Academic Watch", clarify=True)
    add("show students on Attendance Watch", clarify=True)
    add("what is each student's housing status", clarify=True)

    return s


def _plan_failures(scenario: Scenario, routing: dict[str, Any], response: dict[str, Any]) -> list[str]:
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
    missing_filters = scenario.expected_filter_columns - actual_filters
    if missing_filters:
        problems.append(f"missing filter columns {sorted(missing_filters)}; got {sorted(actual_filters)}")
    if (
        operation == "filtered_preview"
        and not scenario.expected_filter_columns
        and "top" not in scenario.question.lower()
        and "bottom" not in scenario.question.lower()
        and "all students" not in scenario.question.lower()
        and response.get("row_count") == 300
    ):
        problems.append("suspicious whole-sheet preview")
    return problems


def run(workbook: Path, *, use_llm: bool, start: int = 1, limit: int | None = None) -> dict[str, Any]:
    loaded = load_excel_workbook(Upload(workbook))
    selected = _selected_sheet(loaded)
    sheet_columns = {name: list(frame.columns) for name, frame in loaded.sheets.items()}
    settings = _settings(use_llm)
    cases = scenarios()[max(start - 1, 0):]
    if limit:
        cases = cases[:limit]

    results = []
    failures = []
    for offset, scenario in enumerate(cases, start=start):
        started = time.monotonic()
        item: dict[str, Any] = {"index": offset, "question": scenario.question}
        try:
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
            problems = _plan_failures(scenario, routing, response)
            item.update(
                {
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
                }
            )
        except Exception as exc:  # noqa: BLE001
            item.update({"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]})
        item["elapsed_seconds"] = round(time.monotonic() - started, 2)
        if not item["ok"]:
            failures.append(item)
        results.append(item)
        print(
            f"{offset:03d}. {'OK' if item['ok'] else 'FAIL'} "
            f"{scenario.question} -> {item.get('plan_source')} {item.get('operation')} "
            f"rows={item.get('row_count')} value={item.get('value')} "
            f"fallback={item.get('fallback_reason') or '-'} "
            f"{'; '.join(item.get('problems') or [])}",
            flush=True,
        )

    return {
        "workbook": str(workbook),
        "selected_sheet": selected,
        "use_llm": use_llm,
        "question_count": len(cases),
        "failure_count": len(failures),
        "failures": failures,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report = run(args.workbook, use_llm=not args.no_llm, start=args.start, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Wrote {args.out} | questions={report['question_count']} "
        f"failures={report['failure_count']}"
    )


if __name__ == "__main__":
    main()
