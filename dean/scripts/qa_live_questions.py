from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.excel_loader import load_excel_workbook
from core.execution_dispatcher import execute_planned_request
from nlp.planner_router import plan_user_request
from ui.settings_panel import load_settings


DEFAULT_WORKBOOK = Path("backups/mock_dean_student_roster_backup_20260519_180123_629899.xlsx")
DEFAULT_REPORT = Path("outputs/qa_live_questions_report.json")

QUESTIONS = [
    "how many seniors are there",
    "show me all seniors",
    "show freshmen with bad standing",
    "how many students are in Health Administration",
    "show me all students in Health Administration",
    "group Health Administration students by advisor",
    "which advisor has the most Health Administration students",
    "top 10 students by GPA",
    "bottom 10 students by GPA",
    "who has GPA below 2.0",
    "how many students have GPA below 2.5",
    "average GPA by major",
    "which major has the worst average GPA",
    "which discipline has the highest average GPA",
    "count students by standing",
    "count students by year",
    "count students by discipline",
    "count students by location",
    "show me Nursing students on Main Campus",
    "show Business juniors",
    "how many online students are there",
    "show students at North Campus",
    "show bad standing students by advisor",
    "which advisor has the most bad standing students",
    "show good standing seniors",
    "average GPA for seniors",
    "average GPA for freshmen",
    "show students majoring in Data Analytics",
    "how many Computer Engineering students are there",
    "group Computer Engineering students by year",
    "show Arts and Sciences students with GPA below 2.5",
    "show Education students on Academic Watch",
    "how many students are on Academic Watch",
    "show students not on Academic Watch",
    "show students with no second major",
    "how many students have a second major",
    "list every major",
    "how many different advisors are there",
    "which location has the most students",
    "show students advised by Dr. Nadia Pierce",
    "count Dr. Nadia Pierce students by major",
    "show Dr. Priya Shah students below 2.5 GPA",
    "which students are in bad standing and below 2.0 GPA",
    "show seniors in Engineering",
    "group seniors by discipline",
    "show Health Campus juniors",
    "average GPA by advisor",
    "which advisor has the lowest average GPA",
    "show the top 5 majors by student count",
    "show the bottom 5 majors by student count",
]


@dataclass
class Upload:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def _selected_sheet(loaded) -> str:
    for name in loaded.sheets:
        if name.lower() in {"student roster", "students"}:
            return name
    return next(iter(loaded.sheets))


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


def run_questions(
    workbook: Path,
    *,
    use_llm: bool,
    limit: int | None = None,
    start: int = 1,
) -> dict[str, Any]:
    loaded = load_excel_workbook(Upload(workbook))
    selected = _selected_sheet(loaded)
    sheet_columns = {name: list(frame.columns) for name, frame in loaded.sheets.items()}
    start = max(start, 1)
    selected_questions = QUESTIONS[start - 1:]
    questions = selected_questions[:limit] if limit else selected_questions
    settings = _settings(use_llm)

    rows = []
    failures = []
    fallback_count = 0
    llm_count = 0
    for index, question in enumerate(questions, start=start):
        started = time.monotonic()
        item: dict[str, Any] = {"index": index, "question": question}
        try:
            routing = plan_user_request(
                user_message=question,
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
                request_summary=question,
            )
            item.update(
                {
                    "intent": routing.get("intent"),
                    "plan_source": routing.get("plan_source"),
                    "llm_used": bool(routing.get("llm_used")),
                    "fallback_reason": routing.get("fallback_reason"),
                    "validation": routing.get("validation"),
                    "plan": routing.get("plan"),
                    "success": bool(response.get("success")),
                    "response_type": response.get("response_type"),
                    "operation": response.get("operation"),
                    "row_count": response.get("row_count"),
                    "value": response.get("value"),
                    "message": response.get("message"),
                    "removed": response.get("removed"),
                }
            )
            if routing.get("fallback_reason"):
                fallback_count += 1
            if routing.get("llm_used"):
                llm_count += 1
            if not response.get("success") or routing.get("intent") in {"clarify", "unsupported", "unavailable"}:
                failures.append(item)
        except Exception as exc:  # noqa: BLE001 - QA report should capture all exceptions
            item.update({"success": False, "exception": f"{type(exc).__name__}: {exc}"})
            failures.append(item)
        rows.append(item)
        elapsed = round(time.monotonic() - started, 2)
        item["elapsed_seconds"] = elapsed
        print(
            f"{index:02d}. {question} -> "
            f"{item.get('plan_source')} {item.get('operation')} "
            f"rows={item.get('row_count')} value={item.get('value')} "
            f"fallback={item.get('fallback_reason') or '-'} "
            f"elapsed={elapsed}s",
            flush=True,
        )

    return {
        "workbook": str(workbook),
        "selected_sheet": selected,
        "use_llm": use_llm,
        "question_count": len(questions),
        "llm_count": llm_count,
        "fallback_count": fallback_count,
        "failure_count": len(failures),
        "failures": failures,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    report = run_questions(args.workbook, use_llm=not args.no_llm, limit=args.limit, start=args.start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Wrote {args.out} | questions={report['question_count']} "
        f"llm={report['llm_count']} fallbacks={report['fallback_count']} "
        f"failures={report['failure_count']}"
    )


if __name__ == "__main__":
    main()
