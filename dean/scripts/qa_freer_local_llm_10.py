from __future__ import annotations

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


WORKBOOK = Path(
    "/Users/albertopaz/azera-formatting/"
    "mock_dean_student_roster_with_assessments_attendance_cleaned.xlsx"
)
OUT = Path("outputs/qa_freer_local_llm_10_report.json")


@dataclass
class Upload:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


ADVISOR_BUILDING_BLOCKS = [
    "Show me students who need attendance support.",
    "Of those students, which advisors have the most cases?",
    "For Dr. Alan Meyer's group, who also has GPA below 2.5?",
    "Summarize why those students may need advisor attention.",
    "What should I check next for this group?",
]

GENERAL_QUESTIONS = [
    "How many students are in the workbook?",
    "What is the average GPA by major?",
    "Which campus location has the most students?",
    "Show the bottom 5 students by Attendance Rate.",
    "What majors are represented among bad standing students?",
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
            "llm_enabled": True,
            "llm_explanations_enabled": True,
            "conversation_llm_enabled": True,
            "planner_full_row_access": True,
            "local_llm_full_row_access": True,
            "local_llm_all_matching_rows": True,
            "planner_model": "llama3.2:3b",
            "explanation_model": "llama3.2:3b",
            "planner_timeout_seconds": 300,
            "max_retries": 1,
        }
    )
    return settings


def _update_state(state: dict[str, Any], routing: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    active = routing.get("active_update")
    if isinstance(active, dict):
        state.update(
            {
                "active_filters": active.get("filters") or [],
                "active_sort": active.get("sort") or {},
                "active_group_by": active.get("group_by") or "",
                "active_limit": active.get("limit"),
                "active_sheet": active.get("sheet") or state.get("active_sheet", ""),
                "last_operation": active.get("operation") or response.get("operation") or "",
            }
        )
    if response.get("top_group"):
        state["last_top_group"] = response["top_group"]
    return state


def _run_one(
    *,
    question: str,
    label: str,
    loaded,
    selected: str,
    sheet_columns: dict[str, list[str]],
    settings: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    routing = plan_user_request(
        user_message=question,
        sheets=loaded.sheets,
        sheet_columns=sheet_columns,
        selected_sheet=selected,
        conversation_state=state,
        settings=settings,
    )
    response = execute_planned_request(
        routing,
        loaded,
        settings,
        request_summary=question,
    )
    next_state = _update_state(dict(state), routing, response)
    item = {
        "label": label,
        "question": question,
        "success": bool(response.get("success")),
        "intent": routing.get("intent"),
        "plan_source": routing.get("plan_source"),
        "llm_used": bool(routing.get("llm_used")),
        "conversation_llm_used": bool(response.get("conversation_llm_used")),
        "fallback_reason": routing.get("fallback_reason"),
        "validation": routing.get("validation"),
        "plan": routing.get("plan"),
        "operation": response.get("operation"),
        "row_count": response.get("row_count"),
        "value": response.get("value"),
        "removed": response.get("removed"),
        "message": response.get("message"),
        "preview_rows": len(response.get("result_preview") or []),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    return item, next_state


def main() -> None:
    loaded = load_excel_workbook(Upload(WORKBOOK))
    selected = _selected_sheet(loaded)
    sheet_columns = {name: list(frame.columns) for name, frame in loaded.sheets.items()}
    settings = _settings()

    results: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    for question in ADVISOR_BUILDING_BLOCKS:
        item, state = _run_one(
            question=question,
            label="advisor_building_block",
            loaded=loaded,
            selected=selected,
            sheet_columns=sheet_columns,
            settings=settings,
            state=state,
        )
        results.append(item)
        print(
            f"{len(results):02d}. {item['label']} | {item['plan_source']} | "
            f"{item['operation']} | rows={item['row_count']} | "
            f"llm={item['llm_used']} conv={item['conversation_llm_used']} | "
            f"{item['elapsed_seconds']}s\n    {item['message']}",
            flush=True,
        )

    for question in GENERAL_QUESTIONS:
        item, _ = _run_one(
            question=question,
            label="general",
            loaded=loaded,
            selected=selected,
            sheet_columns=sheet_columns,
            settings=settings,
            state={},
        )
        results.append(item)
        print(
            f"{len(results):02d}. {item['label']} | {item['plan_source']} | "
            f"{item['operation']} | rows={item['row_count']} | "
            f"llm={item['llm_used']} conv={item['conversation_llm_used']} | "
            f"{item['elapsed_seconds']}s\n    {item['message']}",
            flush=True,
        )

    failures = [
        item for item in results
        if not item["success"] or item["intent"] in {"clarify", "unsupported", "unavailable"}
    ]
    report = {
        "workbook": str(WORKBOOK),
        "selected_sheet": selected,
        "settings": {
            key: settings.get(key)
            for key in (
                "strict_privacy_mode",
                "use_local_llm",
                "conversation_llm_enabled",
                "planner_full_row_access",
                "local_llm_full_row_access",
                "local_llm_all_matching_rows",
                "planner_timeout_seconds",
            )
        },
        "question_count": len(results),
        "llm_count": sum(1 for item in results if item["llm_used"]),
        "conversation_llm_count": sum(1 for item in results if item["conversation_llm_used"]),
        "failure_count": len(failures),
        "failures": failures,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Wrote {OUT} | questions={report['question_count']} "
        f"llm={report['llm_count']} conversation={report['conversation_llm_count']} "
        f"failures={report['failure_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
