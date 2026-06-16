"""Headless UI test of the unified assistant using Streamlit's AppTest.

Runs the real app.py with a seeded Editor login and a loaded workbook (no
browser, no network/LLM), then drives the chat through each path and asserts
the right view renders without exceptions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

WORKBOOK = REPO_ROOT / "outputs" / "mock_dean_student_roster_edited_20260519_180232_331696.xlsx"


class FakeUpload:
    def __init__(self, path: Path):
        self._bytes = path.read_bytes()
        self.name = path.name

    def getvalue(self) -> bytes:
        return self._bytes


# Replace the file_uploader widget with a plain stub that always returns our
# workbook. This keeps the upload loaded across reruns (the real widget resets
# its session_state key to None each run, which a headless test cannot satisfy)
# and frees the "workbook_upload" key so the harness can seed it directly.
st.file_uploader = lambda *args, **kwargs: FakeUpload(WORKBOOK)


def _seed(at: AppTest) -> None:
    # current_user persists across reruns; workbook_upload is a file_uploader
    # key that Streamlit resets to None each rerun, so re-seed before every run
    # (main() reads it before the widget instantiates).
    at.session_state["current_user"] = {"username": "tester", "role": "Editor"}
    at.session_state["workbook_upload"] = FakeUpload(WORKBOOK)


def fresh_app() -> AppTest:
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    _seed(at)
    at.run()
    return at


def submit_text(at: AppTest, text: str) -> AppTest:
    at.chat_input[0].set_value(text)
    _seed(at)
    at.run()
    return at


def click_suggestion(at: AppTest, label: str) -> AppTest:
    for button in at.button:
        if button.label == label:
            button.click()
            break
    else:
        raise AssertionError(f"suggestion not found: {label}")
    _seed(at)
    at.run()
    return at


def ss(at: AppTest, key: str, default=None):
    return at.session_state[key] if key in at.session_state else default


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def main() -> int:
    if not WORKBOOK.exists():
        print(f"FAIL: workbook missing: {WORKBOOK}")
        return 2

    ok = True

    print("1) ASK question (suggestion button)")
    at = fresh_app()
    ok &= check("app booted without exception", not at.exception, str(at.exception))
    at = submit_text(at, "How many students have a GPA below 3.0?")
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("mode is ask_question", ss(at, "assistant_mode") == "ask_question",
                str(ss(at, "assistant_mode")))
    ok &= check("computed a result", bool(ss(at, "ask_description")),
                str(ss(at, "ask_description")))
    ok &= check("row_count is 177", ss(at, "ask_row_count") == 177,
                str(ss(at, "ask_row_count")))

    print("\n2) CHART request — populates the Figures panel (Phase K)")
    at = submit_text(at, "Create a chart of students by Major")
    ok &= check("no exception", not at.exception, str(at.exception))
    figure = ss(at, "latest_figure") or {}
    ok &= check("latest_figure populated", bool(figure), str(figure.get("title")))
    ok &= check("chart field is Major", figure.get("field") == "Major", str(figure.get("field")))
    # The chart now goes to the Figures panel without modifying the workbook,
    # so there should be no pending edit plan.
    ok &= check("no edit_workbook plan staged",
                ss(at, "assistant_mode") != "edit_workbook",
                str(ss(at, "assistant_mode")))

    print("\n2b) EDIT correctly BLOCKED — numeric highlight on string-typed GPA")
    at = submit_text(at, "Highlight students with a GPA below 3.0")
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("planned highlight_rows", (ss(at, "current_plan") or {}).get("commands", [{}])[0].get("action") == "highlight_rows")
    ok &= check("execution safely blocked (validation error)", ss(at, "current_can_execute") is False,
                str(ss(at, "current_validation_error")))

    print("\n3) CLARIFY (vague text)")
    at = submit_text(at, "Clean this up")
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("mode is clarify", ss(at, "assistant_mode") == "clarify",
                str(ss(at, "assistant_mode")))
    ok &= check("clarify question present", bool(ss(at, "clarify_question")),
                str(ss(at, "clarify_question")))

    print("\n4) FOLLOW-UP ('highlight them' after an ask)")
    at = fresh_app()
    at = submit_text(at, "What columns have missing values?")
    ok &= check("ask recorded in memory", bool(ss(at, "assistant_memory", {}).get("last_filters") is not None
                                               or ss(at, "ask_row_count") is not None))
    at = submit_text(at, "Highlight them")
    ok &= check("no exception", not at.exception, str(at.exception))
    mode = ss(at, "assistant_mode")
    # 'missing values' question has no row filter, so 'them' may resolve to a
    # whole-sheet result or ask to clarify — both are acceptable, not a crash.
    ok &= check("follow-up handled (edit or clarify)", mode in {"edit_workbook", "clarify"}, str(mode))

    print("\n5) FOLLOW-UP with a real filter then 'highlight them'")
    at = fresh_app()
    at = submit_text(at, "How many students have a GPA below 3.0?")
    ok &= check("filtered ask recorded", ss(at, "ask_row_count") == 177,
                str(ss(at, "ask_row_count")))
    at = submit_text(at, "Highlight them")
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("mode is edit_workbook", ss(at, "assistant_mode") == "edit_workbook", str(ss(at, "assistant_mode")))
    plan = ss(at, "current_plan") or {}
    cmd = (plan.get("commands") or [{}])[0]
    ok &= check("highlight reuses GPA<3 filter",
                cmd.get("action") == "highlight_rows"
                and cmd.get("conditions", [{}])[0].get("column") == "GPA",
                str(cmd))

    print("\n" + ("ALL PASS" if ok else "SOME FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
