"""End-to-end smoke test for the Phase J conversational loop.

Drives the real Streamlit app headlessly via AppTest and walks through a
multi-turn conversation, asserting that:

  - chat history grows monotonically (no turn is silently dropped),
  - active context accumulates across follow-ups,
  - filter/group questions never trigger an export on their own,
  - an export request produces a confirmation card inside chat,
  - confirming the export appends a download attachment and keeps prior turns.

No browser, no network, no LLM. The local workbook fixture is used.
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

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_students.xlsx"


class FakeUpload:
    def __init__(self, path: Path):
        self._bytes = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self) -> bytes:
        return self._bytes


# The real file_uploader resets to None on every rerun in headless mode; stub
# it so the workbook persists across turns.
st.file_uploader = lambda *args, **kwargs: FakeUpload(FIXTURE)


def _seed(at: AppTest) -> None:
    at.session_state["current_user"] = {"username": "tester", "role": "Editor"}
    at.session_state["workbook_upload"] = FakeUpload(FIXTURE)


def fresh_app() -> AppTest:
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    _seed(at)
    at.run()
    return at


def send(at: AppTest, text: str) -> AppTest:
    at.chat_input[0].set_value(text)
    _seed(at)
    at.run()
    return at


def messages(at: AppTest) -> list[dict]:
    return list(at.session_state["chat_messages"]) if "chat_messages" in at.session_state else []


def user_contents(at: AppTest) -> list[str]:
    return [m["content"] for m in messages(at) if m["role"] == "user"]


def last_assistant(at: AppTest) -> dict | None:
    for m in reversed(messages(at)):
        if m["role"] == "assistant":
            return m
    return None


def filter_columns(at: AppTest) -> set[str]:
    memory = at.session_state["assistant_memory"] if "assistant_memory" in at.session_state else {}
    return {f["column"] for f in (memory.get("active_filters") or [])}


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def main() -> int:
    if not FIXTURE.exists():
        print(f"FAIL: fixture missing: {FIXTURE}")
        return 2

    ok = True

    print("1) Fresh app + workbook loads")
    at = fresh_app()
    ok &= check("app booted without exception", not at.exception, str(at.exception))
    ok &= check("workbook present in cache", at.session_state["cached_loaded"] is not None)
    ok &= check("chat history starts empty", messages(at) == [])

    print("\n2) First question: 'Show me Accounting students'")
    at = send(at, "Show me Accounting students")
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("user message in history", "Show me Accounting students" in user_contents(at))
    asst = last_assistant(at)
    ok &= check("assistant replied", asst is not None and bool(asst.get("content")))
    ok &= check("result attachment present", (asst.get("attachment") or {}).get("type") == "result"
                if asst else False)
    ok &= check("active filter: Department", "Department" in filter_columns(at), str(filter_columns(at)))

    print("\n3) Follow-up: 'now only below 2.5 GPA'")
    at = send(at, "now only below 2.5 GPA")
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("both user prompts preserved", set(user_contents(at)) >= {
        "Show me Accounting students",
        "now only below 2.5 GPA",
    })
    ok &= check("context accumulates: Dept + GPA",
                filter_columns(at) == {"Department", "GPA"}, str(filter_columns(at)))

    print("\n4) Follow-up: 'now only seniors'")
    at = send(at, "now only seniors")
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("all three prompts preserved", set(user_contents(at)) >= {
        "Show me Accounting students",
        "now only below 2.5 GPA",
        "now only seniors",
    })

    print("\n5) Filter questions did NOT trigger export")
    ok &= check("no output file written", not at.session_state["latest_output_file"]
                if "latest_output_file" in at.session_state else True)
    download_attachments = [
        (m.get("attachment") or {}).get("type")
        for m in messages(at) if m["role"] == "assistant"
    ]
    ok &= check("no download cards in chat yet",
                "download" not in download_attachments,
                str(download_attachments))

    print("\n6) Analytical follow-up: 'what is their average GPA'")
    before = len(messages(at))
    at = send(at, "what is their average GPA")
    after = messages(at)
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("history grew", len(after) > before, f"{len(after)} vs {before}")
    # Earlier user turns survive.
    ok &= check("Accounting prompt still in history",
                "Show me Accounting students" in user_contents(at))

    print("\n7) Export request: 'export this list'")
    at = send(at, "export this list")
    asst = last_assistant(at)
    pending = at.session_state["assistant_memory"]["pending_action"] if "assistant_memory" in at.session_state else {}
    ok &= check("export needs confirmation", bool(pending),
                str(pending.get("type") if pending else None))
    ok &= check("confirmation card in chat",
                (asst.get("attachment") or {}).get("type") == "confirmation" if asst else False)

    print("\n8) Confirm the export")
    before = len(messages(at))
    at = send(at, "yes")
    after = messages(at)
    ok &= check("no exception", not at.exception, str(at.exception))
    ok &= check("history grew (success message appended)", len(after) > before)
    ok &= check("pending cleared",
                not (at.session_state["assistant_memory"].get("pending_action") or {}))

    final_assistants = [m for m in after if m["role"] == "assistant"]
    types = [(m.get("attachment") or {}).get("type") for m in final_assistants]
    has_download = "download" in types
    ok &= check("download card present somewhere in chat", has_download, str(types))

    # All prior user turns must still be visible — no replacement.
    ok &= check("conversation history is preserved across all turns",
                set(user_contents(at)) >= {
                    "Show me Accounting students",
                    "now only below 2.5 GPA",
                    "now only seniors",
                    "what is their average GPA",
                    "export this list",
                    "yes",
                })

    print("\n" + ("ALL PASS" if ok else "SOME FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
