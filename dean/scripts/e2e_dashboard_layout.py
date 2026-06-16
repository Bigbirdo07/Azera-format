"""End-to-end smoke test for the Phase K dashboard layout.

Drives the real Streamlit app headlessly and walks the user's full flow:
  upload → ask → follow up → chart → export, asserting that each of the five
  zones gets populated in the right session_state slot.
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


# Stub the real uploader so the test fixture persists across reruns.
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


def headers(at: AppTest) -> str:
    return " ".join(b.value.lower() for b in at.markdown if b.value)


def captions(at: AppTest) -> str:
    out: list[str] = []
    for cap in at.caption:
        try:
            val = cap.value
        except Exception:
            val = ""
        if val:
            out.append(val.lower())
    return " ".join(out)


def assistants(at: AppTest) -> list[dict]:
    msgs = at.session_state["chat_messages"] if "chat_messages" in at.session_state else []
    return [m for m in msgs if m["role"] == "assistant"]


def check(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def main() -> int:
    if not FIXTURE.exists():
        print(f"FAIL: fixture missing: {FIXTURE}")
        return 2

    ok = True

    print("1) App boots with the 5-zone layout")
    at = fresh_app()
    md = headers(at)
    cap = captions(at)
    blob = md + " " + cap
    ok &= check("Workbook zone present", "original workbook" in blob, detail=md[:80])
    ok &= check("Live Output zone present", "live output" in blob)
    ok &= check("Figures zone present", "figures" in blob)
    ok &= check("Export Center zone present", "export center" in blob)
    ok &= check("no exception", not at.exception, str(at.exception))

    print("\n2) Ask 'Show me Accounting students' — Live Output should populate")
    at = send(at, "Show me Accounting students")
    a = assistants(at)
    ok &= check("assistant replied", bool(a))
    last_attach = (a[-1].get("attachment") or {}) if a else {}
    ok &= check("Live Output has a result attachment", last_attach.get("type") == "result",
                str(last_attach.get("type")))
    ok &= check("active filter Department", "Department" in {f["column"] for f in (at.session_state["assistant_memory"].get("active_filters") or [])})

    print("\n3) Follow-up 'now only below 2.5 GPA' — context accumulates")
    at = send(at, "now only below 2.5 GPA")
    filters = {f["column"] for f in (at.session_state["assistant_memory"].get("active_filters") or [])}
    ok &= check("context Dept + GPA", filters == {"Department", "GPA"}, str(filters))
    users = [m["content"] for m in (at.session_state["chat_messages"]) if m["role"] == "user"]
    ok &= check("history retained both prompts",
                "Show me Accounting students" in users and "now only below 2.5 GPA" in users)

    print("\n4) Chart request 'Create a bar chart by advisor' — Figures panel")
    at = send(at, "Create a bar chart by advisor")
    figure = at.session_state["latest_figure"] if "latest_figure" in at.session_state else None
    ok &= check("latest_figure populated", figure is not None,
                str(figure.get("title") if figure else None))
    ok &= check("chart type bar", (figure or {}).get("type") == "bar")
    ok &= check("chart field is Advisor", (figure or {}).get("field") == "Advisor")
    # The query result from step 3 must survive: chart goes to Figures, not Live Output.
    ok &= check("ask_row_count preserved",
                at.session_state["ask_row_count"] is not None)

    print("\n5) Export request: 'Export this list' triggers confirmation")
    at = send(at, "Export this list")
    a = assistants(at)
    pending = at.session_state["assistant_memory"].get("pending_action") or {}
    ok &= check("pending export action stored", bool(pending), str(pending.get("type")))
    last_attach = a[-1].get("attachment") or {} if a else {}
    ok &= check("confirmation card on latest message",
                last_attach.get("type") == "confirmation",
                str(last_attach.get("type")))

    print("\n6) Confirm yes — Export Center should pick up the file")
    at = send(at, "yes")
    pending = at.session_state["assistant_memory"].get("pending_action") or {}
    ok &= check("pending cleared", not pending)
    output_file = at.session_state["latest_output_file"] if "latest_output_file" in at.session_state else None
    ok &= check("output file recorded", bool(output_file), str(output_file))
    export_history = at.session_state["export_history"] if "export_history" in at.session_state else []
    ok &= check("export_history has entry", bool(export_history), str(export_history[:1]))

    print("\n7) Five-zone headers still visible after multiple turns")
    md = headers(at)
    cap = captions(at)
    blob = md + " " + cap
    ok &= check("all five zones still present",
                "original workbook" in blob and "live output" in blob
                and "figures" in blob and "export center" in blob)

    print("\n" + ("ALL PASS" if ok else "SOME FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
