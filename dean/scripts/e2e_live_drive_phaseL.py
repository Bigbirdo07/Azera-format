"""Phase L live drive — runs 7 representative chat turns through AppTest.

This is the same headless driver the tests/conftest::Chat fixture uses, scripted
into a single executable so we can re-verify the end-to-end Phase-L behavior at
any time.

The drive runs with the local LLM disabled — it exercises the deterministic
narration / suggestion / vague-term / correction / log paths. Adding Ollama
later is additive: every check here would still hold.

Usage:
    .venv/bin/python scripts/e2e_live_drive_phaseL.py

Exit code 0 means every assertion passed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from core.interaction_logger import DEFAULT_LOG_PATH  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "synthetic_students.xlsx"


class _FakeUpload:
    def __init__(self, path: Path) -> None:
        self._bytes = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self) -> bytes:
        return self._bytes


# Stub the file uploader so the headless app sees the synthetic workbook.
st.file_uploader = lambda *args, **kwargs: _FakeUpload(FIXTURE)


def banner(label: str) -> None:
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}")


def show(at: AppTest, *keys: str) -> None:
    for key in keys:
        if key not in at.session_state:
            continue
        value = at.session_state[key]
        if isinstance(value, list) and value:
            print(f"  {key}: {value}")
        elif isinstance(value, str) and value:
            print(f"  {key}: {value!r}")
        elif value is not None and not isinstance(value, list):
            print(f"  {key}: {value}")


def send(at: AppTest, text: str) -> AppTest:
    at.chat_input[0].set_value(text)
    at.session_state["current_user"] = {"username": "tester", "role": "Editor"}
    at.session_state["workbook_upload"] = _FakeUpload(FIXTURE)
    at.run()
    return at


def session_value(at: AppTest, key: str, default=None):
    return at.session_state[key] if key in at.session_state else default


def main() -> int:
    if DEFAULT_LOG_PATH.exists():
        DEFAULT_LOG_PATH.unlink()

    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
    at.session_state["current_user"] = {"username": "tester", "role": "Editor"}
    at.session_state["workbook_upload"] = _FakeUpload(FIXTURE)
    at.run()
    print("App loaded — Streamlit AppTest harness ready.")

    # Turn 1 -----------------------------------------------------------------
    banner("Turn 1 — High-confidence query: 'Show me Accounting students'")
    send(at, "Show me Accounting students")
    show(at, "ask_band", "ask_narration", "ask_assumption_note",
         "ask_suggestions", "ask_row_count", "ask_conversation_llm_used")
    assert session_value(at, "ask_band") == "high", "Turn 1 should be high band"
    assert session_value(at, "ask_assumption_note") == "", "Turn 1 should have no assumption note"
    assert session_value(at, "ask_row_count") and session_value(at, "ask_row_count") < 600
    assert session_value(at, "ask_suggestions"), "Turn 1 should suggest next moves"

    # Turn 2 -----------------------------------------------------------------
    banner("Turn 2 — Follow-up replace: 'what about Biology'")
    send(at, "what about Biology")
    show(at, "ask_band", "ask_narration", "ask_row_count")
    assert "Biology" in (session_value(at, "ask_narration") or "")
    assert session_value(at, "ask_row_count") and session_value(at, "ask_row_count") < 600

    # Turn 3 -----------------------------------------------------------------
    banner("Turn 3 — Follow-up refine: 'now only below 2.5 GPA'")
    send(at, "now only below 2.5 GPA")
    show(at, "ask_band", "ask_narration", "ask_row_count")
    assert "GPA" in (session_value(at, "ask_narration") or "")

    # Turn 4 -----------------------------------------------------------------
    banner("Turn 4 — Group: 'group them by advisor'")
    send(at, "group them by advisor")
    show(at, "ask_band", "ask_narration", "ask_suggestions", "ask_row_count")
    assert session_value(at, "ask_operation") == "groupby_count"
    suggestions = session_value(at, "ask_suggestions") or []
    assert any("chart" in s.lower() for s in suggestions), \
        f"Group-by should suggest a chart, got {suggestions}"

    # Turn 5 -----------------------------------------------------------------
    banner("Turn 5 — Medium-band vague query: 'show me struggling students'")
    send(at, "start over")
    send(at, "show me struggling students")
    show(at, "ask_band", "ask_assumption_note", "ask_narration",
         "ask_alternatives", "ask_suggestions", "ask_row_count")
    assert session_value(at, "ask_band") == "medium", \
        f"Turn 5 should be medium, got {session_value(at, 'ask_band')!r}"
    assumption = session_value(at, "ask_assumption_note") or ""
    assert "interpreted" in assumption.lower() or "understood" in assumption.lower(), \
        f"Turn 5 must surface an assumption note, got {assumption!r}"
    alternatives = session_value(at, "ask_alternatives") or []
    assert alternatives, "Turn 5 must offer alternative interpretations"
    rows = session_value(at, "ask_row_count")
    assert isinstance(rows, int) and 0 < rows < 600, (
        f"Turn 5 must NOT return whole sheet — got {rows} rows. "
        "This is the L.12 regression guard."
    )

    # Turn 6 -----------------------------------------------------------------
    banner("Turn 6 — Correction: 'no, I mean students on probation'")
    send(at, "no, I mean students on probation")
    show(at, "ask_band", "ask_narration", "ask_assumption_note",
         "ask_suggestions", "ask_row_count")
    # The corrected query should still execute (not get stuck on the prefix).
    assert session_value(at, "assistant_mode") == "ask_question", \
        f"Turn 6 should produce an answer, got mode={session_value(at, 'assistant_mode')!r}"
    plan_operation = session_value(at, "ask_operation")
    assert plan_operation != "average_column", (
        "Turn 6 must not be mis-parsed as average_column (regression of L.15)."
    )
    rows_corrected = session_value(at, "ask_row_count")
    assert isinstance(rows_corrected, int) and 0 < rows_corrected < 600

    # Turn 7 -----------------------------------------------------------------
    banner("Turn 7 — Medium-band: 'show me overloaded advisors'")
    send(at, "start over")
    send(at, "show me overloaded advisors")
    show(at, "ask_band", "ask_assumption_note", "ask_alternatives",
         "ask_suggestions", "ask_operation")
    assert session_value(at, "ask_operation") == "groupby_count"
    assert "Advisor" in (session_value(at, "ask_assumption_note") or "")

    # Interaction log inspection --------------------------------------------
    banner("Interaction log contents")
    assert DEFAULT_LOG_PATH.exists(), "Interaction log file must exist"
    records = [json.loads(line) for line in DEFAULT_LOG_PATH.read_text().splitlines() if line.strip()]
    print(f"  Records written: {len(records)}")
    found_correction = False
    for index, record in enumerate(records, 1):
        flag = "★ corrects prior" if record.get("corrects_entry_id") else ""
        print(f"  #{index}: band={str(record.get('band')):6} "
              f"intent={str(record.get('intent')):8} "
              f"op={str((record.get('validated_plan') or {}).get('operation')):18} "
              f"msg={record.get('normalized_message')!r} {flag}")
        if record.get("assumption_used"):
            print(f"       assumption: {record['assumption_used']}")
        if record.get("corrects_entry_id"):
            found_correction = True
        # Privacy: no row content leaks.
        blob = json.dumps(record)
        for forbidden in ("@example", "555-", "S0001", "Smith", "Jones", "Maria Lopez"):
            assert forbidden not in blob, f"LEAK: {forbidden!r} in record #{index}"
    assert found_correction, "Turn 6 should be logged with corrects_entry_id"
    print("\n  ✓ no row content leaked into any record")

    # Analyzer ---------------------------------------------------------------
    banner("Rule-mining analyzer")
    from scripts.analyze_interaction_logs import render_report, summarize  # noqa: PLC0415
    summary = summarize(records)
    print(f"  Total turns: {summary['total']}")
    print(f"  By band: {dict(summary['by_band'])}")
    print(f"  By intent: {dict(summary['by_intent'])}")
    print(f"  Repeated assumptions: {dict(summary['repeated_assumptions'])}")
    print(f"  Rule candidates: {len(summary['rule_candidates'])}")
    for message, info in summary["rule_candidates"][:5]:
        print(f"    - {message!r}  count={info['count']}  ops={dict(info['operations'])}")

    report = render_report(summary)
    assert "# Interaction Learning Report" in report

    print("\n✓ Phase-L live drive passed all assertions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
