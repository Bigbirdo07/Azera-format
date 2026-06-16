"""Headless test of composable multi-turn follow-ups through the real app.

Drives the chat with AppTest on the synthetic workbook and checks each answer
against pandas ground truth. Covers the spec's headline flow plus replace /
clear / reset behavior.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from scripts.make_synthetic_workbook import build_dataframe, write_workbook  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_students.xlsx"
write_workbook(FIXTURE)
DF = build_dataframe()


class FakeUpload:
    def __init__(self, path: Path):
        self._bytes = path.read_bytes()
        self.name = path.name

    def getvalue(self) -> bytes:
        return self._bytes


st.file_uploader = lambda *a, **k: FakeUpload(FIXTURE)


def seed(at):
    at.session_state["current_user"] = {"username": "tester", "role": "Editor"}
    at.session_state["workbook_upload"] = FakeUpload(FIXTURE)


def g(at, key):
    return at.session_state[key] if key in at.session_state else None


def send(at, text):
    at.text_input[0].set_value(text)
    for b in at.button:
        if b.label == "Send":
            b.click()
            break
    seed(at)
    at.run()


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    ok = True

    # Ground truth from pandas.
    acc = DF[DF["Department"] == "Accounting"]
    acc_below = acc[acc["GPA"] < 2.5]
    acc_below_senior = acc_below[acc_below["Year"] == "Senior"]
    bio = DF[DF["Department"] == "Biology"]
    n_acc, n_below, n_senior, n_bio = len(acc), len(acc_below), len(acc_below_senior), len(bio)
    print(f"ground truth: Accounting={n_acc} below2.5={n_below} +senior={n_senior} Biology={n_bio}")

    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    seed(at)
    at.run()
    ok &= check("booted", not at.exception, str(at.exception))

    print("\n1) Show me Accounting students")
    send(at, "Show me Accounting students")
    ok &= check("mode ask", g(at, "assistant_mode") == "ask_question", str(g(at, "assistant_mode")))
    ok &= check("Accounting count matches pandas", g(at, "ask_row_count") == n_acc, f"{g(at,'ask_row_count')} vs {n_acc}")
    ok &= check("active context = Department", "Department" in (g(at, "active_context") or ""), g(at, "active_context"))

    print("\n2) Now only below 2.5 GPA  (compose)")
    send(at, "now only below 2.5 GPA")
    ok &= check("composed Accounting AND GPA<2.5", g(at, "ask_row_count") == n_below, f"{g(at,'ask_row_count')} vs {n_below}")
    ok &= check("context shows both filters", "GPA" in (g(at, "active_context") or "") and "Department" in (g(at, "active_context") or ""), g(at, "active_context"))

    print("\n3) Now only seniors  (compose)")
    send(at, "now only seniors")
    ok &= check("composed +Year=Senior", g(at, "ask_row_count") == n_senior, f"{g(at,'ask_row_count')} vs {n_senior}")

    print("\n4) What is their average GPA  (keep filters, aggregate)")
    send(at, "what is their average GPA")
    expected_avg = round(float(acc_below_senior["GPA"].mean()), 4) if n_senior else None
    got = g(at, "ask_value")
    ok &= check("avg GPA over current selection", expected_avg is None or (got is not None and abs(got - expected_avg) < 0.01),
               f"{got} vs {expected_avg}")

    print("\n5) What about Biology  (replace Department, drop GPA/Year? -> replace dept only)")
    send(at, "what about Biology")
    # Replace Department=Accounting with Biology; GPA<2.5 and Year=Senior remain composed.
    bio_ctx = DF[(DF["Department"] == "Biology") & (DF["GPA"] < 2.5) & (DF["Year"] == "Senior")]
    ok &= check("Department replaced to Biology", "Biology" in (g(at, "active_context") or ""), g(at, "active_context"))
    ok &= check("Biology+filters count", g(at, "ask_row_count") == len(bio_ctx), f"{g(at,'ask_row_count')} vs {len(bio_ctx)}")

    print("\n6) Clear that  (drop filters)")
    send(at, "clear that")
    ok &= check("filters cleared", (g(at, "active_context") in (None, "none", "")), g(at, "active_context"))

    print("\n7) Show me Nursing students, then Start over")
    send(at, "Show me Nursing students")
    ok &= check("nursing active", "Nursing" in (g(at, "active_context") or ""), g(at, "active_context"))
    send(at, "start over")
    ok &= check("reset clears context", (g(at, "active_context") in (None, "none", "")), g(at, "active_context"))
    mem = g(at, "assistant_memory") or {}
    ok &= check("memory active_filters empty after reset", not mem.get("active_filters"), str(mem.get("active_filters")))

    print("\n" + ("ALL PASS" if ok else "SOME FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
