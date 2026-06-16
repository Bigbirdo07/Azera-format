"""Headless verification for Phase A+ (sort/limit/group/also) and Phase B (privacy).

Drives the real app via AppTest on the synthetic workbook and checks behavior
against pandas ground truth.
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

from scripts.make_synthetic_workbook import build_dataframe, write_workbook  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_students.xlsx"
write_workbook(FIXTURE)
DF = build_dataframe()


class FakeUpload:
    def __init__(self, path):
        self._bytes = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self):
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


def fresh():
    at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
    seed(at)
    at.run()
    return at


def main() -> int:
    ok = True
    HIDDEN = {"Email", "Phone", "Date of Birth", "Financial Aid Status", "Conduct Status", "Notes"}

    print("PHASE A+  sort / limit / group / also")
    at = fresh()

    print("\n1) below 2.5, then sort by GPA lowest first")
    send(at, "show me students below 2.5 GPA")
    send(at, "sort them by GPA lowest first")
    table = g(at, "ask_table") or []
    gpas = [r.get("GPA") for r in table if r.get("GPA") is not None]
    ok &= check("sort returns rows", len(gpas) > 0, f"op={g(at,'ask_operation')} len={len(table)}")
    ok &= check("preview sorted ascending by GPA", bool(gpas) and gpas == sorted(gpas), str(gpas[:5]))

    print("\n2) just the top 5")
    send(at, "just the top 5")
    n2 = len(g(at, "ask_table") or [])
    ok &= check("limit applied (1..5 rows)", 0 < n2 <= 5, str(n2))

    print("\n3) probation students, then group them by advisor")
    at = fresh()
    send(at, "show me students on probation")
    send(at, "group them by advisor")
    prob = DF[DF["Academic Status"] == "Probation"]
    expected_groups = prob["Advisor"].nunique()
    ok &= check("operation groupby", g(at, "ask_operation") == "groupby_count", str(g(at, "ask_operation")))
    ok &= check("grouped by advisor over probation", len(g(at, "ask_table") or []) == expected_groups,
               f"{len(g(at,'ask_table') or [])} vs {expected_groups}")

    print("\n4) Accounting, then include Biology too (additive -> IN)")
    at = fresh()
    send(at, "show me Accounting students")
    send(at, "include Biology too")
    expected = len(DF[DF["Department"].isin(["Accounting", "Biology"])])
    ok &= check("Department IN [Accounting, Biology]", g(at, "ask_row_count") == expected,
               f"{g(at,'ask_row_count')} vs {expected}")

    print("\nPHASE B  privacy")
    at = fresh()

    print("\n5) aggregate question needs no confirmation")
    send(at, "How many students are in each department?")
    ok &= check("not a confirmation/clarify", g(at, "assistant_mode") == "ask_question", str(g(at, "assistant_mode")))

    print("\n6) student list hides sensitive columns by default")
    send(at, "show me students below 2.5 GPA")
    table = g(at, "ask_table") or []
    shown = set().union(*[set(r.keys()) for r in table]) if table else set()
    ok &= check("no sensitive columns in preview", not (shown & HIDDEN), str(sorted(shown & HIDDEN)))
    ok &= check("preview still has GPA", "GPA" in shown)

    print("\n7) explicit sensitive request requires confirmation")
    send(at, "show me all student emails and GPAs")
    ok &= check("mode clarify (confirmation)", g(at, "assistant_mode") == "clarify", str(g(at, "assistant_mode")))
    mem = g(at, "assistant_memory") or {}
    ok &= check("pending_action stored", (mem.get("pending_action") or {}).get("type") == "show_sensitive",
               str(mem.get("pending_action")))

    print("\n8) confirm -> shows sensitive")
    send(at, "yes")
    table = g(at, "ask_table") or []
    shown = set().union(*[set(r.keys()) for r in table]) if table else set()
    ok &= check("mode ask after confirm", g(at, "assistant_mode") == "ask_question", str(g(at, "assistant_mode")))
    ok &= check("email now shown", "Email" in shown, str(sorted(shown)))
    mem = g(at, "assistant_memory") or {}
    ok &= check("pending cleared after confirm", not mem.get("pending_action"), str(mem.get("pending_action")))

    print("\n9) cancel path clears pending")
    send(at, "show me all student emails")
    ok &= check("pending set again", bool((g(at, "assistant_memory") or {}).get("pending_action")))
    send(at, "no")
    mem = g(at, "assistant_memory") or {}
    ok &= check("pending cleared after cancel", not mem.get("pending_action"), str(mem.get("pending_action")))

    print("\n10) missing column -> no hallucination")
    at = fresh()
    send(at, "what is each student's housing status")
    # housing not a column; should clarify rather than invent.
    ok &= check("clarifies on missing data", g(at, "assistant_mode") in {"clarify"}, str(g(at, "assistant_mode")))

    print("\n" + ("ALL PASS" if ok else "SOME FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
