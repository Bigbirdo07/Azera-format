"""Headless test of the unified-assistant backend (Phase 1).

Drives the full path with NO Streamlit:
  intent_router -> query_planner/edit_planner -> query_engine/validator
  -> session_memory -> followup_resolver

Section A runs rule-only (fast, deterministic). Section B exercises the local
LLM (intent fallback, query fallback, edit plan, plain-English explain) to prove
the timeout fix lets mistral:7b actually serve a request.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.query_engine import run_query  # noqa: E402
from core.session_memory import SessionMemory  # noqa: E402
from nlp.edit_planner import plan_edit  # noqa: E402
from nlp.followup_resolver import resolve_followup  # noqa: E402
from nlp.intent_router import classify_intent  # noqa: E402
from nlp.query_planner import plan_query  # noqa: E402
from nlp.local_model import explain_result_with_model  # noqa: E402

WORKBOOK = REPO_ROOT / "outputs" / "mock_dean_student_roster_edited_20260519_180232_331696.xlsx"
MODEL = "mistral:7b"


def load_workbook():
    xl = pd.ExcelFile(WORKBOOK)
    sheets = {n: pd.read_excel(xl, sheet_name=n) for n in xl.sheet_names}
    sheet_columns = {n: list(df.columns) for n, df in sheets.items()}
    return sheets, sheet_columns, xl.sheet_names[0]


def section_a_rule_only(sheets, sheet_columns, sheet):
    print("=" * 72)
    print("SECTION A — rule-only backend (no LLM)")
    print("=" * 72)
    memory = SessionMemory()

    ask_questions = [
        "How many students have a GPA below 3.0?",
        "Which major has the most students?",
        "What is the average GPA by advisor?",
        "Which columns have missing values?",
        "Are there duplicate Student IDs?",
        "Summarize this sheet.",
    ]
    for q in ask_questions:
        intent = classify_intent(
            user_request=q, sheet_names=list(sheet_columns), sheet_columns=sheet_columns,
            use_local_llm=False, has_previous_result=memory.has_result_set(),
        )
        print(f"\nQ: {q}\n   intent={intent.request_type} ({intent.confidence:.2f}, {intent.source})")
        if intent.request_type != "ask_question":
            print("   ! expected ask_question")
            continue
        qp = plan_query(user_request=q, selected_sheet=sheet, sheet_columns=sheet_columns, use_local_llm=False)
        if qp.needs_clarification:
            print(f"   query planner asked to clarify: {qp.clarification_question}")
            continue
        print(f"   plan: op={qp.query['operation']} group_by={qp.query['group_by']!r} "
              f"value={qp.query['value_column']!r} filters={qp.query['filters']} (conf {qp.confidence:.2f}, {qp.source})")
        result = run_query(qp.query, sheets)
        print(f"   RESULT: {result.description}")
        memory.record_ask(
            request=q, query_plan=qp.query, result_description=result.description,
            row_count=result.row_count, columns_used=result.columns_used, sheet=sheet,
        )

    # Edit commands (rule-only).
    print("\n" + "-" * 72)
    print("Edit commands (rule-only):")
    for cmd in ["Highlight students with GPA below 3.0", "Create a chart of students by Major"]:
        intent = classify_intent(
            user_request=cmd, sheet_names=list(sheet_columns), sheet_columns=sheet_columns,
            use_local_llm=False, has_previous_result=memory.has_result_set(),
        )
        ep = plan_edit(
            user_request=cmd, selected_sheet=sheet, sheet_columns=sheet_columns, sheets=sheets,
            original_file_name=WORKBOOK.name, use_local_llm=False, ollama_model=MODEL,
        )
        print(f"\nE: {cmd}\n   intent={intent.request_type} ({intent.confidence:.2f})")
        print(f"   plan: type={ep.plan_type} conf={ep.confidence:.2f} can_execute={ep.can_execute} "
              f"validation={ep.validation_error or 'ok'}")
        print(f"   summary: {ep.plain_english_summary}")
        if intent.request_type == "edit_workbook" and ep.commands:
            memory.record_edit(request=cmd, edit_plan=ep.to_dict(), sheet=sheet)

    # Follow-up resolution after an ask.
    print("\n" + "-" * 72)
    print("Follow-up resolution:")
    memory.record_ask(
        request="students missing a Second Major", query_plan={"filters": [{"column": "Second Major", "operator": "is_missing"}], "sheet": sheet},
        result_description="121 students are missing a Second Major", row_count=121,
        columns_used=["Second Major"], sheet=sheet,
    )
    for follow in ["Highlight them", "Make a chart of that"]:
        res = resolve_followup(follow, memory)
        print(f"\nF: {follow}\n   is_followup={res.is_followup} resolved={res.resolved} "
              f"filters={res.filters} note={res.referent_note!r}")

    empty = SessionMemory()
    res = resolve_followup("Move them", empty)
    print(f"\nF: Move them (no prior result)\n   needs_clarification={res.needs_clarification} "
          f"q={res.clarification_question!r}")


def section_b_llm(sheets, sheet_columns, sheet):
    print("\n" + "=" * 72)
    print("SECTION B — local LLM paths (mistral:7b)")
    print("=" * 72)

    # Warm up.
    import json as _json, urllib.request as _u
    from nlp.model_prompt import OLLAMA_URL
    t0 = time.perf_counter()
    payload = _json.dumps({"model": MODEL, "prompt": "ping", "stream": False,
                           "options": {"num_predict": 1}}).encode()
    req = _u.Request(f"{OLLAMA_URL}/api/generate", data=payload,
                     headers={"Content-Type": "application/json"}, method="POST")
    with _u.urlopen(req, timeout=300) as r:
        r.read()
    print(f"warm-up: {time.perf_counter()-t0:.1f}s")

    # B1: ask question + plain-English explanation.
    q = "How many students have a GPA below 3.0?"
    qp = plan_query(user_request=q, selected_sheet=sheet, sheet_columns=sheet_columns, use_local_llm=False)
    result = run_query(qp.query, sheets)
    print(f"\n[B1] question: {q}")
    print(f"     pandas result: {result.description}")
    verified = {"operation": result.operation, "value": result.value,
                "row_count": result.row_count, "description": result.description}
    t0 = time.perf_counter()
    text, err = explain_result_with_model(user_question=q, verified_result=verified, model_name=MODEL)
    print(f"     explain ({time.perf_counter()-t0:.1f}s): {text or err}")

    # B2: edit plan through the LLM (complex keyword forces the LLM path).
    cmd = "Create a chart of students grouped by their concentration"
    print(f"\n[B2] edit request: {cmd}")
    t0 = time.perf_counter()
    ep = plan_edit(user_request=cmd, selected_sheet=sheet, sheet_columns=sheet_columns, sheets=sheets,
                   original_file_name=WORKBOOK.name, use_local_llm=True, ollama_model=MODEL)
    print(f"     ({time.perf_counter()-t0:.1f}s) source={ep.source} type={ep.plan_type} "
          f"conf={ep.confidence:.2f} can_execute={ep.can_execute}")
    print(f"     validation: {ep.validation_error or 'ok'}")
    print(f"     summary: {ep.plain_english_summary}")
    if ep.assumptions:
        print(f"     assumptions: {ep.assumptions}")


def main() -> int:
    if not WORKBOOK.exists():
        print(f"FAIL: workbook missing: {WORKBOOK}")
        return 2
    sheets, sheet_columns, sheet = load_workbook()
    print(f"workbook: {WORKBOOK.name}  sheet: {sheet}")
    print(f"columns: {sheet_columns[sheet]}\n")

    section_a_rule_only(sheets, sheet_columns, sheet)
    if "--with-llm" in sys.argv:
        section_b_llm(sheets, sheet_columns, sheet)
    else:
        print("\n(skip SECTION B; pass --with-llm to exercise mistral:7b)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
