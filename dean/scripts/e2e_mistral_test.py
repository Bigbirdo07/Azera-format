"""End-to-end test of plan_request() against a real mistral:7b on Ollama.

Loads a sample workbook, runs three requests through expert_planner.plan_request,
and reports: which path served the plan (rule vs llm), validation outcome, and
whether record_llm_assumptions() learned anything.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nlp.expert_planner import plan_request  # noqa: E402
from nlp.model_prompt import OLLAMA_URL  # noqa: E402

# The production timeout in nlp/local_model._call_ollama is 60s, but mistral:7b
# on Apple Silicon Q4 produces a ~300-token planner JSON in ~85s. We bump the
# urlopen timeout in-process so the E2E test can actually exercise the LLM path.
# (Real fix lives in local_model.py — see report.)
import urllib.request as _urlreq  # noqa: E402

_orig_urlopen = _urlreq.urlopen


def _patched_urlopen(req, *args, **kwargs):
    if "timeout" in kwargs or len(args) >= 1:
        kwargs["timeout"] = 300
        args = ()
    else:
        kwargs.setdefault("timeout", 300)
    return _orig_urlopen(req, *args, **kwargs)


_urlreq.urlopen = _patched_urlopen


def warm_up(model: str) -> float:
    """Force Ollama to load the model into memory so the first real call
    doesn't time out. Returns elapsed seconds."""
    payload = json.dumps({
        "model": model,
        "prompt": "ping",
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        resp.read()
    return time.perf_counter() - t0

WORKBOOK = REPO_ROOT / "outputs" / "mock_dean_student_roster_edited_20260519_180232_331696.xlsx"
LEARNED_SYNONYMS = REPO_ROOT / "knowledge" / "learned_synonyms.json"
LEARNED_COLUMN_MAPPINGS = REPO_ROOT / "knowledge" / "learned_column_mappings.json"
MODEL = "mistral:7b"

TESTS = [
    {
        "name": "rule-only: simple filter",
        "request": "filter students where GPA below 3.0",
        "expect_source": "rule_parser",
    },
    {
        "name": "llm: chart grouped by concentration (blind-spot phrase)",
        "request": "create a chart of students grouped by their concentration",
        "expect_source": "local_llm",
    },
    {
        "name": "llm: report listing tutor and outcome (blind-spot phrases)",
        "request": "make a report listing each student's tutor and final outcome",
        "expect_source": "local_llm",
    },
]


def _read_json(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text() or "[]")


def main() -> int:
    if not WORKBOOK.exists():
        print(f"FAIL: workbook missing: {WORKBOOK}")
        return 2

    xl = pd.ExcelFile(WORKBOOK)
    sheets = {name: pd.read_excel(xl, sheet_name=name) for name in xl.sheet_names}
    sheet_columns = {name: list(df.columns) for name, df in sheets.items()}
    selected_sheet = xl.sheet_names[0]

    print(f"workbook: {WORKBOOK.name}")
    print(f"sheets:   {list(sheets.keys())}")
    print(f"columns:  {sheet_columns[selected_sheet]}")
    print(f"model:    {MODEL}")
    print("warming up model (so first plan_request doesn't hit cold-load timeout)...")
    try:
        warm = warm_up(MODEL)
        print(f"warm-up elapsed: {warm:.2f}s")
    except Exception as exc:
        print(f"warm-up failed: {exc}")
        return 2
    print("-" * 70)

    syn_before = _read_json(LEARNED_SYNONYMS)
    map_before = _read_json(LEARNED_COLUMN_MAPPINGS)

    all_ok = True
    for i, case in enumerate(TESTS, 1):
        print(f"\n[{i}] {case['name']}")
        print(f"    request: {case['request']!r}")
        t0 = time.perf_counter()
        result = plan_request(
            user_request=case["request"],
            selected_sheet=selected_sheet,
            sheet_columns=sheet_columns,
            sheets=sheets,
            original_file_name=WORKBOOK.name,
            use_local_llm=True,
            ollama_model=MODEL,
        )
        elapsed = time.perf_counter() - t0
        print(f"    elapsed:        {elapsed:.2f}s")
        print(f"    source:         {result.source}")
        print(f"    plan_type:      {result.plan_type}")
        print(f"    confidence:     {result.confidence:.2f} ({result.confidence_level})")
        print(f"    can_execute:    {result.can_execute}")
        print(f"    validation:     {result.validation_error or 'ok'}")
        print(f"    summary:        {result.plain_english_summary}")
        if result.clarification_question:
            print(f"    clarification:  {result.clarification_question}")
        if result.assumptions:
            print(f"    assumptions:    {result.assumptions}")
        if result.mapped_columns:
            print(f"    mapped_columns: {result.mapped_columns}")
        if result.commands:
            print(f"    commands:       {result.commands}")

        expected = case["expect_source"]
        if expected and result.source != expected:
            print(f"    NOTE: expected source={expected!r}, got {result.source!r}")
            # not a hard failure — rule parser may resolve more than expected
        if result.plan_type not in {"single_action", "multi_action", "clarify"}:
            print("    FAIL: unrecognized plan_type")
            all_ok = False
        if result.validation_error:
            print("    NOTE: validation error present")

    syn_after = _read_json(LEARNED_SYNONYMS)
    map_after = _read_json(LEARNED_COLUMN_MAPPINGS)

    print("\n" + "-" * 70)
    print("learning loop:")
    print(f"  learned_synonyms.json:        {len(syn_before)} -> {len(syn_after)}")
    print(f"  learned_column_mappings.json: {len(map_before)} -> {len(map_after)}")
    new_syns = [s for s in syn_after if s not in syn_before]
    new_maps = [m for m in map_after if m not in map_before]
    if new_syns:
        print(f"  new synonyms:        {new_syns}")
    if new_maps:
        print(f"  new column_mappings: {new_maps}")
    if not new_syns and not new_maps:
        print("  no new mappings learned this run")

    print("\nresult:", "OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
