"""Dean evaluation harness (supersedes the 20-question dean_pilot_20.py).

Routes each question exactly like the live app — charts -> figure path; generic
reads -> code analyst (LLM); specialized ops -> deterministic dispatcher — and
grades against ground truth that is *computed from the dataframe at runtime* so
it never goes stale when the mock roster changes.

Two failure modes are separated on purpose:
  * routing accuracy  — did the question land on a sensible op / kind?
  * computation accuracy — was the final number/label correct?
Run full (default) to measure computation; set DEAN_EVAL_ROUTING_ONLY=1 to skip
the analyst LLM and measure routing only (fast, no model cost).

Env:
  PILOT_MODEL=llama3.2:3b          model under test
  DEAN_EVAL_ROUTING_ONLY=1         skip analyst LLM, report routing only
  DEAN_EVAL_CATEGORIES=count,...   run only these categories
"""
from __future__ import annotations
import sys, re, os, time, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from core.excel_loader import load_excel_workbook, LoadedWorkbook
from core.data_sources import DataSourceRegistry
from core.execution_dispatcher import execute_planned_request
from nlp.planner_router import plan_user_request
from nlp.code_analyst import analyze, default_llm_call
from ui.figures_panel import is_chart_request, detect_chart_intent

# Mirror ui/chat_panel._ANALYST_GENERIC_OPS: groupby/ranking is intentionally
# excluded so it routes to the deterministic engine (the LLM mis-ranks it).
GENERIC_OPS = {"filtered_preview", "count_rows", "count_unique", "list_unique",
               "average_column", "sum_column", "min_column", "max_column", "average",
               "aggregate", "filter"}

ROUTING_ONLY = os.environ.get("DEAN_EVAL_ROUTING_ONLY") == "1"
ONLY_CATS = {c.strip() for c in os.environ.get("DEAN_EVAL_CATEGORIES", "").split(",") if c.strip()}

ROSTER = '/Users/albertopaz/azera-formatting/mock_dean_student_roster_with_assessments_attendance_cleaned.xlsx'


class Upload:
    def __init__(s, p): s.path = Path(p); s.name = s.path.name
    def getvalue(s): return s.path.read_bytes()


loaded = load_excel_workbook(Upload(ROSTER))
reg = DataSourceRegistry(); reg.set_roster(loaded)
sheets = reg.enriched_sheets() or dict(loaded.sheets)
sel = reg.enriched_roster_sheet or next(iter(sheets))
df = sheets[sel]
enriched = LoadedWorkbook(file_name=loaded.file_name, workbook=loaded.workbook, sheets=sheets, warnings=[])
sheet_columns = {n: list(f.columns) for n, f in sheets.items()}
settings = {"use_local_llm": True, "llm_enabled": True, "strict_privacy_mode": False,
            "code_analyst_enabled": True, "planner_model": "llama3.2:3b", "planner_timeout_seconds": 300}
MODEL = os.environ.get("PILOT_MODEL", "llama3.2:3b")
call = default_llm_call(MODEL, timeout=200)

g = pd.to_numeric(df["GPA"], errors="coerce")
ar = df["Attendance Rate"]


def C(mask) -> str:  # count -> exact int string
    return str(int(mask.sum()))


def num(v, dec=0) -> str:
    return f"{v:.{dec}f}"


# Each question: (category, question, truth-or-route-marker)
#   truth = numeric/string value to find in the answer (graded on computation)
#   "[kind:FIGURE]" / "[kind:NEGATIVE]" = graded on routing bucket
#   "[observe]" = recorded but not hard-graded (specialist/tool routes)
QS: list[tuple[str, str, str]] = [
    # --- simple counts ---------------------------------------------------
    ("count", "How many students are at high risk?", C(df["Risk Level"] == "High Risk")),
    ("count", "How many students are at moderate risk?", C(df["Risk Level"] == "Moderate Risk")),
    ("count", "How many students are in bad standing?", C(df["Standing"] == "Bad Standing")),
    ("count", "How many students are in good standing?", C(df["Standing"] == "Good Standing")),
    ("count", "How many students need attendance support?", C(df["Attendance Category"] == "Needs Attendance Support")),
    ("count", "How many students have great attendance?", C(df["Attendance Category"] == "Great Attendance")),
    ("count", "How many seniors are there?", C(df["Year"] == "Senior")),
    ("count", "How many freshmen are there?", C(df["Year"] == "Freshman")),
    ("count", "How many students study online?", C(df["Location"] == "Online")),
    # NOTE: "Nursing" is both a Discipline and a Major, so a bare "in Nursing" is
    # genuinely ambiguous — the app correctly asks which. Use a discipline-only
    # value (Health Sciences) to test the count path cleanly.
    ("count", "How many students are in the Health Sciences discipline?", C(df["Discipline"] == "Health Sciences")),
    ("count", "How many students are in Engineering?", C(df["Discipline"] == "Engineering")),
    ("count", "How many students have a GPA below 2.0?", C(g < 2.0)),
    ("count", "How many students have a GPA of 3.5 or higher?", C(g >= 3.5)),
    ("count", "How many students have an SAT Total of 1400 or higher?", C(df["SAT Total"] >= 1400)),
    ("count", "How many students have a second major?", C(df["Second Major"].notna())),

    # --- compound filters ------------------------------------------------
    ("compound", "How many seniors are at high risk?", C((df["Year"] == "Senior") & (df["Risk Level"] == "High Risk"))),
    ("compound", "How many students have a GPA below 2.0 and need attendance support?",
        C((g < 2.0) & (df["Attendance Category"] == "Needs Attendance Support"))),
    ("compound", "How many high-risk students are in bad standing?",
        C((df["Risk Level"] == "High Risk") & (df["Standing"] == "Bad Standing"))),
    ("compound", "How many Health Sciences students are at high risk?",
        C((df["Discipline"] == "Health Sciences") & (df["Risk Level"] == "High Risk"))),
    ("compound", "How many online students are seniors?",
        C((df["Location"] == "Online") & (df["Year"] == "Senior"))),
    ("compound", "How many students have a GPA between 2.0 and 3.0?", C((g >= 2.0) & (g < 3.0))),

    # --- distinct counts -------------------------------------------------
    ("distinct", "How many different majors are there?", str(int(df["Major"].nunique()))),
    ("distinct", "How many advisors are there?", str(int(df["Advisor"].nunique()))),
    ("distinct", "How many disciplines are there?", str(int(df["Discipline"].nunique()))),
    ("distinct", "How many campus locations are there?", str(int(df["Location"].nunique()))),

    # --- aggregates ------------------------------------------------------
    ("aggregate", "What is the average GPA?", num(g.mean(), 2)),
    ("aggregate", "What is the highest GPA?", num(g.max(), 2)),
    ("aggregate", "What is the lowest GPA?", num(g.min(), 2)),
    ("aggregate", "What is the average SAT Total?", num(df["SAT Total"].mean(), 0)),
    ("aggregate", "What is the highest SAT Total?", str(int(df["SAT Total"].max()))),
    ("aggregate", "What is the average PSAT Total?", num(df["PSAT Total"].mean(), 0)),

    # --- groupby / rank (string answers) ---------------------------------
    ("groupby", "Which discipline has the lowest average GPA?", df.assign(G=g).groupby("Discipline")["G"].mean().idxmin()),
    ("groupby", "Which discipline has the highest average GPA?", df.assign(G=g).groupby("Discipline")["G"].mean().idxmax()),
    ("groupby", "Which discipline has the highest average attendance rate?", df.groupby("Discipline")["Attendance Rate"].mean().idxmax()),
    # Ties are real: several advisors can share the max. Accept any tied winner.
    ("groupby", "Which advisor has the most high-risk students?",
        "|".join(_hr.index[_hr == _hr.max()]) if not (_hr := df[df["Risk Level"] == "High Risk"].groupby("Advisor").size()).empty else ""),
    ("groupby", "Which year level has the most students?", df["Year"].value_counts().idxmax()),
    ("groupby", "Which campus has the most students?", df["Location"].value_counts().idxmax()),
    ("groupby", "Which discipline has the most students?", df["Discipline"].value_counts().idxmax()),

    # --- percentage ------------------------------------------------------
    ("percentage", "What percentage of students are high risk?", num((df["Risk Level"] == "High Risk").mean() * 100, 0)),
    ("percentage", "What percentage of students are in bad standing?", num((df["Standing"] == "Bad Standing").mean() * 100, 0)),
    ("percentage", "What percentage of students are seniors?", num((df["Year"] == "Senior").mean() * 100, 0)),

    # --- specialist routes (observed, not hard-graded) -------------------
    ("specialist", "Who needs advisor attention?", "[observe]"),
    ("specialist", "Which advisors need support?", "[observe]"),
    ("specialist", "Which advisors are doing a good job?", "[observe]"),
    ("specialist", "Show me the trends in this data", "[observe]"),
    ("specialist", "Compare outcomes across advisors", "[observe]"),

    # --- tool routes (observed) ------------------------------------------
    ("tool", "Average GPA by advisor", "[observe]"),
    ("tool", "Build an advisor intervention dashboard", "[observe]"),
    ("tool", "Give me a data quality summary", "[observe]"),

    # --- figure routes (graded on routing) -------------------------------
    ("figure", "Make a bar chart of students by risk level", "[kind:FIGURE]"),
    ("figure", "Plot the distribution of GPA", "[kind:FIGURE]"),
    ("figure", "Show a pie chart of students by discipline", "[kind:FIGURE]"),
    ("figure", "Graph attendance rate by year", "[kind:FIGURE]"),

    # --- negative / out-of-scope (should NOT answer as a query) ----------
    ("negative", "What's the weather today?", "[kind:NEGATIVE]"),
    ("negative", "asdkfjalskdjf", "[kind:NEGATIVE]"),
    ("negative", "Delete all freshmen from the roster", "[kind:NEGATIVE]"),
    ("negative", "Email every at-risk student their advisor's phone number", "[kind:NEGATIVE]"),
]

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _nums(s: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(s.replace(",", ""))]


def grade_value(truth: str, ans: str) -> bool:
    """Token-aware grading. Numeric truths match by value within a tolerance so
    "2" never spuriously matches "12"/"0.25" (the old substring grader's bug).
    String truths (category labels) match on word boundary, case-insensitive."""
    try:
        t = float(truth)
        tol = 0.01 if "." in truth else 0.5  # ints rounded; allow .5 for rounded means
        return any(abs(n - t) <= tol for n in _nums(ans))
    except ValueError:
        # String truth; "a|b" means any tied winner is acceptable.
        low = ans.lower()
        return any(re.search(r"\b" + re.escape(opt.lower()) + r"\b", low) for opt in truth.split("|"))


def route_and_answer(q: str) -> tuple[str, str, str]:
    """Returns (kind, op, answer)."""
    if is_chart_request(q):
        intent = detect_chart_intent(q, sheet_columns[sel])
        return "FIGURE", "figure", f"field={getattr(intent,'field',None)} type={getattr(intent,'chart_type',None)}"
    r = plan_user_request(user_message=q, sheets=sheets, sheet_columns=sheet_columns,
                          selected_sheet=sel, conversation_state={}, settings={"llm_enabled": False})
    intent = r.get("intent"); plan = r.get("plan") or {}; op = plan.get("operation", "")
    if intent in {"clarify", "unavailable", "unsupported"}:
        return "CLARIFY", op, (r.get("confirmation_reason") or r.get("fallback_reason") or "")[:120]
    if intent != "query":
        return f"INTENT={intent}", op, f"pending={r.get('pending_type')} confirm={r.get('requires_confirmation')}"
    if op in GENERIC_OPS:
        if ROUTING_ONLY:
            return "PLAN", op, "(routing-only)"
        res = analyze(user_message=q, df=df, llm_call=call, verify=False)
        return "ANALYST", op, (res.answer or "(none)").replace("\n", " ")[:200]
    try:
        resp = execute_planned_request(r, enriched, settings, reveal_sensitive=False,
                                       request_summary=q, session_workbook=None)
        return f"TOOL", op, (resp.get("message") or "").replace("\n", " ")[:200]
    except Exception as e:
        return "TOOL", op, f"(dispatch error: {e})"


def _preflight() -> None:
    """In full mode, fail loudly if Ollama isn't reachable on the resolved port.
    Otherwise every analyst answer silently falls back to the rule path and the
    suite grades a dead LLM as a pile of wrong answers. The app's bundled Ollama
    uses 11438+; the system Ollama uses 11434 — point at the right one with
    DEAN_OLLAMA_PORT."""
    if ROUTING_ONLY:
        return
    from nlp.local_model import get_ollama_status
    from nlp.model_prompt import OLLAMA_URL
    status = get_ollama_status(MODEL, timeout=3)
    if not status.available:
        print(f"!! Ollama not usable at {OLLAMA_URL} for `{MODEL}`: {status.user_message}")
        print(f"!! {status.detail or ''}")
        print("!! Full mode would grade the rule-based FALLBACK, not the LLM.")
        print("!! Fix: DEAN_OLLAMA_PORT=11434 (system Ollama) or start the app's bundled Ollama.\n")


def main() -> None:
    _preflight()
    dataset_hash = hashlib.sha1(Path(ROSTER).read_bytes()).hexdigest()[:12]
    print(f"=== MODEL: {MODEL} | dataset: {dataset_hash} | "
          f"mode: {'ROUTING-ONLY' if ROUTING_ONLY else 'FULL'} | rows: {len(df)} ===")

    cats: dict[str, list[bool]] = {}
    rows_out = []
    t_start = time.monotonic()
    for i, (cat, q, truth) in enumerate(QS, 1):
        if ONLY_CATS and cat not in ONLY_CATS:
            continue
        t0 = time.monotonic()
        kind, op, ans = route_and_answer(q)
        dt = time.monotonic() - t0

        graded = False; ok = None
        if truth.startswith("[kind:FIGURE]"):
            graded, ok = True, kind == "FIGURE"
        elif truth.startswith("[kind:NEGATIVE]"):
            graded, ok = True, kind.startswith("CLARIFY") or kind.startswith("INTENT")
        elif truth.startswith("[observe]"):
            graded = False
        elif kind == "PLAN":
            graded = False  # analyst was skipped (routing-only) — no value to grade
        else:
            graded, ok = True, grade_value(truth, ans)

        if graded:
            cats.setdefault(cat, []).append(bool(ok))
        verdict = ""
        if graded:
            verdict = "  ✓" if ok else f"  ✗ expected~{truth}"
        print(f"\nQ{i:>2} [{cat}] {q}")
        print(f"     -> [{kind}] op={op or '-'} | {ans}{verdict}   ({dt:.1f}s)")
        rows_out.append({"q": q, "cat": cat, "kind": kind, "op": op, "ans": ans,
                         "truth": truth, "graded": graded, "ok": ok, "sec": round(dt, 1)})

    total_ok = sum(sum(v) for v in cats.values())
    total_graded = sum(len(v) for v in cats.values())
    print(f"\n{'='*60}\nPER-CATEGORY:")
    for cat in sorted(cats):
        v = cats[cat]
        print(f"  {cat:<12} {sum(v)}/{len(v)}")
    wall = time.monotonic() - t_start
    print(f"{'='*60}")
    print(f"=== {MODEL}: {total_ok}/{total_graded} graded correct | wall {wall:.0f}s "
          f"| mode={'routing-only' if ROUTING_ONLY else 'full'} ===")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report = {"model": MODEL, "dataset_hash": dataset_hash, "routing_only": ROUTING_ONLY,
              "score": [total_ok, total_graded], "wall_seconds": round(wall),
              "per_category": {c: [sum(v), len(v)] for c, v in cats.items()}, "rows": rows_out}
    report_path = out_dir / f"dean_eval_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved baseline -> {report_path.relative_to(Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main()
