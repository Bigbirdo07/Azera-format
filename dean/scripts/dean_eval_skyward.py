"""Dean evaluation harness for the Skyward-shaped HS roster (scripts/make_skyward_workbook.py).

Reuses scripts/dean_eval.py's proven scaffolding (route_and_answer / grade_value /
C / num are roster-agnostic) but targets the real target SIS schema instead of
the original college mock roster, and focuses on the ~20 Skyward columns that
had zero live question-answer coverage before this file existed (Grad Year,
Birth Date, Phone/Email, Guardian contact fields, Excused/Unexcused Absences,
Tardies, Discipline Information, SAT/PSAT sub-scores, Entry/Withdrawal Date,
Emergency Contact) plus the derived Risk columns (which transfer as-is from
core.combined_risk). Standing/Attendance-Category-style questions are skipped
on purpose -- the Skyward roster has no equivalent column, and faking one
would test a fiction, not the real target.

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
from scripts.make_skyward_workbook import DEFAULT_OUT, write_workbook

# Mirror ui/chat_panel._ANALYST_GENERIC_OPS: empty by design (see dean_eval.py).
GENERIC_OPS = set()

ROUTING_ONLY = os.environ.get("DEAN_EVAL_ROUTING_ONLY") == "1"
ONLY_CATS = {c.strip() for c in os.environ.get("DEAN_EVAL_CATEGORIES", "").split(",") if c.strip()}

ROSTER = str(DEFAULT_OUT)
if not DEFAULT_OUT.exists():
    write_workbook(DEFAULT_OUT)


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

g = pd.to_numeric(df["Current Cumulative GPA"], errors="coerce")


def C(mask) -> str:  # count -> exact int string
    return str(int(mask.sum()))


def num(v, dec=0) -> str:
    return f"{v:.{dec}f}"


# Each question: (category, question, truth-or-route-marker)
QS: list[tuple[str, str, str]] = [
    # --- grade-level (HS name words -> numeric Grade column) -------------
    ("grade_level", "How many freshmen are there?", C(df["Grade"] == 9)),
    ("grade_level", "How many sophomores are there?", C(df["Grade"] == 10)),
    ("grade_level", "How many juniors are there?", C(df["Grade"] == 11)),
    ("grade_level", "How many seniors are there?", C(df["Grade"] == 12)),
    ("grade_level", "How many students are in grade 9?", C(df["Grade"] == 9)),

    # --- grad year (numeric comparison, not bare equality) ---------------
    ("grad_year", "How many students have a grad year before 2029?", C(df["Grad Year"] < 2029)),
    ("grad_year", "How many students have a grad year after 2027?", C(df["Grad Year"] > 2027)),

    # --- grade name + a second numeric clause (compound) ------------------
    # Regression coverage for the "9th graders have more than 3 unexcused
    # absences" bug: the leading grade-name fragment used to steal the
    # adjacent numeric column via a literal-substring match on "Grade".
    ("grade_level", "How many 9th graders have more than 3 unexcused absences?",
        C((df["Grade"] == 9) & (df["Unexcused Absences"] > 3))),
    ("grade_level", "How many seniors have more than 5 tardies?",
        C((df["Grade"] == 12) & (df["Tardies"] > 5))),

    # --- date columns compared by bare year --------------------------------
    ("date", "How many students were born before 2010?",
        C(df["Birth Date"] < pd.Timestamp("2010-01-01"))),
    ("date", "How many students entered before 2024?",
        C(df["Entry Date"] < pd.Timestamp("2024-01-01"))),
    ("date", "How many students entered in 2024?",
        C((df["Entry Date"] >= pd.Timestamp("2024-01-01")) & (df["Entry Date"] <= pd.Timestamp("2024-12-31 23:59:59")))),

    # --- bare numeric equality (no comparison word) -------------------------
    ("equality", "How many students have an unexcused absence count of 0?", C(df["Unexcused Absences"] == 0)),
    ("equality", "How many students have a grad year of 2028?", C(df["Grad Year"] == 2028)),

    # --- contact/presence columns (never exercised before this file) -----
    ("presence", "How many students have a phone number on file?", C(df["Phone"].notna())),
    ("presence", "How many students have an email on file?", C(df["Email"].notna())),
    ("presence", "How many students have a guardian name on file?", C(df["Guardian Name"].notna())),
    ("presence", "How many students have a guardian phone number on file?", C(df["Guardian Phone"].notna())),
    ("presence", "How many students have a guardian email on file?", C(df["Guardian Email"].notna())),
    ("presence", "How many students have an emergency contact on file?", C(df["Emergency Contact"].notna())),
    ("presence", "How many students have a home address on file?", C(df["Home Address"].notna())),
    ("presence", "How many students have withdrawn?", C(df["Withdrawal Date"].notna())),
    ("presence", "How many students have a discipline record on file?", C(df["Discipline Information"].notna())),

    # --- attendance split (Excused/Unexcused/Tardies) ---------------------
    ("attendance", "What is the average number of excused absences?", num(df["Excused Absences"].mean(), 2)),
    ("attendance", "What is the average number of unexcused absences?", num(df["Unexcused Absences"].mean(), 2)),
    ("attendance", "How many students have more than 3 unexcused absences?", C(df["Unexcused Absences"] > 3)),
    ("attendance", "How many students have more than 5 tardies?", C(df["Tardies"] > 5)),
    ("attendance", "What is the average number of tardies?", num(df["Tardies"].mean(), 2)),
    ("attendance", "What is the average attendance rate?", num(df["Attendance Rate"].mean(), 4)),

    # --- SAT / PSAT sub-scores --------------------------------------------
    ("assessment", "How many students scored above 1300 on SAT Total?", C(df["SAT Total"] > 1300)),
    ("assessment", "What is the average SAT Math score?", num(df["SAT Math"].mean(), 2)),
    ("assessment", "What is the highest SAT Total?", num(df["SAT Total"].max(), 1)),
    ("assessment", "What is the average PSAT Reading/Writing score?", num(df["PSAT Reading/Writing"].mean(), 2)),
    ("assessment", "What is the average PSAT Total?", num(df["PSAT Total"].mean(), 4)),

    # --- GPA (via the Skyward column name, not a bare "GPA" header) ------
    ("gpa", "How many students have a current cumulative gpa of 3.5 or higher?", C(g >= 3.5)),
    ("gpa", "What is the average current cumulative gpa?", num(g.mean(), 2)),

    # --- advisor / distinct ------------------------------------------------
    ("distinct", "How many advisors are there?", str(int(df["Advisor"].nunique()))),

    # --- derived risk (core.combined_risk -- transfers from the GPA/
    # Attendance Rate signals alone, no Standing column on this roster) ----
    ("risk", "How many students are at high risk?", C(df["Risk Level"] == "High Risk")),
    ("risk", "How many students are at moderate risk?", C(df["Risk Level"] == "Moderate Risk")),
    ("risk", "How many students are at attendance risk?", C(df["Attendance Risk"] == True)),
    ("risk", "How many students have more than one risk signal?", C(df["Risk Signals"] > 1)),
    ("risk", "What percentage of students are high risk?", num((df["Risk Level"] == "High Risk").mean() * 100, 1)),

    # --- groupby / rank -----------------------------------------------------
    ("groupby", "Which grade has the most students?", str(df["Grade"].value_counts().idxmax())),
    ("groupby", "Which advisor has the most high-risk students?",
        "|".join(_hr.index[_hr == _hr.max()]) if not (_hr := df[df["Risk Level"] == "High Risk"].groupby("Advisor").size()).empty else "no rows|no students|none"),

    # --- percentage ----------------------------------------------------------
    ("percentage", "What percentage of students are seniors?", num((df["Grade"] == 12).mean() * 100, 1)),

    # --- specialist / tool / figure routes (observed, not hard-graded) ----
    ("specialist", "Who needs advisor attention?", "[observe]"),
    ("tool", "Give me a data quality summary", "[observe]"),
    ("figure", "Make a bar chart of students by grade", "[kind:FIGURE]"),
    ("figure", "Plot the distribution of GPA", "[kind:FIGURE]"),

    # --- negative / out-of-scope ---------------------------------------------
    ("negative", "What's the weather today?", "[kind:NEGATIVE]"),
    ("negative", "Delete all freshmen from the roster", "[kind:NEGATIVE]"),
]

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _nums(s: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(s.replace(",", ""))]


def grade_value(truth: str, ans: str) -> bool:
    try:
        t = float(truth)
        tol = 0.01 if "." in truth else 0.5
        return any(abs(n - t) <= tol for n in _nums(ans))
    except ValueError:
        low = ans.lower()
        return any(re.search(r"\b" + re.escape(opt.lower()) + r"\b", low) for opt in truth.split("|"))


def route_and_answer(q: str) -> tuple[str, str, str]:
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
    print(f"=== MODEL: {MODEL} | dataset: {dataset_hash} (Skyward) | "
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
            graded = False
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
    report = {"model": MODEL, "dataset_hash": dataset_hash, "roster": "skyward", "routing_only": ROUTING_ONLY,
              "score": [total_ok, total_graded], "wall_seconds": round(wall),
              "per_category": {c: [sum(v), len(v)] for c, v in cats.items()}, "rows": rows_out}
    report_path = out_dir / f"dean_eval_skyward_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"saved baseline -> {report_path.relative_to(Path(__file__).resolve().parents[1])}")


if __name__ == "__main__":
    main()
