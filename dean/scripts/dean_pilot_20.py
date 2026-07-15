"""Ad-hoc 20-question dean pilot: route each question exactly like the live app
(generic reads -> code analyst; specialized ops -> deterministic dispatcher;
charts -> figure path) and grade against computed ground truth."""
from __future__ import annotations
import sys, re, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from core.excel_loader import load_excel_workbook, LoadedWorkbook
from core.data_sources import DataSourceRegistry
from core.execution_dispatcher import execute_planned_request
from nlp.planner_router import plan_user_request
from nlp.code_analyst import analyze, default_llm_call
from ui.figures_panel import is_chart_request, detect_chart_intent

GENERIC_OPS = {"filtered_preview","count_rows","count_unique","list_unique",
               "average_column","sum_column","min_column","max_column",
               "groupby_count","groupby_sum","groupby_average"}

class Upload:
    def __init__(s,p): s.path=Path(p); s.name=s.path.name
    def getvalue(s): return s.path.read_bytes()

loaded = load_excel_workbook(Upload('/Users/albertopaz/azera-formatting/mock_dean_student_roster_with_assessments_attendance_cleaned.xlsx'))
reg = DataSourceRegistry(); reg.set_roster(loaded)
sheets = reg.enriched_sheets() or dict(loaded.sheets)
sel = reg.enriched_roster_sheet or next(iter(sheets))
df = sheets[sel]
enriched = LoadedWorkbook(file_name=loaded.file_name, workbook=loaded.workbook, sheets=sheets, warnings=[])
sheet_columns = {n: list(f.columns) for n,f in sheets.items()}
settings = {"use_local_llm": True, "llm_enabled": True, "strict_privacy_mode": False,
            "code_analyst_enabled": True, "planner_model": "llama3.2:3b", "planner_timeout_seconds": 300}
MODEL = os.environ.get("PILOT_MODEL", "llama3.2:3b")
call = default_llm_call(MODEL, timeout=200)
g = pd.to_numeric(df["GPA"], errors="coerce")

# (question, ground-truth-note or None)
QS = [
 ("How many students are at high risk?", str(int((df["Risk Level"]=="High Risk").sum()))),
 ("What is the average GPA?", f"{g.mean():.2f}"),
 ("How many students have a GPA below 2.0?", str(int((g<2.0).sum()))),
 ("How many students need attendance support?", str(int((df["Attendance Category"]=="Needs Attendance Support").sum()))),
 ("Which discipline has the lowest average GPA?", df.assign(G=g).groupby("Discipline")["G"].mean().idxmin()),
 ("How many students are in bad standing?", str(int((df["Standing"]=="Bad Standing").sum()))),
 ("How many different majors are there?", str(int(df["Major"].nunique()))),
 ("How many seniors are at high risk?", str(int(((df["Year"]=="Senior")&(df["Risk Level"]=="High Risk")).sum()))),
 ("What percentage of students are high risk?", f"{(df['Risk Level']=='High Risk').mean()*100:.0f}"),
 ("Which advisor has the most high-risk students?", df[df["Risk Level"]=="High Risk"].groupby("Advisor").size().idxmax()),
 ("What is the average SAT Total?", f"{df['SAT Total'].mean():.0f}"),
 ("How many students have a GPA below 2.0 and need attendance support?",
    str(int(((g<2.0)&(df["Attendance Category"]=="Needs Attendance Support")).sum()))),
 ("Which discipline has the highest average attendance rate?", df.groupby("Discipline")["Attendance Rate"].mean().idxmax()),
 # specialist / tool routes (graded on routing, not number)
 ("Who needs advisor attention?", "[specialist:intervention]"),
 ("Which advisors need support?", "[specialist:advisor]"),
 ("Which advisors are doing a good job?", "[specialist:advisor]"),
 ("Show me the trends in this data", "[specialist:trend]"),
 ("Average GPA by advisor", "[pivot/groupby]"),
 ("Build an advisor intervention dashboard", "[dashboard]"),
 ("Make a bar chart of students by risk level", "[figure]"),
]

def route_and_answer(q):
    if is_chart_request(q):
        intent = detect_chart_intent(q, sheet_columns[sel])
        return "FIGURE", f"chart intent: field={getattr(intent,'field',None)} metric={getattr(intent,'metric',None)} type={getattr(intent,'chart_type',None)}"
    r = plan_user_request(user_message=q, sheets=sheets, sheet_columns=sheet_columns,
                          selected_sheet=sel, conversation_state={}, settings={"llm_enabled": False})
    intent = r.get("intent"); plan = r.get("plan") or {}; op = plan.get("operation","")
    if intent in {"clarify","unavailable","unsupported"}:
        return "CLARIFY", (r.get("confirmation_reason") or r.get("fallback_reason") or "")[:100]
    if intent != "query":
        return f"INTENT={intent}", f"op={op} pending={r.get('pending_type')} confirm={r.get('requires_confirmation')}"
    if op in GENERIC_OPS:
        res = analyze(user_message=q, df=df, llm_call=call, verify=False)
        return "ANALYST", (res.answer or "(none)").replace("\n"," ")[:160]
    # specialized query op -> deterministic dispatcher
    try:
        resp = execute_planned_request(r, enriched, settings, reveal_sensitive=False,
                                       request_summary=q, session_workbook=None)
        msg = (resp.get("message") or "").replace("\n"," ")
        return f"TOOL:{op}", msg[:160]
    except Exception as e:
        return f"TOOL:{op}", f"(dispatch error: {e})"

print(f"=== MODEL: {MODEL} ===")
correct = graded = 0
analyst_time = 0.0
for i,(q,truth) in enumerate(QS,1):
    t0 = time.monotonic()
    kind, ans = route_and_answer(q)
    dt = time.monotonic() - t0
    if kind == "ANALYST":
        analyst_time += dt
    verdict = ""
    if truth and not truth.startswith("["):
        graded += 1
        ok = truth.lower() in ans.lower()
        correct += ok
        verdict = "  ✓" if ok else f"  ✗ expected~{truth}"
    print(f"\nQ{i:>2}. {q}")
    print(f"     -> [{kind}] {ans}{verdict}   ({dt:.1f}s)")
print(f"\n=== {MODEL}: {correct}/{graded} graded correct | analyst wall-time {analyst_time:.0f}s ===")
