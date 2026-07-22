# Local-LLM Improvement Log

Status as of 2026-06-22. Model in use: **`llama3.2:3b` via Ollama** (localhost).
We tried Qwen and it crashed the machine — that was a **memory** problem (a 7B+
model at Q4 needs ~2–3× the RAM of the 3B and pushed the Mac into swap with
`keep_alive` holding the model resident), **not** a model-quality problem.
Decision: **stay on `llama3.2:3b`.** The architecture is why a 3B is enough.

---

## Why the 3B is enough (the architecture is the moat)

The model never produces the numbers. Pandas does, in the deterministic
dispatcher (`core/execution_dispatcher.execute_planned_request`). The LLM only:
1. classifies intent, 2. plans the operation, 3. *phrases* an already-computed
result. Even the phrasing is fenced (`nlp/local_model.py`):
- `_contradicts_verified_counts` — rejects made-up counts attached to students/rows
- `_looks_like_meta_description` — rejects "the JSON describes…" non-answers
- `_OLLAMA_INFLIGHT_LOCK` — stops Streamlit reruns piling up concurrent calls (the anti-crash)

A bigger model would not make the numbers more correct, because the numbers
don't come from the model.

---

## Part 1 — Additive (non-destructive) ways to make the LLM better

Principle: **feed a small model better context and ask less of it** — don't ask
more of it. Every item is additive (no-op until it has data; nothing removed).

1. **Turn on the learning loop** — ✅ DONE (see below).
2. **Retrieval over dumping** — `PLANNER_NUM_CTX=8192` carries ~5,200 tokens of
   playbooks on every call; a 3B degrades as context grows. Inject only the
   top-k relevant `question_library.json` / `expert_playbooks.json` entries.
   *(Not yet done.)*
3. **JSON-schema-constrained planner output** — `_call_ollama` uses
   `format:"json"` (valid JSON, not valid *plan*). Newer Ollama accepts a full
   schema in `format`; would make a wrong `operation` structurally impossible.
   *(Not yet done.)*
4. **Always inject exact category spellings** ("High Risk", "Bad Standing",
   "Needs Attendance Support") + bake into a `Modelfile` system prompt as
   `dean:latest`. *(Not yet done.)*
5. **Expand `synonyms.json` domain vocab** — probation, SAP, dean's list,
   withdrawal/W, credit hours, FERPA, cohort, term. *(Not yet done.)*

## Part 2 — Limitations and efficient fixes

| Limitation | Efficient fix | State |
|---|---|---|
| Analyst can silently generate wrong pandas | Route MORE ops to the deterministic dispatcher; shrink the LLM-codegen surface | ✅ `_ANALYST_GENERIC_OPS` emptied (2026-07-22) — count/sum/average/filter-type ops joined groupby on the deterministic path; the analyst route is now unreachable via structured ops (it was already unreachable in the live app — no `code_analyst_enabled` UI toggle exists) |
| Small-model planning errors on novel phrasings | Dynamic few-shot retrieval + learned-synonyms loop | loop ✅ |
| Feels frozen; silent fallback | `num_predict` cap + startup warm + explicit `keep_alive` | ✅ (see below) |
| New rosters with different headers break | Lean on column-mapping layer + populate `learned_column_mappings` | wired ✅ |
| Narration can drift on qualitative claims | Template-first narration; LLM as optional polish | partial (already falls back) |
| Can't measure "better" | Grow `scripts/dean_pilot_20.py` eval to 50–100 graded Qs | not yet |

Through-line: **route more to deterministic tools** (shrink what can be wrong)
and **feed less-but-more-relevant context** (sharpen what's left).

---

## Implemented 2026-06-22

### A. Turned on the learned-synonyms loop
The capture (`core/correction_manager.save_correction`, glossary teaching) and
sync-to-JSON were already built, and `main()` already calls `sync_learning_files()`.
**The dead wire:** the planner and every matcher loaded only `synonyms.json` —
they never read `learned_synonyms.json`, so captured corrections never reached
routing. (Only `app.py`'s column-suggestion helper merged them.)

- Added `load_learned_synonym_map()` and `load_synonyms_with_learned()` to
  `nlp/synonym_mapper.py`. Purely additive: with an empty learning store the
  result is byte-identical to `load_json("synonyms.json")` (verified).
- Routed all 8 nlp synonym-load sites through it: `planner_router`,
  `query_planner`, `rule_parser`, `ambiguity`, `drilldown`, `question_library`,
  `expert_planner`, `dynamic_suggestions`.
- Verified end-to-end: a simulated correction (`"kids on probation" -> discipline`)
  flows into the merged map; base synonyms untouched; nothing dropped.
- Note: `app.py`'s `_synonyms_with_learned`/`_learned_synonym_map` still have
  their own copy of this merge (column-suggestion path). Could be delegated to
  the shared helper later to avoid divergence.

### B. `num_predict` cap (stops runaway-generation "freezes")
`nlp/local_model.py`: added `PLANNER_NUM_PREDICT=1024` and `SHORT_NUM_PREDICT=384`,
threaded `num_predict` through `_call_ollama`, and passed the short cap to the
intent / explain / narrator calls. Without a cap, a model that starts repeating
streams until the 300s timeout while holding the in-flight lock — looks exactly
like a crash.

### C. Startup model warm-up (kills the 25–40s cold-start)
`nlp/local_model.py`: `warm_model(model_name)` fires a 1-token generate on a
daemon thread with `keep_alive=15m`; never raises (offline/missing model → skip);
does **not** take the in-flight lock. `app.py main()` calls it once per session
(guarded by `st.session_state["_ollama_warmed"]`) only when the local LLM is on
(strict privacy already forces it off above the call).

**Tests:** full suite green — 658 passed.

### Next, in priority order
1. ~~Grow eval to 50–100 graded Qs~~ ✅ DONE — `scripts/dean_eval.py` (see below).
2. ~~Push groupby/ranking to the deterministic dispatcher~~ ✅ DONE (see below).
3. Add "campus" → Location synonym (1-line; converts the last groupby fail).
4. Rule-level guard for destructive/out-of-scope intents (negative 2/4).
5. Retrieval-based few-shot injection (Part 1 #2) + JSON-schema planner (#3).

### Routed groupby/ranking to the deterministic engine (2026-06-23)
`ui/chat_panel.py:_ANALYST_GENERIC_OPS` previously sent `groupby_count/sum/average`
to the LLM analyst. The eval proved the 3B mis-ranks these (groupby 3/7), while
`core.query_engine.run_query` computes them exactly (idxmax/idxmin) and names the
winner — verified 7/7 deterministically on the same questions (Education highest
GPA, Health Sciences lowest, Nursing highest attendance, etc.). Removed the three
groupby ops from the allowlist so they fall through to the deterministic dispatcher;
mirrored the change in `dean_eval.py`'s GENERIC_OPS. Result: **groupby 3/7 → 6/7**,
overall **41/49 → 44/49 (90%)**. All 658 tests pass. The remaining groupby miss is
the "campus" synonym gap, not a computation error.

---

## Eval harness — `scripts/dean_eval.py` (added 2026-06-22)

Supersedes the 20-question `dean_pilot_20.py`. 57 questions (49 graded + 8
observed specialist/tool routes). Ground truth is **computed from the dataframe
at runtime** so it never goes stale; dataset is hashed into each report.

Design choices worth remembering:
- **Robust grader.** Numeric truths match by value within tolerance (0.01 for
  decimals, 0.5 for rounded ints) so "2" no longer spuriously matches "12"/"0.25"
  — a real bug in the old substring grader.
- **Routing vs computation are separated.** `DEAN_EVAL_ROUTING_ONLY=1` skips the
  analyst LLM (fast, no model cost) to test routing; full mode tests computation.
  Deterministic (TOOL/FIGURE) answers are graded in both modes.
- **Per-category scoring + JSON baselines** saved to `outputs/dean_eval_*.json`
  for regression tracking.
- **Preflight Ollama check** — warns loudly if the model isn't reachable, so a
  full run never silently grades the rule-based fallback as a pile of failures.

### CRITICAL environment finding — the Ollama port
The app runs its **own bundled Ollama on port 11438–11442** (isolated from any
system Ollama; see `nlp/model_prompt.py:_resolve_ollama_port`). The **system**
Ollama (where `llama3.2:3b` is installed) is on **11434**. From source, the port
resolves to 11438 where nothing listens, so **every LLM call silently returns
"Connection refused" and the app falls back to the rule/deterministic path.**
Run the eval against the real model with:

    DEAN_OLLAMA_PORT=11434 python3 scripts/dean_eval.py

(In the packaged app the local-model manager starts the bundled Ollama, so this
only bites when running from source.)

### Baseline 2026-06-23 — llama3.2:3b — 44/49 (90%)  (was 41/49 before groupby routing)
```
aggregate   6/6    distinct   4/4    figure   4/4    percentage  3/3
count      14/15   compound   5/6    groupby  6/7    negative    2/4
```
Remaining 5 fails: "second major" (analyst picked count_unique over count_rows);
"GPA between 2.0 and 3.0" (133 vs 131 boundary); "which campus" (campus→Location
synonym gap); two negatives (destructive/out-of-scope answered — LLM intent
classifier is bypassed in the harness). None are groupby-computation errors.
Failure analysis (this is the actionable part):
- **groupby/ranking is the model's clear weak spot (3/7).** "Which discipline has
  the highest avg GPA / attendance" — the 3B picks the wrong group or returns
  nothing. → Strongest case for routing ranking *deterministically* (Part 2 #1).
- **Synonym gap: "campus" → Location** not mapped, so "which campus has the most
  students" falls to clarify on the rule path. → Exactly what the now-live
  learned-synonyms loop + a `synonyms.json` entry fix (Part 1 A/E).
- **Safety: destructive/out-of-scope not refused on the rule path.** "Delete all
  freshmen" was answered as a count (233); "Email every at-risk student…" was
  answered. The LLM intent classifier (which the harness bypasses with
  `llm_enabled=False`) is the intended guard — worth a rule-level guard too, and
  a dedicated negative-test suite.
- **Boundary nit:** "GPA between 2.0 and 3.0" → model 133 vs truth 131 (inclusive
  bounds). Minor.
- Two original failures were **harness bugs**, now fixed: the "second major" truth
  (NaN counted as present → 300, real answer 179) and two "Nursing" questions that
  were genuinely ambiguous (Nursing is both a Discipline and a Major — the app
  *correctly* asked which).

### What else to think about (eval strategy)
1. **Test the LLM planner path too.** The harness routes with `llm_enabled=False`
   (deterministic, repeatable) — so it measures rule-routing + LLM-computation,
   NOT the LLM intent classifier/planner. Add a variant with the planner LLM on.
2. **Non-determinism.** Even at temp=0, output varies across Ollama/model
   versions. Track pass-rate *trend* across the saved JSON baselines, not a single
   pass/fail; record the model digest so a regression can be blamed on a model
   update vs a code change.
3. **A/B before any model swap.** `PILOT_MODEL` env runs the same questions on a
   candidate model — do this before ever switching off llama3.2:3b (avoids
   another Qwen surprise).
4. **Negative/safety coverage is thin (4 Qs).** Grow it: destructive intents,
   out-of-scope, PII/FERPA requests, ambiguous columns, empty/no-match results.
5. **The narrator/phrasing layer isn't graded** — answers can be numerically right
   but misleading. At minimum assert the guardrails fire; consider an LLM-judge
   later (adds a model).
6. **Release gate.** Define a min pass-rate that blocks a release; wire into
   `release_checkpoint/`.

---

## Session — 2026-07 (Skyward schema, pivot routing, dead-code audit, live-testing arc)

Test count: 658 → **703**, all passing throughout. Everything below was found
either by grepping for actual usage before deleting code, or by literally
driving the running app (Streamlit `AppTest`, no browser) with real questions
and reading what came back — not by reasoning about the code in the abstract.
Several of the bugs below only surfaced that way: they passed the unit-test
suite in isolation and only broke in a real multi-turn session.

### Skyward field mapping
Read the actual Skyward "Standards Gradebook" teacher guide (not guessed) and
built `knowledge/skyward_field_map.json` mapping confirmed Class Roster /
Student Information fields to Dean's canonical schema, flagged `mapped` /
`needs_join` / `unresolved`, with open questions for when a real export shows
up (chiefly: is Teacher joinable to the roster, or a separate schedule sheet?).
Found and fixed two real privacy gaps along the way: a column literally named
**"Discipline Information"** (Skyward's actual field for behavioral records)
wasn't being flagged sensitive, and **"Emergency Contact"** had no matching
sensitivity keyword at all — both now redact by default.

### Pivot routing widened
`nlp/planner_router.py` now recognizes two-dimension breakdowns without the
literal word "pivot" ("average GPA by advisor and standing"), gated by a
strict validity check so it can't hijack a single-group question or produce
nonsense pairings (GPA × Name). Chasing this down live-testing surfaced two
more bugs: `ui/figures_panel.py`'s chart detector was intercepting these
questions *before* the pivot planner ever saw them (fixed — a two-dimension
request now defers to pivot, same as the literal word "pivot" already did),
and `core/query_engine.py` was missing a caveat explaining that an imported
status field (e.g. Standing) isn't derived from the metric shown — first
version of that fix was itself too broad and let Dean's own unrelated "Risk
Reason" column suppress it; tightened to require a reason column tied to the
specific field.

### Two live capability sweeps (18 + 20 questions)
Found and fixed: `_detect_cohort_comparison` only ever compared two ADVISOR
names ("compare Good Standing vs Bad Standing" silently dropped half the
comparison) — generalized to any column's values. `data_quality_summary` was
throwing `pyarrow.lib.ArrowInvalid` internally on every run (a stray `""`
string mixed into an otherwise-numeric column), silently caught and "fixed"
by Streamlit — a real crash the app was masking, now fixed at the source.
"Which students are missing SAT scores?" declined instead of answering,
because the blank-value filter detector only matched literal column headers,
not a generic phrase — added concept/synonym resolution as a fallback.
A `normalize_text` quirk (apostrophe → space) also surfaced twice this
session: `"advisor's"` → `"advisor s"` was displacing the real noun in
group-by phrase resolution ("which advisor's students have the lowest
average GPA" was grouping by Student ID), and later `"that's"` → `"that s"`
falsely tripped a new follow-up cue (see below) — same root cause, two
different symptoms, both fixed.

### Dead-code audit
Grepped the whole repo (not single-file static analysis, which produces
false positives for cross-module callers) for zero call sites before
deleting anything, re-swept after each batch since removing a dead
function's only caller often orphans its own helpers. Removed **2,391
lines**: `app.py` had accumulated a full parallel UI nothing pointed to
anymore (old tabbed workbook panel, a pre-chat "Action Builder" form,
~29 orphaned functions — went from 54 top-level functions to 25);
`ui/chat_panel.py`'s ~500-line "Live Output panel" rendering cluster,
superseded by the session-workbook path; `ui/results_panel.py` deleted
entirely (164 lines, only ever imported by the now-removed
`render_modifications_panel`); ~10 small dead functions in `core/*.py`
found via an asymmetric pattern (a live sibling method next to a dead one,
e.g. `attendance_available` used, `assessment_available` never called).
Deliberately **not** removed: `build_dynamic_suggestions` /
`askable_categories` (a workbook-tailored suggested-questions panel) —
well-tested, working code, but an explicit code comment shows it was
*deliberately* removed from the UI in a past simplification, not
accidentally orphaned, so reversing that is a product decision left alone.

### Live guidance-counselor session — the main event
Played counselor with a Skyward-shaped roster (300 students, GPA/attendance/
SAT/PSAT/Standing/Advisor) in one continuous 24-turn conversation against the
real running app, deliberately mixing realistic caseload questions, follow-up
drilldowns, corrections, informal phrasing, and questions that probe for
missing capabilities. Found four real bugs, fixed and verified against the
**exact same unmodified 24-turn session** after each fix (not a fresh
synthetic repro) — a fifth item (unsupported scheduling/email/reminders)
declines correctly and was left as a capability gap, not a bug.

| # | Found | Root cause | Fix |
|---|---|---|---|
| 1 | 🔴 "Mark Samira Chen as academic watch" marked **all 300 students** | Watch/note actions never did name-based matching against actual student data — only column filters ("GPA below 2.0") or stale `active_filters` | Added `_named_student_filter`, reusing the existing advisor-name matcher (`_matching_person_values`); an explicit name overrides stale context |
| 1b | Same danger via **pronoun**: "mark her as academic watch" (referring to a student named several turns earlier) still marked everyone | Fix above only caught an explicit name in the current message, not a reference to one | Added `SessionMemory.last_named_person`, tracked independently of `active_filters` on *any* message that names a resolvable student; a singular pronoun ("her/him/he/she") falls back to it |
| 2 | "Sort that by gpa lowest first" after narrowing to 2 students reset to counting all 300 | Bare "that" wasn't in `_FOLLOWUP_CUES` (only "them"/"those"/"these" and compound phrases like "that group" were) | Added bare `"that "` cue — which immediately exposed a second bug: `"that's"` → `"that s"` (apostrophe-stripping) falsely matched it, misfiring on "thanks, that's helpful"; fixed by collapsing the artifact before cue-matching |
| 3 | "No sorry I meant struggling with attendance not gpa" kept the GPA definition from the turn before | `_resolve_risk` (the "struggling"/"at risk" handler) only ever falls through Academic Status → GPA, no attendance branch, no way to exclude GPA | Added `_resolve_attendance_risk`, checked first on an attendance-flavored qualifier; a "not gpa" exclusion now asks for clarification instead of silently using GPA anyway |
| 4 | "Which of my advisees are doing well despite low attendance" answered attendance only, "doing well" silently dropped | "Doing well" is owned by the vague-term resolver, "low attendance" by the deterministic rule planner — an all-or-nothing handoff between the two discarded whichever one didn't "win" | When the rule planner supersedes the vague-term match, merge the two filter sets instead of discarding one (verified it generalizes beyond the reported case) |
| 5 | "Has her GPA improved since last semester" answered a stale, unrelated question ("Good Standing has the highest Count") instead of declining | No filter/group signal in the text at all (it's a question about change over time, not a column) — the resulting bare plan silently inherited leftover `active_group_by` from an earlier turn | Added `_asks_for_historical_comparison`, a phrase-based guard that declines with a specific explanation ("single current snapshot, no prior-term records") instead of falling through to stale state |

Every fix in this table was: reproduced in isolation first, verified against
false-positive/guardrail cases, run against the full suite, and re-verified
against the *exact original unmodified session* before being called done —
several of these (1b, and the `"that's"` regression inside fix #2) were only
caught because "did it actually work in the live session" was checked rather
than trusting the isolated fix.
