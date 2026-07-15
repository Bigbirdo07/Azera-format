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
| Analyst can silently generate wrong pandas | Route MORE ops to the deterministic dispatcher; shrink the LLM-codegen surface | not yet |
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
