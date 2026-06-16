"""Free-form local data analyst: write pandas code, run it, answer.

This is the "like Claude on a spreadsheet" path. Instead of forcing the model
to emit a fixed-operation plan blind (see ``nlp.planner_router``), we let the
model SEE the schema + a small sample, write Python against the real DataFrame,
run that code in a locked-down namespace, show it the output, and iterate until
it answers in plain English.

Everything runs locally and in-process — the workbook never leaves the machine.
The model only ever touches a *copy* of the frame, the original is untouched,
and the sandbox blocks imports / file / network access so generated code cannot
do anything but compute over `df`.

Read-only by design: this path never writes the workbook. Edits/exports keep the
deterministic confirm-and-write-new-workbook pipeline in the planner.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.privacy import is_hidden_by_default

MAX_ITERATIONS = 4
MAX_OUTPUT_CHARS = 4000
SAMPLE_ROWS = 5
EXEC_TIMEOUT_SECONDS = 10  # advisory; enforced by the caller's thread budget

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class AnalysisResult:
    answer: str
    code_steps: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    error: str | None = None
    iterations: int = 0
    grounded: bool = False  # True only if real code ran and produced output
    plan: str = ""          # the model's stated approach, when plan-first ran
    verified: bool | None = None  # True=cross-check agreed, False=disagreed, None=couldn't check
    confidence: str = "unknown"   # "high" | "medium" | "low"

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer.strip())


# --- prompt -----------------------------------------------------------------


def _schema_block(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique(dropna=True)
        lines.append(f"- {col!r} ({dtype}, {nunique} distinct)")
    return "\n".join(lines)


def _sample_block(df: pd.DataFrame, rows: int = SAMPLE_ROWS) -> str:
    """A few example rows, with hidden-by-default columns masked in the preview.

    Code can still read those columns; we just don't paste their raw values
    (emails, IDs) into the prompt by default.
    """
    preview = df.head(rows).copy()
    for col in preview.columns:
        if is_hidden_by_default(col):
            preview[col] = "<hidden>"
    return preview.to_string(index=False, max_colwidth=24)


def _match_column(word: str, columns: list[str]) -> str | None:
    """Map a noun from the question to an actual column name (loose match)."""
    target = word.strip().lower().rstrip("s")  # crude singularize: advisors->advisor
    for col in columns:
        norm = col.lower()
        if norm == word.strip().lower() or norm.rstrip("s") == target or target in norm.split():
            return col
    return None


# Sentence-shape cues. Each entry is (regex, hint-builder). The hint builder
# receives the regex match + the column list and returns a guidance string (or
# None). These are *soft* interpretation aids prepended to the prompt — they
# tell a small model what SHAPE of pandas operation the phrasing implies, which
# is exactly where llama3.2:3b picks the wrong strategy on its own.
_PER_GROUP_AVG_RE = re.compile(
    r"\b(below|under|above|over|less than|greater than|more than|worse than|better than)\b"
    r"[^.]*?\baver(?:age|ge)?|mean\b[^.]*?\b(their|its|each|own|the same)\b",
    re.IGNORECASE,
)
_GROUPBY_RE = re.compile(r"\b(?:per|by|for each|grouped by|broken down by|in each|across)\s+([a-z]+)", re.IGNORECASE)
_RANK_RE = re.compile(r"\b(most|highest|largest|greatest|top|fewest|lowest|least|smallest|bottom|rank(?:ed)?)\b", re.IGNORECASE)
_AVG_RE = re.compile(r"\b(average|avg|mean|median)\b", re.IGNORECASE)
_COUNT_RE = re.compile(r"\b(how many|number of|count of|count|total number)\b", re.IGNORECASE)
_DISTINCT_RE = re.compile(r"\b(distinct|unique|different|how many kinds|how many types)\b", re.IGNORECASE)
_PROPORTION_RE = re.compile(r"\b(percent|percentage|proportion|share|fraction|rate|%)\b", re.IGNORECASE)
_OWN_GROUP_RE = re.compile(r"\b(?:their|its)\s+(?:own\s+)?([a-z]+)(?:'s)?\b", re.IGNORECASE)
# "which/what <group> has the <superlative> [average/sum] [<metric>] …"
_WHICH_RANK_RE = re.compile(
    r"\b(?:which|what)\s+(?P<group>[a-z][a-z ]*?)\s+(?:has|have|had)\s+the\s+"
    r"(?P<sup>highest|lowest|most|fewest|least|best|worst|greatest|largest|smallest|top|bottom)\b"
    r"(?P<rest>.*)",
    re.IGNORECASE,
)
_AGG_WORD_TO_FN = {"average": "mean", "avg": "mean", "mean": "mean",
                   "median": "median", "sum": "sum", "total": "sum"}
_AGG_WORD_RE = re.compile(r"\b(average|avg|mean|median|sum|total)\b", re.IGNORECASE)
_MIN_SUPERLATIVES = {"lowest", "fewest", "least", "worst", "smallest", "bottom"}


def _which_rank_hint(text: str, columns: list[str]) -> str | None:
    """The single highest-value advisor pattern: "which X has the most/highest …".

    A small model reflexively reaches for value_counts() here, which counts ROWS
    — wrong when the metric is an average, a sum, or a *filtered* count. We
    resolve the group column and (if present) the metric column and hand back the
    exact one-liner, with an explicit don't-use-value_counts warning.
    """
    match = _WHICH_RANK_RE.search(text)
    if not match:
        return None
    group_phrase = match.group("group").strip()
    group_col = _match_column(group_phrase.split()[-1], columns) or _match_column(group_phrase, columns)
    if not group_col:
        return None
    idx = "idxmin" if match.group("sup").lower() in _MIN_SUPERLATIVES else "idxmax"
    rest = match.group("rest") or ""
    agg = _AGG_WORD_RE.search(rest)
    metric_col = next(
        (c for c in columns
         if c != group_col and re.search(rf"\b{re.escape(c.lower())}\b", rest.lower())),
        None,
    )
    if agg and metric_col:
        fn = _AGG_WORD_TO_FN[agg.group(1).lower()]
        return (
            f"To find which {group_col} has the {match.group('sup')} {agg.group(1)} "
            f"{metric_col}, aggregate then pick the winner: "
            f"df.groupby('{group_col}')['{metric_col}'].{fn}().{idx}(). "
            f"Do NOT use value_counts() — that counts rows, not the {agg.group(1)}."
        )
    return (
        f"To find which {group_col} has the {match.group('sup')} students: if the "
        f"question names a subset (e.g. at-risk, on probation), filter to it FIRST, "
        f"then count per group: df[<condition>].groupby('{group_col}').size().{idx}() "
        f"(omit the filter if the question is about all students). "
        f"Do NOT pass a condition into value_counts()."
    )


def build_intent_hints(message: str, columns: list[str]) -> list[str]:
    """Translate sentence structure into operation-shape hints for the model.

    Examples of what it catches:
      "average GPA per department"        -> group by Department, take the mean
      "below their own department's avg"  -> per-group comparison via transform
      "which advisor has the most"        -> rank / idxmax over a groupby count
    """
    hints: list[str] = []
    text = message or ""

    # Highest-value pattern first so it leads the list: "which X has the most/
    # highest/lowest ..." (the value_counts trap).
    which_rank = _which_rank_hint(text, columns)
    if which_rank:
        hints.append(which_rank)

    # "X per/by/for-each Y" -> group by Y.
    for m in _GROUPBY_RE.finditer(text):
        col = _match_column(m.group(1), columns)
        if col:
            hints.append(f'"{m.group(0)}" means group by `{col}` — use df.groupby("{col}").')

    # "below/above their OWN group's average" -> per-group comparison. This is
    # the exact shape the model botched: it needs transform(), not mean().
    if _PER_GROUP_AVG_RE.search(text):
        group_col = None
        own = _OWN_GROUP_RE.search(text)
        if own:
            group_col = _match_column(own.group(1), columns)
        target = f'`{group_col}`' if group_col else "that group"
        hints.append(
            "Comparing each row to its OWN group's average needs a group transform, "
            "not a plain mean. Pattern: "
            f'grp_avg = df.groupby({target if group_col else "<group>"})["<value>"].transform("mean"); '
            'then filter df[df["<value>"] < grp_avg].'
        )

    if _RANK_RE.search(text):
        hints.append(
            'Words like most/highest/fewest/top/rank ask for a ranking: compute the '
            'per-group value, then .sort_values() and take the top row, or use '
            '.idxmax()/.idxmin() to get the single winner.'
        )
    if _DISTINCT_RE.search(text):
        hints.append('"distinct/unique" means count separate values with .nunique() (not len).')
    elif _COUNT_RE.search(text):
        hints.append('"how many / number of" means count rows: len(df_filtered) or .shape[0].')
    if _AVG_RE.search(text):
        hints.append('"average/mean" -> .mean(); "median" -> .median(), on the numeric column named in the question.')
    if _PROPORTION_RE.search(text):
        hints.append('A percentage/rate is a ratio: matching_count / total_count (multiply by 100 for percent).')

    # Dedupe while preserving order.
    seen: set[str] = set()
    out = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


_MISSING_TOKENS = {"n/a", "na", "none", "null", "nan", "-", "--", "unknown", "tbd", "?", "n.a."}


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Deterministically clean a COPY of the frame so the model works on tidy
    data instead of being told (and often failing) to clean it itself.

    Applies only safe, value-preserving fixes:
      - numbers stored as text  -> pd.to_numeric(errors='coerce')
      - stray whitespace in text -> stripped (NaN preserved)
      - 'N/A'/'-'/'null' placeholders -> real NaN

    Deliberately does NOT change letter-casing of categoricals — that's a
    judgment call (is "Nursing" the same as "nursing"?) left to an advisory note.
    Returns the cleaned copy and a list of human-readable actions taken.
    """
    out = df.copy()
    actions: list[str] = []
    try:
        from core.schema import infer_column_types

        types = infer_column_types(df)
    except Exception:
        types = {}

    for column, info in types.items():
        if info.get("analysis_dtype") == "numeric" and not pd.api.types.is_numeric_dtype(out[column]):
            coerced = pd.to_numeric(out[column], errors="coerce")
            newly_blank = int(coerced.isna().sum() - out[column].isna().sum())
            out[column] = coerced
            suffix = f" ({newly_blank} non-numeric value(s) became blank)" if newly_blank > 0 else ""
            actions.append(f"converted `{column}` from text to numbers{suffix}")

    for column in out.columns:
        if out[column].dtype != object:
            continue
        stripped = out[column].str.strip()  # .str preserves NaN
        sentinel_mask = stripped.str.casefold().isin(_MISSING_TOKENS)
        cleaned = stripped.mask(sentinel_mask)
        if not cleaned.equals(out[column]):
            bits = []
            if not stripped.equals(out[column]):
                bits.append("trimmed whitespace")
            if bool(sentinel_mask.any()):
                bits.append("treated placeholder values (N/A, -, …) as blank")
            out[column] = cleaned
            if bits:
                actions.append(f"`{column}`: " + " and ".join(bits))

    return out, actions


def casing_advisory(df: pd.DataFrame, max_notes: int = 5) -> list[str]:
    """Categorical columns whose values collapse under lower-casing — a judgment
    call we surface but do not auto-apply."""
    notes: list[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        lowered = values.str.lower()
        raw_unique, low_unique = values.nunique(), lowered.nunique()
        if 0 < low_unique < raw_unique and low_unique <= 50:
            notes.append(
                f"`{column}` has values differing only by letter-case "
                f"({raw_unique} distinct → {low_unique} when lower-cased). If those are the "
                f"same category, apply .str.lower() before grouping/filtering."
            )
        if len(notes) >= max_notes:
            break
    return notes


def build_cleaning_block(actions: list[str], advisory: list[str]) -> str:
    """Render what was auto-cleaned plus any remaining judgment-call advisories."""
    if not actions and not advisory:
        return ""
    lines: list[str] = []
    if actions:
        lines.append("The data was auto-cleaned before you see it: " + "; ".join(actions) + ".")
    for note in advisory:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def build_value_hints(message: str, df: pd.DataFrame, max_unique: int = 25, max_hints: int = 5) -> str:
    """Tell the model which column a mentioned category value lives in.

    Prevents the worst silent error: filtering the WRONG column for a value that
    exists elsewhere (e.g. 'on probation' → Financial Aid Status, which has no
    'Probation', so the code prints 0 — grounded, self-consistent, and wrong).
    For values in one column it pins the filter; for values in several it lists
    them so the model picks an intended one rather than an unrelated column.
    """
    norm_msg = f" {_normalize_for_match(message)} "
    value_to_columns: dict[str, list[str]] = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        values = df[column].dropna().astype(str).unique()
        if not (0 < len(values) <= max_unique):
            continue
        for value in values:
            v = value.strip()
            if len(v) < 3:
                continue
            if f" {_normalize_for_match(v)} " in norm_msg:
                value_to_columns.setdefault(v, [])
                if column not in value_to_columns[v]:
                    value_to_columns[v].append(column)

    lines: list[str] = []
    single_conditions: list[str] = []  # (column unambiguous) value conditions
    for value, cols in list(value_to_columns.items())[:max_hints]:
        if len(cols) == 1:
            lines.append(f"- '{value}' is a value in column `{cols[0]}` — filter df[df['{cols[0]}'] == '{value}'].")
            single_conditions.append(f"(df['{cols[0]}'] == '{value}')")
        else:
            joined = ", ".join(f"`{c}`" for c in cols)
            lines.append(f"- '{value}' appears in columns {joined} — filter one of those (not any other column).")
    if not lines:
        return ""
    block = "Where the mentioned values live (use these exact columns):\n" + "\n".join(lines) + "\n"
    # Multi-condition guard: when the question names values from 2+ different
    # columns, a small model often keeps only one. Hand it the combined filter so
    # it can't drop a condition.
    distinct_columns = {cols[0] for v, cols in value_to_columns.items() if len(cols) == 1}
    if len(single_conditions) >= 2 and len(distinct_columns) >= 2:
        combined = " & ".join(single_conditions)
        block += (f"This question has MULTIPLE conditions — combine them ALL with &, "
                  f"don't drop any: df[{combined}].\n")
    return block


def build_value_domains(df: pd.DataFrame, max_unique: int = 6, max_cols: int = 10) -> str:
    """List the exact values of *low-cardinality* categorical columns.

    Without this, the model invents near-miss spellings — 'Need Attendance
    Support' instead of the real 'Needs Attendance Support' — and silently
    filters to zero rows. Showing the real domain pins the exact strings.

    Deliberately conservative (<=6 distinct, <=10 columns): a small model is
    sensitive to prompt size, so dumping big domains (21 advisors, 29 majors)
    bloats the prompt and degrades unrelated questions. We only pin the short
    enumerations where exact spelling actually matters for a filter.
    Hidden-by-default columns (names/emails) are skipped.
    """
    lines: list[str] = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]) or is_hidden_by_default(column):
            continue
        values = df[column].dropna().astype(str).unique()
        if 0 < len(values) <= max_unique:
            shown = ", ".join(repr(str(v)) for v in values)
            lines.append(f"- {column}: {shown}")
        if len(lines) >= max_cols:
            break
    if not lines:
        return ""
    return "Exact values for category columns (use these spellings verbatim):\n" + "\n".join(lines) + "\n"


def build_system_prompt(df: pd.DataFrame) -> str:
    domains = build_value_domains(df)
    domains_block = f"\n{domains}" if domains else ""
    return (
        "You are a data analyst answering questions about a pandas DataFrame "
        "named `df`. To answer, write a SINGLE Python code block that computes "
        "the answer and prints it with print(). You will see the printed output, "
        "then you can write another code block to refine, or give the final "
        "answer.\n\n"
        "Rules:\n"
        "- The DataFrame is already loaded as `df`. `pd` and `np` are available.\n"
        "- Do NOT import anything, read/write files, or use input().\n"
        "- Reference columns by their exact names shown below.\n"
        "- For category filters, use the exact values listed — do not guess spellings.\n"
        "- Always print() what you want to see — output is captured from stdout.\n"
        "- When you have the answer, reply WITHOUT a code block, in one or two "
        "plain-English sentences. Include the concrete numbers.\n\n"
        f"Columns:\n{_schema_block(df)}\n"
        f"{domains_block}\n"
        f"Sample rows:\n{_sample_block(df)}\n"
    )


_CODEY_RE = re.compile(r"print\(|df\[|df\.|\.groupby|\.idxm|\)\.count\(|```|=\s*df")


def _looks_like_code(text: str) -> bool:
    """True if the model handed back code where we asked for a sentence."""
    return bool(_CODEY_RE.search(text or ""))


def _answer_consistent(answer: str, output: str) -> bool:
    """Does the prose answer actually match what the code printed?

    Targets the "right code, wrong words" failure: a small model runs correct
    code (prints 'Freshman' / 'Dr. Patel' / '244') then states a different value
    in prose. We only judge when the output is a single short scalar/label —
    multi-line tables are left alone (too noisy to compare cheaply).
    """
    out = (output or "").strip()
    if not out or "\n" in out or len(out) > 40:
        return True
    nums_out = _numbers(out)
    if nums_out:
        return bool(nums_out & _numbers(answer))
    return out.lower() in (answer or "").lower()


def _present(output: str) -> str:
    """Turn a captured stdout into a user-facing answer when the model fails to
    phrase one itself. Short single-line outputs are shown as-is; multi-line
    outputs (tables) get a lead-in."""
    text = (output or "").strip()
    if not text:
        return ""
    return text if "\n" not in text else f"Here is the result:\n{text}"


def _extract_code(text: str) -> str | None:
    match = _CODE_BLOCK_RE.search(text or "")
    if match:
        return match.group(1).strip()
    # Some small models emit bare code with no fence. Treat a short reply that
    # carries clear code tokens (print(/df[/.groupby/…) as code. We deliberately
    # do NOT use sentence punctuation as a signal: string literals legitimately
    # contain '. ' (e.g. names like 'Dr. Patel'), which the old heuristic misread
    # as prose — silently dropping correct code for any advisor-name question.
    stripped = (text or "").strip()
    if stripped and stripped.count("\n") <= 8 and _looks_like_code(stripped):
        return stripped
    return None


# --- sandbox ----------------------------------------------------------------


# Modules the model may "import" — they're already in the namespace, so the
# import is a no-op redirect to the module we control. Everything else (os, sys,
# subprocess, socket, shutil, pathlib, requests, importlib, ...) is rejected.
_ALLOWED_IMPORTS = {
    "pandas": pd, "numpy": np, "math": __import__("math"),
    "re": re, "datetime": __import__("datetime"),
    "statistics": __import__("statistics"),
    "collections": __import__("collections"),
    "itertools": __import__("itertools"),
    "json": __import__("json"),
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in _ALLOWED_IMPORTS:
        return _ALLOWED_IMPORTS[root]
    raise ImportError(f"import of '{name}' is not allowed in the analyst sandbox")


_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "format": format, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min, "print": print,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "slice": slice, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
    "zip": zip, "True": True, "False": False, "None": None,
    "__import__": _safe_import,
}

# Block the escape hatches that don't go through __import__: file/eval/exec and
# attribute-introspection tricks. Benign `import pandas` is NOT matched here
# (the leading word boundary excludes the dunder `__import__`); it is routed
# through `_safe_import` above.
_FORBIDDEN_RE = re.compile(
    r"\b(__import__|open|exec|eval|compile|globals|locals|vars|getattr|"
    r"setattr|delattr|input|breakpoint|__\w+__)\b"
)


def make_namespace(df: pd.DataFrame) -> dict[str, Any]:
    """A fresh execution namespace over a *copy* of df (original untouched)."""
    return {
        "__builtins__": _SAFE_BUILTINS,
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }


def run_sandboxed(
    code: str,
    df: pd.DataFrame | None = None,
    *,
    namespace: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Execute model-written code in a locked-down namespace. Returns (stdout, error).

    Pass ``namespace`` to persist variables across steps (notebook-style): a
    name bound in one step is still defined in the next, so the model doesn't
    re-derive (or lose) intermediate results. When omitted, a fresh namespace
    over ``df`` is created. The namespace exposes only df/pd/np and a safe
    builtins subset; a static check rejects file/eval/attribute escapes. This is
    a local convenience guard, not a hostile-code sandbox — but it stops the
    common ways generated code reaches the filesystem or network.
    """
    forbidden = _FORBIDDEN_RE.search(code or "")
    if forbidden:
        return "", f"Blocked disallowed call: {forbidden.group(0)}"
    if namespace is None:
        if df is None:
            raise ValueError("run_sandboxed needs either df or namespace")
        namespace = make_namespace(df)
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(compile(code, "<analyst>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # surfaced back to the model so it can self-correct
        out = buffer.getvalue()
        return out[:MAX_OUTPUT_CHARS], f"{type(exc).__name__}: {exc}"
    return buffer.getvalue()[:MAX_OUTPUT_CHARS], None


# --- loop -------------------------------------------------------------------


def _render_history(history: list[dict[str, Any]], max_turns: int = 3) -> str:
    """Render the recent conversation so the model can resolve follow-ups.

    Includes the prior question, the code that answered it, and the answer, so a
    reference like "those" / "just the Biology ones" can be reconstructed by
    extending the earlier filter rather than starting blind.
    """
    if not history:
        return ""
    lines = ["Conversation so far (oldest first) — use this to resolve "
             "references like 'those', 'them', 'that', 'just the …':"]
    for turn in history[-max_turns:]:
        lines.append(f"\nEarlier question: {turn.get('question', '')}")
        if turn.get("code"):
            lines.append(f"Code that answered it:\n{turn['code']}")
        lines.append(f"Answer was: {turn.get('answer', '')}")
    return "\n".join(lines) + "\n\n"


_MULTISTEP_RE = re.compile(
    r"\b(their own|each|per|average|median|compared|relative to)\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _looks_multistep(message: str) -> bool:
    """Cheap heuristic: is this worth a planning pre-step? Multi-clause or
    per-group/aggregate questions benefit; simple lookups don't."""
    text = message or ""
    if _MULTISTEP_RE.search(text):
        return True
    return " and " in text.lower() and len(text.split()) >= 12


def make_plan(user_message: str, df: pd.DataFrame, llm_call: Callable[[str], str]) -> str:
    """Ask the model for a terse numbered approach before it writes code. This
    nudges a small model to decompose instead of lunging at a one-liner."""
    prompt = (
        "You are about to answer a question with pandas over a DataFrame `df`.\n"
        f"Columns: {', '.join(map(str, df.columns))}\n"
        f"Question: {user_message}\n\n"
        "List the steps to compute the answer — numbered, terse, at most 3 steps. "
        "Name the columns and operations (filter / groupby / transform / mean / "
        "count / sort). Do NOT write code yet."
    )
    try:
        return (llm_call(prompt) or "").strip()[:500]
    except Exception:
        return ""


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for token in _NUMBER_RE.findall(text or ""):
        try:
            out.add(round(float(token), 4))
        except ValueError:
            continue
    return out


def _outputs_agree(primary: str, check: str) -> bool | None:
    """Compare two stdout blocks by the numbers they contain. Returns True if
    they share a value, False if both have numbers but none overlap, None if the
    comparison isn't meaningful (no numbers on a side)."""
    a, b = _numbers(primary), _numbers(check)
    if not a or not b:
        return None
    return bool(a & b)


def verify_answer(
    user_message: str,
    df: pd.DataFrame,
    llm_call: Callable[[str], str],
    primary_code: str,
    primary_output: str,
) -> tuple[bool | None, str]:
    """Independently re-derive the result a DIFFERENT way and compare.

    Runs in a FRESH namespace so it can't reuse the primary run's variables.
    Conservative: a verification that errors or yields no number is treated as
    'couldn't check' (None), not as a contradiction — we only return False when
    a clean cross-check produces numbers that disagree.
    """
    prompt = (
        "Double-check an analysis of DataFrame `df`.\n"
        f"Columns: {', '.join(map(str, df.columns))}\n"
        f"Question: {user_message}\n"
        f"The answer was computed by:\n{primary_code}\n"
        f"which printed:\n{primary_output}\n\n"
        "Write a SHORT, DIFFERENT piece of Python (a different method) that "
        "recomputes the key number to confirm it. Print only that number. "
        "`pd`/`np` are available; do not import or read files."
    )
    try:
        reply = llm_call(prompt) or ""
    except Exception:
        return None, ""
    code = _extract_code(reply)
    if not code:
        return None, ""
    out, err = run_sandboxed(code, namespace=make_namespace(df))
    if err or not out.strip():
        return None, out
    return _outputs_agree(primary_output, out), out


def _assess_confidence(result: "AnalysisResult") -> str:
    if not result.grounded:
        return "low"
    if result.verified is False:
        return "low"
    error_steps = sum(1 for o in result.outputs if "ERROR:" in o)
    if result.verified is True and error_steps == 0:
        return "high"
    if result.verified is None and error_steps == 0 and result.iterations <= 2:
        return "high"
    return "medium"


def analyze(
    *,
    user_message: str,
    df: pd.DataFrame,
    llm_call: Callable[[str], str],
    history: list[dict[str, Any]] | None = None,
    glossary: str = "",
    clean_hints: bool = True,
    plan_first: bool = False,
    verify: bool = True,
    max_iterations: int = MAX_ITERATIONS,
) -> AnalysisResult:
    """Run the write-code → execute → observe → answer loop.

    `llm_call` takes a single prompt string and returns the model's raw text
    (free-form, NOT json mode). The whole transcript is rebuilt into one prompt
    each turn so this works with the stateless Ollama /generate endpoint.

    `history` is the recent conversation (list of {question, answer, code}) so
    the analyst can answer follow-ups in context instead of one-shot.

    `glossary` is a prebuilt block of durable definitions/column aliases the
    user has taught (see nlp.glossary), applied so the model shares their
    vocabulary across sessions.
    """
    # Deterministically clean a copy first, then run everything (schema, sample,
    # code, verification) against the tidy frame.
    work_df = df
    if clean_hints:
        work_df, actions = clean_frame(df)
        cleaning_block = build_cleaning_block(actions, casing_advisory(work_df))
    else:
        cleaning_block = ""

    system = build_system_prompt(work_df)
    if glossary:
        system += "\n" + glossary
    if cleaning_block:
        system += "\n" + cleaning_block
    value_hints = build_value_hints(user_message, work_df)
    if value_hints:
        system += "\n" + value_hints
    hints = build_intent_hints(user_message, list(work_df.columns))
    if hints:
        system += "\nInterpretation hints (apply only if they fit the question):\n" + \
                  "\n".join(f"- {h}" for h in hints) + "\n"
    if history:
        system += (
            "\nThis is a multi-turn conversation. If the new question refers to "
            "a previous result, reconstruct the earlier filter from the code "
            "shown and combine it with the new condition.\n"
        )
    transcript = _render_history(history or []) + f"Question: {user_message}\n"
    result = AnalysisResult(answer="")

    # Plan-then-code: get a terse approach first so the model decomposes instead
    # of lunging at a wrong one-liner. OFF by default — measured worse on
    # llama3.2:3b, whose self-generated plan is as flawed as its code and can
    # override the (better) intent hints. Keep the lever for stronger models.
    if plan_first and _looks_multistep(user_message):
        result.plan = make_plan(user_message, work_df, llm_call)
        if result.plan:
            transcript += f"Approach to follow:\n{result.plan}\n"

    ran_successfully = False
    answered = False
    last_good_output = ""  # stdout of the most recent step that ran cleanly
    # One persistent namespace for the whole turn: variables bound in step 1
    # survive into step 2 (notebook-style), so the model stops losing state.
    namespace = make_namespace(work_df)

    for step in range(1, max_iterations + 1):
        result.iterations = step
        prompt = f"{system}\n{transcript}\nYour turn:"
        try:
            reply = llm_call(prompt) or ""
        except Exception as exc:
            result.error = f"model call failed: {exc}"
            return result

        code = _extract_code(reply)
        if code is None:
            # No code block → the model wants to give a final answer. Only trust
            # it if real code has actually run; otherwise it is answering from
            # imagination (the fabrication failure). Push it back to write code.
            if ran_successfully:
                # Trust the model's prose only if it actually IS prose. When it
                # hands back code-junk (a known small-model failure), fall back
                # to the last value real code computed.
                if (reply.strip() and not _looks_like_code(reply)
                        and _answer_consistent(reply, last_good_output)):
                    result.answer = reply.strip()
                else:
                    # Code-junk, OR prose that contradicts what the code printed
                    # (right code, hallucinated words) — surface the computed value.
                    result.answer = _present(last_good_output)
                result.grounded = True
                answered = True
                break
            transcript += (
                f"\nYou replied without running code:\n{reply.strip()[:300]}\n"
                "Do NOT answer from memory. Write a Python code block that "
                "computes the answer from `df` and prints it.\n"
            )
            continue

        stdout, error = run_sandboxed(code, namespace=namespace)
        result.code_steps.append(code)
        observation = stdout if stdout.strip() else "(no output)"
        if error:
            observation = f"{observation}\nERROR: {error}"
        elif stdout.strip():
            ran_successfully = True
            last_good_output = stdout
        result.outputs.append(observation)
        transcript += (
            f"\nStep {step} code:\n```python\n{code}\n```\n"
            f"Output:\n{observation}\n"
        )

    if not answered:
        # Ran out of iterations without a plain-English answer — ask for one,
        # grounded in everything computed so far.
        closing = (
            f"{system}\n{transcript}\n"
            "You are out of code steps. Using the outputs above, answer the "
            "question now in one or two plain-English sentences with the numbers. "
            "Do NOT write more code."
        )
        try:
            closing_answer = (llm_call(closing) or "").strip()
            result.grounded = ran_successfully
            if (closing_answer and not _looks_like_code(closing_answer)
                    and _answer_consistent(closing_answer, last_good_output)):
                result.answer = closing_answer
            else:
                # Model wouldn't phrase it, or its prose contradicts the output —
                # surface the computed value.
                result.answer = _present(last_good_output)
        except Exception as exc:
            result.error = f"model call failed: {exc}"

    # Self-verification: independently recompute the key number a different way.
    # Only worth it when grounded and there's an actual number to confirm.
    if result.grounded and verify and result.code_steps and _numbers(last_good_output):
        result.verified, _ = verify_answer(
            user_message, work_df, llm_call, result.code_steps[-1], last_good_output,
        )
    result.confidence = _assess_confidence(result)
    if not result.answer and not result.error:
        result.error = "no answer produced"
    return result


def default_llm_call(model_name: str, timeout: int = 120) -> Callable[[str], str]:
    """Local Ollama caller in free-text (non-JSON) mode for codegen."""
    from nlp.local_model import _call_ollama

    def _call(prompt: str) -> str:
        raw, error = _call_ollama(prompt, model_name, timeout=timeout, json_mode=False)
        if error or raw is None:
            raise RuntimeError(error or "no response")
        return raw

    return _call
