"""Mine logs/interaction_learning.jsonl for promotion candidates.

Three categories are mined:

1. **LLM-cache candidates** — phrases that the LLM planner was used for, where
   the same validated plan was produced repeatedly. These are candidates to
   convert into rule-parser handlers so the LLM never gets called for them
   again.

2. **Synonym-gap candidates** — phrases that the rule parser punted on
   (intent ∈ {clarify, unavailable, unsupported}) repeatedly, where extracting
   the noun tokens from the phrase suggests a missing column synonym
   (e.g. "what is their housing status" repeats 9× → "housing" not in
   knowledge/synonyms.json).

3. **Correction signals** — entries with ``user_corrected=True`` or a
   non-null ``corrects_entry_id``: explicit user-supplied training pairs.

The default mode emits a Markdown report. ``--json`` emits machine-readable
output. ``--apply-synonyms`` auto-writes synonym promotions to the local
SQLite learning DB — only for clusters whose noun tokens already match an
existing concept (i.e. the cheap-and-safe subset). Anything ambiguous is
left in the report for a human to review.

The script NEVER writes to the workbook or the chat log; it only reads the
interaction log and (with explicit opt-in) appends to the learning DB.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = REPO_ROOT / "logs" / "interaction_learning.jsonl"

PUNT_INTENTS = {"clarify", "unavailable", "unsupported"}

# Common stop tokens we strip when guessing what column a clarify-cluster is
# trying to reference. The set is intentionally small — over-stripping turns
# "show their housing" into "show", which is useless.
_STOP_TOKENS = {
    "a", "an", "the", "is", "are", "of", "for", "to", "and", "or", "in", "on",
    "by", "with", "from", "as", "their", "his", "her", "its", "our", "your",
    "this", "that", "these", "those", "what", "which", "who", "whose", "where",
    "when", "how", "why", "show", "list", "tell", "me", "us", "give", "find",
    "filter", "students", "student", "each", "every", "all", "any", "some",
    "be", "do", "does", "did", "have", "has", "had", "can", "could", "would",
    "should", "may", "might", "will", "shall", "?", ".", ",",
}


@dataclass
class LlmCacheCandidate:
    phrase: str
    count: int
    canonical_plan: dict[str, Any]
    plan_variance: int  # how many distinct plans this phrase produced
    sessions: int       # how many distinct sessions saw it


@dataclass
class SynonymGapCandidate:
    phrase: str
    count: int
    noun_tokens: list[str]
    suggested_concept: str | None       # matched if any noun matches a known concept
    unmatched_tokens: list[str]         # nouns that didn't map to any known concept
    sessions: int


@dataclass
class CorrectionExample:
    original_phrase: str
    correction_message: str
    incorrect_plan: dict[str, Any]
    corrected_plan: dict[str, Any] | None  # may be None if no follow-up was logged


@dataclass
class PromotionReport:
    log_path: Path
    record_count: int
    llm_cache: list[LlmCacheCandidate] = field(default_factory=list)
    synonym_gaps: list[SynonymGapCandidate] = field(default_factory=list)
    corrections: list[CorrectionExample] = field(default_factory=list)


# -- IO ---------------------------------------------------------------------


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# -- Mining -----------------------------------------------------------------


def _normalize_plan_for_grouping(plan: dict[str, Any] | None) -> str:
    """Hashable signature of the plan structure (operation + filters + group_by)."""
    if not plan:
        return ""
    signature = {
        "operation": plan.get("operation"),
        "filters": plan.get("filters") or [],
        "group_by": plan.get("group_by"),
        "value_column": plan.get("value_column"),
    }
    return json.dumps(signature, sort_keys=True, default=str)


def mine_llm_cache_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    min_count: int = 3,
) -> list[LlmCacheCandidate]:
    """Phrases the LLM planner handled multiple times that converged on one plan."""
    by_phrase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("llm_used"):
            continue
        if not row.get("safe_for_rule_mining"):
            continue
        if row.get("validation_status") != "passed":
            continue
        phrase = (row.get("normalized_message") or "").strip()
        if not phrase:
            continue
        by_phrase[phrase].append(row)

    candidates: list[LlmCacheCandidate] = []
    for phrase, entries in by_phrase.items():
        if len(entries) < min_count:
            continue
        plan_counts: Counter[str] = Counter()
        plan_objects: dict[str, dict[str, Any]] = {}
        for entry in entries:
            plan = entry.get("validated_plan") or {}
            signature = _normalize_plan_for_grouping(plan)
            plan_counts[signature] += 1
            plan_objects.setdefault(signature, plan)
        canonical_sig, _ = plan_counts.most_common(1)[0]
        sessions = len({entry.get("session_id") for entry in entries if entry.get("session_id")})
        candidates.append(LlmCacheCandidate(
            phrase=phrase,
            count=len(entries),
            canonical_plan=plan_objects[canonical_sig],
            plan_variance=len(plan_counts),
            sessions=sessions,
        ))
    candidates.sort(key=lambda c: (-c.count, c.phrase))
    return candidates


def _suggest_concept_for_phrase(
    phrase: str,
    concept_lookup: dict[str, str],
) -> str | None:
    """Find the longest concept-lookup key that appears in the phrase.

    Returns None if no key matches, or if the two longest matches map to
    different concepts and are the same length (genuine ambiguity).
    """
    lowered = phrase.lower()
    matches: list[tuple[int, str]] = []
    for key, concept in concept_lookup.items():
        if not key:
            continue
        # Word-boundary check so "us" doesn't match "housing".
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", lowered):
            matches.append((len(key), concept))
    if not matches:
        return None
    matches.sort(key=lambda item: -item[0])
    longest_len = matches[0][0]
    same_length = {concept for length, concept in matches if length == longest_len}
    if len(same_length) > 1:
        return None  # tied at the top with different concepts → ambiguous
    return matches[0][1]


def _noun_tokens(phrase: str) -> list[str]:
    """Best-effort extraction of content nouns from a normalized phrase."""
    raw = re.findall(r"[a-z0-9]+", phrase.lower())
    return [token for token in raw if token not in _STOP_TOKENS and len(token) > 2]


def mine_synonym_gap_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    min_count: int = 3,
    known_concepts: dict[str, list[str]] | None = None,
) -> list[SynonymGapCandidate]:
    """Phrases that the parser kept giving up on — likely missing synonyms."""
    by_phrase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("intent") not in PUNT_INTENTS:
            continue
        phrase = (row.get("normalized_message") or "").strip()
        if not phrase:
            continue
        # Skip pure confirmation noise ("yes", "no") — those punt for unrelated reasons.
        if phrase in {"yes", "no", "ok", "okay"}:
            continue
        by_phrase[phrase].append(row)

    concepts = known_concepts or {}
    concept_lookup: dict[str, str] = {}
    for concept, phrases in concepts.items():
        concept_lookup[concept.lower()] = concept
        for entry in phrases:
            concept_lookup[entry.lower()] = concept

    candidates: list[SynonymGapCandidate] = []
    for phrase, entries in by_phrase.items():
        if len(entries) < min_count:
            continue
        tokens = _noun_tokens(phrase)
        unmatched = [t for t in tokens if t not in concept_lookup]
        # Prefer the LONGEST concept-lookup key that appears as a substring of
        # the original phrase, so "housing status" (which is a full synonym
        # under housing_status) beats the lone token "status" (which would
        # otherwise pull enrollment_status). Token-level matches still fire
        # for short phrases like "filter by housing" where the bigram is just
        # the single noun.
        suggested = _suggest_concept_for_phrase(phrase, concept_lookup)
        sessions = len({entry.get("session_id") for entry in entries if entry.get("session_id")})
        candidates.append(SynonymGapCandidate(
            phrase=phrase,
            count=len(entries),
            noun_tokens=tokens,
            suggested_concept=suggested,
            unmatched_tokens=unmatched,
            sessions=sessions,
        ))
    candidates.sort(key=lambda c: (-c.count, c.phrase))
    return candidates


def mine_corrections(
    rows: Iterable[dict[str, Any]],
) -> list[CorrectionExample]:
    """Build the list of correction signals, deduped by (original phrase,
    correction message, both plan signatures) so replayed e2e scenarios
    collapse into a single entry."""
    rows_list = list(rows)
    by_id = {row.get("id"): row for row in rows_list if row.get("id")}
    examples: list[CorrectionExample] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows_list:
        if not (row.get("user_corrected") or row.get("corrects_entry_id")):
            continue
        target_id = row.get("corrects_entry_id")
        target = by_id.get(target_id) if target_id else None
        example = CorrectionExample(
            original_phrase=(target or {}).get("normalized_message", "") or row.get("normalized_message", ""),
            correction_message=row.get("correction_message") or row.get("user_message") or "",
            incorrect_plan=(target or {}).get("validated_plan") or {},
            corrected_plan=row.get("validated_plan"),
        )
        key = (
            example.original_phrase,
            example.correction_message,
            _normalize_plan_for_grouping(example.incorrect_plan),
            _normalize_plan_for_grouping(example.corrected_plan),
        )
        if key in seen:
            continue
        seen.add(key)
        examples.append(example)
    return examples


def build_report(
    rows: list[dict[str, Any]],
    *,
    log_path: Path,
    min_count: int,
    known_concepts: dict[str, list[str]] | None,
) -> PromotionReport:
    return PromotionReport(
        log_path=log_path,
        record_count=len(rows),
        llm_cache=mine_llm_cache_candidates(rows, min_count=min_count),
        synonym_gaps=mine_synonym_gap_candidates(
            rows, min_count=min_count, known_concepts=known_concepts,
        ),
        corrections=mine_corrections(rows),
    )


# -- Rendering --------------------------------------------------------------


def render_markdown(report: PromotionReport) -> str:
    lines: list[str] = []
    lines.append(f"# Promotion candidates from `{report.log_path.name}`")
    lines.append(f"Scanned {report.record_count} records.\n")

    lines.append("## LLM-cache candidates")
    if not report.llm_cache:
        lines.append("_None at the current min-count threshold._\n")
    else:
        lines.append("Phrases the LLM planner handled repeatedly with a stable plan.")
        lines.append("If the plan is correct, lift them into the rule parser so the LLM stops being called.\n")
        lines.append("| Phrase | Count | Sessions | Plan variants | Canonical plan |")
        lines.append("|---|---:|---:|---:|---|")
        for c in report.llm_cache:
            plan_brief = (
                f"`{c.canonical_plan.get('operation','?')}` "
                f"filters={len(c.canonical_plan.get('filters') or [])} "
                f"group_by={c.canonical_plan.get('group_by') or '—'}"
            )
            lines.append(f"| `{c.phrase}` | {c.count} | {c.sessions} | {c.plan_variance} | {plan_brief} |")
        lines.append("")

    lines.append("## Synonym-gap candidates")
    if not report.synonym_gaps:
        lines.append("_None at the current min-count threshold._\n")
    else:
        lines.append("Phrases that consistently fell through to clarify/unavailable/unsupported.")
        lines.append("Likely cause: a noun in the phrase doesn't map to any known concept.\n")
        lines.append("| Phrase | Count | Sessions | Noun tokens | Gap noun(s) | Suggested concept |")
        lines.append("|---|---:|---:|---|---|---|")
        for c in report.synonym_gaps:
            suggested = c.suggested_concept or "_(needs a new concept)_"
            tokens = ", ".join(c.noun_tokens) or "—"
            gap = ", ".join(c.unmatched_tokens) or "—"
            lines.append(f"| `{c.phrase}` | {c.count} | {c.sessions} | {tokens} | {gap} | {suggested} |")
        lines.append("")

    lines.append("## User corrections")
    if not report.corrections:
        lines.append("_No corrections recorded._\n")
    else:
        lines.append("Highest-value signal: the user explicitly fixed an interpretation.\n")
        for ex in report.corrections:
            lines.append(f"- **Original phrase:** `{ex.original_phrase}`")
            lines.append(f"  **Correction said:** `{ex.correction_message}`")
            if ex.incorrect_plan:
                lines.append(f"  **Was interpreted as:** `{ex.incorrect_plan.get('operation','?')}` "
                             f"filters={len(ex.incorrect_plan.get('filters') or [])}")
            if ex.corrected_plan:
                lines.append(f"  **Corrected to:** `{ex.corrected_plan.get('operation','?')}` "
                             f"filters={len(ex.corrected_plan.get('filters') or [])}")
            lines.append("")
    return "\n".join(lines)


def render_json(report: PromotionReport) -> str:
    payload = {
        "log_path": str(report.log_path),
        "record_count": report.record_count,
        "llm_cache_candidates": [vars(c) for c in report.llm_cache],
        "synonym_gap_candidates": [vars(c) for c in report.synonym_gaps],
        "corrections": [vars(c) for c in report.corrections],
    }
    return json.dumps(payload, indent=2, default=str)


# -- Auto-apply (opt-in) -----------------------------------------------------


def apply_synonym_promotions(
    candidates: list[SynonymGapCandidate],
    *,
    confidence_threshold: int = 5,
) -> list[tuple[str, str]]:
    """Auto-promote ONLY when a candidate phrase contains a noun that already
    maps to an existing concept. We're adding a NEW phrasing for a KNOWN
    concept, never inventing a concept. Returns the (phrase, concept) pairs
    that were written so the caller can log them.
    """
    # Imported lazily so the script can be invoked in environments without the
    # full app dependency set when running as a read-only report.
    from core.correction_manager import add_learned_synonym, sync_learning_files

    applied: list[tuple[str, str]] = []
    for c in candidates:
        if c.count < confidence_threshold:
            continue
        if not c.suggested_concept:
            continue
        # Refuse to auto-write when the phrase still has unmatched nouns —
        # surfacing the suggestion in the report is fine (it's a hint), but
        # committing it to the DB would mismap phrases like "housing status"
        # to enrollment_status when housing is the actual gap.
        if c.unmatched_tokens:
            continue
        add_learned_synonym(
            phrase=c.phrase,
            mapped_concept=c.suggested_concept,
            source=f"promote_learned_patterns:count={c.count}",
        )
        applied.append((c.phrase, c.suggested_concept))
    if applied:
        sync_learning_files()
    return applied


# -- CLI --------------------------------------------------------------------


def _load_known_concepts() -> dict[str, list[str]]:
    """Read the baseline + learned synonym files so suggestion lookups know
    what concepts already exist."""
    knowledge_dir = REPO_ROOT / "knowledge"
    baseline: dict[str, list[str]] = {}
    baseline_path = knowledge_dir / "synonyms.json"
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            baseline = {}
    learned_path = knowledge_dir / "learned_synonyms.json"
    if learned_path.exists():
        try:
            learned = json.loads(learned_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            learned = []
        # learned_synonyms.json is a list of {phrase, mapped_concept, ...}.
        for entry in learned if isinstance(learned, list) else []:
            phrase = entry.get("phrase")
            concept = entry.get("mapped_concept")
            if not phrase or not concept:
                continue
            baseline.setdefault(concept, []).append(phrase)
    return baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH,
                        help="Path to interaction_learning.jsonl")
    parser.add_argument("--min-count", type=int, default=3,
                        help="Minimum cluster size to include (default 3)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of Markdown")
    parser.add_argument("--apply-synonyms", action="store_true",
                        help="Auto-write synonym promotions to the learning DB "
                             "(only when phrase matches an existing concept)")
    parser.add_argument("--apply-threshold", type=int, default=5,
                        help="Minimum count required for --apply-synonyms (default 5)")
    args = parser.parse_args(argv)

    rows = read_log(args.log)
    known = _load_known_concepts()
    report = build_report(rows, log_path=args.log, min_count=args.min_count, known_concepts=known)

    if args.apply_synonyms:
        applied = apply_synonym_promotions(
            report.synonym_gaps,
            confidence_threshold=args.apply_threshold,
        )
        print(f"Wrote {len(applied)} synonym promotion(s) to the learning DB.")
        for phrase, concept in applied:
            print(f"  + {phrase!r} -> {concept!r}")
        if not applied:
            print("  (no candidates met the threshold + concept-match criteria)")
        return 0

    output = render_json(report) if args.json else render_markdown(report)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
