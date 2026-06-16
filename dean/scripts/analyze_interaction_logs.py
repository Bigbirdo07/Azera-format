"""Rule-mining report over the sanitized interaction log.

Reads `logs/interaction_learning.jsonl` and writes
`outputs/interaction_learning_report.md` with:

  - most common user message patterns
  - low-confidence prompts
  - frequently corrected prompts
  - repeated assumptions
  - candidate deterministic rules
  - prompts that often clarify
  - prompts that caused validation failure

Usage:
    .venv/bin/python scripts/analyze_interaction_logs.py
    .venv/bin/python scripts/analyze_interaction_logs.py --log <path> --out <path>

The input log is sanitized at write time (see core/interaction_logger), so this
script only ever sees redacted text. Records with `safe_for_rule_mining = False`
are excluded from rule-candidate aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.interaction_logger import DEFAULT_LOG_PATH, read_records  # noqa: E402


DEFAULT_REPORT_PATH = REPO_ROOT / "outputs" / "interaction_learning_report.md"


def _top(counter: Counter, n: int = 10) -> list[tuple[str, int]]:
    return counter.most_common(n)


def _format_count_table(rows: list[tuple[str, int]]) -> str:
    if not rows:
        return "_(none)_"
    lines = ["| Count | Phrase / value |", "|---:|---|"]
    for value, count in rows:
        truncated = value if len(value) <= 80 else value[:77] + "…"
        lines.append(f"| {count} | {truncated} |")
    return "\n".join(lines)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the log into the buckets we report on."""
    total = len(records)
    by_intent = Counter(str(r.get("intent")) for r in records)
    by_plan_source = Counter(str(r.get("plan_source")) for r in records)
    by_band = Counter(str(r.get("band")) for r in records)

    # Normalized message patterns by band.
    safe_records = [r for r in records if r.get("safe_for_rule_mining")]
    safe_messages = Counter(r.get("normalized_message") or "" for r in safe_records)
    medium_messages = Counter(
        r.get("normalized_message") or ""
        for r in safe_records
        if r.get("band") == "medium"
    )
    low_messages = Counter(
        r.get("normalized_message") or ""
        for r in records
        if (r.get("band") == "low" or r.get("intent") == "clarify")
    )
    validation_failures = [r for r in records if r.get("validation_status") == "failed"]
    llm_records = [r for r in records if r.get("plan_source") == "llm"]

    # Repeated assumptions in medium-band records.
    repeated_assumptions = Counter(
        r.get("assumption_used") or ""
        for r in records
        if r.get("band") == "medium" and r.get("assumption_used")
    )

    # Corrections: a record whose `corrects_entry_id` references an earlier id.
    index_by_id = {r.get("id"): r for r in records if r.get("id")}
    corrections: list[tuple[dict, dict | None]] = []
    for record in records:
        target_id = record.get("corrects_entry_id")
        if not target_id:
            continue
        corrections.append((record, index_by_id.get(target_id)))

    corrected_phrases = Counter()
    correction_patterns = Counter()
    for current, original in corrections:
        if original:
            corrected_phrases[original.get("normalized_message") or ""] += 1
        correction_patterns[current.get("normalized_message") or ""] += 1

    # Candidate rules: vague phrases (frequently appearing in medium band with
    # consistent validated operations) suggest a deterministic shortcut.
    candidates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "operations": Counter(), "filter_columns": Counter(),
                 "example_assumption": ""}
    )
    for record in safe_records:
        if record.get("band") not in {"medium", "high"}:
            continue
        message = record.get("normalized_message") or ""
        if not message:
            continue
        bucket = candidates[message]
        bucket["count"] += 1
        bucket["operations"][record.get("operation") or ""] += 1
        for condition in (record.get("validated_plan") or {}).get("filters") or []:
            column = condition.get("column")
            if column:
                bucket["filter_columns"][column] += 1
        if not bucket["example_assumption"] and record.get("assumption_used"):
            bucket["example_assumption"] = record["assumption_used"]

    rule_candidates = [
        (message, info) for message, info in candidates.items()
        if info["count"] >= 2 and len(info["operations"]) == 1
    ]
    rule_candidates.sort(key=lambda item: item[1]["count"], reverse=True)

    return {
        "total": total,
        "by_intent": by_intent,
        "by_plan_source": by_plan_source,
        "by_band": by_band,
        "safe_messages": safe_messages,
        "medium_messages": medium_messages,
        "low_messages": low_messages,
        "repeated_assumptions": repeated_assumptions,
        "corrected_phrases": corrected_phrases,
        "correction_patterns": correction_patterns,
        "validation_failures": validation_failures,
        "llm_records": llm_records,
        "rule_candidates": rule_candidates,
    }


def render_report(summary: dict[str, Any]) -> str:
    """Produce the markdown report from a summary dict."""
    lines: list[str] = [
        "# Interaction Learning Report",
        "",
        f"Total turns logged: **{summary['total']}**",
        "",
        "## Distribution",
        "",
        "### By intent",
        _format_count_table(summary["by_intent"].most_common()),
        "",
        "### By plan source",
        _format_count_table(summary["by_plan_source"].most_common()),
        "",
        "### By confidence band",
        _format_count_table(summary["by_band"].most_common()),
        "",
        "## Most common user phrasing",
        "",
        "Pulled from records flagged `safe_for_rule_mining` (no PII detected).",
        "",
        _format_count_table(_top(summary["safe_messages"])),
        "",
        "## Medium-confidence prompts",
        "",
        "These are the prompts that triggered the assume-and-offer path. Frequent",
        "ones are candidates for promotion to deterministic rules.",
        "",
        _format_count_table(_top(summary["medium_messages"])),
        "",
        "## Low-confidence / clarify prompts",
        "",
        "Phrasings that the system asked the user to clarify, ordered by frequency.",
        "",
        _format_count_table(_top(summary["low_messages"])),
        "",
        "## Repeated assumptions (medium band)",
        "",
        _format_count_table(_top(summary["repeated_assumptions"])),
        "",
        "## Corrections",
        "",
        f"Turns with a follow-up correction: **{sum(summary['correction_patterns'].values())}**",
        "",
        "### Phrasings most often corrected",
        _format_count_table(_top(summary["corrected_phrases"])),
        "",
        "### Phrasings the user used to correct",
        _format_count_table(_top(summary["correction_patterns"])),
        "",
        "## Validation failures",
        "",
        f"Turns where the validator rejected a plan: **{len(summary['validation_failures'])}**",
        "",
        "## Candidate deterministic rules",
        "",
    ]
    if not summary["rule_candidates"]:
        lines.append("_(none yet — not enough repeat traffic.)_")
    else:
        for message, info in summary["rule_candidates"][:20]:
            operations = ", ".join(op or "(unknown)" for op, _ in info["operations"].most_common())
            columns = ", ".join(col for col, _ in info["filter_columns"].most_common())
            lines.extend([
                f"### Pattern: `{message[:80]}`",
                f"- Occurrences: **{info['count']}**",
                f"- Resolved operation: {operations or 'n/a'}",
                f"- Filter columns seen: {columns or 'n/a'}",
            ])
            if info["example_assumption"]:
                lines.append(f"- Example assumption: {info['example_assumption']}")
            lines.append("")
    lines.append("")
    lines.append("---")
    lines.append("Generated by `scripts/analyze_interaction_logs.py`.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the sanitized interaction learning log.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    records = read_records(args.log)
    if not records:
        print(f"No records found at {args.log}.")
        return 0

    summary = summarize(records)
    report = render_report(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote report ({len(records)} records → {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
