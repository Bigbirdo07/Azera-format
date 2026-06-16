"""v0.1 health check: run the test suite, planner eval, and e2e smoke scripts,
then print a single pass/fail summary.

    .venv/bin/python scripts/health_check.py            # full
    .venv/bin/python scripts/health_check.py --quick    # skip slow e2e UI scripts
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(label: str, args: list[str]) -> tuple[str, bool, str]:
    proc = subprocess.run([PY, *args], cwd=REPO_ROOT, capture_output=True, text=True)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    summary = next(
        (ln for ln in reversed(lines)
         if any(k in ln for k in ("passed", "failed", "cases passed", "ALL PASS", "SOME FAIL"))),
        lines[-1] if lines else "",
    )
    return label, proc.returncode == 0, summary


def main() -> int:
    quick = "--quick" in sys.argv
    checks = [
        ("pytest suite", ["-m", "pytest", "-q"]),
        ("planner eval (rules-only)", ["scripts/eval_planner.py"]),
        ("planner eval (--dispatch)", ["scripts/eval_planner.py", "--dispatch"]),
    ]
    if not quick:
        checks += [
            ("e2e: follow-up composition", ["scripts/e2e_followup.py"]),
            ("e2e: phase A/B", ["scripts/e2e_phaseAB.py"]),
            ("e2e: plan execution", ["scripts/e2e_plan_exec.py"]),
            ("e2e: plan execution (UI)", ["scripts/e2e_plan_exec_ui.py"]),
        ]

    print(f"Running {len(checks)} checks ({'quick' if quick else 'full'})...\n")
    results = []
    for label, args in checks:
        name, ok, tail = _run(label, args)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<32} {tail}")
        results.append(ok)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed == len(results):
        print("HEALTH CHECK: PASS — v0.1 looks healthy.")
        return 0
    print("HEALTH CHECK: FAIL — see failing checks above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
