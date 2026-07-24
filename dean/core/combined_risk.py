"""Combined risk scoring — fold per-signal flags into one Risk Level column.

Signals (each independent, all optional depending on what data is loaded):
- GPA Risk          → GPA < 2.0
- Attendance Risk   → Attendance Rate < 90% (from core.attendance)
- Standing Risk     → Academic Standing in {Warning, Probation, At Risk}
- Assessment Risk   → PSAT/SAT Benchmark Status indicates below benchmark

Aggregation:
- Risk Signals = count of True signals
- Risk Level = High (>=2) | Moderate (==1) | Low (==0)

All inputs are looked up case-insensitively; missing columns simply skip
that signal — a roster with no Academic Standing column still gets the
GPA + attendance signals.
"""

from __future__ import annotations

import pandas as pd


GPA_RISK_THRESHOLD = 2.0
ATTENDANCE_RISK_THRESHOLD = 90.0
AT_RISK_STANDING_VALUES = {"warning", "probation", "at risk", "academic warning",
                           "academic probation"}
BELOW_BENCHMARK_VALUES = {"below benchmark", "did not meet", "did not meet benchmark",
                          "below", "not met"}
HIGH_RISK_SIGNAL_COUNT = 2
MODERATE_RISK_SIGNAL_COUNT = 1


def attach_combined_risk(
    roster: pd.DataFrame,
    *,
    gpa_threshold: float | None = None,
    attendance_threshold: float | None = None,
    high_count: int | None = None,
    moderate_count: int | None = None,
    sat_math_threshold: float | None = None,
    sat_ebrw_threshold: float | None = None,
    psat_math_threshold: float | None = None,
    psat_reading_writing_threshold: float | None = None,
) -> pd.DataFrame:
    """Add Risk Signals + Risk Level + per-signal boolean columns + a
    human-readable Risk Reason to the roster. Returns a copy — never
    mutates the input frame.

    All threshold kwargs are optional. When omitted, the module constants
    apply (preserves historical behaviour for tests + existing callers);
    when supplied, they let the caller plumb ``RiskSettings`` straight in.
    """
    gpa_t = gpa_threshold if gpa_threshold is not None else GPA_RISK_THRESHOLD
    att_t = attendance_threshold if attendance_threshold is not None else ATTENDANCE_RISK_THRESHOLD
    high_t = high_count if high_count is not None else HIGH_RISK_SIGNAL_COUNT
    mod_t = moderate_count if moderate_count is not None else MODERATE_RISK_SIGNAL_COUNT

    if roster is None or roster.empty:
        return roster

    out = roster.copy()
    signals: list[pd.Series] = []
    # Per-row reason fragments — joined into a single Risk Reason column
    # at the bottom so we can explain WHY a student lights up.
    reason_fragments: list[pd.Series] = []

    gpa_col = _find_column(out.columns, ("gpa", "cumulative gpa", "cum gpa"))
    if gpa_col:
        gpa_numeric = pd.to_numeric(out[gpa_col], errors="coerce")
        out["GPA Risk"] = gpa_numeric < gpa_t
        signals.append(out["GPA Risk"].fillna(False))
        reason_fragments.append(_label_when_true(
            out["GPA Risk"].fillna(False), f"GPA below {_fmt_threshold(gpa_t)}",
        ))

    if "Attendance Rate" in out.columns:
        attendance_rate = pd.to_numeric(out["Attendance Rate"], errors="coerce")
        # Attendance Rate is stored as a 0-1 fraction (0.94, not 94); the
        # threshold (module default / RiskSettings) is always expressed on a
        # 0-100 scale, matching the UI and this module's own docstring ("<
        # 90%"). Comparing the two scales directly meant every row with any
        # attendance data satisfied "< 90" and was silently flagged at risk
        # regardless of actual attendance -- confirmed on every roster used
        # this session, college and Skyward alike.
        att_threshold = att_t / 100.0 if att_t > 1 else att_t
        out["Attendance Risk"] = attendance_rate < att_threshold
        signals.append(out["Attendance Risk"].fillna(False))
        reason_fragments.append(_label_when_true(
            out["Attendance Risk"].fillna(False),
            f"Attendance below {_fmt_threshold(att_t)}%",
        ))

    standing_col = _find_column(out.columns, ("academic standing", "standing", "status"))
    if standing_col:
        normalized = out[standing_col].astype(str).str.strip().str.lower()
        out["Standing Risk"] = normalized.isin(AT_RISK_STANDING_VALUES)
        signals.append(out["Standing Risk"].fillna(False))
        reason_fragments.append(_standing_reason(out[standing_col], out["Standing Risk"].fillna(False)))

    assessment_signal, assessment_reason = _assessment_risk_signal(
        out,
        sat_math_threshold=sat_math_threshold,
        sat_ebrw_threshold=sat_ebrw_threshold,
        psat_math_threshold=psat_math_threshold,
        psat_reading_writing_threshold=psat_reading_writing_threshold,
    )
    if assessment_signal is not None:
        out["Assessment Risk"] = assessment_signal
        signals.append(out["Assessment Risk"].fillna(False))
        reason_fragments.append(assessment_reason)

    if not signals:
        out["Risk Signals"] = 0
        out["Risk Level"] = "Low Risk"
        out["Risk Reason"] = ""
        return out

    counts = signals[0].astype(int)
    for s in signals[1:]:
        counts = counts + s.astype(int)
    out["Risk Signals"] = counts
    out["Risk Level"] = counts.apply(
        lambda n: _risk_level_for_count(n, high_t, mod_t),
    )
    out["Risk Reason"] = _join_reason_fragments(reason_fragments)
    return out


def _risk_level_for_count(n: int, high_t: int = 2, mod_t: int = 1) -> str:
    if n >= high_t:
        return "High Risk"
    if n >= mod_t:
        return "Moderate Risk"
    return "Low Risk"


def _label_when_true(mask: pd.Series, label: str) -> pd.Series:
    return mask.map(lambda flag: label if bool(flag) else "")


def _standing_reason(values: pd.Series, mask: pd.Series) -> pd.Series:
    return values.astype(str).where(mask, "").map(
        lambda value: f"Academic Standing = {value.strip()}" if value.strip() else "",
    )


def _join_reason_fragments(fragments: list[pd.Series]) -> pd.Series:
    if not fragments:
        return pd.Series([""] * 0)
    out = fragments[0]
    for frag in fragments[1:]:
        out = out + frag.map(lambda v: ("; " + v) if v else "")
        # Joining a non-empty fragment to an empty one shouldn't prepend "; ".
    # Strip stray leading separators (when only later fragments were True).
    return out.str.replace(r"^;\s*", "", regex=True).str.strip()


def _fmt_threshold(value: float) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return f"{value:.1f}" if value < 10 else f"{int(value)}"
    return f"{value:g}"


def _assessment_risk_signal(
    out: pd.DataFrame,
    *,
    sat_math_threshold: float | None,
    sat_ebrw_threshold: float | None,
    psat_math_threshold: float | None,
    psat_reading_writing_threshold: float | None,
) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    signal = pd.Series(False, index=out.index)
    reasons = pd.Series("", index=out.index, dtype=object)

    existing_col = _find_column(out.columns, ("assessment risk",))
    if existing_col:
        existing = out[existing_col]
        if existing.dtype == bool:
            existing_mask = existing.fillna(False)
        else:
            norm = existing.astype(str).str.strip().str.lower()
            existing_mask = norm.isin({"true", "yes", "y", "1", "risk", "at risk", "below benchmark"})
        signal = signal | existing_mask
        reason_col = _find_column(out.columns, ("assessment reason",))
        if reason_col:
            reasons = _append_reason_series(reasons, out[reason_col].astype(str).where(existing_mask, ""))
        else:
            reasons = _append_reason(reasons, existing_mask, "Assessment below benchmark")

    bench_col = _find_column(out.columns, ("benchmark status", "psat benchmark",
                                            "sat benchmark", "assessment benchmark"))
    if bench_col:
        normalized = out[bench_col].astype(str).str.strip().str.lower()
        mask = normalized.isin(BELOW_BENCHMARK_VALUES) | normalized.str.contains("below|not meet", regex=True)
        signal = signal | mask
        reasons = _append_reason_series(
            reasons,
            out[bench_col].astype(str).where(mask, "").map(
                lambda value: f"Benchmark Status = {value}" if value else "",
            ),
        )

    for colname, label in (
        ("Math Benchmark Met", "Math benchmark not met"),
        ("Reading Benchmark Met", "Reading benchmark not met"),
    ):
        col = _find_column(out.columns, (colname.lower(),))
        if col:
            mask = out[col].map(_not_met_bool).fillna(False)
            signal = signal | mask
            reasons = _append_reason(reasons, mask, label)

    threshold_specs = (
        ("SAT Math", sat_math_threshold, "SAT Math below configured benchmark"),
        ("SAT EBRW", sat_ebrw_threshold, "SAT EBRW below configured benchmark"),
        ("PSAT Math", psat_math_threshold, "PSAT Math below configured benchmark"),
        ("PSAT Reading/Writing", psat_reading_writing_threshold, "PSAT Reading/Writing below configured benchmark"),
    )
    for colname, threshold, label in threshold_specs:
        if threshold in (None, ""):
            continue
        col = _find_column(out.columns, (colname.lower(),))
        if not col:
            continue
        mask = pd.to_numeric(out[col], errors="coerce") < float(threshold)
        signal = signal | mask.fillna(False)
        reasons = _append_reason(reasons, mask.fillna(False), label)

    if not bool(signal.any()):
        return None, None
    return signal, reasons.str.replace(r"^;\s*", "", regex=True).str.strip()


def _not_met_bool(value) -> bool | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"false", "no", "n", "0", "not met", "below benchmark", "did not meet"}:
        return True
    if normalized in {"true", "yes", "y", "1", "met"}:
        return False
    return None


def _append_reason(reasons: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    return _append_reason_series(reasons, mask.map(lambda flag: label if bool(flag) else ""))


def _append_reason_series(reasons: pd.Series, additions: pd.Series) -> pd.Series:
    return reasons + additions.map(lambda value: ("; " + value) if value else "")


def _find_column(columns, variants: tuple[str, ...]) -> str | None:
    norm_map = {str(c).strip().lower(): c for c in columns}
    for variant in variants:
        if variant in norm_map:
            return norm_map[variant]
    for variant in variants:
        for norm, original in norm_map.items():
            if variant in norm:
                return original
    return None
