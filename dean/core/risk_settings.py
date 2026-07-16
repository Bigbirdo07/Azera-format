"""Risk thresholds — the numbers behind the assistant's at-risk language.

Defaults match the spec. The settings panel exposes each as a numeric input
so a school can tune them (e.g., a tighter 92% attendance threshold for a
demanding district). Every place that reads these thresholds also reads the
``mention_when_used()`` text, so the assistant can narrate WHAT it counted as
risk on a given turn ("I interpreted attendance risk as Attendance Rate below
90%.").

Stored in ``st.session_state["risk_settings"]`` as a serialised dict so a
new session falls back to defaults cleanly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskSettings:
    """Configurable thresholds used by attendance + combined-risk scoring."""
    # Performance.
    gpa_risk_threshold: float = 2.0
    # Attendance — rates are percentages.
    attendance_risk_threshold: float = 90.0
    severe_attendance_risk_threshold: float = 80.0
    # Counter-based concerns (when only Days Absent / Tardies are available).
    unexcused_absence_concern: int = 3
    tardy_concern: int = 5
    # Combined risk → Risk Level ladder.
    high_risk_signal_count: int = 2
    moderate_risk_signal_count: int = 1
    # Optional assessment benchmarks. None means the workbook must provide
    # benchmark fields before assessment risk is inferred.
    sat_math_benchmark_threshold: float | None = None
    sat_ebrw_benchmark_threshold: float | None = None
    psat_math_benchmark_threshold: float | None = None
    psat_reading_writing_benchmark_threshold: float | None = None

    # ---- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RiskSettings":
        if not data:
            return cls()
        defaults = cls()
        merged = {**defaults.to_dict(), **data}
        try:
            return cls(**merged)
        except TypeError:
            # Unknown keys in stored dict — fall back to defaults so a
            # legacy session can't crash the app on load.
            return defaults

    # ---- narration helpers ------------------------------------------------

    def mention_gpa_risk(self) -> str:
        return f"GPA below {_fmt_num(self.gpa_risk_threshold)}"

    def mention_attendance_risk(self) -> str:
        return f"Attendance Rate below {_fmt_num(self.attendance_risk_threshold)}%"


def _fmt_num(value: float | int) -> str:
    """Drop the trailing .0 from whole numbers so "2.0" → "2.0" stays the
    way a counsellor writes it, but "90.0" doesn't become "90.00" anywhere."""
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        # Keep one decimal for GPA-like (≤5) but bare integer for percent-
        # like (≥10) — matches the way humans write each.
        return f"{value:.1f}" if value < 10 else f"{int(value)}"
    return f"{value:g}"


# ---- session-state helpers (thin Streamlit wrapper) -----------------------


def load_risk_settings(session_state) -> RiskSettings:
    """Read the active RiskSettings from a Streamlit-like session_state. The
    `session_state` argument is duck-typed (anything supporting ``get`` /
    ``__setitem__``) so the helper is testable without Streamlit."""
    stored = session_state.get("risk_settings") if hasattr(session_state, "get") else None
    return RiskSettings.from_dict(stored)


def save_risk_settings(session_state, settings: RiskSettings) -> None:
    session_state["risk_settings"] = settings.to_dict()
