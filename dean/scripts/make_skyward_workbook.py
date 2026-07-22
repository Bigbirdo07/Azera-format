"""Generate a synthetic roster shaped like a real Skyward SIS export.

Built from `knowledge/skyward_field_map.json` (itself built from
`skyward/Standards_Gradebook_Teacher_Guide.pdf`) -- every "mapped" field in
that map gets a column here, using the field map's literal Skyward label as
the header, so Dean's schema/sensitivity/query layers can be validated
against something shaped like a real export rather than the hand-invented
mock roster used elsewhere. "needs_join" fields (Current Schedule, Course
Grades, Academic History -- per-course, one-to-many) and "unresolved" fields
are intentionally skipped; see knowledge/skyward_field_map.json for why.

Skyward tracks attendance as 4 distinct categories (Excused/Unexcused/
Tardy/Other), not one blended rate -- this generator keeps them separate and
derives Attendance Rate from the raw counts, matching a real export shape.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "skyward_roster.xlsx"

FIRST_NAMES = ["Maria", "James", "Aisha", "Chen", "David", "Sofia", "Omar", "Grace", "Liam", "Nina"]
LAST_NAMES = ["Lopez", "Smith", "Khan", "Wang", "Brown", "Garcia", "Hassan", "Park", "Murphy", "Cohen"]
ADVISORS = [f"Dr. {name}" for name in ["Reyes", "Okafor", "Nguyen", "Patel", "Brooks", "Santos", "Cohen"]]
GRADES = [9, 10, 11, 12]
CONDUCT_ENTRIES = ["Tardiness pattern", "Dress code violation", "Classroom disruption"]


def _student_id(i: int) -> str:
    return f"SKY{100000 + i}"


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, delta_days))


def build_dataframe(seed: int = 42, n: int = 250) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    today = date(2026, 7, 22)

    for i in range(1, n + 1):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        grade = rng.choice(GRADES)
        grad_year = 2026 + (12 - grade)
        birth_year = today.year - (grade - 9) - 14
        birth_date = _random_date(rng, date(birth_year, 1, 1), date(birth_year, 12, 31))

        excused = rng.randint(0, 6)
        unexcused = rng.randint(0, 4)
        tardies = rng.randint(0, 8)
        days_present = 170 - excused - unexcused
        attendance_rate = round(days_present / 170, 4)

        # Most students have no conduct record on file.
        conduct = rng.choice(CONDUCT_ENTRIES) if rng.random() < 0.08 else ""

        # A small minority have withdrawn.
        withdrawn = rng.random() < 0.03
        entry_date = _random_date(rng, date(today.year - (grade - 9), 8, 1), date(today.year - (grade - 9), 8, 15))
        withdrawal_date = _random_date(rng, entry_date, today) if withdrawn else None

        rows.append({
            "Student ID": _student_id(i),
            "Name": f"{first} {last}",
            "Grade": grade,
            "Grad Year": grad_year,
            "Birth Date": birth_date,
            "Phone": f"555-{rng.randint(100,999)}-{rng.randint(1000,9999)}",
            "Email": f"{first.lower()}.{last.lower()}{i}@student.example.edu",
            "Home Address": f"{rng.randint(100,9999)} {rng.choice(['Maple','Oak','Elm','Cedar'])} St",
            "Current Cumulative GPA": round(rng.uniform(1.5, 4.0), 2),
            "Advisor": rng.choice(ADVISORS),
            "Guardian Name": f"{rng.choice(FIRST_NAMES)} {last}",
            "Guardian Phone": f"555-{rng.randint(100,999)}-{rng.randint(1000,9999)}",
            "Guardian Email": f"{last.lower()}.family{i}@example.com",
            "Excused Absences": excused,
            "Unexcused Absences": unexcused,
            "Tardies": tardies,
            "Attendance Rate": attendance_rate,
            "Discipline Information": conduct,
            "SAT Math": rng.randint(400, 800) if grade >= 11 else None,
            "SAT EBRW": rng.randint(400, 800) if grade >= 11 else None,
            "SAT Total": None,
            "PSAT Math": rng.randint(400, 760) if grade in (9, 10) else None,
            "PSAT Reading/Writing": rng.randint(400, 760) if grade in (9, 10) else None,
            "PSAT Total": None,
            "Entry Date": entry_date,
            "Withdrawal Date": withdrawal_date,
            "Emergency Contact": f"{rng.choice(FIRST_NAMES)} {last} ({rng.choice(['Mother','Father','Guardian'])})",
        })

    df = pd.DataFrame(rows)
    sat_mask = df["SAT Math"].notna()
    df.loc[sat_mask, "SAT Total"] = df.loc[sat_mask, "SAT Math"] + df.loc[sat_mask, "SAT EBRW"]
    psat_mask = df["PSAT Math"].notna()
    df.loc[psat_mask, "PSAT Total"] = df.loc[psat_mask, "PSAT Math"] + df.loc[psat_mask, "PSAT Reading/Writing"]
    return df


def write_workbook(path: Path = DEFAULT_OUT, seed: int = 42, n: int = 250) -> Path:
    df = build_dataframe(seed=seed, n=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, sheet_name="Students", index=False)
    return path


if __name__ == "__main__":
    out = write_workbook()
    print(f"Wrote {out}")
