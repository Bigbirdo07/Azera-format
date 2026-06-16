"""Generate the academic-workflow test workbook (Phase N.10).

A deterministic roster designed for the school-style four-step workflow:
  1. Show me all teachers that teach Biology.
  2. How many of their students have GPA above 2.00?
  3. Which students under each professor are below 2.00?
  4. Mark these students under Academic Watch + Export.

Columns:
  Student ID, Student Name, Teacher, Department, Major, GPA,
  Academic Standing, Academic Watch, Follow Up Needed.

Properties guaranteed by the seed:
  - 4 distinct Biology teachers, each with 25+ students.
  - Roughly 25% of students have GPA < 2.0 (mix of Probation/Warning/Good).
  - Academic Watch starts blank for every row.
  - Follow Up Needed starts as 'No'.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "academic_roster.xlsx"

DEPARTMENTS = {
    "Biology": ["Dr. Adler", "Dr. Beck", "Dr. Cho", "Dr. Diaz"],
    "Chemistry": ["Dr. Ellis", "Dr. Foster"],
    "Mathematics": ["Dr. Gomez", "Dr. Hanson"],
    "English": ["Dr. Ito", "Dr. Jansen"],
    "History": ["Dr. Kim"],
}
MAJORS_BY_DEPT = {
    "Biology": ["Biology", "Biochemistry", "Pre-Med"],
    "Chemistry": ["Chemistry", "Biochemistry"],
    "Mathematics": ["Mathematics", "Statistics", "Applied Math"],
    "English": ["English", "Creative Writing"],
    "History": ["History", "Political Science"],
}
STANDINGS = ["Good Standing", "Warning", "Probation", "At Risk"]


def _student_id(i: int) -> str:
    return f"S{1000 + i:04d}"


def build_dataframe(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    next_id = 0
    for department, teachers in DEPARTMENTS.items():
        # Biology gets a heavier roster so the workflow has volume to work with.
        per_teacher = 28 if department == "Biology" else 18
        for teacher in teachers:
            for _ in range(per_teacher):
                next_id += 1
                # 25% below 2.0, 25% borderline (2.0–2.5), 50% above 2.5.
                bucket = rng.random()
                if bucket < 0.25:
                    gpa = round(rng.uniform(0.8, 1.99), 2)
                elif bucket < 0.50:
                    gpa = round(rng.uniform(2.0, 2.49), 2)
                else:
                    gpa = round(rng.uniform(2.5, 4.0), 2)
                # Standing weakly correlated with GPA so the test mixes signals.
                if gpa < 1.8:
                    standing = rng.choice(["Probation", "At Risk", "Warning"])
                elif gpa < 2.0:
                    standing = rng.choice(["Warning", "Probation", "Good Standing"])
                else:
                    standing = rng.choice(["Good Standing", "Good Standing", "Warning"])
                rows.append({
                    "Student ID": _student_id(next_id),
                    "Student Name": f"Student {next_id:04d}",
                    "Teacher": teacher,
                    "Department": department,
                    "Major": rng.choice(MAJORS_BY_DEPT[department]),
                    "GPA": gpa,
                    "Academic Standing": standing,
                    "Academic Watch": "",
                    "Follow Up Needed": "No",
                })
    return pd.DataFrame(rows)


def write_workbook(path: Path = DEFAULT_OUT, seed: int = 42) -> Path:
    df = build_dataframe(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(path, sheet_name="Students", index=False)
    return path


if __name__ == "__main__":
    out = write_workbook()
    print(f"Wrote {out}")
