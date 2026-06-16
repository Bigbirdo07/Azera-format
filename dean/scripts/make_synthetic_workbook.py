"""Generate a deterministic synthetic student workbook for tests and demos.

NOT real student data. Fixed seed so tests can compute ground truth with pandas
and compare the assistant's answers against it. Includes sensitive columns
(Name/Email/Phone) so privacy behavior can be exercised too.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "synthetic_students.xlsx"

DEPARTMENTS = ["Business", "Biology", "Nursing", "Computer Science", "Accounting", "Psychology"]
MAJORS = {
    "Business": ["Management", "Marketing", "Finance"],
    "Biology": ["Molecular Biology", "Ecology"],
    "Nursing": ["Nursing"],
    "Computer Science": ["Software Engineering", "Data Science"],
    "Accounting": ["Accounting"],
    "Psychology": ["Clinical Psychology", "Cognitive Science"],
}
YEARS = ["Freshman", "Sophomore", "Junior", "Senior"]
STATUSES = ["Good Standing", "Good Standing", "Good Standing", "Warning", "Probation", "At Risk"]
ADVISORS = [f"Dr. {name}" for name in ["Reyes", "Okafor", "Nguyen", "Patel", "Brooks", "Santos", "Cohen"]]
GRAD = ["Not Graduated", "Not Graduated", "Not Graduated", "Graduated"]
FIRST_NAMES = ["Maria", "James", "Aisha", "Chen", "David", "Sofia", "Omar", "Grace", "Liam", "Nina"]
LAST_NAMES = ["Lopez", "Smith", "Khan", "Wang", "Brown", "Garcia", "Hassan", "Park", "Murphy", "Cohen"]
AID = ["None", "Pell Grant", "Subsidized Loan", "Scholarship", "Work Study"]
CONDUCT = ["Clear", "Clear", "Clear", "Warning Issued", "Probation"]


def build_dataframe(seed: int = 42, n: int = 600) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        dept = rng.choice(DEPARTMENTS)
        major = rng.choice(MAJORS[dept])
        year = rng.choice(YEARS)
        status = rng.choice(STATUSES)
        # At-risk/probation students skew lower GPA so tests have signal.
        if status in {"Probation", "At Risk"}:
            gpa = round(rng.uniform(1.2, 2.6), 2)
        elif status == "Warning":
            gpa = round(rng.uniform(2.0, 3.0), 2)
        else:
            gpa = round(rng.uniform(2.5, 4.0), 2)
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        year_born = 2007 - YEARS.index(year)
        rows.append({
            "Student ID": f"S{i:05d}",
            "First Name": first,
            "Last Name": last,
            "Name": f"{first} {last}",
            "Email": f"{first.lower()}.{last.lower()}{i}@university.edu",
            "Phone": f"555-{rng.randint(1000, 9999)}",
            "Date of Birth": f"{year_born}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "Department": dept,
            "Major": major,
            "Year": year,
            "GPA": gpa,
            "Academic Status": status,
            "Advisor": rng.choice(ADVISORS),
            "Credits Completed": rng.randint(0, 130),
            "Graduation Status": rng.choice(GRAD),
            "Financial Aid Status": rng.choice(AID),
            "Conduct Status": rng.choice(CONDUCT),
            "Notes": "",
        })
    return pd.DataFrame(rows)


def write_workbook(path: Path = DEFAULT_OUT, seed: int = 42, n: int = 600) -> Path:
    df = build_dataframe(seed=seed, n=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Students", index=False)
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    written = write_workbook(out)
    frame = build_dataframe()
    print(f"Wrote {written} ({len(frame)} rows)")
    print("Department counts:\n", frame["Department"].value_counts().to_string())
