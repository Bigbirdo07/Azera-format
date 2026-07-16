"""Phase C: pandas query engine accuracy and operator coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from core.query_engine import QueryExecutionError, run_query


def _count(sheets, filters):
    return run_query({"operation": "count_rows", "sheet": "Students", "filters": filters}, sheets).value


def test_department_count_matches_pandas(sheets, gt):
    result = run_query({"operation": "groupby_count", "sheet": "Students", "group_by": "Department"}, sheets)
    counts = {row["Department"]: row["Count"] for row in result.table}
    expected = gt.groupby("Department").size().to_dict()
    assert counts == expected


def test_filter_count_matches_pandas(sheets, gt):
    got = _count(sheets, [{"column": "Department", "operator": "equals", "value": "Accounting"}])
    assert got == int((gt["Department"] == "Accounting").sum())


def test_count_rows_returns_visible_summary_table(sheets, gt):
    result = run_query(
        {
            "operation": "count_rows",
            "sheet": "Students",
            "filters": [{"column": "Year", "operator": "equals", "value": "Senior"}],
        },
        sheets,
    )
    expected = int((gt["Year"] == "Senior").sum())
    assert result.value == expected
    assert result.table == [{"Metric": "Matching records", "Value": expected}]


def test_cohort_summary_returns_profile_metrics():
    sheets = {
        "Students": pd.DataFrame(
            {
                "Advisor": ["Dr. Nadia Pierce", "Dr. Nadia Pierce", "Dr. Victor Ford"],
                "Standing": ["Good Standing", "Bad Standing", "Good Standing"],
                "Year": ["Senior", "Junior", "Senior"],
                "Major": ["Health Administration", "Nursing", "Marketing"],
                "Location": ["North Campus", "Main Campus", "Online"],
                "GPA": [3.2, 2.4, 3.7],
                "Attendance Rate": [94.0, 82.0, 99.0],
                "Attendance Category": ["Great Attendance", "Needs Attendance Support", "Great Attendance"],
                "Days Absent": [2, 12, 1],
            }
        )
    }
    result = run_query(
        {
            "operation": "cohort_summary",
            "sheet": "Students",
            "filters": [{"column": "Advisor", "operator": "equals", "value": "Dr. Nadia Pierce"}],
        },
        sheets,
    )
    metrics = {row["Metric"]: row["Value"] for row in result.table}
    assert result.row_count == 2
    assert metrics["Students"] == 2
    assert metrics["Average GPA"] == 2.8
    assert metrics["Median GPA"] == 2.8
    assert metrics["GPA below 2.5"] == 1
    assert metrics["Needs Attendance Support"] == 1
    assert metrics["Days Absent above 10"] == 1
    assert metrics["Standing: Good Standing"] == 1
    assert metrics["Standing: Bad Standing"] == 1


def test_cohort_comparison_returns_dean_metrics():
    sheets = {
        "Students": pd.DataFrame(
            {
                "Advisor": ["Dr. Nadia Pierce", "Dr. Nadia Pierce", "Prof. Omar Sloan"],
                "Standing": ["Good Standing", "Bad Standing", "Good Standing"],
                "Year": ["Senior", "Junior", "Senior"],
                "Location": ["North Campus", "Main Campus", "Online"],
                "Major": ["Health Administration", "Nursing", "Marketing"],
                "GPA": [3.2, 2.4, 3.7],
                "Attendance Rate": [0.94, 0.82, 0.99],
                "Attendance Category": ["Great Attendance", "Needs Attendance Support", "Great Attendance"],
                "SAT Total": [1200, 980, 1300],
                "PSAT Total": [1100, 900, 1250],
            }
        )
    }
    result = run_query(
        {
            "operation": "cohort_comparison",
            "sheet": "Students",
            "filters": [
                {
                    "column": "Advisor",
                    "operator": "in",
                    "value": ["Dr. Nadia Pierce", "Prof. Omar Sloan"],
                }
            ],
            "group_by": "Advisor",
        },
        sheets,
    )
    by_advisor = {row["Advisor"]: row for row in result.table}
    assert by_advisor["Dr. Nadia Pierce"]["Students"] == 2
    assert by_advisor["Dr. Nadia Pierce"]["Average GPA"] == 2.8
    assert by_advisor["Dr. Nadia Pierce"]["Bad Standing"] == 1
    assert by_advisor["Dr. Nadia Pierce"]["Needs Attendance Support"] == 1
    assert by_advisor["Dr. Nadia Pierce"]["Top Year"] in {"Senior", "Junior"}
    assert by_advisor["Dr. Nadia Pierce"]["Top Location"] in {"North Campus", "Main Campus"}
    assert by_advisor["Prof. Omar Sloan"]["Students"] == 1


def test_average_gpa_matches_pandas(sheets, gt):
    result = run_query(
        {"operation": "average_column", "sheet": "Students", "value_column": "GPA",
         "filters": [{"column": "Department", "operator": "equals", "value": "Accounting"}]},
        sheets,
    )
    expected = round(float(gt.loc[gt["Department"] == "Accounting", "GPA"].mean()), 4)
    assert abs(result.value - expected) < 0.01


def test_count_unique_matches_pandas(sheets, gt):
    result = run_query({"operation": "count_unique", "sheet": "Students", "value_column": "Major"}, sheets)
    assert result.value == gt["Major"].nunique()


def test_count_unique_honors_filters(mini):
    result = run_query(
        {
            "operation": "count_unique",
            "sheet": "Students",
            "value_column": "Name",
            "filters": [{"column": "Dept", "operator": "equals", "value": "Bio"}],
        },
        mini,
    )
    assert result.value == 2


def test_groupby_none_limit_returns_all_groups():
    sheets = {"Students": pd.DataFrame({
        "Group": [f"G{i:02d}" for i in range(12)],
        "Value": list(range(12)),
    })}
    result = run_query(
        {"operation": "groupby_count", "sheet": "Students", "group_by": "Group", "limit": None},
        sheets,
    )
    assert len(result.table) == 12


def test_groupby_count_can_sort_by_generated_count_column(sheets):
    result = run_query(
        {
            "operation": "groupby_count",
            "sheet": "Students",
            "group_by": "Major",
            "sort": {"column": "Count", "direction": "desc"},
            "limit": 5,
        },
        sheets,
    )
    assert len(result.table) == 5
    counts = [row["Count"] for row in result.table]
    assert counts == sorted(counts, reverse=True)


def test_filtered_preview_none_limit_returns_all_matching_rows(mini):
    result = run_query(
        {
            "operation": "filtered_preview",
            "sheet": "Students",
            "filters": [{"column": "Dept", "operator": "equals", "value": "Bio"}],
            "limit": None,
        },
        mini,
    )
    assert result.row_count == 3
    assert len(result.table) == 3
    assert result.preview_truncated is False


# --- new operators (deterministic small frame) -------------------------------


@pytest.fixture()
def mini():
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Aaron", "Cara", None],
        "Dept": ["Bio", "Acc", "Bio", "Acc", "Bio"],
        "GPA": [3.1, 2.2, 1.9, 3.8, 2.5],
        "Notes": ["x", "", None, "y", "  "],
    })
    return {"Students": df}


def _mini_count(mini, column, operator, value=None):
    cond = {"column": column, "operator": operator}
    if value is not None:
        cond["value"] = value
    return run_query({"operation": "count_rows", "sheet": "Students", "filters": [cond]}, mini).value


def test_starts_with(mini):
    assert _mini_count(mini, "Name", "starts_with", "A") == 2


def test_ends_with(mini):
    assert _mini_count(mini, "Name", "ends_with", "n") == 1


def test_between(mini):
    assert _mini_count(mini, "GPA", "between", [2.0, 3.0]) == 2


def test_not_in(mini):
    assert _mini_count(mini, "Dept", "not_in", ["Bio"]) == 2


def test_is_blank(mini):
    assert _mini_count(mini, "Notes", "is_blank") == 3  # "", None, "  "


def test_is_not_blank(mini):
    assert _mini_count(mini, "Notes", "is_not_blank") == 2


def test_unknown_operator_raises_clean(mini):
    with pytest.raises(QueryExecutionError):
        _mini_count(mini, "Dept", "frobnicate", "x")


def test_unknown_column_raises_clean(mini):
    with pytest.raises(QueryExecutionError):
        run_query({"operation": "count_rows", "sheet": "Students",
                   "filters": [{"column": "Nope", "operator": "equals", "value": 1}]}, mini)


# --- column projection -------------------------------------------------------


def test_filtered_preview_projects_to_selected_columns(mini):
    result = run_query(
        {
            "operation": "filtered_preview",
            "sheet": "Students",
            "filters": [],
            "select_columns": ["Name", "GPA"],
            "limit": 10,
        },
        mini,
    )
    assert result.table, "expected at least one row in preview"
    for row in result.table:
        assert set(row.keys()) == {"Name", "GPA"}


def test_filtered_preview_projection_preserves_user_order(mini):
    result = run_query(
        {
            "operation": "filtered_preview",
            "sheet": "Students",
            "filters": [],
            "select_columns": ["GPA", "Name"],
            "limit": 1,
        },
        mini,
    )
    assert list(result.table[0].keys()) == ["GPA", "Name"]


def test_filtered_preview_projection_with_filter(mini):
    result = run_query(
        {
            "operation": "filtered_preview",
            "sheet": "Students",
            "filters": [{"column": "Dept", "operator": "equals", "value": "Bio"}],
            "select_columns": ["Name"],
            "limit": 10,
        },
        mini,
    )
    assert all(set(row.keys()) == {"Name"} for row in result.table)
    assert result.row_count == 3  # Alice, Aaron, (None) all in Bio


def test_filtered_preview_projection_rejects_unknown_column(mini):
    with pytest.raises(QueryExecutionError):
        run_query(
            {
                "operation": "filtered_preview",
                "sheet": "Students",
                "select_columns": ["Nope"],
            },
            mini,
        )


# --- list_unique -------------------------------------------------------------


def test_list_unique_returns_distinct_values(mini):
    result = run_query(
        {"operation": "list_unique", "sheet": "Students", "value_column": "Dept"},
        mini,
    )
    assert result.operation == "list_unique"
    assert result.row_count == 2
    assert [row["Dept"] for row in result.table] == ["Acc", "Bio"]


def test_list_unique_honors_filters(mini):
    result = run_query(
        {
            "operation": "list_unique",
            "sheet": "Students",
            "value_column": "Name",
            "filters": [{"column": "Dept", "operator": "equals", "value": "Bio"}],
        },
        mini,
    )
    # Bio members are Alice, Aaron, and the None-Name row (dropped by dropna).
    assert sorted(row["Name"] for row in result.table) == ["Aaron", "Alice"]


def test_list_unique_requires_column(mini):
    with pytest.raises(QueryExecutionError):
        run_query({"operation": "list_unique", "sheet": "Students"}, mini)


# --- percent_rows ----------------------------------------------------------


def test_percent_rows_matches_share_of_total(mini):
    result = run_query(
        {
            "operation": "percent_rows",
            "sheet": "Students",
            "filters": [{"column": "Dept", "operator": "equals", "value": "Bio"}],
        },
        mini,
    )
    # mini has 5 rows; 3 are Bio (Alice, Aaron, the None-Name row).
    assert result.row_count == 3
    assert result.value == 60.0


def test_percent_rows_no_filters_is_100_percent(mini):
    result = run_query(
        {"operation": "percent_rows", "sheet": "Students", "filters": []},
        mini,
    )
    assert result.value == 100.0
    assert result.row_count == 5


def test_percent_rows_empty_sheet():
    import pandas as pd
    sheets = {"Students": pd.DataFrame({"Name": [], "Dept": []})}
    result = run_query(
        {"operation": "percent_rows", "sheet": "Students",
         "filters": [{"column": "Dept", "operator": "equals", "value": "Bio"}]},
        sheets,
    )
    assert result.value == 0.0
    assert result.row_count == 0


# --- OR filter mode --------------------------------------------------------


def test_filter_mode_any_combines_with_or(mini):
    """Bio OR GPA<2.0 should be a strict superset of each individual filter."""
    result = run_query(
        {
            "operation": "count_rows",
            "sheet": "Students",
            "filters": [
                {"column": "Dept", "operator": "equals", "value": "Bio"},
                {"column": "GPA", "operator": "less_than", "value": 2.0},
            ],
            "filter_mode": "any",
        },
        mini,
    )
    # Bio rows: 3 (Alice 3.1, Aaron 1.9, None 2.5).
    # GPA<2.0: 1 (Aaron 1.9).
    # OR: Bio + Bob's row only if Bob<2.0 — Bob is 2.2 → no.
    # So OR = Bio rows ∪ {Aaron} = 3 (Aaron is already in Bio).
    assert result.value == 3


def test_filter_mode_all_is_default_and_strict(mini):
    """Same filters under default 'all' must be a subset of the OR result."""
    result = run_query(
        {
            "operation": "count_rows",
            "sheet": "Students",
            "filters": [
                {"column": "Dept", "operator": "equals", "value": "Bio"},
                {"column": "GPA", "operator": "less_than", "value": 2.0},
            ],
        },
        mini,
    )
    # Bio AND GPA<2.0 = only Aaron.
    assert result.value == 1


def test_filter_mode_any_with_percent(mini):
    result = run_query(
        {
            "operation": "percent_rows",
            "sheet": "Students",
            "filters": [
                {"column": "Dept", "operator": "equals", "value": "Acc"},
                {"column": "GPA", "operator": "greater_than", "value": 3.5},
            ],
            "filter_mode": "any",
        },
        mini,
    )
    # Acc rows: 2 (Bob, Cara). GPA>3.5: 1 (Cara 3.8). OR = 2 (already in Acc).
    assert result.row_count == 2
    assert result.value == 40.0


# --- between (range) -------------------------------------------------------


def test_between_filter_inclusive(mini):
    result = run_query(
        {
            "operation": "count_rows",
            "sheet": "Students",
            "filters": [{"column": "GPA", "operator": "between", "value": [2.0, 3.0]}],
        },
        mini,
    )
    # mini GPAs: 3.1, 2.2, 1.9, 3.8, 2.5 → in [2.0, 3.0]: 2.2 + 2.5 = 2
    assert result.value == 2


# --- status/standing caveat -------------------------------------------------
# When a numeric metric is grouped or pivoted by an imported status/standing
# label with no accompanying "Reason" column, the description should say so --
# otherwise a case like "GPA 3.06 but Bad Standing" reads as a data error
# rather than the normal real-world mismatch it is (Standing isn't computed
# from GPA; see core.query_engine._status_column_caveat).

def test_groupby_average_by_status_column_includes_caveat(sheets):
    result = run_query(
        {"operation": "groupby_average", "sheet": "Students", "group_by": "Academic Status", "value_column": "GPA"},
        sheets,
    )
    assert "existing label" in result.description
    assert "Academic Status" in result.description


def test_groupby_average_by_non_status_column_has_no_caveat(sheets):
    result = run_query(
        {"operation": "groupby_average", "sheet": "Students", "group_by": "Department", "value_column": "GPA"},
        sheets,
    )
    assert "existing label" not in result.description


def test_groupby_count_has_no_caveat_even_by_status_column(sheets):
    result = run_query(
        {"operation": "groupby_count", "sheet": "Students", "group_by": "Academic Status"},
        sheets,
    )
    assert "existing label" not in result.description


def test_pivot_average_crossed_with_status_column_includes_caveat(sheets):
    result = run_query(
        {
            "operation": "pivot_table_summary",
            "sheet": "Students",
            "pivot_rows": "Department",
            "pivot_columns": "Academic Status",
            "value_column": "GPA",
            "metric": "average",
        },
        sheets,
    )
    assert "existing label" in result.description


def test_status_caveat_suppressed_when_reason_column_present(sheets):
    frame = sheets["Students"].copy()
    frame["Academic Status Reason"] = ""
    result = run_query(
        {"operation": "groupby_average", "sheet": "Students", "group_by": "Academic Status", "value_column": "GPA"},
        {"Students": frame},
    )
    assert "existing label" not in result.description


def test_unrelated_reason_column_does_not_suppress_the_caveat(sheets):
    # Caught live: a workbook enriched with Dean's own combined-risk "Risk
    # Reason" column was silently suppressing the Standing caveat, even
    # though Risk Reason explains Dean's computed risk score, not why an
    # imported Academic Status label disagrees with GPA.
    frame = sheets["Students"].copy()
    frame["Risk Reason"] = ""
    result = run_query(
        {"operation": "groupby_average", "sheet": "Students", "group_by": "Academic Status", "value_column": "GPA"},
        {"Students": frame},
    )
    assert "existing label" in result.description


# --- data quality summary dtype -------------------------------------------
# Caught live: a "" string mixed into the otherwise-numeric "Missing %"
# column broke Streamlit's Arrow serialization of the result table
# (pyarrow.lib.ArrowInvalid). The fully-duplicated-rows row must use a
# numeric placeholder, not a string, to keep the column a single dtype.

def test_data_quality_summary_missing_percent_column_is_numeric(sheets):
    result = run_query({"operation": "data_quality_summary", "sheet": "Students"}, sheets)
    values = [row["Missing %"] for row in result.table]
    assert all(v is None or isinstance(v, (int, float)) for v in values)
    table_df = pd.DataFrame(result.table)
    assert table_df["Missing %"].dtype.kind == "f"
