"""Conversation memory for the unified assistant.

Holds a privacy-safe summary of the most recent turn so the assistant can
resolve follow-up references ("highlight them", "make a chart of that") without
re-asking. This object stores only metadata: the request text, the detected
mode, the structured plan, the filters used, column names, an output sheet name,
a short result description and a row count. It never stores raw student rows.

It is intentionally free of any Streamlit import so it can be unit-tested
headless and held inside st.session_state by the UI layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _describe_referent(plan: dict[str, Any], filters: list[dict[str, Any]]) -> str:
    """Build a short human-readable label for the prior result set."""
    if not plan:
        return ""
    operation = plan.get("operation", "")
    group_by = plan.get("group_by") or ""
    value_column = plan.get("value_column") or ""
    parts: list[str] = []
    if operation == "count_unique" and value_column:
        parts.append(f"distinct {value_column.lower()} values")
    elif operation.startswith("groupby_") and group_by:
        verb = {"groupby_count": "rows",
                "groupby_sum": "sums",
                "groupby_average": "averages"}.get(operation, "groups")
        parts.append(f"{verb} grouped by {group_by}")
        if value_column and operation != "groupby_count":
            parts.append(f"of {value_column}")
    elif operation == "filtered_preview":
        parts.append("matching rows")
    elif operation == "count_rows":
        parts.append("matching rows (count)")

    if filters:
        condition_bits = []
        for f in filters:
            column = f.get("column", "")
            op = f.get("operator", "")
            value = f.get("value")
            op_phrase = {
                "less_than": "<", "less_or_equal": "≤",
                "greater_than": ">", "greater_or_equal": "≥",
                "equals": "=", "not_equals": "≠",
            }.get(op, op)
            if op in {"is_missing", "is_not_missing", "is_blank", "is_not_blank"}:
                condition_bits.append(f"{column} {op_phrase}")
            elif isinstance(value, list):
                condition_bits.append(f"{column} {op_phrase} {value}")
            else:
                condition_bits.append(f"{column} {op_phrase} {value}")
        parts.append("where " + " AND ".join(condition_bits))
    return " ".join(parts).strip()


@dataclass
class SessionMemory:
    last_request: str = ""
    last_mode: str = ""  # ask_question | edit_workbook | clarify
    last_query_plan: dict[str, Any] = field(default_factory=dict)
    last_edit_plan: dict[str, Any] = field(default_factory=dict)
    last_filters: list[dict[str, Any]] = field(default_factory=list)
    last_result_description: str = ""
    last_result_row_count: int | None = None
    last_columns_used: list[str] = field(default_factory=list)
    last_sheet: str = ""
    last_output_sheet: str = ""
    last_summary_table: list[dict[str, Any]] = field(default_factory=list)
    last_chart_recommendation: dict[str, Any] = field(default_factory=dict)
    # Composable conversation context.
    active_filters: list[dict[str, Any]] = field(default_factory=list)
    active_sheet: str = ""
    last_operation: str = ""
    active_sort: dict[str, Any] = field(default_factory=dict)
    active_group_by: str = ""
    active_limit: int | None = None
    pending_action: dict[str, Any] = field(default_factory=dict)
    # Phase P: result-shape signals so follow-up drilldown ("which students
    # are those?", "show me students in that department") can act on the
    # right axis without re-asking.
    last_result_type: str = ""           # "row_level" | "group_level" | "aggregate"
    last_referent_label: str = ""        # human description of the prior result set
    last_group_winner_column: str = ""   # e.g., "Department" when the prior plan
    last_group_winner_value: str = ""    # was groupby_avg + limit=1, with "Education" as top
    last_row_filter: list[dict[str, Any]] = field(default_factory=list)  # row-level conditions
                                                                            # without the group_by axis
    # Most recently mentioned specific individual (e.g. {"column": "Name",
    # "value": "Samira Chen"}), independent of active_filters -- so a later
    # singular pronoun ("mark her as academic watch") can resolve to that
    # person even after several unrelated turns have overwritten
    # active_filters. Set on ANY message that names a resolvable single
    # student, not just watch/note actions; never auto-cleared by a topic
    # switch, only by reset_all().
    last_named_person: dict[str, Any] = field(default_factory=dict)
    # Free-form code-analyst conversation: a compact rolling log of recent
    # turns so the analyst can resolve follow-ups ("just the Biology ones",
    # "break that down by year"). Privacy-safe — stores the question, the answer
    # text, and the final code snippet, never raw student rows.
    recent_turns: list[dict[str, Any]] = field(default_factory=list)

    def has_result_set(self) -> bool:
        """True when there is a previous filtered/queried result a follow-up
        like 'them' or 'those students' could refer to."""
        return bool(self.active_filters) or self.last_result_row_count is not None

    def set_active_filters(self, filters: list[dict[str, Any]], sheet: str = "") -> None:
        self.active_filters = list(filters)
        self.last_filters = list(filters)  # kept in sync for followup_resolver
        if sheet:
            self.active_sheet = sheet

    def clear_filters(self) -> None:
        self.active_filters = []
        self.last_filters = []

    def reset_all(self) -> None:
        fresh = SessionMemory()
        for key, value in fresh.__dict__.items():
            setattr(self, key, value)

    def record_ask(
        self,
        *,
        request: str,
        query_plan: dict[str, Any],
        result_description: str,
        row_count: int | None,
        columns_used: list[str],
        sheet: str,
        summary_table: list[dict[str, Any]] | None = None,
        top_group: dict[str, Any] | None = None,
    ) -> None:
        self.last_request = request
        self.last_mode = "ask_question"
        self.last_query_plan = query_plan
        effective = list(query_plan.get("filters", []) or [])
        self.last_filters = effective
        self.active_filters = effective
        self.last_operation = query_plan.get("operation", "") or self.last_operation
        self.active_sort = query_plan.get("sort") or {}
        self.active_group_by = query_plan.get("group_by") or ""
        self.active_limit = query_plan.get("limit")
        self.last_result_description = result_description
        self.last_result_row_count = row_count
        self.last_columns_used = list(columns_used)
        self.last_sheet = sheet or query_plan.get("sheet", "")
        self.active_sheet = self.last_sheet
        self.last_summary_table = summary_table or []
        # Phase P: tag the result shape and remember the winner of a top-N
        # group query so subsequent follow-ups can drill into it.
        operation = self.last_operation
        group_by = self.active_group_by
        self.last_row_filter = [
            dict(f) for f in effective
            if not group_by or f.get("column") != group_by
        ]
        if operation in {"groupby_count", "groupby_sum", "groupby_average"}:
            self.last_result_type = "group_level"
        elif operation in {"count_rows", "count_unique", "average_column",
                           "sum_column", "min_column", "max_column",
                           "missing_summary", "duplicate_check",
                           "data_quality_summary"}:
            self.last_result_type = "aggregate"
        elif operation == "filtered_preview":
            self.last_result_type = "row_level"
        else:
            self.last_result_type = ""
        self.last_referent_label = _describe_referent(query_plan, effective)
        # If the plan was a groupby_* and the result is sorted (either by an
        # explicit limit=1 or by a sort directive), remember the top group's
        # column + value so phrases like "in that department" can resolve to
        # {column equals value}. We prefer the explicit ``top_group`` field
        # the dispatcher passes (captured pre-redaction) so privacy-redacted
        # group columns still produce a usable winner; otherwise fall back to
        # summary_table[0] when nothing was redacted.
        self.last_group_winner_column = ""
        self.last_group_winner_value = ""
        if (operation in {"groupby_count", "groupby_sum", "groupby_average"}
                and group_by):
            winner_value = None
            if isinstance(top_group, dict) and top_group.get("column") == group_by:
                winner_value = top_group.get("value")
            elif self.last_summary_table:
                winner_value = self.last_summary_table[0].get(group_by)
            if winner_value is not None and str(winner_value).strip():
                self.last_group_winner_column = group_by
                self.last_group_winner_value = str(winner_value)

    def record_edit(
        self,
        *,
        request: str,
        edit_plan: dict[str, Any],
        sheet: str = "",
        output_sheet: str = "",
    ) -> None:
        self.last_request = request
        self.last_mode = "edit_workbook"
        self.last_edit_plan = edit_plan
        commands = edit_plan.get("commands", []) or []
        # Carry forward the filters from the plan so a later "now chart that"
        # can reuse the same selection.
        for command in commands:
            if command.get("conditions"):
                self.last_filters = list(command["conditions"])
                self.last_sheet = command.get("sheet", sheet)
                break
        else:
            self.last_sheet = sheet or self.last_sheet
        self.last_output_sheet = output_sheet or self._first_output_sheet(commands)

    def record_clarify(self, *, request: str) -> None:
        self.last_request = request
        self.last_mode = "clarify"

    def record_analyst_turn(
        self, *, question: str, answer: str, code: str = "", max_turns: int = 6,
    ) -> None:
        """Append one free-form analyst turn to the rolling conversation log and
        mirror it onto the shared last_* fields so the rest of the assistant
        (follow-up resolver, suggestions) stays in sync."""
        self.last_request = question
        self.last_mode = "ask_question"
        self.last_result_description = answer
        self.recent_turns = (
            self.recent_turns + [{"question": question, "answer": answer, "code": code}]
        )[-max_turns:]

    @staticmethod
    def _first_output_sheet(commands: list[dict[str, Any]]) -> str:
        for command in commands:
            if command.get("output_sheet"):
                return str(command["output_sheet"])
        return ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SessionMemory":
        if not data:
            return cls()
        known = {key: data[key] for key in cls().__dict__ if key in data}
        return cls(**known)
