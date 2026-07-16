"""Phase C: composable conversation state (follow-up memory)."""

from __future__ import annotations


def _cols(filters):
    return {f.get("column") for f in filters}


def test_first_turn_filter(chat, gt):
    chat.send("Show me Accounting students")
    assert chat.get("assistant_mode") == "ask_question"
    assert chat.get("ask_row_count") == int((gt["Department"] == "Accounting").sum())
    assert _cols(chat.memory()["active_filters"]) == {"Department"}


def test_additive_followup(chat, gt):
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    expected = int(((gt["Department"] == "Accounting") & (gt["GPA"] < 2.5)).sum())
    assert chat.get("ask_row_count") == expected
    assert _cols(chat.memory()["active_filters"]) == {"Department", "GPA"}


def test_third_filter(chat, gt):
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    chat.send("now only seniors")
    mask = (gt["Department"] == "Accounting") & (gt["GPA"] < 2.5) & (gt["Year"] == "Senior")
    assert chat.get("ask_row_count") == int(mask.sum())
    assert _cols(chat.memory()["active_filters"]) == {"Department", "GPA", "Year"}


def test_replacement_filter(chat, gt):
    chat.send("Show me Accounting students")
    chat.send("what about Biology")
    filters = chat.memory()["active_filters"]
    depts = [f for f in filters if f["column"] == "Department"]
    assert len(depts) == 1 and depts[0]["value"] == "Biology"


def test_thanks_thats_helpful_does_not_misfire_as_a_followup(chat, gt):
    # Caught live immediately after adding the "that" follow-up cue:
    # normalize_text strips the apostrophe in "that's", leaving a stray
    # "that s" that would otherwise match the bare "that " cue and re-run
    # the prior query instead of just... not doing anything to a closing
    # remark.
    chat.send("Show me Accounting students")
    chat.send("thanks, that's really helpful")
    assert chat.get("assistant_mode") != "ask_question"


def test_sort_that_follow_up_preserves_active_filters(chat, gt):
    # Caught live: "sort that by gpa lowest first" -- bare "that", not
    # "those"/"them"/"these" -- was classified as a brand-new question
    # (only compound phrases like "that group" matched), so the active
    # filters were silently dropped and the sort landed on the full
    # unfiltered roster instead of the narrowed result.
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    chat.send("sort that by gpa lowest first")
    expected = int(((gt["Department"] == "Accounting") & (gt["GPA"] < 2.5)).sum())
    assert chat.get("ask_row_count") == expected
    assert _cols(chat.memory()["active_filters"]) == {"Department", "GPA"}
    assert chat.memory()["active_sort"] == {"column": "GPA", "direction": "asc"}


def test_additive_include(chat, gt):
    chat.send("Show me Accounting students")
    chat.send("include Biology too")
    expected = int(gt["Department"].isin(["Accounting", "Biology"]).sum())
    assert chat.get("ask_row_count") == expected


def test_sort_followup(chat):
    chat.send("Show me Nursing students")
    chat.send("sort them by GPA lowest first")
    assert _cols(chat.memory()["active_filters"]) == {"Department"}
    table = chat.get("ask_table") or []
    gpas = [r["GPA"] for r in table if r.get("GPA") is not None]
    assert gpas and gpas == sorted(gpas)


def test_limit_followup(chat):
    chat.send("Show me students below 2.5 GPA")
    chat.send("just the top 5")
    assert 0 < len(chat.get("ask_table") or []) <= 5
    assert chat.memory()["active_limit"] == 5


def test_groupby_followup(chat, gt):
    chat.send("Show me students on probation")
    chat.send("group them by advisor")
    assert chat.get("ask_operation") == "groupby_count"
    expected = gt.loc[gt["Academic Status"] == "Probation", "Advisor"].nunique()
    assert len(chat.get("ask_table") or []) == expected


def test_average_followup_uses_subset(chat, gt):
    chat.send("Show me Accounting students")
    chat.send("what is their average GPA")
    expected = round(float(gt.loc[gt["Department"] == "Accounting", "GPA"].mean()), 4)
    assert abs(chat.get("ask_value") - expected) < 0.01


def test_clear_filters(chat):
    chat.send("Show me Accounting students")
    chat.send("clear that")
    assert chat.memory()["active_filters"] == []


def test_start_over_resets_everything(chat):
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    chat.send("start over")
    mem = chat.memory()
    assert mem["active_filters"] == []
    assert not mem.get("active_sort")
    assert not mem.get("active_group_by")
    assert mem.get("active_limit") is None
    assert not mem.get("pending_action")


def test_missing_field_no_hallucination(chat):
    chat.send("what is each student's housing status")
    assert chat.get("assistant_mode") == "clarify"
    assert "housing" in (chat.get("clarify_question") or "").lower()
