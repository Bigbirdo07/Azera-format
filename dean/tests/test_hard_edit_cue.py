"""has_hard_edit_cue: which phrasings route a message to the structural-edit
planner (highlight/format/chart/...) instead of the read-only query planner."""

from __future__ import annotations

from nlp.conversation import has_hard_edit_cue


def test_bare_color_word_is_not_a_hard_edit_cue():
    # Caught live: "show me students by their favorite color" was hijacked
    # into a highlight-edit confirmation on an unrelated column, because bare
    # "color" was in the cue list. A question that merely contains the word
    # "color" is not a formatting request.
    assert not has_hard_edit_cue("Show me students by their favorite color")
    assert not has_hard_edit_cue("what is their favorite color")
    assert not has_hard_edit_cue("which students have color blindness noted")


def test_genuine_color_formatting_requests_still_trigger():
    assert has_hard_edit_cue("Color the low GPA rows red")
    assert has_hard_edit_cue("please color these students yellow")
    assert has_hard_edit_cue("color coded by standing")
    assert has_hard_edit_cue("highlight these rows in red")
