"""Phase H: the chat UI and the eval harness share one planning path."""

from __future__ import annotations

import ui.chat_panel as chat_panel
from scripts.eval_planner import run_eval


def _routing(**overrides):
    base = {
        "plan_source": "rules", "intent": "query", "confidence": 0.9, "plan": None,
        "requires_confirmation": False, "confirmation_reason": None, "pending_type": None,
        "warnings": [], "llm_used": False, "validation": {"status": "passed", "errors": []},
        "fallback_reason": None, "active_update": None, "context_note": "", "reveal_sensitive": None,
    }
    base.update(overrides)
    return base


# 1-2: chat goes through planner_router --------------------------------------


def test_chat_invokes_planner_router(chat, monkeypatch):
    calls = {"n": 0}
    real = chat_panel.plan_user_request

    def spy(**kwargs):
        calls["n"] += 1
        return real(**kwargs)

    monkeypatch.setattr(chat_panel, "plan_user_request", spy)
    chat.send("Show me Accounting students")
    assert calls["n"] == 1
    assert chat.get("assistant_mode") == "ask_question"


def test_unified_followup_filter(chat, gt):
    chat.send("Show me Accounting students")
    chat.send("now only below 2.5 GPA")
    expected = int(((gt["Department"] == "Accounting") & (gt["GPA"] < 2.5)).sum())
    assert chat.get("ask_row_count") == expected
    assert {f["column"] for f in chat.memory()["active_filters"]} == {"Department", "GPA"}


def test_unified_replacement_filter(chat):
    chat.send("Show me Accounting students")
    chat.send("what about Biology")
    depts = [f for f in chat.memory()["active_filters"] if f["column"] == "Department"]
    assert len(depts) == 1 and depts[0]["value"] == "Biology"


# 7-9: clarify / unavailable / unsupported from router ------------------------


def test_chat_handles_clarification(chat, monkeypatch):
    monkeypatch.setattr(chat_panel, "plan_user_request",
                        lambda **k: _routing(plan_source="clarification", intent="clarify",
                                             confidence=0.0, confirmation_reason="Please be more specific."))
    chat.send("something vague")
    assert chat.get("assistant_mode") == "clarify"
    assert "specific" in (chat.get("clarify_question") or "")


def test_chat_handles_unavailable(chat, monkeypatch):
    monkeypatch.setattr(chat_panel, "plan_user_request",
                        lambda **k: _routing(intent="unavailable", confirmation_reason="No housing data."))
    chat.send("what is their housing status")
    assert chat.get("assistant_mode") == "clarify"
    assert not chat.get("latest_output_file")


def test_chat_handles_unsupported(chat, monkeypatch):
    monkeypatch.setattr(chat_panel, "plan_user_request",
                        lambda **k: _routing(intent="unsupported", confirmation_reason="Not supported."))
    chat.send("do something impossible")
    assert chat.get("assistant_mode") == "clarify"


# 16-18: LLM route through UI (mocked) ----------------------------------------


def test_llm_route_through_ui(chat, monkeypatch):
    plan = {"operation": "count_rows", "sheet": "Students", "filters": [], "plain_english_question": "count"}
    monkeypatch.setattr(chat_panel, "plan_user_request",
                        lambda **k: _routing(plan_source="llm", intent="query", llm_used=True,
                                             plan=plan, active_update={"filters": [], "sheet": "Students"}))
    chat.send("who needs attention")
    assert chat.get("assistant_mode") == "ask_question"
    assert chat.get("routing_debug")["llm_used"] is True
    assert chat.get("routing_debug")["plan_source"] == "llm"


def test_llm_disabled_never_calls_model(sheets, columns, monkeypatch):
    # The local LLM is on by default (Dean now treats it as a first-class,
    # not opt-in, capability); this verifies the "never calls the model when
    # off" guarantee still holds for a user who explicitly turns it off.
    # Exercised at the planner level directly -- the Streamlit AppTest harness
    # shares process-global caches (get_ollama_manager is @st.cache_resource)
    # across separate test sessions, which makes it an unreliable place to
    # assert this specific invariant.
    import nlp.planner_router as router
    from nlp.planner_router import plan_user_request

    def boom(config):
        raise AssertionError("LLM must not be called when disabled")

    monkeypatch.setattr(router, "_default_llm_call", boom)

    routing = plan_user_request(
        user_message="Show me Accounting students",
        sheets=sheets,
        sheet_columns={"Students": columns},
        selected_sheet="Students",
        conversation_state=None,
        settings={"use_local_llm": False, "llm_enabled": False},
    )
    assert routing["intent"] == "query"
    assert routing["llm_used"] is False


def test_invalid_llm_plan_rejected_in_ui(chat, monkeypatch):
    monkeypatch.setattr(chat_panel, "plan_user_request",
                        lambda **k: _routing(plan_source="clarification", intent="clarify", confidence=0.0,
                                             confirmation_reason="That plan referenced a column I don't have.",
                                             validation={"status": "failed", "errors": ["nonexistent column"]}))
    chat.send("filter by Housing")
    assert chat.get("assistant_mode") == "clarify"
    assert not chat.get("latest_output_file")  # nothing executed


# 10-15: context + pending through unified path -------------------------------


def test_active_context_survives_unified(chat, gt):
    chat.send("Show me Nursing students")
    chat.send("now only seniors")
    cols = {f["column"] for f in chat.memory()["active_filters"]}
    assert cols == {"Department", "Year"}


def test_pending_export_confirm_creates_file(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    assert (chat.memory().get("pending_action") or {}).get("type") == "export"
    chat.send("yes, export")
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_pending_note_confirm_creates_workbook(chat):
    chat.send("show me Accounting students")
    chat.send("add note: Advisor follow-up needed")
    chat.send("yes, do it")
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_pending_field_update_confirm(chat):
    chat.send("show me Accounting students")
    chat.send("set Follow Up Needed to Yes")
    chat.send("yes, do it")
    assert chat.get("latest_output_file")
    assert not chat.memory().get("pending_action")


def test_cancel_clears_pending(chat):
    chat.send("show me Accounting students")
    chat.send("export this list with emails")
    chat.send("no")
    assert not chat.memory().get("pending_action")
    assert not chat.get("latest_output_file")


def test_yes_with_no_pending_is_noop(chat):
    chat.send("yes")
    assert not chat.memory().get("pending_action")
    assert not chat.get("latest_output_file")


# 19: debug state --------------------------------------------------------------


def test_debug_state_includes_routing_and_execution(chat):
    chat.send("Show me Accounting students")
    routing = chat.get("routing_debug")
    assert routing and "plan_source" in routing and "llm_used" in routing
    assert routing["intent"] == "query" and routing["validation_status"] == "passed"


# eval dispatch ----------------------------------------------------------------


def test_eval_harness_dispatch_mode_runs():
    rows = run_eval(llm_enabled=False, dispatch=True)
    assert not [r for r in rows if not r["pass"]]
    # at least some query cases were actually executed through the dispatcher
    assert any(r["dispatched"] is True for r in rows)
