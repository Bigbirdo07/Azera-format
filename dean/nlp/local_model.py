from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from nlp.llm_json_parser import (
    LLMCommandError,
    ParsedPlan,
    parse_llm_plan,
)
from nlp.model_prompt import (
    OLLAMA_URL,
    build_conversational_prompt,
    build_expert_planner_prompt,
    build_explain_prompt,
    build_intent_prompt,
    build_query_planner_prompt,
)
from nlp.privacy_filter import check_local_model_request_privacy
from nlp.system_resources import ollama_call_is_safe
from core.privacy_guard import PrivacyGuardError, assert_loopback_url


@dataclass(frozen=True)
class LocalPlannerResult:
    """Result of calling the local Ollama expert planner."""

    plan: ParsedPlan | None
    error: str | None = None
    source: str = "local_llm"


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    running: bool
    model_available: bool
    user_message: str
    detail: str | None = None


def get_ollama_status(model_name: str, timeout: int = 2) -> OllamaStatus:
    try:
        assert_loopback_url(OLLAMA_URL)
    except PrivacyGuardError as exc:
        return OllamaStatus(
            available=False,
            running=False,
            model_available=False,
            user_message="Local model setup is blocked by the localhost-only privacy check.",
            detail=str(exc),
        )

    try:
        request = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return OllamaStatus(
            available=False,
            running=False,
            model_available=False,
            user_message="Ollama is not running locally. The app will keep using the built-in rule-based parser.",
            detail=str(exc),
        )

    models = [item.get("name") for item in payload.get("models", [])]
    model_available = model_name in models
    if not model_available:
        return OllamaStatus(
            available=False,
            running=True,
            model_available=False,
            user_message=f"Ollama is running, but model `{model_name}` is not installed.",
            detail=f"Available models: {', '.join(models) if models else 'none'}",
        )
    return OllamaStatus(
        available=True,
        running=True,
        model_available=True,
        user_message=f"Ollama is ready with model `{model_name}`.",
    )


def test_ollama_connection(model_name: str) -> tuple[bool, str]:
    status = get_ollama_status(model_name, timeout=3)
    return status.available, status.user_message


def plan_from_local_model(
    *,
    user_request: str,
    model_name: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
    mapped_columns: dict[str, str] | None = None,
) -> LocalPlannerResult:
    """Call the local Ollama planner and return a validated ParsedPlan envelope.

    The LLM only receives: the user request, sheet names, column names, mapped
    column concepts, the allowed actions, and the few-shot examples. Nothing
    about workbook data, file paths, or logs is included.
    """
    privacy_check = check_local_model_request_privacy(user_request)
    if not privacy_check.allowed:
        reasons = ", ".join(privacy_check.reasons)
        return LocalPlannerResult(
            plan=None,
            error=(
                "Local model fallback was skipped because the request may contain "
                f"sensitive data: {reasons}."
            ),
        )

    try:
        assert_loopback_url(OLLAMA_URL)
    except PrivacyGuardError as exc:
        return LocalPlannerResult(plan=None, error=str(exc))

    prompt = build_expert_planner_prompt(
        user_request=user_request,
        sheet_names=sheet_names,
        sheet_columns=sheet_columns,
        mapped_columns=mapped_columns,
    )
    raw_response, error = _call_ollama(prompt, model_name)
    if error or raw_response is None:
        return LocalPlannerResult(plan=None, error=error)

    try:
        plan = parse_llm_plan(raw_response, sheet_columns)
    except LLMCommandError as exc:
        return LocalPlannerResult(plan=None, error=str(exc))

    return LocalPlannerResult(plan=plan)


# Planner-sized JSON cold-start: llama3.2:3b ~25-40s, mistral:7b ~85-115s on
# Apple Silicon Q4. Sustained load (thermal throttling) slows both further, so
# a tight 60s timeout silently fails every LLM call back to the rule parser.
# Budget generously; the call still returns as soon as the model is done.
OLLAMA_TIMEOUT_SECONDS = 300

# Streamlit reruns on every interaction. If a slow LLM call is already running,
# a second rerun would spawn another one and pile up — that's what locks up the
# machine. This non-blocking lock makes the second caller fall back to the rule
# parser instead of queueing another model invocation.
_OLLAMA_INFLIGHT_LOCK = threading.Lock()


@dataclass(frozen=True)
class IntentResult:
    request_type: str | None
    confidence: float
    reason: str = ""
    error: str | None = None


def classify_intent_with_model(
    *,
    user_request: str,
    model_name: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
) -> IntentResult:
    """Ask the local model to classify ask/edit/clarify. Privacy-safe: only the
    message plus sheet and column names are sent."""
    try:
        assert_loopback_url(OLLAMA_URL)
    except PrivacyGuardError as exc:
        return IntentResult(None, 0.0, error=str(exc))

    prompt = build_intent_prompt(
        user_request=user_request,
        sheet_names=sheet_names,
        sheet_columns=sheet_columns,
    )
    raw, error = _call_ollama(prompt, model_name, num_ctx=SHORT_NUM_CTX, num_predict=SHORT_NUM_PREDICT)
    if error or raw is None:
        return IntentResult(None, 0.0, error=error)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return IntentResult(None, 0.0, error=f"Intent JSON parse failed: {exc}")
    request_type = payload.get("request_type")
    if request_type not in {"ask_question", "edit_workbook", "clarify"}:
        return IntentResult(None, 0.0, error=f"Invalid request_type: {request_type!r}")
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return IntentResult(request_type, confidence, reason=str(payload.get("reason", "")))


def plan_query_from_local_model(
    *,
    user_request: str,
    model_name: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
    mapped_columns: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Produce an ask_question query plan. Returns (query_dict, error)."""
    privacy_check = check_local_model_request_privacy(user_request)
    if not privacy_check.allowed:
        reasons = ", ".join(privacy_check.reasons)
        return None, f"Query skipped (possible sensitive data): {reasons}."
    try:
        assert_loopback_url(OLLAMA_URL)
    except PrivacyGuardError as exc:
        return None, str(exc)

    prompt = build_query_planner_prompt(
        user_request=user_request,
        sheet_names=sheet_names,
        sheet_columns=sheet_columns,
        mapped_columns=mapped_columns,
    )
    raw, error = _call_ollama(prompt, model_name)
    if error or raw is None:
        return None, error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Query JSON parse failed: {exc}"
    payload["request_type"] = "ask_question"
    return payload, None


def explain_result_with_model(
    *,
    user_question: str,
    verified_result: dict[str, Any],
    model_name: str,
) -> tuple[str | None, str | None]:
    """Have the model phrase an already-computed result in plain English.

    The model receives only the question and the verified result summary (never
    raw rows), and is instructed not to recompute any numbers.
    """
    try:
        assert_loopback_url(OLLAMA_URL)
    except PrivacyGuardError as exc:
        return None, str(exc)
    prompt = build_explain_prompt(
        user_question=user_question,
        verified_result=verified_result,
    )
    raw, error = _call_ollama(prompt, model_name, json_mode=False, num_ctx=SHORT_NUM_CTX, num_predict=SHORT_NUM_PREDICT)
    if error or raw is None:
        return None, error
    return raw.strip(), None


# Markers that mean the local model described its own prompt/JSON envelope
# instead of answering (a common small-model failure). If any appears, we reject
# the reply and let the dispatcher fall back to the deterministic message.
_META_DESCRIPTION_MARKERS = (
    "json", "array of objects", "appears to be a response", "is structured as",
    "api or a web application", "metadata about", "payload",
    "active_context", "allowed_next_actions", "hidden_sensitive_fields",
    "verified_result", "understood_plan", "privacy_boundary", "row_sample",
    "row_sample_policy", "result_rows", "group_by", "the data structure", "key-value",
    "the provided json", "the json data", "data: an array",
)


def _looks_like_meta_description(text: str) -> bool:
    """True when the reply is describing the prompt's structure, not answering."""
    low = text.lower()
    return any(marker in low for marker in _META_DESCRIPTION_MARKERS)


_COUNT_NOUN_RE = re.compile(
    r"\b(\d{1,6})\s+"
    r"(student|students|record|records|row|rows|match|matches|matching|"
    r"advisor|advisors|teacher|teachers|department|departments|major|majors|"
    r"group|groups|category|categories|people|person|"
    r"freshman|freshmen|sophomore|sophomores|junior|juniors|senior|seniors)\b",
    re.IGNORECASE,
)
_MORE_COUNT_NOUN_RE = re.compile(
    r"\b(\d{1,6})\s+more\s+"
    r"(advisor|advisors|teacher|teachers|department|departments|major|majors|"
    r"group|groups|category|categories|student|students|record|records|row|rows)\b",
    re.IGNORECASE,
)
_HIDDEN_FIELDS_RE = re.compile(
    r"\b(hidden|redacted|withheld)\b.*\b(field|fields|column|columns|information)\b|"
    r"\b(field|fields|column|columns|information)\b.*\b(hidden|redacted|withheld)\b",
    re.IGNORECASE,
)


def _verified_numbers(verified_result: dict[str, Any], row_sample: list[dict[str, Any]] | None) -> set[int]:
    """Numbers the narrator may safely repeat.

    The conversational model is allowed to phrase results, but not to invent
    counts. We allow the verified row_count/value plus integer values present in
    the redacted result rows, such as group Count values.
    """
    allowed: set[int] = set()
    for key in ("row_count", "value"):
        value = verified_result.get(key)
        if isinstance(value, int):
            allowed.add(value)
        elif isinstance(value, float) and value.is_integer():
            allowed.add(int(value))
    for row in row_sample or []:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if isinstance(value, int):
                allowed.add(value)
            elif isinstance(value, float) and value.is_integer():
                allowed.add(int(value))
    return allowed


def _contradicts_verified_counts(
    text: str,
    verified_result: dict[str, Any],
    row_sample: list[dict[str, Any]] | None,
) -> bool:
    """Reject narrator replies that attach an unverified count to rows/students."""
    allowed = _verified_numbers(verified_result, row_sample)
    if not allowed:
        return False
    visible_rows = len(row_sample or [])
    for match in _MORE_COUNT_NOUN_RE.finditer(text or ""):
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        if visible_rows == 0 or number >= visible_rows:
            return True
    for match in _COUNT_NOUN_RE.finditer(text or ""):
        try:
            number = int(match.group(1))
        except ValueError:
            continue
        if number not in allowed:
            return True
    return False


def _mentions_hidden_fields_without_any_hidden(
    text: str,
    hidden_sensitive_fields: list[str] | None,
) -> bool:
    return not hidden_sensitive_fields and bool(_HIDDEN_FIELDS_RE.search(text or ""))


def converse_about_result_with_model(
    *,
    user_question: str,
    understood_plan: str,
    verified_result: dict[str, Any],
    model_name: str,
    active_context: dict[str, Any] | None = None,
    hidden_sensitive_fields: list[str] | None = None,
    allowed_next_actions: list[str] | None = None,
    row_sample: list[dict[str, Any]] | None = None,
    row_sample_policy: str = "redacted_name_safe_rows",
) -> tuple[str | None, str | None]:
    """Conversational narrator that runs AFTER validated execution.

    Receives the user question, the interpretation, the verified result summary,
    active context, hidden-field names, allowed next actions, and a bounded row
    sample so it can name specific students when the user asks for them. The row
    sample is redacted/name-safe by default; admins can explicitly enable full
    local row access. Returns a 1–3 sentence plain-text reply or (None, error).
    """
    try:
        assert_loopback_url(OLLAMA_URL)
    except PrivacyGuardError as exc:
        return None, str(exc)
    prompt = build_conversational_prompt(
        user_question=user_question,
        understood_plan=understood_plan,
        verified_result=verified_result,
        active_context=active_context,
        hidden_sensitive_fields=hidden_sensitive_fields,
        allowed_next_actions=allowed_next_actions,
        row_sample=row_sample,
        row_sample_policy=row_sample_policy,
    )
    raw, error = _call_ollama(prompt, model_name, json_mode=False, num_ctx=SHORT_NUM_CTX, num_predict=SHORT_NUM_PREDICT)
    if error or raw is None:
        return None, error
    reply = raw.strip()
    if not reply or _looks_like_meta_description(reply):
        # The model described the JSON envelope instead of answering — drop it
        # so the dispatcher uses the deterministic, validated phrasing instead.
        return None, "narrator produced a meta/JSON description, not an answer"
    if _contradicts_verified_counts(reply, verified_result, row_sample):
        return None, "narrator contradicted verified workbook counts"
    if _mentions_hidden_fields_without_any_hidden(reply, hidden_sensitive_fields):
        return None, "narrator mentioned hidden fields when none were hidden"
    return reply, None


PLANNER_NUM_CTX = 8192
"""Context window large enough to carry the expert-planner payload
(~5,200 tokens of playbooks + rules + few-shots, see knowledge/*.json)."""

SHORT_NUM_CTX = 2048
"""Context window for short prompts (intent classification, narrator,
explanation). Keeping these calls small frees ~500 MB of KV cache per call
on Apple Silicon when the model is loaded with default keep_alive=5m."""

ANALYST_NUM_CTX = 4096
"""Context window for the code-analyst loop (nlp/code_analyst.py). Its system
prompt (schema + sample rows + hints) measures ~1,000 tokens on the real
roster; multi-step iterations (up to MAX_ITERATIONS) and 3-turn history can
add another ~2,000-2,500 in a worst-case turn. 4096 keeps comfortable
headroom against truncation while still cutting KV-cache memory roughly in
half versus the 8192 default the analyst previously inherited unintentionally
(it never set num_ctx explicitly)."""

PLANNER_NUM_PREDICT = 1024
"""Hard ceiling on tokens the model may generate for a planner/code reply.
Without a cap, a small model that starts repeating itself will stream until the
300s timeout while holding the in-flight lock — which looks exactly like a
freeze. A validated plan or pandas snippet fits comfortably under 1024 tokens."""

SHORT_NUM_PREDICT = 384
"""Output ceiling for the short calls (intent JSON, 1-3 sentence narrator,
plain-English explanation). These never need to be long; capping them tightly
bounds worst-case latency."""


def _call_ollama(
    prompt: str,
    model_name: str,
    timeout: int = OLLAMA_TIMEOUT_SECONDS,
    json_mode: bool = True,
    num_ctx: int = PLANNER_NUM_CTX,
    num_predict: int = PLANNER_NUM_PREDICT,
) -> tuple[str | None, str | None]:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"

    safe, reason = ollama_call_is_safe()
    if not safe:
        return None, f"Skipped local model call: {reason} (using rule parser)."

    if not _OLLAMA_INFLIGHT_LOCK.acquire(blocking=False):
        return None, "Local Ollama is already busy with another request; using the rule parser."

    try:
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"Local Ollama request failed: {exc}"
    finally:
        _OLLAMA_INFLIGHT_LOCK.release()

    text = response_payload.get("response", "") if isinstance(response_payload, dict) else ""
    if not text:
        return None, "Local Ollama returned an empty response."
    return text, None


# How long Ollama keeps the warmed model resident after the warm-up ping. Long
# enough to cover a user reading the screen / uploading a workbook before they
# ask their first question; real queries refresh this each time they run.
WARM_KEEP_ALIVE = "15m"


def warm_model(model_name: str, timeout: int = OLLAMA_TIMEOUT_SECONDS) -> None:
    """Fire-and-forget: ask Ollama to load the model into memory now so the first
    real question doesn't pay the 25-40s cold-start.

    Runs on a daemon thread and never raises: if Ollama is offline, the model is
    missing, or the loopback check fails, warming is silently skipped and the app
    falls back to its normal cold-start-then-rule-parser behavior. It does NOT
    take the in-flight lock — a 1-token generate is cheap, and holding the lock
    through the model load would needlessly bounce a concurrent first question to
    the rule parser.
    """

    def _run() -> None:
        try:
            assert_loopback_url(OLLAMA_URL)
        except PrivacyGuardError:
            return
        payload = {
            "model": model_name,
            "prompt": "ok",
            "stream": False,
            "keep_alive": WARM_KEEP_ALIVE,
            "options": {"temperature": 0, "num_predict": 1},
        }
        try:
            request = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response.read()
        except Exception:
            pass  # offline / model missing — warming is best-effort

    threading.Thread(target=_run, name="ollama-warm", daemon=True).start()
