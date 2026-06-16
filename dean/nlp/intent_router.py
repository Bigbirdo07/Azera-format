"""Intent router for the unified assistant.

Classifies every user message into one of three safe paths:
    ask_question   - read-only; pandas computes an answer
    edit_workbook  - changes/creates something; needs a validated plan + confirm
    clarify        - too vague or refers to something that does not exist

Rule-based first (fast, deterministic, free), with a local-Ollama fallback only
when the rules are not confident. The router never sees spreadsheet rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlp.local_model import classify_intent_with_model
from nlp.synonym_mapper import normalize_text


ASK_QUESTION = "ask_question"
EDIT_WORKBOOK = "edit_workbook"
CLARIFY = "clarify"

RULE_CONFIDENCE = 0.85
WEAK_CONFIDENCE = 0.45
LLM_FALLBACK_THRESHOLD = 0.6

# Phrases that are inherently underspecified — always clarify.
VAGUE_PHRASES = (
    "clean this up",
    "clean it up",
    "clean up",
    "fix it",
    "fix the bad ones",
    "fix them up",
    "make it better",
    "make it nice",
    "make it good",
    "do something",
    "do the report",
    "do the thing",
    "help me",
    "organize them",
    "organize it",
    "sort it out",
    "use the right column",
    "the right one",
    "handle it",
)

# References to a prior result. If there is no prior result, clarify.
BACKREFERENCE_PHRASES = (
    "from before",
    "that thing",
    "earlier",
    "previous result",
    "last one",
    "the ones from before",
)

# Strong edit signals: a verb/keyword that implies changing or creating output.
EDIT_KEYWORDS = (
    "highlight",
    "color",
    "colour",
    "fill",
    "bold",
    "move ",
    "put ",
    "send ",
    "copy ",
    "create",
    "add ",
    "insert",
    "make a",
    "make an",
    "make it a",
    "build",
    "generate",
    "format",
    "reformat",
    "freeze",
    "autofit",
    "auto fit",
    "auto-fit",
    "export",
    "save as",
    "download",
    "sort ",
    "reorder",
    "remove duplicate",
    "dedupe",
    "drop duplicate",
    "apply",
    "conditional format",
    "sumifs",
    "sumif",
    "countifs",
    "vlookup",
    "xlookup",
    "formula",
    "flag",
    "new sheet",
    "new tab",
    "another sheet",
    "another tab",
    "summary sheet",
    "report",
    "chart",
    "graph",
    "plot",
)

# Strong ask signals: question wording that does not change the file.
ASK_KEYWORDS = (
    "how many",
    "how much",
    "number of",
    "count of",
    "which ",
    "who ",
    "whose ",
    "what ",
    "are there",
    "is there",
    "do we have",
    "do any",
    "summarize",
    "summarise",
    "summary of",
    "overview",
    "top ",
    "average",
    "mean ",
    "total ",
    "highest",
    "lowest",
    "most ",
    "least ",
    "what looks wrong",
    "anything wrong",
    "missing values",
    "missing data",
    "duplicate",
    "list ",
    "show ",
    "find ",
)

ASK_QUESTION_STARTERS = (
    "how",
    "which",
    "who",
    "whose",
    "what",
    "are",
    "is",
    "do",
    "does",
    "can",
    "where",
    "when",
    "why",
)


@dataclass(frozen=True)
class IntentClassification:
    request_type: str
    confidence: float
    source: str  # "rule" | "local_llm" | "fallback"
    reason: str = ""


def classify_intent(
    *,
    user_request: str,
    sheet_names: list[str],
    sheet_columns: dict[str, list[str]],
    use_local_llm: bool = False,
    ollama_model: str = "llama3.2:3b",
    has_previous_result: bool = False,
) -> IntentClassification:
    text = normalize_text(user_request)

    # 1. Inherently vague -> clarify.
    if _matches_any(text, VAGUE_PHRASES):
        return IntentClassification(CLARIFY, WEAK_CONFIDENCE, "rule", "Vague request.")

    # 2. References a prior result that does not exist -> clarify.
    if _matches_any(text, BACKREFERENCE_PHRASES) and not has_previous_result:
        return IntentClassification(
            CLARIFY, WEAK_CONFIDENCE, "rule", "Refers to a previous result that does not exist."
        )

    has_edit = _matches_any(text, EDIT_KEYWORDS)
    has_ask = _matches_any(text, ASK_KEYWORDS) or _starts_with_question(text)

    # 3. Decide from rule signals.
    #    "show / list / find" are read-only words that also appear in EDIT via
    #    chart/report keywords; an explicit edit verb wins over a bare ask word.
    if has_edit and not _is_pure_question(text):
        return IntentClassification(EDIT_WORKBOOK, RULE_CONFIDENCE, "rule", "Edit verb detected.")
    if has_ask:
        return IntentClassification(ASK_QUESTION, RULE_CONFIDENCE, "rule", "Question wording detected.")
    if has_edit:
        return IntentClassification(EDIT_WORKBOOK, RULE_CONFIDENCE, "rule", "Edit verb detected.")

    # 4. Rules unsure -> optional local LLM fallback.
    if use_local_llm:
        llm = classify_intent_with_model(
            user_request=user_request,
            model_name=ollama_model,
            sheet_names=sheet_names,
            sheet_columns=sheet_columns,
        )
        if llm.request_type and llm.confidence >= LLM_FALLBACK_THRESHOLD:
            return IntentClassification(
                llm.request_type, llm.confidence, "local_llm", llm.reason or "Model classification."
            )

    # 5. Still unsure -> clarify rather than guess.
    return IntentClassification(
        CLARIFY, WEAK_CONFIDENCE, "fallback", "Could not confidently classify the request."
    )


def _matches_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _starts_with_question(text: str) -> bool:
    first = text.split(" ", 1)[0] if text else ""
    return first in ASK_QUESTION_STARTERS


def _is_pure_question(text: str) -> bool:
    """True when the message is clearly a question even though it also contains
    an edit-flavored noun like 'report' or 'chart' (e.g. 'how many are in the
    report')."""
    return _starts_with_question(text) and not _matches_any(
        text,
        (
            "highlight",
            "create",
            "make a",
            "make an",
            "build",
            "generate",
            "format",
            "move ",
            "put ",
            "add ",
            "freeze",
            "export",
            "sort ",
        ),
    )
