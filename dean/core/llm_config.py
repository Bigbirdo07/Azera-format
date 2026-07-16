"""Local-LLM configuration.

The local model is always optional. Strict privacy mode force-disables it.
This module centralizes the settings and maps the app's existing settings dict
onto the Phase G config shape so nothing else has to change.
"""

from __future__ import annotations

from typing import Any

DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "llm_enabled": False,
    "llm_provider": "ollama",
    "ollama_base_url": "http://localhost:11434",
    "planner_model": "llama3.2:3b",
    "explanation_model": "llama3.2:3b",
    "planner_temperature": 0.1,
    "explanation_temperature": 0.3,
    "planner_timeout_seconds": 20,
    "max_retries": 1,
    "llm_explanations_enabled": False,
    "conversation_llm_enabled": False,
    "planner_full_row_access": False,
    "local_llm_full_row_access": False,
    "local_llm_all_matching_rows": False,
}


def from_app_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Map the existing app settings dict onto the Phase G config.

    Backward compatible: `use_local_llm` -> `llm_enabled`, `ollama_model` ->
    planner/explanation model. Strict privacy mode forces the model off.
    """
    settings = settings or {}
    config = dict(DEFAULT_LLM_CONFIG)
    strict = bool(settings.get("strict_privacy_mode", False))
    enabled = bool(settings.get("llm_enabled", settings.get("use_local_llm", False)))
    if strict:
        enabled = False
    model = settings.get("planner_model") or settings.get("ollama_model") or config["planner_model"]
    # The conversational narrator runs *after* validated execution. Strict
    # privacy still force-disables it (Option A: maximum safety; deterministic
    # narration remains).
    conversation_enabled = bool(
        enabled and settings.get("conversation_llm_enabled", False)
    )
    full_row_access = bool(
        conversation_enabled
        and settings.get("local_llm_full_row_access", False)
        and not strict
    )
    all_matching_rows = bool(
        full_row_access
        and settings.get("local_llm_all_matching_rows", False)
        and not strict
    )
    planner_full_row_access = bool(
        enabled
        and settings.get("planner_full_row_access", False)
        and not strict
    )
    config.update(
        {
            "llm_enabled": enabled,
            "planner_model": model,
            "explanation_model": settings.get("explanation_model") or model,
            "llm_explanations_enabled": bool(
                enabled and settings.get("llm_explanations_enabled", settings.get("use_local_llm", False))
            ),
            "conversation_llm_enabled": conversation_enabled,
            "planner_full_row_access": planner_full_row_access,
            "local_llm_full_row_access": full_row_access,
            "local_llm_all_matching_rows": all_matching_rows,
        }
    )
    for key in ("ollama_base_url", "planner_temperature", "explanation_temperature",
                "planner_timeout_seconds", "max_retries"):
        if key in settings:
            config[key] = settings[key]
    return config
