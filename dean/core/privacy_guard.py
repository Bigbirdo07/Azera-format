from __future__ import annotations

from urllib.parse import urlparse


ALLOWED_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
ALLOWED_OLLAMA_PORTS = {11434, 11438, 11439, 11440, 11441, 11442}


class PrivacyGuardError(ValueError):
    pass


def assert_loopback_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http":
        raise PrivacyGuardError("Only plain local HTTP is allowed for Ollama localhost access.")
    if parsed.hostname not in ALLOWED_LOCAL_HOSTS:
        raise PrivacyGuardError("Remote Ollama endpoints are blocked. Use localhost only.")
    if parsed.port not in ALLOWED_OLLAMA_PORTS:
        raise PrivacyGuardError(
            f"Only local Ollama ports 11434 and the range 11438-11442 are allowed. Port {parsed.port} blocked."
        )


def local_only_security_summary(use_local_llm: bool, ollama_url: str) -> list[str]:
    status = [
        "Cloud APIs: disabled",
        "Telemetry: disabled",
        "Analytics: disabled",
        "Remote logging: disabled",
        "Spreadsheet rows sent to model: only a name-safe result sample to the chat narrator (Strict Privacy Mode sends none)",
        f"Local LLM: {'enabled' if use_local_llm else 'disabled'}",
        f"Ollama endpoint: {ollama_url}",
    ]
    try:
        assert_loopback_url(ollama_url)
    except PrivacyGuardError as exc:
        status.append(f"Security issue: {exc}")
    else:
        status.append("Ollama endpoint check: localhost loopback interface verified")
    return status
