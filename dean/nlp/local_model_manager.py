import subprocess
import os
import sys
import time
import urllib.request
import urllib.error
import json
import platform
import atexit
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import streamlit as st

from nlp.model_prompt import OLLAMA_PORT, OLLAMA_URL


@dataclass
class LLMRuntimeStatus:
    mode: str          # "disabled" | "rule_only" | "bundled_ollama" | "system_ollama"
    endpoint: str
    port: int
    model_name: str
    server_running: bool
    model_available: bool
    fallback_used: bool
    error_message: str | None = None
    privacy_status: str = "local-only"


class BundledOllamaManager:
    def __init__(self):
        self.port = OLLAMA_PORT
        self.url = OLLAMA_URL
        self.process = None
        self.status = LLMRuntimeStatus(
            mode="rule_only",
            endpoint=self.url,
            port=self.port,
            model_name="llama3.2:3b",
            server_running=False,
            model_available=False,
            fallback_used=False,
            error_message="Ollama manager is not initialized."
        )

    def _get_binary_path(self) -> tuple[Path | None, str | None]:
        base_dir = Path(__file__).resolve().parents[1]
        sys_name = platform.system()
        machine = platform.machine().lower()

        if sys_name == "Darwin":
            if machine in {"arm64", "aarch64"}:
                binary_name = "ollama-darwin-arm64"
            elif machine in {"x86_64", "amd64"}:
                binary_name = "ollama-darwin-amd64"
            else:
                return None, f"Unsupported macOS CPU architecture: {machine}"
        elif sys_name == "Windows":
            if machine in {"x86_64", "amd64"}:
                binary_name = "ollama-windows-amd64.exe"
            else:
                return None, f"Unsupported Windows CPU architecture: {machine}"
        elif sys_name == "Linux":
            if machine in {"x86_64", "amd64"}:
                binary_name = "ollama-linux-amd64"
            else:
                return None, f"Unsupported Linux CPU architecture: {machine}"
        else:
            return None, f"Unsupported operating system: {sys_name}"

        bin_path = base_dir / "bin" / binary_name
        if not bin_path.exists():
            return None, f"Bundled binary missing: '{binary_name}' not found under dean/bin/."
        return bin_path, None

    def start(self, model_name: str, debug_mode: bool = False) -> bool:
        # Prevent starting a duplicate process if already running with correct model
        if self.process and self.process.poll() is None and self.status.model_name == model_name:
            return True

        self.status = LLMRuntimeStatus(
            mode="rule_only",
            endpoint=self.url,
            port=self.port,
            model_name=model_name,
            server_running=False,
            model_available=False,
            fallback_used=False
        )

        binary_path, err = self._get_binary_path()
        if err:
            # Check if we are running in development or have explicit system fallback permissions
            allow_fallback = os.environ.get("DEAN_ALLOW_SYSTEM_OLLAMA_FALLBACK", "").lower() == "true"
            is_packaged = hasattr(sys, "_MEIPASS")
            
            if (not is_packaged and self.port == 11434) or allow_fallback:
                return self._check_system_ollama(model_name)

            self.status = LLMRuntimeStatus(
                mode="rule_only",
                endpoint=self.url,
                port=self.port,
                model_name=model_name,
                server_running=False,
                model_available=False,
                fallback_used=False,
                error_message=f"Bundled LLM could not start: {err}"
            )
            return False

        base_dir = Path(__file__).resolve().parents[1]
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{self.port}"
        env["OLLAMA_MODELS"] = str(models_dir)

        # Configure debug logging redirection
        stdout_fd = subprocess.DEVNULL
        stderr_fd = subprocess.DEVNULL
        if debug_mode:
            log_dir = base_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "ollama.log"
            try:
                stdout_fd = open(log_file, "a", encoding="utf-8")
                stderr_fd = stdout_fd
            except Exception as e:
                print(f"Failed to open log file {log_file}: {e}")

        try:
            self.process = subprocess.Popen(
                [str(binary_path), "serve"],
                env=env,
                stdout=stdout_fd,
                stderr=stderr_fd
            )
            
            # Wait for responsive server & validate model tags list
            for _ in range(15):
                if self.process.poll() is not None:
                    # Subprocess terminated prematurely
                    self.status = LLMRuntimeStatus(
                        mode="rule_only",
                        endpoint=self.url,
                        port=self.port,
                        model_name=model_name,
                        server_running=False,
                        model_available=False,
                        fallback_used=False,
                        error_message="Ollama server crashed on startup."
                    )
                    return False
                try:
                    request = urllib.request.Request(f"{self.url}/api/tags", method="GET")
                    with urllib.request.urlopen(request, timeout=1) as response:
                        if response.status == 200:
                            payload = json.loads(response.read().decode("utf-8"))
                            models = [item.get("name") for item in payload.get("models", [])]
                            if model_name in models or f"{model_name}:latest" in models:
                                self.status = LLMRuntimeStatus(
                                    mode="bundled_ollama",
                                    endpoint=self.url,
                                    port=self.port,
                                    model_name=model_name,
                                    server_running=True,
                                    model_available=True,
                                    fallback_used=False
                                )
                                return True
                            else:
                                self.stop()
                                self.status = LLMRuntimeStatus(
                                    mode="rule_only",
                                    endpoint=self.url,
                                    port=self.port,
                                    model_name=model_name,
                                    server_running=True,
                                    model_available=False,
                                    fallback_used=False,
                                    error_message=f"Bundled Ollama started, but model `{model_name}` was not found in dean/models."
                                )
                                return False
                except Exception:
                    time.sleep(0.5)

            self.stop()
            self.status = LLMRuntimeStatus(
                mode="rule_only",
                endpoint=self.url,
                port=self.port,
                model_name=model_name,
                server_running=False,
                model_available=False,
                fallback_used=False,
                error_message="Ollama server started but failed to respond within 7.5s."
            )
        except Exception as e:
            self.status = LLMRuntimeStatus(
                mode="rule_only",
                endpoint=self.url,
                port=self.port,
                model_name=model_name,
                server_running=False,
                model_available=False,
                fallback_used=False,
                error_message=f"Failed to launch bundled Ollama: {e}"
            )
        return False

    def _check_system_ollama(self, model_name: str) -> bool:
        try:
            request = urllib.request.Request(f"{self.url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=1) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    models = [item.get("name") for item in payload.get("models", [])]
                    model_ok = model_name in models or f"{model_name}:latest" in models
                    
                    self.status = LLMRuntimeStatus(
                        mode="system_ollama" if model_ok else "rule_only",
                        endpoint=self.url,
                        port=self.port,
                        model_name=model_name,
                        server_running=True,
                        model_available=model_ok,
                        fallback_used=True,
                        error_message=None if model_ok else f"System Ollama running, but model `{model_name}` is not installed."
                    )
                    return model_ok
        except Exception as e:
            self.status = LLMRuntimeStatus(
                mode="rule_only",
                endpoint=self.url,
                port=self.port,
                model_name=model_name,
                server_running=False,
                model_available=False,
                fallback_used=True,
                error_message=f"System Ollama at {self.url} is not running: {e}"
            )
        return False

    def update_state(self, settings: dict[str, Any], debug_mode: bool = False):
        strict = settings.get("strict_privacy_mode", True)
        enabled = settings.get("use_local_llm", False) and not strict
        model_name = settings.get("ollama_model", "llama3.2:3b")

        if enabled:
            if not self.is_running_with(model_name):
                if self.process or self.status.server_running:
                    self.stop()
                self.start(model_name, debug_mode)
        else:
            if self.process or self.status.server_running:
                self.stop()
            self.status = LLMRuntimeStatus(
                mode="disabled" if strict else "rule_only",
                endpoint=self.url,
                port=self.port,
                model_name=model_name,
                server_running=False,
                model_available=False,
                fallback_used=False,
                error_message="Local LLM is disabled by user settings."
            )

    def is_running_with(self, model_name: str) -> bool:
        if not self.status.server_running:
            return False
        if self.status.model_name != model_name:
            return False
        if self.process and self.process.poll() is not None:
            return False
        return True

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self.status = LLMRuntimeStatus(
            mode="rule_only",
            endpoint=self.url,
            port=self.port,
            model_name=self.status.model_name,
            server_running=False,
            model_available=False,
            fallback_used=False,
            error_message="Ollama server stopped."
        )


@st.cache_resource
def get_ollama_manager() -> BundledOllamaManager:
    manager = BundledOllamaManager()
    atexit.register(manager.stop)
    return manager
