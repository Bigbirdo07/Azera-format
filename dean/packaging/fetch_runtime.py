#!/usr/bin/env python3
"""Populate ``bin/`` (Ollama engine) and ``models/`` (model weights) so the
packaged app ships fully self-contained — the target machine downloads nothing
and never needs a separate Ollama install.

Two pieces, both required (binary alone is not enough — without weights the
engine starts but has no model to run):

  1. The Ollama binary  -> dean/bin/ollama-<platform>          (~30-50 MB)
  2. The model weights   -> dean/models/  (OLLAMA_MODELS layout) (~2 GB)

Usage:
    # Use the locally-installed ollama for THIS platform (recommended), then
    # pre-bake the model into ./models:
    .venv/bin/python packaging/fetch_runtime.py --model llama3.2:3b

    # Point at a specific ollama binary instead of the one on PATH:
    .venv/bin/python packaging/fetch_runtime.py --ollama-bin /opt/homebrew/bin/ollama

The model bake runs an isolated ``ollama serve`` with OLLAMA_MODELS pointed at
``dean/models`` and pulls the model into it, so the weights are committed to the
bundle layout rather than the user's ``~/.ollama``.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = REPO_ROOT / "bin"
MODELS_DIR = REPO_ROOT / "models"
BAKE_PORT = 11455  # isolated port used only while pre-baking the model


def target_binary_name() -> str:
    """Match the names ``core`` / ``nlp`` expect under dean/bin/."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine in {"arm64", "aarch64"}:
            return "ollama-darwin-arm64"
        return "ollama-darwin-amd64"
    if system == "Windows":
        return "ollama-windows-amd64.exe"
    if system == "Linux":
        return "ollama-linux-amd64"
    raise SystemExit(f"Unsupported platform for bundling: {system}/{machine}")


def resolve_ollama_bin(explicit: str | None) -> Path:
    candidate = explicit or shutil.which("ollama")
    if not candidate:
        raise SystemExit(
            "No ollama binary found. Install Ollama (https://ollama.com/download) "
            "or pass --ollama-bin /path/to/ollama."
        )
    path = Path(candidate).resolve()  # follow Homebrew/symlinks to the real binary
    if not path.exists():
        raise SystemExit(f"ollama binary not found at {path}")
    return path


def copy_binary(ollama_bin: Path) -> Path:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    dest = BIN_DIR / target_binary_name()
    shutil.copy2(ollama_bin, dest)
    dest.chmod(0o755)
    print(f"[bin]   copied {ollama_bin} -> {dest}")
    return dest


def _wait_until_up(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _model_present(port: int, model: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=5) as resp:
            import json

            names = {m.get("name", "") for m in json.load(resp).get("models", [])}
        return model in names or f"{model}:latest" in names
    except Exception:
        return False


def bake_model(ollama_bin: Path, model: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OLLAMA_MODELS"] = str(MODELS_DIR)
    env["OLLAMA_HOST"] = f"127.0.0.1:{BAKE_PORT}"

    print(f"[model] starting isolated ollama serve on :{BAKE_PORT} (OLLAMA_MODELS={MODELS_DIR})")
    serve = subprocess.Popen(
        [str(ollama_bin), "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_until_up(BAKE_PORT):
            raise SystemExit("Timed out waiting for the baking ollama server to start.")
        print(f"[model] pulling {model} (this downloads ~2 GB the first time)…")
        subprocess.run([str(ollama_bin), "pull", model], env=env, check=True)
        if not _model_present(BAKE_PORT, model):
            raise SystemExit(f"Pull finished but {model} is not present in {MODELS_DIR}.")
        print(f"[model] baked {model} into {MODELS_DIR}")
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=10)
        except subprocess.TimeoutExpired:
            serve.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.2:3b", help="Model tag to bake (default: llama3.2:3b)")
    parser.add_argument("--ollama-bin", default=None, help="Path to an ollama binary (default: from PATH)")
    parser.add_argument("--skip-binary", action="store_true", help="Do not (re)copy the ollama binary")
    parser.add_argument("--skip-model", action="store_true", help="Do not (re)bake the model weights")
    args = parser.parse_args()

    ollama_bin = resolve_ollama_bin(args.ollama_bin)
    print(f"[info]  using ollama binary: {ollama_bin}")

    if not args.skip_binary:
        copy_binary(ollama_bin)
    if not args.skip_model:
        bake_model(ollama_bin, args.model)

    size = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file()) if MODELS_DIR.exists() else 0
    print(f"[done]  bin/ + models/ ready. models/ size: {size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
