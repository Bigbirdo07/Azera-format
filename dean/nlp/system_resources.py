"""Memory guard for local LLM calls.

This machine has crashed twice from Ollama memory pressure (RAM/swap
exhaustion) on an 8GB Mac. Before every Ollama call, `_call_ollama` in
nlp/local_model.py checks `ollama_call_is_safe()` and falls back to the rule
parser instead of risking another crash.
"""

from __future__ import annotations

import platform
import subprocess

MIN_FREE_MB_DEFAULT = 800
"""Once llama3.2:3b is resident (kept warm via keep_alive), a generate call
mostly needs headroom for activations, not another full model load -- on this
8GB machine, steady-state operation with the model loaded commonly sits
around 900MB-1GB free, which is stable, not crash-adjacent. This threshold
targets the genuinely dangerous zone (near-zero free, imminent swap storm)
rather than blocking ordinary warm-model operation."""

MAX_SWAP_PERCENT = 75.0
"""Independent bad signal even when 'available' memory looks fine — macOS
overcommits and reports available memory generously while already swapping
heavily."""


def available_memory_mb() -> float | None:
    """Free system memory in MB, or None if it can't be determined."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return None


def swap_percent() -> float | None:
    """Percent of swap currently in use, or None if it can't be determined."""
    try:
        import psutil

        return psutil.swap_memory().percent
    except Exception:
        return None


def ollama_call_is_safe(min_free_mb: float = MIN_FREE_MB_DEFAULT) -> tuple[bool, str | None]:
    """Whether it's safe to make an Ollama call right now.

    Fails open (returns safe=True) when memory can't be read at all — no data
    beats blocking every call blindly. Once readable, blocks conservatively.
    """
    free_mb = available_memory_mb()
    if free_mb is not None and free_mb < min_free_mb:
        return False, f"only {free_mb:.0f}MB free (need {min_free_mb:.0f}MB)"

    swap = swap_percent()
    if swap is not None and swap > MAX_SWAP_PERCENT:
        return False, f"swap usage at {swap:.0f}% (limit {MAX_SWAP_PERCENT:.0f}%)"

    return True, None


def list_top_processes(limit: int = 12) -> list[dict]:
    """Top memory-consuming processes on this machine, for a "free up
    memory" admin panel. Excludes this process, Ollama (never offer to quit
    the thing the AI needs to run), and anything not owned by the current
    user (excludes system/root daemons the admin shouldn't be touching)."""
    import getpass
    import os

    try:
        import psutil
    except Exception:
        return []

    current_user = getpass.getuser()
    this_pid = os.getpid()
    rows: list[dict] = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "username"]):
        try:
            info = proc.info
            pid = info.get("pid")
            name = info.get("name") or ""
            if pid == this_pid:
                continue
            if "ollama" in name.lower():
                continue
            if info.get("username") != current_user:
                continue
            mem_info = info.get("memory_info")
            if mem_info is None:
                continue
            mem_mb = mem_info.rss / (1024 * 1024)
            rows.append({"pid": pid, "name": name, "mem_mb": mem_mb})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    rows.sort(key=lambda r: r["mem_mb"], reverse=True)
    return rows[:limit]


def quit_process_gracefully(pid: int, name: str) -> tuple[bool, str | None]:
    """Ask an application to quit via the OS's normal quit signal -- never a
    force-kill. On macOS this sends a real Apple Event, so a well-behaved app
    runs its own quit handler and can still prompt to save unsaved work,
    exactly as if the user had quit it themselves."""
    system = platform.system()
    if system != "Darwin":
        return False, "Graceful quit is only supported on macOS right now."

    try:
        result = subprocess.run(
            ["osascript", "-e", f'tell application "{name}" to quit'],
            timeout=10,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timed out waiting for {name} to quit."
    except Exception as exc:
        return False, f"Could not quit {name}: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or "unknown error"
        return False, f"Could not quit {name}: {detail}"
    return True, None
