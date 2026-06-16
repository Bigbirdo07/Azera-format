"""Frozen entry point for the packaged Dean desktop app.

PyInstaller analyzes this file. At runtime it:
  1. Locates the bundle root (``sys._MEIPASS`` when frozen) and puts the app
     source on ``sys.path`` so ``app.py`` and its ``core``/``nlp``/``ui``/
     ``database`` imports resolve.
  2. Points ``DEAN_DATA_DIR`` at a writable per-user directory and seeds it
     (config + knowledge defaults) on first launch, so nothing mutable is ever
     written inside the read-only application bundle.
  3. Makes the bundled Ollama binary executable.
  4. Launches the Streamlit server (headless) and opens the browser.

The whole thing stays offline: Streamlit binds to localhost and the bundled
Ollama binary + model are used in-process. No download or system install is
required on the target machine.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import threading
import time
import webbrowser
from pathlib import Path


APP_NAME = "Dean"
DEFAULT_SERVER_PORT = 8501


def bundle_root() -> Path:
    """Directory that holds the shipped (read-only) source + bin/ + models/."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def user_data_dir() -> Path:
    """Per-user writable directory for the database, exports, logs, settings."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = (Path(xdg) if xdg else Path.home() / ".local" / "share") / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def seed_defaults(root: Path, data_dir: Path) -> None:
    """Copy shipped default config + knowledge into the data dir on first run.

    Only fills in files that don't exist yet, so user edits are never
    overwritten on subsequent launches.
    """
    for folder in ("config", "knowledge"):
        src = root / folder
        if not src.is_dir():
            continue
        dst = data_dir / folder
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.is_file() and not (dst / item.name).exists():
                shutil.copy2(item, dst / item.name)


def ensure_binary_executable(root: Path) -> None:
    """PyInstaller may strip the +x bit when shipping the Ollama binary as data."""
    bin_dir = root / "bin"
    if not bin_dir.is_dir():
        return
    for binary in bin_dir.iterdir():
        if binary.is_file():
            mode = binary.stat().st_mode
            binary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _select_server_port(preferred: int = DEFAULT_SERVER_PORT) -> int:
    """Use the preferred port when free; otherwise let the OS pick one."""
    import socket

    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return int(sock.getsockname()[1])
    return preferred


def open_browser_when_ready(port: int) -> None:
    """Open the default browser once the Streamlit server is accepting traffic."""
    import socket

    deadline = time.time() + 60
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(f"http://localhost:{port}")
                return
        time.sleep(0.5)


def main() -> int:
    root = bundle_root()

    # Make the shipped app source importable (app.py + core/nlp/ui/database).
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data_dir = user_data_dir()
    os.environ["DEAN_DATA_DIR"] = str(data_dir)
    seed_defaults(root, data_dir)
    ensure_binary_executable(root)

    # Writes that slip through as plain relative paths land in the data dir,
    # never inside the bundle.
    os.chdir(data_dir)

    server_port = _select_server_port()
    threading.Thread(target=open_browser_when_ready, args=(server_port,), daemon=True).start()

    # Hand control to Streamlit's CLI, pointed at the bundled app.py.
    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(root / "app.py"),
        "--server.port",
        str(server_port),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
        "--global.developmentMode",
        "false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
