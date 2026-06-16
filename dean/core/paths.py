"""Central resolver for the app's writable data directory.

In a source checkout (development, tests) ``DEAN_DATA_DIR`` is unset, so every
helper resolves under the repository root exactly as the code did before this
module existed — behavior is unchanged.

In a packaged build the launcher (``packaging/run_dean.py``) sets
``DEAN_DATA_DIR`` to a per-user location (e.g. ``~/Library/Application
Support/Dean``). All mutable state — the SQLite database, exported workbooks,
backups, learned-pattern JSON, logs, and the privacy settings file — then lives
there, so it survives app updates and never gets written inside the read-only
application bundle.

This module imports only the standard library, so it is safe to import from
anywhere (``core`` and ``database`` packages alike) without creating cycles.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository / bundle root. In a frozen PyInstaller build the source tree is
# shipped under ``sys._MEIPASS`` and this resolves there (read-only defaults);
# in a checkout it resolves to the repo root.
BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the writable per-install data directory, creating it if needed."""
    env = os.environ.get("DEAN_DATA_DIR")
    root = Path(env).expanduser() if env else BUNDLE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_dir(*parts: str) -> Path:
    """Return (and create) a subdirectory under the writable data root."""
    directory = data_root().joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
