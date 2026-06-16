# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Dean desktop app (onedir).

Build:   .venv/bin/pyinstaller dean.spec --noconfirm
Output:  dist/Dean/  (ship this whole folder; on macOS also dist/Dean.app)

Design notes
------------
* onedir (not onefile): the ~2 GB bundled model must NOT be re-extracted to a
  temp dir on every launch. onedir keeps bin/ and models/ on disk next to the
  app.
* The app's own source (app.py + core/nlp/ui/database) ships as DATA and is
  imported at runtime via the sys.path injection in packaging/run_dean.py. This
  keeps the code transparent/inspectable (a selling point for "no hidden cloud
  calls") and avoids PyInstaller missing Streamlit's dynamically-run app graph.
* Third-party deps (streamlit, pandas, openpyxl, …) are frozen via collect_all
  so they're present even though the app modules that import them aren't
  analyzed.
* bin/ (Ollama engine) and models/ (weights) are bundled so the target machine
  installs and downloads nothing.
"""

import sys
from pathlib import Path

from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_all, copy_metadata

REPO = Path(SPECPATH)
ICON_ICNS = str(REPO / "packaging" / "Dean.icns")  # macOS
ICON_ICO = str(REPO / "packaging" / "Dean.ico")    # Windows

datas = []
binaries = []
hiddenimports = []

# --- Third-party runtime deps -------------------------------------------------
for pkg in ("streamlit", "pandas", "openpyxl", "altair", "pyarrow", "numpy"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # pragma: no cover - build-time diagnostics
        print(f"[spec] collect_all({pkg}) skipped: {exc}")

# Streamlit (and several deps) read their version via importlib.metadata.
for pkg in ("streamlit", "pandas", "numpy", "pyarrow", "altair", "openpyxl", "packaging"):
    try:
        datas += copy_metadata(pkg)
    except Exception as exc:  # pragma: no cover
        print(f"[spec] copy_metadata({pkg}) skipped: {exc}")

# Single-file data (2-tuples). The package source dirs and the bin/models
# bundles are added as Tree() TOCs further down (passed to COLLECT), since
# Tree entries are 3-tuples that don't belong in Analysis(datas=...).
datas += [
    ("app.py", "."),
    (".streamlit/config.toml", ".streamlit"),
]

# --- Directory trees: app source + seed assets + Ollama engine/weights --------
# Shipped as DATA and imported at runtime via the sys.path injection in
# packaging/run_dean.py. database/ carries schema.sql; config/ + knowledge/ are
# seeded into the user data dir on first launch.
trees = []
for tree_dir in ("core", "nlp", "ui", "database", "config", "knowledge"):
    if (REPO / tree_dir).is_dir():
        trees.append(Tree(str(REPO / tree_dir), prefix=tree_dir, excludes=["__pycache__", "*.pyc"]))

if (REPO / "bin").is_dir():
    trees.append(Tree(str(REPO / "bin"), prefix="bin"))
else:
    print("[spec] WARNING: bin/ is empty — run packaging/fetch_runtime.py first.")
if (REPO / "models").is_dir():
    trees.append(Tree(str(REPO / "models"), prefix="models"))
else:
    print("[spec] WARNING: models/ is empty — run packaging/fetch_runtime.py first.")


a = Analysis(
    ["packaging/run_dean.py"],
    pathex=[str(REPO)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["streamlit.web.cli", "streamlit.runtime.scriptrunner.magic_funcs"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pytest", "IPython"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Dean",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Windowed release: no terminal window on launch — double-click just opens
    # the browser. Set console=True if you need to see Streamlit logs while
    # debugging a build.
    console=False,
    disable_windowed_traceback=False,
    icon=ICON_ICO if sys.platform.startswith("win") else ICON_ICNS,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    *trees,
    strip=False,
    upx=False,
    name="Dean",
)

# macOS .app wrapper (double-clickable). Harmless on other platforms.
app = BUNDLE(
    coll,
    name="Dean.app",
    icon=ICON_ICNS,
    bundle_identifier="com.azera.dean",
    info_plist={
        "CFBundleName": "Dean",
        "CFBundleDisplayName": "Dean",
        "NSHighResolutionCapable": True,
        # Keeps the bundled localhost Ollama/Streamlit from being blocked by ATS.
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    },
)
