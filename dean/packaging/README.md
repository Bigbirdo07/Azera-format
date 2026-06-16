# Packaging Dean as a self-contained desktop app

The goal: a folder/installer a school can run with **no Python, no Ollama
install, and no internet** — the engine and the model ship inside.

## What gets bundled

| Piece | Source | Lands in bundle as | Size |
|---|---|---|---|
| Ollama engine | `bin/ollama-<platform>` | `_internal/bin/` | ~30–50 MB |
| Model weights | `models/` (OLLAMA_MODELS layout) | `_internal/models/` | ~2 GB |
| App source | `app.py`, `core/`, `nlp/`, `ui/`, `database/` | `_internal/…` (data) | small |
| Python + deps | venv (streamlit, pandas, openpyxl…) | frozen by PyInstaller | ~200 MB |

Binary **and** weights are both required. The binary alone starts the engine
but has no model to run (`local_model_manager.py` errors with *"model not found
in dean/models"*).

## Build

```bash
# Full build (copies local ollama binary, bakes ~2 GB model, then packages):
packaging/build.sh

# Reuse an already-baked models/ dir:
packaging/build.sh --skip-model
```

Output:
- `dist/Dean/` — ship this whole folder.
- `dist/Dean.app` — macOS double-clickable wrapper.

### Steps individually

```bash
# 1. Populate bin/ + models/ (uses the ollama on PATH for this platform)
.venv/bin/python packaging/fetch_runtime.py --model llama3.2:3b

# 2. Package
.venv/bin/pyinstaller dean.spec --noconfirm
```

## Where user data lives (not in the bundle)

The launcher (`packaging/run_dean.py`) sets `DEAN_DATA_DIR` to a per-user dir so
the database, exports, logs, and settings survive app updates:

- macOS: `~/Library/Application Support/Dean/`
- Windows: `%LOCALAPPDATA%\Dean\`
- Linux: `~/.local/share/Dean/`

`config/` and `knowledge/` defaults are seeded there on first launch. In a
source checkout `DEAN_DATA_DIR` is unset, so everything resolves under the repo
root exactly as before (dev/tests unchanged). Path routing lives in
`core/paths.py`.

## Before shipping to a school

- **macOS:** `codesign` + **notarize** `Dean.app`. Unsigned apps are blocked by
  Gatekeeper and look broken/untrustworthy.
- **Windows:** build on Windows; Authenticode-sign the `.exe` to avoid
  SmartScreen warnings. (Run `fetch_runtime.py` on Windows with an `ollama.exe`
  available so the correct binary is bundled.)
- **Llama license:** include the Llama 3.2 Community License + "Built with
  Llama" attribution in the installer (redistributing weights is permitted under
  it; Ollama itself is MIT).
- **Prove the offline claim:** launch the built app on a clean machine with
  Wi-Fi/Ethernet **off** and confirm chat + workbook flows work end to end.
- For a windowed release, set `console=False` in `dean.spec` (keep `True` while
  debugging so Streamlit logs are visible).
