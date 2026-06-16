#!/bin/bash
# Build the self-contained Dean desktop app.
#
# Produces dist/Dean/ (and dist/Dean.app on macOS) with the Ollama engine and
# the model weights bundled in — the target machine downloads nothing and needs
# no separate Ollama install.
#
# Usage:
#   packaging/build.sh                 # full build (binary + ~2GB model + app)
#   packaging/build.sh --skip-model    # reuse an already-baked models/ dir
#
# Run from the repo root.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY=".venv/bin/python"
PIP=".venv/bin/pip"

if [ ! -x "$PY" ]; then
  echo "error: $PY not found. Create the venv first (.venv)." >&2
  exit 1
fi

echo "==> Ensuring build dependencies (pyinstaller)"
"$PIP" install --quiet "pyinstaller>=6.0"

FETCH_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --skip-model) FETCH_ARGS+=("--skip-model") ;;
    --skip-binary) FETCH_ARGS+=("--skip-binary") ;;
  esac
done

echo "==> Populating bin/ (Ollama engine) and models/ (weights)"
"$PY" packaging/fetch_runtime.py ${FETCH_ARGS[@]+"${FETCH_ARGS[@]}"}

echo "==> Cleaning previous build output"
rm -rf build dist

echo "==> Running PyInstaller"
.venv/bin/pyinstaller dean.spec --noconfirm

echo ""
echo "==> Build complete."
echo "    Folder app : $ROOT/dist/Dean/"
if [ -d "$ROOT/dist/Dean.app" ]; then
  echo "    macOS .app : $ROOT/dist/Dean.app"
fi
echo ""
echo "Next steps for distribution:"
echo "  - macOS : codesign + notarize Dean.app (unsigned apps are Gatekeeper-blocked)."
echo "  - Windows: build on Windows; sign the .exe to avoid SmartScreen warnings."
echo "  - Test on a clean machine with networking OFF to prove the offline story."
