#!/bin/bash
# Double-click this file in Finder to launch the Offline University Spreadsheet
# Assistant in your browser. Close the Terminal window (or press Ctrl+C) to stop it.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/streamlit run app.py
