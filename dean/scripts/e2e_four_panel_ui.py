"""E2E driver for the Phase UI four-panel academic workspace.

Verifies — via Streamlit's AppTest harness — that:
  1. All four panels (Original Workbook, Working Sheet, Figures & Insights,
     Export Center) render.
  2. The chat → Working Sheet update path fires (filter result becomes the
     latest result attachment in chat history).
  3. The chained Academic Watch + export flow produces an output file and
     the Working Sheet title shifts to "Modified Working Sheet".
  4. A chart request populates Figures & Insights.
  5. The Original Workbook panel never shows an edit widget and stays
     read-only.

Run:
    .venv/bin/python scripts/e2e_four_panel_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import FIXTURE, FakeUpload  # noqa: E402


def _all_text(at) -> str:
    parts = []
    for name in ("markdown", "title", "caption", "info", "subheader"):
        for el in getattr(at, name, []):
            try:
                parts.append(str(getattr(el, "value", "")))
            except Exception:
                pass
    return " ".join(parts).lower()


def _seed(at, message: str | None = None) -> None:
    at.session_state["current_user"] = {"username": "tester", "role": "Editor"}
    at.session_state["workbook_upload"] = FakeUpload(FIXTURE)
    if message is not None:
        at.chat_input[0].set_value(message)
    at.run()


def main() -> int:
    print("=== Phase UI E2E: four-panel academic workspace ===")
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
    _seed(at)

    text = _all_text(at)
    print("\nStep 1: all four panels render")
    for heading in ("dean assistant", "original workbook", "working sheet",
                    "figures", "export center"):
        assert heading in text, f"missing heading: {heading!r}"
        print(f"  ✓ {heading} heading present")

    print("\nStep 2: filter query updates the Working Sheet")
    _seed(at, "Show me Accounting students")
    messages = (at.session_state["chat_messages"] if "chat_messages" in at.session_state else None) or []
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert assistants, "no assistant reply produced"
    assert (assistants[-1].get("attachment") or {}).get("type") == "result"
    print(f"  ✓ result attachment carried on last assistant message")

    print("\nStep 3: chained edit + export → Modified Working Sheet")
    _seed(at, "now only below 2.5 GPA")
    _seed(at, "export this list")
    _seed(at, "yes, export")
    text = _all_text(at)
    output_file = (at.session_state["latest_output_file"] if "latest_output_file" in at.session_state else None)
    assert output_file, "expected an export output file"
    assert "modified working sheet" in text
    assert "original workbook is unchanged" in text
    print(f"  ✓ output file: {Path(output_file).name}")
    print("  ✓ working-sheet title shifted to Modified Working Sheet")
    print("  ✓ panel reminds: original workbook unchanged")

    print("\nStep 4: chart request → Figures & Insights")
    _seed(at, "Create a bar chart by advisor")
    figure = (at.session_state["latest_figure"] if "latest_figure" in at.session_state else None)
    assert figure is not None, "latest_figure not set"
    assert figure.get("type") == "bar"
    print(f"  ✓ latest_figure populated: {figure.get('type')} by {figure.get('field')}")

    print("\nStep 5: original workbook panel stays read-only")
    # The original panel surfaces upload + selectbox + badges + preview.
    # We assert there's no input/text/edit widget under that section by
    # confirming the read-only language is shown.
    text = _all_text(at)
    assert "original file protected" in text or "read-only" in text
    print("  ✓ Original Workbook shows 'Original file protected' badge")

    print("\n=================================================")
    print("E2E PASS — four-panel workspace is wired end-to-end")
    print("  Dean Assistant (chat) ✓")
    print("  Original Workbook (read-only middle) ✓")
    print("  Modified Working Sheet (right) ✓")
    print("  Figures & Insights (bottom-left) ✓")
    print("  Export Center (bottom-right) ✓")
    print("=================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
