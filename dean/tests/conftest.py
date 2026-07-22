"""Shared pytest fixtures for the dean-office assistant test suite.

Deterministic synthetic workbook + pandas ground truth, plus a `chat` fixture
that drives the real Streamlit app headlessly via AppTest (no browser, no LLM).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from core.excel_loader import load_excel_workbook  # noqa: E402
from scripts.make_synthetic_workbook import build_dataframe, write_workbook  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_students.xlsx"


class FakeUpload:
    def __init__(self, path: Path):
        self._bytes = Path(path).read_bytes()
        self.name = Path(path).name

    def getvalue(self) -> bytes:
        return self._bytes


def _ensure_fixture() -> None:
    if not FIXTURE.exists():
        write_workbook(FIXTURE)


_ensure_fixture()
# The headless tests can't drive a real file uploader, so the widget always
# returns our synthetic workbook.
st.file_uploader = lambda *args, **kwargs: FakeUpload(FIXTURE)


@pytest.fixture(autouse=True)
def _isolate_outputs(tmp_path, monkeypatch):
    """Keep confirmed-action outputs and the audit log in a temp dir per test."""
    from core import confirmed_actions as ca
    from core import failure_log as fl
    from core import interaction_logger as il
    from core import session_workbook as sw

    monkeypatch.setattr(ca, "DEFAULT_OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(ca, "DEFAULT_AUDIT_PATH", tmp_path / "logs" / "audit_log.jsonl")
    # AppTest-driven UI tests call ensure_session_workbook(), which writes
    # session_<timestamp>.xlsx to DEFAULT_OUTPUT_DIR. Route those writes to
    # tmp_path too so the project's outputs/ directory stays clean.
    monkeypatch.setattr(sw, "DEFAULT_OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(sw, "DEFAULT_AUDIT_PATH", tmp_path / "logs" / "audit_log.jsonl")
    # Same pattern: AppTest runs append to interaction_learning.jsonl through
    # log_interaction(), which uses DEFAULT_LOG_PATH when no override is given.
    # Route those writes to tmp_path so the project's logs/ stays clean and
    # the real interaction log isn't polluted with synthetic test phrasings.
    monkeypatch.setattr(il, "DEFAULT_LOG_PATH", tmp_path / "logs" / "interaction_learning.jsonl")
    # Same again for failure_log.log_failure (the "Things I couldn't answer"
    # triage queue) -- previously unpatched, so every test run that exercises
    # a clarify/unsupported/failed path permanently wrote fixture strings
    # ("do something impossible", "something vague", ...) into the real
    # logs/failed_asks.jsonl, drowning out genuine user-reported gaps.
    monkeypatch.setattr(fl, "DEFAULT_PATH", tmp_path / "logs" / "failed_asks.jsonl")
    yield


@pytest.fixture(scope="session")
def gt() -> pd.DataFrame:
    """Ground-truth DataFrame (same seed as the fixture workbook)."""
    return build_dataframe()


@pytest.fixture()
def sheets() -> dict[str, pd.DataFrame]:
    """Workbook as loaded by the app (string-typed columns, like production)."""
    return load_excel_workbook(FakeUpload(FIXTURE)).sheets


@pytest.fixture()
def columns(sheets) -> list[str]:
    return list(sheets["Students"].columns)


class Chat:
    """Thin wrapper around AppTest that types messages into the assistant."""

    def __init__(self) -> None:
        self.at = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=120)
        self._seed()
        self.at.run()

    def _seed(self) -> None:
        self.at.session_state["current_user"] = {"username": "tester", "role": "Editor"}
        self.at.session_state["workbook_upload"] = FakeUpload(FIXTURE)
        # Exercise the diagnostics ("Advanced details") surface, which is hidden
        # from end users but asserted on by the layout tests.
        self.at.session_state["show_workspace_details"] = True

    def send(self, text: str) -> "Chat":
        self.at.chat_input[0].set_value(text)
        self._seed()
        self.at.run()
        return self

    def get(self, key, default=None):
        return self.at.session_state[key] if key in self.at.session_state else default

    def memory(self) -> dict:
        return self.get("assistant_memory") or {}

    @property
    def exception(self):
        return self.at.exception


@pytest.fixture()
def chat() -> Chat:
    return Chat()
