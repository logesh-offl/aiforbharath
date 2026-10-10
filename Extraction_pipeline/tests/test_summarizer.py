"""Unit tests for summarizer.

The real edge0 engine (19GB checkpoint, SSD expert offload) is too
heavy to load in automated tests, so these tests mock the pieces
summarizer.summarize() depends on (ChatSession, the engine's tokenizer)
and assert the prompt construction / dispatch logic is correct. A
manual smoke test against the real model is documented in the README.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

import summarizer


def test_resolve_model_dir_prefers_explicit_argument(monkeypatch):
    monkeypatch.setenv(summarizer.DEFAULT_MODEL_ENV_VAR, "/env/path")
    assert summarizer.resolve_model_dir("/explicit/path") == "/explicit/path"


def test_resolve_model_dir_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv(summarizer.DEFAULT_MODEL_ENV_VAR, "/env/path")
    assert summarizer.resolve_model_dir(None) == "/env/path"


def test_resolve_model_dir_raises_when_nothing_available(monkeypatch):
    monkeypatch.delenv(summarizer.DEFAULT_MODEL_ENV_VAR, raising=False)
    with pytest.raises(ValueError):
        summarizer.resolve_model_dir(None)


def test_summarize_returns_empty_string_for_empty_text():
    engine = MagicMock()
    assert summarizer.summarize("   ", engine) == ""
    assert summarizer.summarize("", engine) == ""


def test_run_instruction_returns_empty_string_for_empty_text():
    engine = MagicMock()
    assert summarizer.run_instruction("   ", engine, "list action items") == ""
    assert summarizer.run_instruction("", engine, None) == ""


def test_summarize_builds_prompt_and_decodes_mocked_output():
    fake_engine = MagicMock()
    fake_engine.name = "edge0-35b"
    fake_engine._tok.decode.return_value = "This is the mocked summary."

    with patch("edge0.server.chat.ChatSession") as MockSession:
        instance = MockSession.return_value
        instance.run.return_value = ([1, 2, 3], {"usage": {}})

        result = summarizer.summarize(
            "Some long extracted OCR text.", fake_engine, max_tokens=64)

    assert result == "This is the mocked summary."

    # Verify the ChatSession was built with our engine and a ChatRequest
    # containing the system prompt + the extracted text as the user turn.
    args, kwargs = MockSession.call_args
    session_engine, session_req = args
    assert session_engine is fake_engine
    assert session_req.model == "edge0-35b"
    assert session_req.max_tokens == 64
    assert session_req.messages[0].role == "system"
    assert session_req.messages[0].content == summarizer.SUMMARY_SYSTEM_PROMPT
    assert session_req.messages[1].role == "user"
    assert "Some long extracted OCR text." in session_req.messages[1].content
    fake_engine._tok.decode.assert_called_once_with([1, 2, 3])


def test_run_instruction_with_custom_instruction_uses_instruction_prompt():
    fake_engine = MagicMock()
    fake_engine.name = "edge0-35b"
    fake_engine._tok.decode.return_value = "2026-01-01, $42.00"

    with patch("edge0.server.chat.ChatSession") as MockSession:
        instance = MockSession.return_value
        instance.run.return_value = ([4, 5, 6], {"usage": {}})

        result = summarizer.run_instruction(
            "Invoice dated 2026-01-01 for $42.00.",
            fake_engine,
            instruction="extract only dates and amounts",
            max_tokens=64,
        )

    assert result == "2026-01-01, $42.00"

    args, kwargs = MockSession.call_args
    _session_engine, session_req = args
    assert session_req.messages[0].role == "system"
    assert (session_req.messages[0].content
            == summarizer.CUSTOM_INSTRUCTION_SYSTEM_PROMPT)
    assert session_req.messages[1].role == "user"
    assert "extract only dates and amounts" in session_req.messages[1].content
    assert "Invoice dated 2026-01-01" in session_req.messages[1].content


def test_run_instruction_falls_back_to_summary_when_instruction_blank():
    fake_engine = MagicMock()
    fake_engine.name = "edge0-35b"
    fake_engine._tok.decode.return_value = "summary text"

    with patch("edge0.server.chat.ChatSession") as MockSession:
        instance = MockSession.return_value
        instance.run.return_value = ([1], {"usage": {}})

        summarizer.run_instruction("some text", fake_engine, instruction="   ")

    args, kwargs = MockSession.call_args
    _session_engine, session_req = args
    assert session_req.messages[0].content == summarizer.SUMMARY_SYSTEM_PROMPT
