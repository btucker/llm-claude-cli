"""Tests for the llm-claude-code plugin."""

import llm
import pytest
from unittest.mock import patch, MagicMock
import json

from llm_claude_code import ClaudeCode, __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"


def test_model_registration():
    """Test that the model can be registered."""
    model = ClaudeCode()
    assert model.model_id == "claude-code"
    assert model.can_stream is True


def test_model_in_registry():
    """Test that the model appears in the LLM registry."""
    models = [m.model_id for m in llm.get_models()]
    assert "claude-code" in models


def test_execute_non_streaming():
    """Test non-streaming execution."""
    model = ClaudeCode()
    prompt = llm.Prompt(prompt="Hello", model=model)
    response = MagicMock(spec=llm.Response)

    mock_result = MagicMock()
    mock_result.stdout = json.dumps({"type": "result", "result": "Hello there!"})
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        result = list(model.execute(prompt, stream=False, response=response))
        assert result == ["Hello there!"]
        mock_run.assert_called_once()


def test_execute_streaming():
    """Test streaming execution."""
    model = ClaudeCode()
    prompt = llm.Prompt(prompt="Hello", model=model)
    response = MagicMock(spec=llm.Response)

    mock_stdout = [
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi"}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": " there"}]}}),
    ]

    mock_process = MagicMock()
    mock_process.stdout = iter(mock_stdout)
    mock_process.wait.return_value = 0

    with patch("subprocess.Popen", return_value=mock_process) as mock_popen:
        result = list(model.execute(prompt, stream=True, response=response))
        assert result == ["Hi", " there"]
        mock_popen.assert_called_once()


def test_cli_not_found():
    """Test error handling when CLI is not found."""
    model = ClaudeCode()
    prompt = llm.Prompt(prompt="Hello", model=model)
    response = MagicMock(spec=llm.Response)

    with patch("subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(llm.ModelError, match="Claude Code CLI not found"):
            list(model.execute(prompt, stream=False, response=response))
