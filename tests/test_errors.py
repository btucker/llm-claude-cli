"""Tests for error handling."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

import llm
from llm_claude_code import ClaudeCode, ClaudeCodeOptions


class TestErrorHandling:
    """Tests for error handling in execute methods."""

    def test_cli_not_found_error(self, mock_llm_response):
        """Test error when Claude CLI is not found."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(llm.ModelError) as exc_info:
                list(model.execute(prompt, stream=False, response=mock_llm_response))

            assert "Claude Code CLI not found" in str(exc_info.value)

    def test_cli_not_found_streaming(self, mock_llm_response):
        """Test error when Claude CLI is not found in streaming mode."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        with patch("subprocess.Popen", side_effect=FileNotFoundError()):
            with pytest.raises(llm.ModelError) as exc_info:
                list(model.execute(prompt, stream=True, response=mock_llm_response))

            assert "Claude Code CLI not found" in str(exc_info.value)

    def test_timeout_error(self, mock_llm_response):
        """Test error on CLI timeout."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(timeout=1)

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            with pytest.raises(llm.ModelError) as exc_info:
                list(model.execute(prompt, stream=False, response=mock_llm_response))

            assert "timed out" in str(exc_info.value)

    def test_cli_error_non_zero_return(self, mock_subprocess_run, mock_llm_response):
        """Test error when CLI returns non-zero exit code."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Some error occurred",
        )

        with pytest.raises(llm.ModelError) as exc_info:
            list(model.execute(prompt, stream=False, response=mock_llm_response))

        assert "Claude Code CLI error" in str(exc_info.value)
        assert "Some error occurred" in str(exc_info.value)

    def test_empty_response_no_error(self, mock_subprocess_run, mock_llm_response):
        """Test that empty response doesn't raise error."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        # Should not raise, just return empty
        result = list(model.execute(prompt, stream=False, response=mock_llm_response))
        assert result == []


class TestStreamingErrorHandling:
    """Tests for error handling in streaming mode."""

    def test_streaming_timeout_kills_process(self, mock_llm_response):
        """Test that timeout in streaming mode kills the process."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(timeout=1)

        mock_process = MagicMock()
        # Return a line then empty string to end the readline loop
        mock_process.stdout.readline = MagicMock(side_effect=['{"type": "test"}\n', ''])
        mock_process.wait = MagicMock(side_effect=subprocess.TimeoutExpired("cmd", 1))
        mock_process.kill = MagicMock()

        with patch("subprocess.Popen", return_value=mock_process):
            with pytest.raises(llm.ModelError) as exc_info:
                list(model.execute(prompt, stream=True, response=mock_llm_response))

            assert "timed out" in str(exc_info.value)
            mock_process.kill.assert_called_once()


class TestExtractTextFromResponse:
    """Tests for _extract_text_from_response method."""

    def test_extracts_result_string(self):
        """Test extraction of result as string."""
        model = ClaudeCode(model_id="test")
        data = {"result": "Direct result text"}

        result = model._extract_text_from_response(data)

        assert result == "Direct result text"

    def test_extracts_content_array(self):
        """Test extraction from content array."""
        model = ClaudeCode(model_id="test")
        data = {
            "content": [
                {"type": "text", "text": "First "},
                {"type": "text", "text": "Second"},
            ]
        }

        result = model._extract_text_from_response(data)

        assert result == "First Second"

    def test_extracts_string_content(self):
        """Test extraction when content contains strings."""
        model = ClaudeCode(model_id="test")
        data = {"content": ["String content"]}

        result = model._extract_text_from_response(data)

        assert result == "String content"

    def test_extracts_message_content(self):
        """Test extraction from message.content."""
        model = ClaudeCode(model_id="test")
        data = {
            "message": {
                "content": [
                    {"type": "text", "text": "Message text"}
                ]
            }
        }

        result = model._extract_text_from_response(data)

        assert result == "Message text"

    def test_returns_empty_for_no_content(self):
        """Test returns empty string when no content found."""
        model = ClaudeCode(model_id="test")
        data = {"other": "data"}

        result = model._extract_text_from_response(data)

        assert result == ""

    def test_ignores_non_text_blocks(self):
        """Test ignores content blocks that aren't text type."""
        model = ClaudeCode(model_id="test")
        data = {
            "content": [
                {"type": "image", "data": "..."},
                {"type": "text", "text": "Only this"},
            ]
        }

        result = model._extract_text_from_response(data)

        assert result == "Only this"
