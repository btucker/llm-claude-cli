"""Tests for streaming functionality."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from llm_claude_cli import ClaudeCode, ClaudeCodeOptions


class MockProcess:
    """Mock subprocess.Popen for streaming tests."""

    def __init__(self, lines, returncode=0):
        self.lines = iter(lines)
        self.returncode = returncode
        self._stdout = self

    @property
    def stdout(self):
        return self

    @property
    def stderr(self):
        return MagicMock()

    def readline(self):
        try:
            return next(self.lines)
        except StopIteration:
            return ""

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class TestStreaming:
    """Tests for streaming response handling."""

    def test_streaming_command_includes_verbose(self, mock_llm_response):
        """Test streaming mode includes --verbose flag required by CLI for stream-json with -p."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "result", "result": "done", "usage": {}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)) as mock_popen:
            list(model.execute(prompt, stream=True, response=mock_llm_response))

        # Verify --verbose is included in the command
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert "--verbose" in cmd, f"--verbose not found in command: {cmd}"
        assert "--output-format" in cmd
        stream_json_index = cmd.index("--output-format") + 1
        assert cmd[stream_json_index] == "stream-json"

    def test_streaming_text_delta(self, mock_llm_response):
        """Test streaming handles text_delta events."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n',
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}\n',
            '{"type": "result", "result": "", "session_id": "123", "usage": {"input_tokens": 10, "output_tokens": 5}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=True, response=mock_llm_response))

        assert "Hello" in result
        assert " world" in result

    def test_streaming_sets_usage(self, mock_llm_response):
        """Test streaming sets token usage from result event."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "result", "result": "done", "usage": {"input_tokens": 100, "output_tokens": 50}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            list(model.execute(prompt, stream=True, response=mock_llm_response))

        mock_llm_response.set_usage.assert_called_once_with(input=100, output=50)

    def test_streaming_handles_assistant_message(self, mock_llm_response):
        """Test streaming handles assistant message events."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Response text"}]}}\n',
            '{"type": "result", "result": "", "usage": {}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=True, response=mock_llm_response))

        assert "Response text\n\n" in result

    def test_streaming_handles_message_event(self, mock_llm_response):
        """Test streaming handles message events with content."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "message", "content": [{"type": "text", "text": "Message content"}]}\n',
            '{"type": "result", "result": "", "usage": {}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=True, response=mock_llm_response))

        assert "Message content" in result

    def test_streaming_ignores_invalid_json(self, mock_llm_response):
        """Test streaming ignores lines that aren't valid JSON."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            "not valid json\n",
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "valid"}}\n',
            "another invalid line\n",
            '{"type": "result", "result": "", "usage": {}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=True, response=mock_llm_response))

        assert "valid" in result

    def test_streaming_handles_empty_lines(self, mock_llm_response):
        """Test streaming handles empty lines gracefully."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            "\n",
            "   \n",
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "text"}}\n',
            "\n",
            '{"type": "result", "result": "", "usage": {}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=True, response=mock_llm_response))

        assert "text" in result

    def test_streaming_ignores_result_text(self, mock_llm_response):
        """Test streaming ignores result text (it duplicates content from other events)."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}}\n',
            '{"type": "result", "result": "Hello", "usage": {}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=True, response=mock_llm_response))

        assert result == ["Hello\n\n"]


class TestNonStreaming:
    """Tests for non-streaming response handling (which uses streaming internally)."""

    def test_non_streaming_collects_all_output(self, mock_llm_response):
        """Test non-streaming collects all output and returns at once."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}\n',
            '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}\n',
            '{"type": "result", "result": "", "usage": {"input_tokens": 10, "output_tokens": 5}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=False, response=mock_llm_response))

        # Non-streaming should return all content in a single chunk
        assert len(result) == 1
        assert "Hello" in result[0]
        assert " world" in result[0]

    def test_non_streaming_sets_usage(self, mock_llm_response):
        """Test non-streaming sets token usage."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "result", "result": "done", "usage": {"input_tokens": 25, "output_tokens": 15}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            list(model.execute(prompt, stream=False, response=mock_llm_response))

        mock_llm_response.set_usage.assert_called_once_with(input=25, output=15)

    def test_non_streaming_handles_empty_response(self, mock_llm_response):
        """Test non-streaming handles empty response gracefully."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        lines = [
            '{"type": "result", "result": "", "usage": {}}\n',
        ]

        with patch("subprocess.Popen", return_value=MockProcess(lines)):
            result = list(model.execute(prompt, stream=False, response=mock_llm_response))

        # Should return empty list when no content
        assert result == []
