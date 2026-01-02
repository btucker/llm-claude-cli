"""Tests for conversation handling."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_claude_code import ClaudeCode, ClaudeCodeOptions


class TestConversationResume:
    """Tests for conversation resumption via session_id."""

    def test_resumes_with_session_id(self, mock_subprocess_run, mock_llm_response):
        """Test that execute uses --resume when session_id is available."""
        model = ClaudeCode(model_id="claude-code")

        # Create a mock previous response with session_id
        prev_response = MagicMock()
        prev_response._claude_session_id = "previous-session-123"
        prev_response.prompt = MagicMock()
        prev_response.prompt.prompt = "Previous question"
        prev_response.text = MagicMock(return_value="Previous answer")

        # Create conversation with previous response
        conversation = MagicMock()
        conversation.responses = [prev_response]

        prompt = MagicMock()
        prompt.prompt = "Follow-up question"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response, conversation=conversation))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--resume" in cmd
        assert "previous-session-123" in cmd

    def test_falls_back_to_context_without_session_id(self, mock_subprocess_run, mock_llm_response):
        """Test fallback to text context when no session_id available."""
        model = ClaudeCode(model_id="claude-code")

        # Create a mock previous response WITHOUT session_id
        prev_response = MagicMock(spec=[])  # No _claude_session_id
        prev_response.prompt = MagicMock()
        prev_response.prompt.prompt = "Previous question"
        prev_response.text = MagicMock(return_value="Previous answer")

        conversation = MagicMock()
        conversation.responses = [prev_response]

        prompt = MagicMock()
        prompt.prompt = "Follow-up question"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response, conversation=conversation))

        cmd = mock_subprocess_run.call_args[0][0]
        # Should NOT have --resume
        assert "--resume" not in cmd
        # Should have context embedded in prompt
        prompt_text = cmd[2]
        assert "Previous question" in prompt_text
        assert "Previous answer" in prompt_text
        assert "Follow-up question" in prompt_text

    def test_no_resume_for_first_message(self, mock_subprocess_run, mock_llm_response):
        """Test that first message in conversation doesn't use --resume."""
        model = ClaudeCode(model_id="claude-code")

        # Empty conversation (first message)
        conversation = MagicMock()
        conversation.responses = []

        prompt = MagicMock()
        prompt.prompt = "First question"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "session_id": "new-session", "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response, conversation=conversation))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--resume" not in cmd

    def test_manual_resume_option(self, mock_subprocess_run, mock_llm_response):
        """Test manual resume option takes precedence."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(resume="manual-session-id")

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        # Should have the manual resume, not auto-resume
        resume_indices = [i for i, x in enumerate(cmd) if x == "--resume"]
        assert len(resume_indices) >= 1
        # Check that manual-session-id is in the command
        assert "manual-session-id" in cmd


class TestConversationContext:
    """Tests for building conversation context."""

    def test_builds_context_from_multiple_turns(self, mock_subprocess_run, mock_llm_response):
        """Test context building with multiple conversation turns."""
        model = ClaudeCode(model_id="claude-code")

        # Create multiple previous responses WITHOUT session_id
        prev_response1 = MagicMock(spec=[])
        prev_response1.prompt = MagicMock()
        prev_response1.prompt.prompt = "Question 1"
        prev_response1.text = MagicMock(return_value="Answer 1")

        prev_response2 = MagicMock(spec=[])
        prev_response2.prompt = MagicMock()
        prev_response2.prompt.prompt = "Question 2"
        prev_response2.text = MagicMock(return_value="Answer 2")

        conversation = MagicMock()
        conversation.responses = [prev_response1, prev_response2]

        prompt = MagicMock()
        prompt.prompt = "Question 3"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response, conversation=conversation))

        cmd = mock_subprocess_run.call_args[0][0]
        prompt_text = cmd[2]

        # All turns should be in the context
        assert "Human: Question 1" in prompt_text
        assert "Assistant: Answer 1" in prompt_text
        assert "Human: Question 2" in prompt_text
        assert "Assistant: Answer 2" in prompt_text
        assert "Human: Question 3" in prompt_text

    def test_context_order_is_correct(self, mock_subprocess_run, mock_llm_response):
        """Test that conversation context maintains correct order."""
        model = ClaudeCode(model_id="claude-code")

        prev_response = MagicMock(spec=[])
        prev_response.prompt = MagicMock()
        prev_response.prompt.prompt = "First"
        prev_response.text = MagicMock(return_value="Response")

        conversation = MagicMock()
        conversation.responses = [prev_response]

        prompt = MagicMock()
        prompt.prompt = "Second"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response, conversation=conversation))

        cmd = mock_subprocess_run.call_args[0][0]
        prompt_text = cmd[2]

        # Verify order
        first_idx = prompt_text.index("Human: First")
        response_idx = prompt_text.index("Assistant: Response")
        second_idx = prompt_text.index("Human: Second")

        assert first_idx < response_idx < second_idx
