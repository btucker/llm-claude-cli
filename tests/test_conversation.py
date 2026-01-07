"""Tests for conversation handling."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from llm_claude_cli import ClaudeCode, ClaudeCodeOptions


def create_mock_popen(events=None):
    """Create a properly configured mock Popen for streaming tests."""
    if events is None:
        events = [
            '{"type": "result", "result": "test", "usage": {"input_tokens": 10, "output_tokens": 5}}',
        ]

    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = events + [""]  # Empty string ends iteration
    mock_process.wait.return_value = 0
    return mock_process


class TestSessionPersistence:
    """Tests for session persistence based on conversation management."""

    def test_no_session_persistence_for_one_off_prompt(self, mock_llm_response):
        """Test that one-off prompts use --no-session-persistence."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "One-off question"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = create_mock_popen()

            # Execute without conversation (one-off prompt)
            list(model.execute(prompt, stream=True, response=mock_llm_response, conversation=None))

            cmd = mock_popen.call_args[0][0]
            assert "--no-session-persistence" in cmd
            assert "--session-id" not in cmd

    def test_session_id_for_conversation(self, mock_llm_response):
        """Test that conversations use --session-id with deterministic UUID."""
        model = ClaudeCode(model_id="claude-code")

        conversation = MagicMock()
        conversation.id = "test-conversation-123"
        conversation.responses = []

        prompt = MagicMock()
        prompt.prompt = "Question in conversation"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = create_mock_popen()

            list(model.execute(prompt, stream=True, response=mock_llm_response, conversation=conversation))

            cmd = mock_popen.call_args[0][0]
            assert "--no-session-persistence" not in cmd
            assert "--session-id" in cmd

            # Verify the session ID is deterministic
            session_id_idx = cmd.index("--session-id")
            session_id = cmd[session_id_idx + 1]
            expected_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"llm-claude-cli:{conversation.id}"))
            assert session_id == expected_uuid

    def test_same_conversation_id_produces_same_session_id(self, mock_llm_response):
        """Test that the same conversation ID always produces the same session UUID."""
        model = ClaudeCode(model_id="claude-code")

        with patch("subprocess.Popen") as mock_popen:
            # First call - uses --session-id
            mock_popen.return_value = create_mock_popen()

            conversation1 = MagicMock()
            conversation1.id = "my-conversation"
            conversation1.responses = []

            prompt1 = MagicMock()
            prompt1.prompt = "First message"
            prompt1.system = None
            prompt1.schema = None
            prompt1.options = ClaudeCodeOptions()

            list(model.execute(prompt1, stream=True, response=mock_llm_response, conversation=conversation1))
            cmd1 = mock_popen.call_args[0][0]
            assert "--session-id" in cmd1
            session_id_idx1 = cmd1.index("--session-id")
            session_id1 = cmd1[session_id_idx1 + 1]

            # Reset mock
            mock_popen.reset_mock()
            mock_popen.return_value = create_mock_popen()

            # Second call with same conversation ID - uses --resume
            conversation2 = MagicMock()
            conversation2.id = "my-conversation"  # Same ID
            conversation2.responses = [MagicMock()]  # Has previous responses

            prompt2 = MagicMock()
            prompt2.prompt = "Second message"
            prompt2.system = None
            prompt2.schema = None
            prompt2.options = ClaudeCodeOptions()

            list(model.execute(prompt2, stream=True, response=mock_llm_response, conversation=conversation2))
            cmd2 = mock_popen.call_args[0][0]
            # Second call uses --resume instead of --session-id
            assert "--resume" in cmd2
            resume_idx = cmd2.index("--resume")
            session_id2 = cmd2[resume_idx + 1]

            # Session IDs should be identical
            assert session_id1 == session_id2

    def test_different_conversation_ids_produce_different_session_ids(self, mock_llm_response):
        """Test that different conversation IDs produce different session UUIDs."""
        model = ClaudeCode(model_id="claude-code")

        with patch("subprocess.Popen") as mock_popen:
            # First conversation
            mock_popen.return_value = create_mock_popen()

            conversation1 = MagicMock()
            conversation1.id = "conversation-a"
            conversation1.responses = []

            prompt1 = MagicMock()
            prompt1.prompt = "Message A"
            prompt1.system = None
            prompt1.schema = None
            prompt1.options = ClaudeCodeOptions()

            list(model.execute(prompt1, stream=True, response=mock_llm_response, conversation=conversation1))
            cmd1 = mock_popen.call_args[0][0]
            session_id_idx1 = cmd1.index("--session-id")
            session_id1 = cmd1[session_id_idx1 + 1]

            # Reset mock
            mock_popen.reset_mock()
            mock_popen.return_value = create_mock_popen()

            # Second conversation with different ID
            conversation2 = MagicMock()
            conversation2.id = "conversation-b"  # Different ID
            conversation2.responses = []

            prompt2 = MagicMock()
            prompt2.prompt = "Message B"
            prompt2.system = None
            prompt2.schema = None
            prompt2.options = ClaudeCodeOptions()

            list(model.execute(prompt2, stream=True, response=mock_llm_response, conversation=conversation2))
            cmd2 = mock_popen.call_args[0][0]
            session_id_idx2 = cmd2.index("--session-id")
            session_id2 = cmd2[session_id_idx2 + 1]

            # Session IDs should be different
            assert session_id1 != session_id2


class TestSessionIdFormat:
    """Tests for session ID format validation."""

    def test_session_id_is_valid_uuid(self, mock_llm_response):
        """Test that generated session ID is a valid UUID."""
        model = ClaudeCode(model_id="claude-code")

        conversation = MagicMock()
        conversation.id = "any-conversation-id"
        conversation.responses = []

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions()

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = create_mock_popen()

            list(model.execute(prompt, stream=True, response=mock_llm_response, conversation=conversation))

            cmd = mock_popen.call_args[0][0]
            session_id_idx = cmd.index("--session-id")
            session_id = cmd[session_id_idx + 1]

            # Should be a valid UUID (won't raise)
            parsed = uuid.UUID(session_id)
            assert parsed.version == 5  # uuid5
