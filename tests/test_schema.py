"""Tests for schema/structured output functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_claude_code import ClaudeCode, ClaudeCodeOptions


class TestSchemaSupport:
    """Tests for JSON schema support."""

    def test_schema_added_to_command(self, mock_subprocess_run, mock_llm_response):
        """Test that schema is added to command as --json-schema."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Create a dog"
        prompt.system = None
        prompt.options = ClaudeCodeOptions()
        prompt.schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": {"content": [{"type": "text", "text": "{\\"name\\": \\"Buddy\\"}"}]}, "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--json-schema" in cmd

    def test_schema_uses_non_streaming(self, mock_subprocess_run, mock_llm_response):
        """Test that schema requests use non-streaming mode."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Create a dog"
        prompt.system = None
        prompt.options = ClaudeCodeOptions()
        prompt.schema = {"type": "object"}

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "usage": {}}',
            stderr="",
        )

        # Even with stream=True, schema should use non-streaming
        list(model.execute(prompt, stream=True, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--output-format" in cmd
        json_idx = cmd.index("--output-format")
        assert cmd[json_idx + 1] == "json"

    def test_schema_extracts_structured_output(self, mock_subprocess_run, mock_llm_response):
        """Test that schema execution extracts structured output."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Create a dog"
        prompt.system = None
        prompt.options = ClaudeCodeOptions()
        prompt.schema = {"type": "object"}

        # Response with structured_output field
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"structured_output": {"name": "Buddy", "age": 3}, "usage": {}}',
            stderr="",
        )

        result = list(model.execute(prompt, stream=False, response=mock_llm_response))

        # Should return JSON string
        assert len(result) == 1
        parsed = json.loads(result[0])
        assert parsed["name"] == "Buddy"
        assert parsed["age"] == 3

    def test_schema_extracts_from_result_content(self, mock_subprocess_run, mock_llm_response):
        """Test schema extraction from result.content array."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Create a dog"
        prompt.system = None
        prompt.options = ClaudeCodeOptions()
        prompt.schema = {"type": "object"}

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": {"content": [{"type": "text", "text": "{\\"name\\": \\"Max\\"}"}]}, "usage": {}}',
            stderr="",
        )

        result = list(model.execute(prompt, stream=False, response=mock_llm_response))

        assert len(result) == 1
        parsed = json.loads(result[0])
        assert parsed["name"] == "Max"

    def test_schema_captures_session_id(self, mock_subprocess_run, mock_llm_response):
        """Test that schema execution captures session_id."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Create a dog"
        prompt.system = None
        prompt.options = ClaudeCodeOptions()
        prompt.schema = {"type": "object"}

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "session_id": "schema-session", "usage": {}}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        assert mock_llm_response._claude_session_id == "schema-session"


class TestExtractStructuredOutput:
    """Tests for _extract_structured_output method."""

    def test_extracts_structured_output_field(self):
        """Test extraction from structured_output field."""
        model = ClaudeCode(model_id="test")
        data = {"structured_output": {"key": "value"}}

        result = model._extract_structured_output(data)

        assert result == {"key": "value"}

    def test_extracts_from_result_structured_output(self):
        """Test extraction from result.structured_output."""
        model = ClaudeCode(model_id="test")
        data = {"result": {"structured_output": {"nested": True}}}

        result = model._extract_structured_output(data)

        assert result == {"nested": True}

    def test_extracts_json_from_result_content(self):
        """Test extraction of JSON from result.content text."""
        model = ClaudeCode(model_id="test")
        data = {
            "result": {
                "content": [
                    {"type": "text", "text": '{"parsed": true}'}
                ]
            }
        }

        result = model._extract_structured_output(data)

        assert result == {"parsed": True}

    def test_extracts_json_from_messages(self):
        """Test extraction of JSON from messages array."""
        model = ClaudeCode(model_id="test")
        data = {
            "messages": [
                {
                    "content": [
                        {"type": "text", "text": '{"from_messages": true}'}
                    ]
                }
            ]
        }

        result = model._extract_structured_output(data)

        assert result == {"from_messages": True}

    def test_returns_none_for_invalid_json(self):
        """Test returns None when no valid JSON found."""
        model = ClaudeCode(model_id="test")
        data = {
            "result": {
                "content": [
                    {"type": "text", "text": "not json"}
                ]
            }
        }

        result = model._extract_structured_output(data)

        assert result is None

    def test_returns_none_for_empty_data(self):
        """Test returns None for empty data."""
        model = ClaudeCode(model_id="test")

        result = model._extract_structured_output({})

        assert result is None
