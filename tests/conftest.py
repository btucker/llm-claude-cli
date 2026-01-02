"""Pytest fixtures for llm-claude-cli tests."""

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_subprocess_popen():
    """Mock subprocess.Popen for streaming tests."""
    with patch("subprocess.Popen") as mock_popen:
        yield mock_popen


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run for non-streaming tests."""
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def sample_streaming_events():
    """Sample streaming JSON events from Claude Code CLI."""
    return [
        {"type": "system", "session_id": "test-session-123"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}},
        {"type": "result", "result": "!", "session_id": "test-session-123", "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]


@pytest.fixture
def sample_json_response():
    """Sample JSON response from Claude Code CLI."""
    return {
        "result": "This is a test response.",
        "session_id": "test-session-456",
        "usage": {
            "input_tokens": 15,
            "output_tokens": 8,
        },
    }


@pytest.fixture
def sample_schema_response():
    """Sample JSON response with structured output."""
    return {
        "result": {
            "content": [
                {"type": "text", "text": '{"name": "Buddy", "age": 3, "breed": "Golden Retriever"}'}
            ]
        },
        "session_id": "test-session-789",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 12,
        },
    }


@pytest.fixture
def mock_llm_prompt():
    """Create a mock llm.Prompt object."""
    prompt = MagicMock()
    prompt.prompt = "Test prompt"
    prompt.system = None
    prompt.options = None
    # Explicitly set schema to None to avoid MagicMock being passed to json.dumps
    prompt.schema = None
    return prompt


@pytest.fixture
def mock_llm_response():
    """Create a mock llm.Response object."""
    response = MagicMock()
    response.set_usage = MagicMock()
    return response


@pytest.fixture
def mock_llm_conversation():
    """Create a mock llm.Conversation object."""
    conversation = MagicMock()
    conversation.responses = []
    return conversation
