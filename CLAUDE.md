# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest tests/test_model.py

# Run a specific test
uv run pytest tests/test_model.py::TestClaudeCodeModel::test_model_init -v
```

## Architecture

This is an [LLM](https://llm.datasette.io/) plugin that wraps the Claude Code CLI (`claude`) as a subprocess, exposing Claude models through LLM's interface.

**llm_claude_cli.py** - Single-file plugin containing:
- `CLAUDE_MODELS` - Model definitions (claude-cli, claude-cli-opus, claude-cli-sonnet, claude-cli-haiku)
- `ClaudeCodeOptions` - Pydantic options class for CLI flags (timeout, allowedTools, add_dir, etc.)
- `ClaudeCode` - Main model class implementing `llm.Model`
- `register_models()` - LLM hook for registration

**Key execution paths:**
- Streaming: Uses `--output-format stream-json --verbose` with subprocess.Popen, parsing NDJSON events
- Schema/structured output: Uses `--output-format json` with subprocess.run, extracts from `structured_output` or content blocks
- Non-streaming: Internally uses streaming but collects all output before returning

**System prompt behavior:**
- Simple queries (no agentic options, no schema): Empty system prompt for direct responses
- Agentic queries (tools, permissions, schema): Uses Claude Code's default system prompt
- `use_default_system_prompt` option overrides auto-detection
- `append_system_prompt` appends to default instead of replacing

**Session management:**
- Conversations generate deterministic UUIDs from LLM conversation ID via `uuid.uuid5()`
- First message in a conversation uses `--session-id`, subsequent messages use `--resume`
- One-off prompts use `--no-session-persistence`

## Testing Conventions

Tests mock `subprocess.Popen` (streaming) and `subprocess.run` (non-streaming) via fixtures in `tests/conftest.py`. The `create_mock_popen()` helper in test_model.py creates properly configured mock processes with streaming events.
