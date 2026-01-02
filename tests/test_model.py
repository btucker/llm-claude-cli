"""Tests for ClaudeCode model class."""

import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from llm_claude_code import ClaudeCode, ClaudeCodeOptions, CLAUDE_MODELS, register_models


class TestClaudeCodeModel:
    """Tests for ClaudeCode model initialization and properties."""

    def test_model_init(self):
        """Test model initialization with basic parameters."""
        model = ClaudeCode(model_id="claude-code", claude_model=None)
        assert model.model_id == "claude-code"
        assert model.claude_model is None
        assert model.aliases == ()

    def test_model_init_with_claude_model(self):
        """Test model initialization with specific claude model."""
        model = ClaudeCode(model_id="claude-code-opus", claude_model="opus", aliases=("cc-opus",))
        assert model.model_id == "claude-code-opus"
        assert model.claude_model == "opus"
        assert model.aliases == ("cc-opus",)

    def test_model_properties(self):
        """Test model class properties."""
        model = ClaudeCode(model_id="test")
        assert model.can_stream is True
        assert model.needs_key is None
        assert model.key_env_var is None
        assert model.supports_schema is True

    def test_model_aliases_property(self):
        """Test aliases property returns correct tuple."""
        model = ClaudeCode(model_id="test", aliases=("alias1", "alias2"))
        assert model.aliases == ("alias1", "alias2")


class TestClaudeCodeOptions:
    """Tests for ClaudeCodeOptions class."""

    def test_default_options(self):
        """Test default option values."""
        options = ClaudeCodeOptions()
        assert options.max_tokens is None
        assert options.system_prompt is None
        assert options.use_default_system_prompt is None
        assert options.timeout == 300
        assert options.allowedTools is None
        assert options.disallowedTools is None
        assert options.max_turns is None
        assert options.add_dir is None
        assert options.permission_mode is None
        assert options.resume is None
        assert options.cwd is None
        assert options.mcp_config is None
        assert options.verbose is False

    def test_custom_options(self):
        """Test setting custom option values."""
        options = ClaudeCodeOptions(
            max_tokens=1000,
            system_prompt="Be helpful",
            timeout=600,
            allowedTools="Read,Write",
            max_turns=5,
            verbose=True,
        )
        assert options.max_tokens == 1000
        assert options.system_prompt == "Be helpful"
        assert options.timeout == 600
        assert options.allowedTools == "Read,Write"
        assert options.max_turns == 5
        assert options.verbose is True


class TestRegisterModels:
    """Tests for model registration."""

    def test_register_models_called(self):
        """Test that register_models registers all defined models."""
        register = MagicMock()
        register_models(register)

        assert register.call_count == len(CLAUDE_MODELS)

    def test_register_models_with_aliases(self):
        """Test that models are registered with correct aliases."""
        registered_models = []

        def capture_register(model, aliases=None):
            registered_models.append((model, aliases))

        register_models(capture_register)

        # Check that aliases are passed correctly
        for model, aliases in registered_models:
            assert isinstance(model, ClaudeCode)
            if aliases:
                assert isinstance(aliases, tuple)

    def test_claude_models_definition(self):
        """Test CLAUDE_MODELS contains expected models."""
        model_ids = [m["model_id"] for m in CLAUDE_MODELS]
        assert "claude-code" in model_ids
        assert "claude-code-opus" in model_ids
        assert "claude-code-sonnet" in model_ids
        assert "claude-code-haiku" in model_ids


class TestExecuteBasics:
    """Tests for basic execute functionality."""

    def test_execute_builds_basic_command(self, mock_subprocess_run, mock_llm_prompt, mock_llm_response):
        """Test that execute builds correct basic command."""
        model = ClaudeCode(model_id="claude-code")

        # Setup mock
        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test", "session_id": "123"}',
            stderr="",
        )

        # Execute
        result = list(model.execute(mock_llm_prompt, stream=False, response=mock_llm_response))

        # Verify command
        call_args = mock_subprocess_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "Test prompt" in cmd

    def test_execute_with_model_flag(self, mock_subprocess_run, mock_llm_prompt, mock_llm_response):
        """Test that execute adds --model flag for specific models."""
        model = ClaudeCode(model_id="claude-code-opus", claude_model="opus")

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(mock_llm_prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--model" in cmd
        assert "opus" in cmd

    def test_execute_uses_empty_system_prompt_by_default(self, mock_subprocess_run, mock_llm_prompt, mock_llm_response):
        """Test that execute uses empty system prompt by default for simple responses."""
        model = ClaudeCode(model_id="claude-code")

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(mock_llm_prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == ""

    def test_execute_with_custom_system_prompt(self, mock_subprocess_run, mock_llm_prompt, mock_llm_response):
        """Test that execute uses provided system prompt."""
        model = ClaudeCode(model_id="claude-code")
        mock_llm_prompt.system = "You are helpful"

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(mock_llm_prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--system-prompt" in cmd
        assert "You are helpful" in cmd

    def test_execute_with_default_system_prompt(self, mock_subprocess_run, mock_llm_response):
        """Test that use_default_system_prompt=True skips system prompt flag."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        options = ClaudeCodeOptions(use_default_system_prompt=True)
        prompt.options = options

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--system-prompt" not in cmd

    def test_execute_with_append_system_prompt(self, mock_subprocess_run, mock_llm_response):
        """Test that append_system_prompt=True uses --append-system-prompt."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        options = ClaudeCodeOptions(system_prompt="Extra instructions", append_system_prompt=True)
        prompt.options = options

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--append-system-prompt" in cmd
        assert "Extra instructions" in cmd
        assert "--system-prompt" not in cmd

    def test_execute_auto_enables_default_prompt_for_schema(self, mock_subprocess_run, mock_llm_response):
        """Test that schema triggers use of default system prompt."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = {"type": "object"}  # Has schema
        prompt.options = ClaudeCodeOptions()

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "{}"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        # Should NOT have --system-prompt when schema is present (uses default)
        assert "--system-prompt" not in cmd or cmd[cmd.index("--system-prompt") + 1] != ""

    def test_execute_auto_enables_default_prompt_for_permission_mode(self, mock_subprocess_run, mock_llm_response):
        """Test that permission_mode triggers use of default system prompt."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(permission_mode="plan")

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        # Should NOT have empty --system-prompt when permission_mode is set
        assert "--system-prompt" not in cmd

    def test_execute_auto_enables_default_prompt_for_add_dir(self, mock_subprocess_run, mock_llm_response):
        """Test that add_dir triggers use of default system prompt."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(add_dir=["./other"])

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--system-prompt" not in cmd

    def test_execute_explicit_false_overrides_auto_detection(self, mock_subprocess_run, mock_llm_response):
        """Test that explicit use_default_system_prompt=False overrides auto-detection."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        # Has agentic option but explicitly disables default prompt
        prompt.options = ClaudeCodeOptions(permission_mode="plan", use_default_system_prompt=False)

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        # Should have empty --system-prompt despite permission_mode
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == ""


class TestExecuteOptions:
    """Tests for execute with various options."""

    def test_execute_with_max_tokens(self, mock_subprocess_run, mock_llm_response):
        """Test max_tokens option is passed correctly."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(max_tokens=500)

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--max-tokens" in cmd
        assert "500" in cmd

    def test_execute_with_allowed_tools(self, mock_subprocess_run, mock_llm_response):
        """Test allowedTools option is passed correctly."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(allowedTools="Read,Write,Glob")

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--allowedTools" in cmd
        assert "Read,Write,Glob" in cmd

    def test_execute_with_add_dir(self, mock_subprocess_run, mock_llm_response):
        """Test add_dir option adds multiple --add-dir flags."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(add_dir=["../frontend", "../backend"])

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert cmd.count("--add-dir") == 2
        assert "../frontend" in cmd
        assert "../backend" in cmd

    def test_execute_with_verbose(self, mock_subprocess_run, mock_llm_response):
        """Test verbose option is passed correctly."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(verbose=True)

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        cmd = mock_subprocess_run.call_args[0][0]
        assert "--verbose" in cmd

    def test_execute_with_cwd(self, mock_subprocess_run, mock_llm_response):
        """Test cwd option sets working directory."""
        model = ClaudeCode(model_id="claude-code")

        prompt = MagicMock()
        prompt.prompt = "Test"
        prompt.system = None
        prompt.schema = None
        prompt.options = ClaudeCodeOptions(cwd="/custom/path")

        mock_subprocess_run.return_value = MagicMock(
            returncode=0,
            stdout='{"result": "test"}',
            stderr="",
        )

        list(model.execute(prompt, stream=False, response=mock_llm_response))

        call_kwargs = mock_subprocess_run.call_args[1]
        assert call_kwargs.get("cwd") == "/custom/path"
