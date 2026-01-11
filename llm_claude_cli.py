"""
LLM plugin for Claude models via Claude Code CLI.

This plugin allows using Claude models through the Claude Code CLI tool,
leveraging your existing Claude Code subscription without requiring an API key.
"""

import json
import subprocess
import uuid
from typing import Any, Iterator, Optional

import llm
from pydantic import Field


# Model definitions with their capabilities
CLAUDE_MODELS = [
    {
        "model_id": "claude-cli",
        "aliases": ["ccli"],
        "name": "Claude CLI (default model)",
        "claude_model": None,  # Use CLI default
    },
    {
        "model_id": "claude-cli-opus",
        "aliases": ["ccli-opus"],
        "name": "Claude Opus 4.5",
        "claude_model": "opus",
    },
    {
        "model_id": "claude-cli-sonnet",
        "aliases": ["ccli-sonnet"],
        "name": "Claude Sonnet 4.5",
        "claude_model": "sonnet",
    },
    {
        "model_id": "claude-cli-haiku",
        "aliases": ["ccli-haiku"],
        "name": "Claude Haiku 4.5",
        "claude_model": "haiku",
    },
]


class ClaudeCodeOptions(llm.Options):
    """Options for Claude Code CLI models."""

    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt (replaces Claude Code's default prompt)",
    )
    use_default_system_prompt: Optional[bool] = Field(
        default=None,
        description="Use Claude Code's default agentic system prompt (auto-detected if not set)",
    )
    append_system_prompt: bool = Field(
        default=False,
        description="If True, append system_prompt to Claude Code's default instead of replacing",
    )
    timeout: int = Field(
        default=300,
        description="Timeout in seconds for CLI execution",
    )
    allowedTools: Optional[str] = Field(
        default=None,
        description="Comma-separated list of allowed tools (e.g., 'Read,Write,Bash')",
    )
    disallowedTools: Optional[str] = Field(
        default=None,
        description="Comma-separated list of disallowed tools",
    )
    add_dir: Optional[str] = Field(
        default=None,
        description="Additional directory for Claude to access (can be specified multiple times via CLI)",
    )
    permission_mode: Optional[str] = Field(
        default=None,
        description="Permission mode: 'default', 'plan', 'acceptEdits', 'bypassPermissions', 'delegate', 'dontAsk'",
    )
    cwd: Optional[str] = Field(
        default=None,
        description="Working directory for Claude Code execution",
    )
    mcp_config: Optional[str] = Field(
        default=None,
        description="Path to MCP configuration file",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging output",
    )


class ClaudeCode(llm.Model):
    """Model class for Claude via Claude Code CLI."""

    can_stream = True
    needs_key = None
    key_env_var = None
    supports_schema = True

    class Options(ClaudeCodeOptions):
        pass

    def __init__(
        self, model_id: str, claude_model: Optional[str] = None, aliases: tuple = ()
    ):
        self.model_id = model_id
        self.claude_model = claude_model
        self._aliases = aliases

    @property
    def aliases(self):
        return self._aliases

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation] = None,
    ) -> Iterator[str]:
        """Execute the prompt using Claude Code CLI."""
        prompt_text = prompt.prompt

        # Build CLI command
        cmd = ["claude", "-p", prompt_text, "--tools", "default"]

        # Add model if specified
        if self.claude_model:
            cmd.extend(["--model", self.claude_model])

        # Handle session persistence based on LLM's conversation management
        if conversation is None:
            # One-off prompt, don't persist session
            cmd.append("--no-session-persistence")
        else:
            # In a conversation - generate deterministic UUID from conversation ID
            session_uuid = uuid.uuid5(
                uuid.NAMESPACE_DNS, f"llm-claude-cli:{conversation.id}"
            )
            # First message uses --session-id, subsequent messages use --resume
            if len(conversation.responses) == 0:
                cmd.extend(["--session-id", str(session_uuid)])
            else:
                cmd.extend(["--resume", str(session_uuid)])

        # Handle system prompt
        # Use empty system prompt only for simple queries
        # Use default system prompt when agentic features are enabled
        schema = getattr(prompt, "schema", None)
        has_agentic_options = prompt.options and any(
            [
                prompt.options.permission_mode,
                prompt.options.add_dir,
                prompt.options.allowedTools,
                prompt.options.disallowedTools,
                prompt.options.mcp_config,
            ]
        )

        # Determine if we should use Claude Code's default system prompt
        if prompt.options and prompt.options.use_default_system_prompt is not None:
            # Explicit setting takes precedence
            use_default = prompt.options.use_default_system_prompt
        else:
            # Auto-detect: use default for agentic features or schema
            use_default = has_agentic_options or schema is not None

        append = prompt.options.append_system_prompt if prompt.options else False

        if not use_default:
            system_text = prompt.system or (
                prompt.options.system_prompt if prompt.options else None
            )
            if system_text:
                if append:
                    cmd.extend(["--append-system-prompt", system_text])
                else:
                    cmd.extend(["--system-prompt", system_text])
            else:
                # Simple query: empty system prompt for direct responses
                cmd.extend(["--system-prompt", ""])

        # Add options
        # Default allowedTools includes WebSearch; user can override
        allowed_tools = "WebSearch"
        if prompt.options:
            if prompt.options.allowedTools:
                allowed_tools = prompt.options.allowedTools
            if prompt.options.disallowedTools:
                cmd.extend(["--disallowedTools", prompt.options.disallowedTools])
            if prompt.options.add_dir:
                cmd.extend(["--add-dir", prompt.options.add_dir])
            if prompt.options.permission_mode:
                cmd.extend(["--permission-mode", prompt.options.permission_mode])
            if prompt.options.mcp_config:
                cmd.extend(["--mcp-config", prompt.options.mcp_config])
            if prompt.options.verbose:
                cmd.append("--verbose")

        cmd.extend(["--allowedTools", allowed_tools])

        # Add schema if provided
        schema = getattr(prompt, "schema", None)
        if schema:
            cmd.extend(["--json-schema", json.dumps(schema)])

        timeout = prompt.options.timeout if prompt.options else 300
        cwd = prompt.options.cwd if prompt.options and prompt.options.cwd else None

        # When using schema, always use non-streaming for reliable structured output
        if schema:
            yield from self._execute_with_schema(cmd, timeout, response, schema, cwd)
        elif stream:
            yield from self._execute_streaming(cmd, timeout, response, cwd)
        else:
            yield from self._execute_non_streaming(cmd, timeout, response, cwd)

    def _execute_streaming(
        self, cmd: list, timeout: int, response: llm.Response, cwd: Optional[str] = None
    ) -> Iterator[str]:
        """Execute with streaming JSON output."""
        # --verbose is required for stream-json with -p mode
        cmd.extend(["--output-format", "stream-json", "--verbose"])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=cwd,
            )

            input_tokens = 0
            output_tokens = 0

            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")

                    # Handle different event types
                    if event_type == "assistant":
                        # Assistant message with content
                        message = event.get("message", {})
                        for content_block in message.get("content", []):
                            if content_block.get("type") == "text":
                                text = content_block.get("text", "")
                                if text:
                                    yield text + "\n\n"
                    elif event_type == "content_block_delta":
                        # Streaming text delta
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text
                    elif event_type == "result":
                        # Extract usage info (don't yield result text -
                        # it duplicates content already yielded from other events)
                        usage = event.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                    elif event_type == "message":
                        # Message event with potential text content
                        content = event.get("content", [])
                        for block in content:
                            if block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    yield text

                except json.JSONDecodeError:
                    # Not JSON, might be raw text output
                    continue

            process.wait(timeout=timeout)

            # Set token usage if available
            if input_tokens or output_tokens:
                response.set_usage(input=input_tokens, output=output_tokens)

        except subprocess.TimeoutExpired:
            process.kill()
            raise llm.ModelError("Claude Code CLI timed out")
        except FileNotFoundError:
            raise llm.ModelError(
                "Claude Code CLI not found. Please install it: https://docs.anthropic.com/en/docs/claude-code"
            )

    def _execute_with_schema(
        self,
        cmd: list,
        timeout: int,
        response: llm.Response,
        schema: dict,
        cwd: Optional[str] = None,
    ) -> Iterator[str]:
        """Execute with JSON schema for structured output."""
        cmd.extend(["--output-format", "json"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise llm.ModelError(f"Claude Code CLI error: {stderr}")

            stdout = result.stdout.strip()
            if not stdout:
                return

            try:
                data = json.loads(stdout)

                # Extract structured output from response
                structured_output = self._extract_structured_output(data)
                if structured_output:
                    # Return the structured output as a JSON string
                    if isinstance(structured_output, str):
                        # Check if it's already valid JSON
                        try:
                            parsed = json.loads(structured_output)
                            response.response_json = parsed
                            yield structured_output
                        except json.JSONDecodeError:
                            # Not JSON, yield as-is
                            yield structured_output
                    else:
                        response.response_json = structured_output
                        yield json.dumps(structured_output)
                else:
                    # Fallback to regular text extraction
                    text = self._extract_text_from_response(data)
                    if text:
                        yield text

                # Extract usage information
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                if input_tokens or output_tokens:
                    response.set_usage(input=input_tokens, output=output_tokens)

            except json.JSONDecodeError:
                # Fallback: treat as plain text
                yield stdout

        except subprocess.TimeoutExpired:
            raise llm.ModelError("Claude Code CLI timed out")
        except FileNotFoundError:
            raise llm.ModelError(
                "Claude Code CLI not found. Please install it: https://docs.anthropic.com/en/docs/claude-code"
            )

    def _extract_structured_output(self, data: dict) -> Optional[Any]:
        """Extract structured output from Claude Code JSON response."""
        # Check for structured_output field (Claude Code specific)
        if "structured_output" in data:
            return data["structured_output"]

        # Check in result field
        if "result" in data:
            result = data["result"]
            if isinstance(result, dict):
                if "structured_output" in result:
                    return result["structured_output"]
                # Check in content array within result
                content = result.get("content", [])
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            # Try to parse as JSON
                            try:
                                return json.loads(text)
                            except json.JSONDecodeError:
                                continue

        # Check in messages array
        messages = data.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            continue

        return None

    def _execute_non_streaming(
        self, cmd: list, timeout: int, response: llm.Response, cwd: Optional[str] = None
    ) -> Iterator[str]:
        """Execute without streaming - collects all output then returns at once."""
        # Use stream-json internally (works better in nested claude sessions)
        # but collect all output and return at once
        collected_text = []
        for chunk in self._execute_streaming(cmd, timeout, response, cwd):
            collected_text.append(chunk)
        if collected_text:
            yield "".join(collected_text)

    def _extract_text_from_response(self, data: dict) -> str:
        """Extract text content from Claude Code JSON response."""
        # Try 'result' field first (common in stream-json final output)
        if "result" in data:
            result = data["result"]
            if isinstance(result, str):
                return result

        # Try 'content' array
        content = data.get("content", [])
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)

        if texts:
            return "".join(texts)

        # Try message.content
        message = data.get("message", {})
        if message:
            content = message.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))

        return "".join(texts)


@llm.hookimpl
def register_models(register):
    """Register Claude Code models with LLM."""
    for model_info in CLAUDE_MODELS:
        model = ClaudeCode(
            model_id=model_info["model_id"],
            claude_model=model_info.get("claude_model"),
            aliases=tuple(model_info.get("aliases", [])),
        )
        register(model, aliases=model.aliases)
