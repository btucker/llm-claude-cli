"""
LLM plugin for Claude models via Claude Code CLI.

This plugin allows using Claude models through the Claude Code CLI tool,
leveraging your existing Claude Code subscription without requiring an API key.
"""

import json
import subprocess
from typing import Any, Iterator, Optional

import llm
from pydantic import Field


# Model definitions with their capabilities
CLAUDE_MODELS = [
    {
        "model_id": "claude-code",
        "aliases": ["cc"],
        "name": "Claude Code (default model)",
        "claude_model": None,  # Use CLI default
    },
    {
        "model_id": "claude-code-opus",
        "aliases": ["cc-opus", "opus"],
        "name": "Claude Opus 4",
        "claude_model": "opus",
    },
    {
        "model_id": "claude-code-sonnet",
        "aliases": ["cc-sonnet", "sonnet"],
        "name": "Claude Sonnet 4",
        "claude_model": "sonnet",
    },
    {
        "model_id": "claude-code-haiku",
        "aliases": ["cc-haiku", "haiku"],
        "name": "Claude Haiku 3.5",
        "claude_model": "haiku",
    },
]


class ClaudeCodeOptions(llm.Options):
    """Options for Claude Code CLI models."""

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt to append to Claude Code's default prompt",
    )
    replace_system_prompt: bool = Field(
        default=False,
        description="If True, fully replace the default system prompt instead of appending",
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
    max_turns: Optional[int] = Field(
        default=None,
        description="Maximum number of agentic turns",
    )


class ClaudeCode(llm.Model):
    """Model class for Claude via Claude Code CLI."""

    can_stream = True
    needs_key = None
    key_env_var = None
    supports_schema = True

    class Options(ClaudeCodeOptions):
        pass

    def __init__(self, model_id: str, claude_model: Optional[str] = None, aliases: tuple = ()):
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
        # Build the prompt text
        prompt_text = prompt.prompt

        # Build conversation context if available
        if conversation and conversation.responses:
            context_parts = []
            for prev_response in conversation.responses:
                context_parts.append(f"Human: {prev_response.prompt.prompt}")
                context_parts.append(f"Assistant: {prev_response.text()}")
            context_parts.append(f"Human: {prompt_text}")
            prompt_text = "\n\n".join(context_parts)

        # Build CLI command
        cmd = ["claude", "-p", prompt_text]

        # Add model if specified
        if self.claude_model:
            cmd.extend(["--model", self.claude_model])

        # Handle system prompt - prefer prompt.system, fallback to options
        system_text = None
        replace_system = False
        if prompt.system:
            system_text = prompt.system
        elif prompt.options and prompt.options.system_prompt:
            system_text = prompt.options.system_prompt
            replace_system = prompt.options.replace_system_prompt

        if system_text:
            if replace_system:
                # Fully replace Claude Code's default system prompt
                cmd.extend(["--system-prompt", system_text])
            else:
                # Append to Claude Code's default system prompt (preserves agentic behavior)
                cmd.extend(["--append-system-prompt", system_text])

        # Add options
        if prompt.options:
            if prompt.options.max_tokens:
                cmd.extend(["--max-tokens", str(prompt.options.max_tokens)])
            if prompt.options.allowedTools:
                cmd.extend(["--allowedTools", prompt.options.allowedTools])
            if prompt.options.disallowedTools:
                cmd.extend(["--disallowedTools", prompt.options.disallowedTools])
            if prompt.options.max_turns:
                cmd.extend(["--max-turns", str(prompt.options.max_turns)])

        # Add schema if provided
        schema = getattr(prompt, "schema", None)
        if schema:
            cmd.extend(["--json-schema", json.dumps(schema)])

        timeout = prompt.options.timeout if prompt.options else 300

        # When using schema, always use non-streaming for reliable structured output
        if schema:
            yield from self._execute_with_schema(cmd, timeout, response, schema)
        elif stream:
            yield from self._execute_streaming(cmd, timeout, response)
        else:
            yield from self._execute_non_streaming(cmd, timeout, response)

    def _execute_streaming(
        self, cmd: list, timeout: int, response: llm.Response
    ) -> Iterator[str]:
        """Execute with streaming JSON output."""
        cmd.extend(["--output-format", "stream-json"])

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
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
                                    yield text
                    elif event_type == "content_block_delta":
                        # Streaming text delta
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield text
                    elif event_type == "result":
                        # Final result with usage info
                        result_text = event.get("result", "")
                        if result_text and isinstance(result_text, str):
                            yield result_text
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
        self, cmd: list, timeout: int, response: llm.Response, schema: dict
    ) -> Iterator[str]:
        """Execute with JSON schema for structured output."""
        cmd.extend(["--output-format", "json"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
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
                            json.loads(structured_output)
                            yield structured_output
                        except json.JSONDecodeError:
                            # Not JSON, yield as-is
                            yield structured_output
                    else:
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
        self, cmd: list, timeout: int, response: llm.Response
    ) -> Iterator[str]:
        """Execute with JSON output (non-streaming)."""
        cmd.extend(["--output-format", "json"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise llm.ModelError(f"Claude Code CLI error: {stderr}")

            stdout = result.stdout.strip()
            if not stdout:
                return

            try:
                data = json.loads(stdout)

                # Extract text from response
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
