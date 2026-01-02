"""LLM plugin for using Anthropic models through Claude Code CLI."""

import llm
import subprocess
import json
from typing import Iterator

__version__ = "0.1.0"


@llm.hookimpl
def register_models(register):
    register(ClaudeCode())


class ClaudeCode(llm.Model):
    model_id = "claude-code"
    can_stream = True

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: llm.Conversation | None = None,
    ) -> Iterator[str]:
        """Execute a prompt using the Claude Code CLI."""
        messages = []

        if conversation:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})

        messages.append({"role": "user", "content": prompt.prompt})

        cmd = ["claude", "--output-format", "stream-json", "-p", prompt.prompt]

        if not stream:
            cmd[2] = "json"

        try:
            if stream:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("type") == "assistant":
                                content = data.get("message", {}).get("content", [])
                                for block in content:
                                    if block.get("type") == "text":
                                        yield block.get("text", "")
                        except json.JSONDecodeError:
                            continue
                process.wait()
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                data = json.loads(result.stdout)
                if data.get("type") == "result":
                    yield data.get("result", "")
        except subprocess.CalledProcessError as e:
            raise llm.ModelError(f"Claude Code CLI error: {e.stderr}")
        except FileNotFoundError:
            raise llm.ModelError(
                "Claude Code CLI not found. Please install it: npm install -g @anthropic-ai/claude-code"
            )
