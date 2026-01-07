# llm-claude-cli

[![PyPI](https://img.shields.io/pypi/v/llm-claude-cli.svg)](https://pypi.org/project/llm-claude-cli/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/btucker/llm-claude-cli/blob/main/LICENSE)

LLM plugin for using Claude models through the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code).

This plugin allows you to use Claude models via the `claude` command-line tool, leveraging your existing Claude Code subscription without requiring a separate API key.

## Installation

First, ensure you have [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated:

```bash
npm install -g @anthropic-ai/claude-code
claude auth login
```

Then install this plugin:

```bash
llm install llm-claude-cli
```

Or install from source:

```bash
pip install -e .
```

## Usage

### Basic usage

```bash
# Use the default Claude CLI model
llm -m claude-cli "What is the capital of France?"

# Use specific model variants
llm -m claude-cli-opus "Explain quantum computing"
llm -m claude-cli-sonnet "Write a haiku about programming"
llm -m claude-cli-haiku "Quick summary of Python decorators"
```

### Available models

| Model ID | Aliases | Description |
|----------|---------|-------------|
| `claude-cli` | `ccli` | Default Claude CLI model |
| `claude-cli-opus` | `ccli-opus` | Claude Opus 4.5 |
| `claude-cli-sonnet` | `ccli-sonnet` | Claude Sonnet 4.5 |
| `claude-cli-haiku` | `ccli-haiku` | Claude Haiku 4.5 |

### Using aliases

```bash
llm -m ccli "Hello, Claude!"
llm -m ccli-opus "Deep analysis of this code"
llm -m ccli-sonnet "Explain this concept"
llm -m ccli-haiku "Quick answer"
```

### Options

```bash
# Set timeout (in seconds, default 300)
llm -m claude-cli "Complex task" -o timeout 600

# Limit allowed tools
llm -m claude-cli "Read file.txt" -o allowedTools "Read,Glob"

# Block specific tools
llm -m claude-cli "Help me code" -o disallowedTools "Bash(rm:*),Bash(sudo:*)"

# Add additional directory for Claude to access
llm -m claude-cli "Review the frontend" -o add_dir "../frontend"

# Set working directory
llm -m claude-cli "Run the tests" -o cwd "/path/to/project"

# Set permission mode (default, acceptEdits, bypassPermissions)
llm -m claude-cli "Refactor this" -o permission_mode "acceptEdits"

# Use custom MCP configuration
llm -m claude-cli "Use my tools" -o mcp_config "/path/to/mcp.json"

# Enable verbose logging
llm -m claude-cli "Debug this" -o verbose true
```

### System Prompts

The plugin auto-detects when to use Claude Code's default agentic system prompt:

- **Simple queries** (no agentic options, no schema): Uses an empty system prompt for direct responses
- **Agentic queries** (tools, permissions, schema, etc.): Uses Claude Code's default system prompt

You can customize this behavior:

```bash
# Simple query - uses empty system prompt by default
llm -m claude-cli "What is the capital of France?"

# Custom system prompt (replaces the default)
llm -m claude-cli "Hello" --system "You are a pirate"

# Append to Claude Code's default system prompt instead of replacing
# This preserves agentic capabilities while adding your instructions
llm -m claude-cli "Analyze this code" --system "Focus on security" -o append_system_prompt true

# Explicitly control whether to use Claude Code's default system prompt
llm -m claude-cli "Help me code" -o use_default_system_prompt true
```

### Conversations

Use LLM's conversation features for multi-turn interactions:

```bash
# Start a conversation (gets assigned an ID automatically)
llm -m claude-cli "My name is Alice"

# Continue the most recent conversation
llm -m claude-cli "What's my name?" -c

# View conversation history and IDs
llm logs list

# Continue a specific conversation by ID
llm -m claude-cli "Tell me more" --cid 01abc123...
```

The plugin generates a deterministic Claude session ID from the LLM conversation ID, so continuing a conversation automatically resumes the corresponding Claude session. For one-off prompts (without `-c` or `--cid`), the plugin passes `--no-session-persistence` to Claude to avoid creating orphan sessions.

### Streaming

Streaming is enabled by default. The plugin uses Claude Code's `stream-json` output format for real-time token streaming.

```bash
# Streaming output (default)
llm -m claude-cli "Tell me a story"

# Disable streaming
llm -m claude-cli "Quick answer" --no-stream
```

### Schemas (Structured Output)

This plugin supports LLM's schema feature for structured JSON output. Use the `--schema` flag to specify the expected output format:

```bash
# Simple schema with field types
llm -m claude-cli "Invent a cool dog" --schema 'name, age int, breed'

# Complex schema
llm -m claude-cli "List 3 countries" --schema '
  countries: [{
    name,
    capital,
    population int
  }]
'

# Using a JSON schema file
llm -m claude-cli "Extract info" --schema-file schema.json
```

The plugin uses Claude Code's `--json-schema` flag to ensure structured output conformance.

### Python API

```python
import llm

# Get a model
model = llm.get_model("claude-cli")

# Basic prompt
response = model.prompt("What is Python?")
print(response.text())

# With system prompt
response = model.prompt("Explain decorators", system="You are a Python expert")
print(response.text())

# Streaming
for chunk in model.prompt("Tell me a story", stream=True):
    print(chunk, end="", flush=True)

# Conversation
conversation = model.conversation()
response1 = conversation.prompt("My name is Alice")
response2 = conversation.prompt("What's my name?")
print(response2.text())  # Should mention Alice

# Schema for structured output
import json

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "breed": {"type": "string"}
    },
    "required": ["name", "age", "breed"]
}

response = model.prompt("Invent a cool dog", schema=schema)
dog = json.loads(response.text())
print(f"Name: {dog['name']}, Age: {dog['age']}, Breed: {dog['breed']}")
```

## How it works

This plugin invokes the Claude Code CLI (`claude`) as a subprocess with the `-p` flag for non-interactive prompts. It supports:

- **Streaming**: Uses `--output-format stream-json` for real-time NDJSON streaming
- **Non-streaming**: Uses `--output-format json` for complete responses
- **Schemas**: Uses `--json-schema` for structured JSON output conformance
- **System prompts**: Auto-detects when to use Claude Code's default; supports `--system-prompt` (replace) or `--append-system-prompt`
- **Conversations**: Integrates with LLM's `-c`/`--cid` flags; uses `--no-session-persistence` for one-off prompts
- **Model selection**: Passes `--model` flag to select Opus, Sonnet, or Haiku
- **Directory access**: Uses `--add-dir` for multi-directory projects
- **Permission control**: Uses `--permission-mode` and tool filtering
- **MCP integration**: Uses `--mcp-config` for custom tool servers

## Requirements

- Python 3.10+
- [LLM](https://llm.datasette.io/) 0.26+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

## License

MIT License - see [LICENSE](LICENSE) for details.
