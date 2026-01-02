# llm-claude-code

[![PyPI](https://img.shields.io/pypi/v/llm-claude-code.svg)](https://pypi.org/project/llm-claude-code/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/btucker/llm-claude-code/blob/main/LICENSE)

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
llm install llm-claude-code
```

Or install from source:

```bash
pip install -e .
```

## Usage

### Basic usage

```bash
# Use the default Claude Code model
llm -m claude-code "What is the capital of France?"

# Use specific model variants
llm -m claude-code-opus "Explain quantum computing"
llm -m claude-code-sonnet "Write a haiku about programming"
llm -m claude-code-haiku "Quick summary of Python decorators"
```

### Available models

| Model ID | Aliases | Description |
|----------|---------|-------------|
| `claude-code` | `cc` | Default Claude Code model |
| `claude-code-opus` | `cc-opus`, `opus` | Claude Opus 4 |
| `claude-code-sonnet` | `cc-sonnet`, `sonnet` | Claude Sonnet 4 |
| `claude-code-haiku` | `cc-haiku`, `haiku` | Claude Haiku 3.5 |

### Using aliases

```bash
llm -m cc "Hello, Claude!"
llm -m opus "Deep analysis of this code"
llm -m sonnet "Explain this concept"
llm -m haiku "Quick answer"
```

### Options

```bash
# Set maximum tokens
llm -m claude-code "Summarize this" -o max_tokens 500

# Set timeout (in seconds, default 300)
llm -m claude-code "Complex task" -o timeout 600

# Set system prompt
llm -m claude-code "Analyze this code" -o system_prompt "You are a code reviewer"

# Limit allowed tools
llm -m claude-code "Read file.txt" -o allowedTools "Read,Glob"

# Set max turns for agentic operations
llm -m claude-code "Build a web app" -o max_turns 10
```

### Streaming

Streaming is enabled by default. The plugin uses Claude Code's `stream-json` output format for real-time token streaming.

```bash
# Streaming output (default)
llm -m claude-code "Tell me a story"

# Disable streaming
llm -m claude-code "Quick answer" --no-stream
```

### Schemas (Structured Output)

This plugin supports LLM's schema feature for structured JSON output. Use the `--schema` flag to specify the expected output format:

```bash
# Simple schema with field types
llm -m claude-code "Invent a cool dog" --schema 'name, age int, breed'

# Complex schema
llm -m claude-code "List 3 countries" --schema '
  countries: [{
    name,
    capital,
    population int
  }]
'

# Using a JSON schema file
llm -m claude-code "Extract info" --schema-file schema.json
```

The plugin uses Claude Code's `--json-schema` flag to ensure structured output conformance.

### Python API

```python
import llm

# Get a model
model = llm.get_model("claude-code")

# Basic prompt
response = model.prompt("What is Python?")
print(response.text())

# With options
response = model.prompt(
    "Explain decorators",
    max_tokens=500,
    system_prompt="You are a Python expert"
)
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
- **Conversations**: Maintains context by including previous turns in the prompt
- **Model selection**: Passes `--model` flag to select Opus, Sonnet, or Haiku

## Requirements

- Python 3.10+
- [LLM](https://llm.datasette.io/) 0.26+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

## License

MIT License - see [LICENSE](LICENSE) for details.
