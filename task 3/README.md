# Task 3 — Multi-Tool Autonomous AI Agent

## Problem statement

A language model alone cannot do reliable arithmetic, look up current facts, or read local files. An agent that can call tools can, but each tool must be safe against bad or malicious input.

## Solution overview

Run a console-based LangChain agent that selects from web search, a safe calculator, and a restricted local file reader. The agent can make several tool calls before it writes a final answer, and every tool call is shown in a trace.

## Architecture / workflow

1. `agent.py` builds three LangChain tools (`web_search`, `calculator`, `local_file_reader`) and a Gemini chat model, and creates the agent with `create_agent`.
2. For each question the agent decides which tools to call, in any order and as many times as needed, and bases its final answer on the tool observations.
3. Each tool call is recorded in a trace (tool name, input, output). The console prints the trace, with each observation cut to 500 characters, followed by the final answer.
4. Tool errors come back to the agent as text observations, so a bad input does not end the session.

## Safety and trace design

- The calculator evaluates a Python AST allow-list (numbers, `+ - * / % **`, parentheses), not `eval()`. Expressions are limited to 200 characters, and division by zero, overflow (for example `10**400`), complex results (for example `(-8)**0.5`) and non-finite results are rejected with a calculator error.
- The local reader resolves paths and requires them to stay below `sample_files/`, which blocks `../` escapes, absolute paths and symlinks pointing outside. It accepts only `.txt` and `.md` files up to 100,000 bytes.
- DuckDuckGo search returns up to 3 results; failures return a tool observation rather than terminating the process.
- The console trace reports only observable tool names, inputs, and truncated observations. It deliberately does not expose private chain-of-thought.

## Files

- `agent.py` — LangChain `create_agent` setup and console loop.
- `tools.py` — validated tool implementations.
- `sample_files/` — the only directory the reader is allowed to access.
- `tests/` — offline tests.

## Technologies

Python, LangChain (`langchain` 1.x `create_agent`, `langchain-google-genai`), Gemini, `ddgs` for DuckDuckGo search, `python-dotenv`, pytest.

## Installation

From this directory, inside an activated Python 3.13 environment:

```text
pip install -r requirements.txt
```

## Environment variables

Set these in the root `.env` (one level above this folder):

| Variable | Required | Default |
| --- | --- | --- |
| `GOOGLE_API_KEY` | yes | none |
| `GEMINI_MODEL` | no | `gemini-2.5-flash` |

Web search additionally needs internet access.

## Example usage

```text
python agent.py
```

Try: `Read project_notes.txt, calculate (18 * 7) / 3, and summarize the note.` The trace should show a file-reader action, a calculator action, their observations, and then the synthesized answer. (Which tools the model chooses has not been confirmed in a live Gemini run; the tools themselves are covered by the offline tests.)

## Testing

```text
pytest -q
```

The tests need no API key and no internet. They cover calculator results and rejected inputs, every file-access restriction above, the search tool with a fake search backend (results, no results, failure), the three LangChain tools and their traces, and the agent wiring (model, tools and final answer) with a fake agent, plus the empty-question and missing-key errors.

Not covered: real tool selection by Gemini, which needs a live key, and real DuckDuckGo results, which need internet access.
