# Task 3 — Multi-Tool Autonomous AI Agent

## Objective

Run a console-based LangChain agent that selects from web search, a safe calculator, and a restricted local file reader. The agent can make several tool calls before it writes a final answer.

## Safety and trace design

- The calculator evaluates a Python AST allow-list, not `eval()`.
- The local reader resolves paths and requires them to remain below `sample_files/`; it accepts only small `.txt` and `.md` files.
- DuckDuckGo search failures return a tool observation rather than terminating the process.
- The console trace reports only observable tool names, inputs, and truncated observations. It deliberately does not expose private chain-of-thought.

## Files

- `agent.py` — LangChain `create_agent` setup and console loop.
- `tools.py` — validated tool implementations.
- `sample_files/` — the only directory the reader is allowed to access.

## Run

```powershell
pip install -r requirements.txt
python agent.py
pytest -q
```

Try: `Read project_notes.txt, calculate (18 * 7) / 3, and summarize the note.` The trace should show a file-reader action, a calculator action, their observations, and then the synthesized answer. A Gemini API key is required for the agent; DuckDuckGo also requires internet access.
