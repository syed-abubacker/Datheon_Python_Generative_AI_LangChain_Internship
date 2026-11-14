# Datheon Python + Generative AI & LangChain Internship

Four independent, beginner-friendly Python projects covering structured LLM extraction, retrieval-augmented generation (RAG), tool-using agents, and a Streamlit chat application. They use Google Gemini as the primary provider and never contain API keys.

## Projects

| Project | What it demonstrates |
| --- | --- |
| `task 1` | Gemini structured output, parameterized prompts, Pydantic validation, and resilient JSON extraction. |
| `task 2` | PDF/text loading, chunking, Gemini embeddings, Chroma retrieval, grounded answers, and page citations. |
| `task 3` | A LangChain tool-calling console agent with safe web search, arithmetic, and constrained file access. |
| `task 4` | A Streamlit chat UI with session memory and real Gemini streaming, reusing Task 2's RAG module. |

## Prerequisites

Use Python **3.12 or 3.13** in a virtual environment. Python 3.14 is very new and may not yet be supported by every binary dependency, particularly ChromaDB; do not replace a system Python installation. Install a supported version side-by-side and create an isolated environment.

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy `.env.example` to `.env` at the repository root and add the key. `.env` is ignored by Git.
3. In PowerShell, create an environment (commands use quoted paths because directory names have spaces):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

## Run

```powershell
Set-Location "task 1"; pip install -r requirements.txt; python main.py sample_inputs/sample_resume.txt --kind resume
Set-Location "..\\task 2"; pip install -r requirements.txt; python ingest.py sample_documents; python chat.py
Set-Location "..\\task 3"; pip install -r requirements.txt; python agent.py
Set-Location "..\\task 4"; pip install -r requirements.txt; streamlit run app.py
```

## Test

Run each task's offline unit tests after installing its requirements:

```powershell
Set-Location "task 1"; pytest -q
Set-Location "..\\task 2"; pytest -q
Set-Location "..\\task 3"; pytest -q
Set-Location "..\\task 4"; pytest -q
```

The tests deliberately mock or avoid LLM requests. Perform a manual integration test after setting `GOOGLE_API_KEY`.

## Design choices and limitations

- `google-genai` is the maintained Google Python SDK; `google-generativeai` is legacy.
- The default `gemini-2.5-flash` balances cost and capability. Google API availability, quotas, and free-tier limits vary by account; check the current [Gemini pricing page](https://ai.google.dev/gemini-api/docs/pricing/).
- RAG answers are constrained by retrieved chunks and use a relevance threshold, but LLM output is probabilistic. Review answers for high-stakes uses.
- Chroma indexes are local and ignored by Git. Ingest documents again on a new machine.
- Task 3's DuckDuckGo search needs internet access; all agent tools can fail gracefully.

## Layout

```text
task 1/  structured extraction
task 2/  reusable RAG pipeline
task 3/  multi-tool agent
task 4/  Streamlit RAG chat
```
