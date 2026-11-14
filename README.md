# Datheon Python + Generative AI & LangChain Internship

![Offline unit tests](https://github.com/syed-abubacker/Datheon_Python_Generative_AI_LangChain_Internship/actions/workflows/tests.yml/badge.svg)

## Overview

This repository holds the four assignments of the Datheon Python + Generative AI + LangChain internship: structured LLM extraction, retrieval-augmented generation (RAG), a tool-using agent, and a Streamlit chat application. Each task is an independent, beginner-friendly Python project in its own folder, with its own `requirements.txt`, tests and README. They use Google Gemini as the model provider and never contain API keys.

## Projects

| Project | What it demonstrates | Documentation |
| --- | --- | --- |
| `task 1` | Gemini structured output, parameterized prompts, Pydantic validation, and resilient JSON extraction. | [task 1/README.md](task%201/README.md) |
| `task 2` | PDF/text loading, chunking, Gemini embeddings, Chroma retrieval, grounded answers, and page citations. | [task 2/README.md](task%202/README.md) |
| `task 3` | A LangChain tool-calling console agent with web search, a safe calculator, and constrained file access. | [task 3/README.md](task%203/README.md) |
| `task 4` | A Streamlit chat UI with session memory and Gemini streaming, reusing Task 2's RAG module. | [task 4/README.md](task%204/README.md) |

## Technologies

Python, Google Gemini (`gemini-2.5-flash` by default), LangChain (`langchain`, `langchain-google-genai`, `langchain-chroma`, `langchain-text-splitters`), the `google-genai` SDK, Pydantic, ChromaDB, Streamlit, `ddgs` (DuckDuckGo search), pytest, and GitHub Actions.

## Repository structure

```text
.
├── .env.example               template for the required environment variables
├── .github/workflows/tests.yml   CI: runs each task's offline tests
├── task 1/                    structured extraction (main.py, prompts.py, schemas.py, sample_inputs/)
├── task 2/                    reusable RAG pipeline (rag.py, ingest.py, chat.py, sample_documents/)
├── task 3/                    multi-tool agent (agent.py, tools.py, sample_files/)
└── task 4/                    Streamlit RAG chat (app.py, backend.py)
```

Every task folder also contains `requirements.txt`, `pytest.ini`, `tests/` and its own `README.md`.

## Installation and environment setup

CI runs on Python **3.13**; 3.12 is expected to work but is not exercised here. Python 3.14 is very new and may not yet be supported by every binary dependency, particularly ChromaDB. Use an isolated virtual environment and do not replace a system Python.

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create and activate a virtual environment, then create `.env` at the repository root from the template.

   PowerShell:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   Copy-Item .env.example .env
   ```

   macOS / Linux:

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   cp .env.example .env
   ```

3. Edit `.env` and set `GOOGLE_API_KEY`. `.env` is ignored by Git; never commit it.

| Variable | Required | Default | Used by |
| --- | --- | --- | --- |
| `GOOGLE_API_KEY` | yes, for any live Gemini call | none | all tasks |
| `GEMINI_MODEL` | no | `gemini-2.5-flash` | all tasks |
| `GEMINI_EMBEDDING_MODEL` | no | `models/gemini-embedding-001` | tasks 2 and 4 |

Each task has its own requirements file, so install them per task (the commands below work in PowerShell and bash).

## Running each task

```text
cd "task 1"; pip install -r requirements.txt; python main.py sample_inputs/sample_resume.txt --kind resume; cd ..
cd "task 2"; pip install -r requirements.txt; python ingest.py sample_documents; python chat.py; cd ..
cd "task 3"; pip install -r requirements.txt; python agent.py; cd ..
cd "task 4"; pip install -r requirements.txt; streamlit run app.py; cd ..
```

Task 1 also accepts `--kind invoice` with `sample_inputs/sample_invoice.txt`. See each task's README for example questions and expected behavior.

## Testing

Each task's tests run offline and need no API key. From a task folder, after installing its requirements:

```text
pytest -q
```

Gemini is replaced by fakes in the tests. Tasks 2 and 4 exercise a real local ChromaDB with deterministic fake embeddings, and Task 4 also drives the Streamlit UI with Streamlit's `AppTest`. GitHub Actions runs all four suites on every push and pull request to `main`.

## Limitations

- The default `gemini-2.5-flash` balances cost and capability. Availability, quotas and free-tier limits vary by account; check the current [Gemini pricing page](https://ai.google.dev/gemini-api/docs/pricing/).
- RAG answers are constrained by retrieved chunks and a relevance threshold, but LLM output is probabilistic. Review answers for high-stakes uses.
- Chroma indexes are local and ignored by Git. Run ingestion again on a new machine; ingesting again on the same machine replaces the previous index.
- Task 3's DuckDuckGo search needs internet access; every agent tool reports failures instead of crashing.
- `langchain-community` (used for the PDF and text loaders) is being sunset upstream and prints a deprecation warning.

## License

See [LICENSE](LICENSE).
