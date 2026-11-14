# Task 2 — Document Q&A Assistant with RAG

## Problem statement

An LLM answers confidently even when it does not know. Questions about private documents need answers grounded only in those documents, with a way to check where each answer came from.

## Solution overview

Index local PDF, TXT, and Markdown documents and answer questions only from relevant retrieved chunks. PDF page metadata survives loading, splitting, retrieval, and citation formatting, so every answer lists `source, page N`. If no chunk is relevant enough, the assistant says so instead of guessing.

## Architecture / workflow

1. **Ingest** (`ingest.py`): `load_documents` reads `.pdf`, `.txt` and `.md` files (other files are ignored). `split_documents` uses `RecursiveCharacterTextSplitter` (900 characters, 150 overlap) and keeps each file's metadata on every chunk. Gemini creates the embeddings and Chroma persists them in a local `chroma_db/` folder. Running ingestion again replaces the previous index instead of adding duplicates.
2. **Ask** (`chat.py`): `retrieve_documents` fetches the 4 most similar chunks and keeps those with a relevance score of at least 0.35. If none remain, the assistant returns the fixed "not enough context" message without calling the model.
3. **Answer**: otherwise Gemini receives only the retrieved text and is told not to use outside knowledge. `source_labels` turns metadata into citations (PDF page numbers are shown one-based).

## Files

- `rag.py` — shared RAG implementation, also imported by Task 4.
- `ingest.py` — index builder.
- `chat.py` — interactive console interface.
- `sample_documents/company_policy.txt` — safe example document.
- `tests/` — offline tests.

## Technologies

Python, LangChain (`langchain-community` loaders, `langchain-text-splitters`, `langchain-chroma`, `langchain-google-genai`), ChromaDB, Gemini embeddings and chat model, `pypdf`, `python-dotenv`, pytest.

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
| `GEMINI_EMBEDDING_MODEL` | no | `models/gemini-embedding-001` |

## Example usage

```text
python ingest.py sample_documents
python chat.py
```

Example question: `What are the core collaboration hours?` The sample document supports the answer `10:00 to 16:00 India Standard Time`, and a source label such as `company_policy.txt, page 1` should be printed. A question with no sufficiently relevant chunk receives the explicit insufficient-context response instead of a made-up answer. (The live answer text has not been confirmed in a Gemini run; the retrieval, threshold and citation logic is covered by the offline tests.)

`ingest.py` also accepts `--index PATH` to store the index elsewhere, but `chat.py` always reads the default `chroma_db/` folder, so keep the default for the console chat.

## Notes and limitations

- Place PDFs and text files in any input folder. The generated `chroma_db/` is local, contains derived embeddings, and is excluded from Git.
- The 0.35 threshold uses Chroma's default L2-based relevance function, which assumes normalized embeddings. It has not been tuned against live Gemini embeddings, so adjust `minimum_score` in `retrieve_documents` if answers are refused too often or too rarely.
- `langchain-community` is being sunset upstream and prints a deprecation warning.

## Testing

```text
pytest -q
```

The tests need no API key. A deterministic fake embedding replaces Gemini, while the rest runs for real: loading text, Markdown and generated PDF files, chunking with metadata, a real local Chroma index, score filtering, grounded prompt construction with a fake LLM, citations, the no-context path (no model call), re-ingestion without duplicates, and error cases (missing folder, empty folder, missing index, missing key).

Not covered: real Gemini embeddings and answers, which need a live key.
