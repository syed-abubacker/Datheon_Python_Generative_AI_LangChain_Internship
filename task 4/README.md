# Task 4 — Interactive Streamlit RAG Chatbot

## Problem statement

The Task 2 assistant runs only in a terminal. People expect a browser chat window where they can load documents, ask follow-up questions, see answers appear as they are written, and check the sources.

## Solution overview

A browser-based document chatbot with conversation history that survives Streamlit reruns and model streaming. This is not a static mock-up: `st.write_stream` displays chunks from Gemini's streaming API as they arrive. Every answer shows the source labels of the chunks that were sent to the model.

## Architecture / workflow

1. **Index**: in the sidebar the user enters a folder (default: Task 2's `sample_documents`) and clicks **Index documents**. `backend.build_index` loads, splits and embeds the files with Task 2's functions into a local `task 4/chroma_db/` index (replacing any earlier index), and the app opens it into `st.session_state`.
2. **Retrieve**: for each question, `retrieved_sources` retrieves the relevant chunks once and returns both the chunks and their citations.
3. **Stream**: `stream_answer` sends the chunks, the last 6 chat messages (for follow-up wording only, not as evidence) and the question to Gemini and yields the real streamed chunks.
4. **Display**: the answer and its sources are stored in `st.session_state.messages` and shown on every rerun. **Clear conversation** empties the history.

## Integration with Task 2

`backend.py` adds the sibling `task 2` folder to the import path and imports its document loaders, splitter, embeddings, Chroma functions, retrieval threshold, insufficient-context message, and citation formatter. It does not duplicate the RAG implementation. The adapter only supplies UI-oriented indexing, conversation formatting, and streaming. Task 4 keeps its own index folder, separate from Task 2's.

## Files

- `app.py` — Streamlit user interface and session-state handling.
- `backend.py` — adapter around Task 2 plus Gemini streaming.
- `tests/` — offline tests.

## Technologies

Python, Streamlit, LangChain (`langchain-google-genai`, `langchain-chroma`, loaders and splitter via Task 2), ChromaDB, Gemini, `python-dotenv`, pytest.

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
streamlit run app.py
```

Use the sidebar to enter a folder path and click **Index documents**, then ask a question such as `What are the core collaboration hours?`. Streamlit keeps user and assistant messages in `st.session_state`, so the conversation stays visible while the app reruns; it is not saved across a browser reload.

## Important behavior

The chatbot will not call Gemini when no retrieved chunks meet Task 2's relevance threshold; it displays the explicit insufficient-context message instead. Indexing, model streaming, and API errors appear in the UI rather than being hidden. Asking a question before indexing shows a warning.

## Testing

```text
pytest -q
```

The tests need no API key. They start the app with Streamlit's `AppTest` and drive the full flow with a fake Gemini model and a deterministic fake embedding: empty startup state, a question before indexing, indexing a missing folder, indexing a real folder with a real local Chroma index, asking, streamed output, source citations, conversation history passed to the next question, clearing the history, and model errors shown in the UI. Backend tests cover streaming, the 6-message history limit, the missing key, and rebuilding the index without duplicates. The real server was also started headless and answered its health check.

Not covered: real Gemini streaming and answer quality, which need a live key.
