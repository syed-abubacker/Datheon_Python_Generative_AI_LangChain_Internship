# Task 4 — Interactive Streamlit RAG Chatbot

## Objective

Provide a browser-based document chatbot with persistent session history and actual model streaming. This is not a static mock-up: `st.write_stream` displays chunks from Gemini's streaming API.

## Integration with Task 2

`backend.py` adds the sibling `task 2` folder to the import path and imports its document loaders, splitter, embeddings, Chroma functions, retrieval threshold, insufficient-context message, and citation formatter. It does not duplicate the RAG implementation. The adapter only supplies UI-oriented indexing, conversation formatting, and streaming.

## Run

From this directory, after activating an environment and configuring the root `.env`:

```powershell
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

Use the sidebar to enter a folder path and click **Index documents**. The default points to Task 2's sample documents. Ask a question in the chat box. Streamlit stores user/assistant messages in `st.session_state`, so the conversation remains visible across reruns. The assistant shows the same retrieved source labels that informed its answer.

## Important behavior

The chatbot will not call Gemini when no retrieved chunks meet Task 2's relevance threshold; it displays the explicit insufficient-context message instead. Indexing, model streaming, and API errors appear in the UI rather than being hidden.
