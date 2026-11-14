# Task 2 — Document Q&A Assistant with RAG

## Objective

Index local PDF, TXT, and Markdown documents and answer questions only from relevant retrieved chunks. PDF loader page metadata survives loading, splitting, retrieval, and citation formatting.

## Architecture

`ingest.py` calls the reusable functions in `rag.py`: loaders read files, `RecursiveCharacterTextSplitter` makes overlapping chunks, Gemini creates embeddings, and Chroma persists the index locally. `chat.py` retrieves the most relevant chunks, applies a relevance threshold, asks Gemini for a grounded response, and prints `source, page N` citations.

## Files

- `rag.py` — shared RAG implementation, also imported by Task 4.
- `ingest.py` — index builder.
- `chat.py` — interactive console interface.
- `sample_documents/company_policy.txt` — safe example document.

## Run

```powershell
pip install -r requirements.txt
python ingest.py sample_documents
python chat.py
pytest -q
```

Example question: `What are the core collaboration hours?` The sample document supports the answer `10:00 to 16:00 India Standard Time`; a source label is included. A question with no sufficiently relevant chunk receives the explicit insufficient-context response instead of a made-up answer.

## Notes

Place PDFs and text files in any input folder. The generated `chroma_db/` is local, contains derived embeddings, and is excluded from Git. Real retrieval and answer generation need `GOOGLE_API_KEY` in the root `.env`.
