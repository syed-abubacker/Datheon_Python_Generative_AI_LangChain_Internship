"""Task 4 adapter: reuse Task 2 RAG rather than copying it."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

TASK_2_DIRECTORY = Path(__file__).resolve().parents[1] / "task 2"
if str(TASK_2_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TASK_2_DIRECTORY))

from rag import (  # noqa: E402
    NOT_ENOUGH_CONTEXT,
    create_vector_store,
    load_documents,
    retrieve_documents,
    source_labels,
    split_documents,
)

DEFAULT_INDEX_DIRECTORY = Path(__file__).parent / "chroma_db"


def build_index(
    documents_directory: str | Path, index_directory: str | Path = DEFAULT_INDEX_DIRECTORY
) -> tuple[int, int]:
    """Use Task 2 loading/chunking/embedding code and return document/chunk counts."""
    documents = load_documents(documents_directory)
    chunks = split_documents(documents)
    create_vector_store(chunks, index_directory)
    return len(documents), len(chunks)


def retrieved_sources(vector_store: Any, question: str) -> tuple[list[Document], list[str]]:
    """Retrieve once so the UI can show exactly the sources sent to Gemini."""
    documents = retrieve_documents(vector_store, question)
    return documents, source_labels(documents)


def stream_answer(question: str, history: list[dict[str, str]], documents: list[Document]) -> Iterable[str]:
    """Yield actual Gemini chunks; no synthetic or pre-written streaming is used."""
    if not documents:
        yield NOT_ENOUGH_CONTEXT
        return
    context = "\n\n---\n\n".join(document.page_content for document in documents)
    history_text = "\n".join(f"{message['role'].title()}: {message['content']}" for message in history[-6:])
    prompt = f"""Answer only from the document context below. If it is insufficient, say exactly:
{NOT_ENOUGH_CONTEXT}
Keep the answer concise. Previous conversation is for follow-up wording only; it is not evidence.

Previous conversation:
{history_text}

Document context:
{context}

User question: {question}
"""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Configure the root .env file.")
    model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
    for chunk in model.stream(prompt):
        content = str(chunk.content)
        if content:
            yield content
