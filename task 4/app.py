"""Streamlit UI for a document-grounded, streaming Gemini chat."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from backend import DEFAULT_INDEX_DIRECTORY, build_index, retrieved_sources, stream_answer

TASK_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(TASK_ROOT / ".env")

st.set_page_config(page_title="Document RAG Chat", page_icon="💬", layout="centered")
st.title("💬 Document RAG Chat")
st.caption("Answers are generated from your indexed documents and include the retrieved sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Documents")
    default_path = str(TASK_ROOT / "task 2" / "sample_documents")
    documents_directory = st.text_input("Folder containing PDFs, TXT, or Markdown", value=default_path)
    if st.button("Index documents", use_container_width=True):
        with st.spinner("Reading and embedding documents..."):
            try:
                document_count, chunk_count = build_index(documents_directory, DEFAULT_INDEX_DIRECTORY)
                from rag import open_vector_store  # Imported from Task 2 through backend path setup.
                st.session_state.vector_store = open_vector_store(DEFAULT_INDEX_DIRECTORY)
                st.success(f"Indexed {document_count} file pages into {chunk_count} chunks.")
            except Exception as error:
                st.error(f"Indexing failed: {error}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))

question = st.chat_input("Ask a question about the indexed documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            st.warning("Index a document folder from the sidebar before asking a question.")
        else:
            try:
                documents, sources = retrieved_sources(st.session_state.vector_store, question)
                answer = st.write_stream(stream_answer(question, st.session_state.messages[:-1], documents))
                if sources:
                    st.caption("Sources: " + "; ".join(sources))
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as error:
                st.error(f"Unable to answer: {error}")
