"""Build a local Chroma index from a folder of PDF, TXT, or Markdown files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag import DEFAULT_PERSIST_DIRECTORY, create_vector_store, load_documents, split_documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Index documents for Task 2 RAG.")
    parser.add_argument("documents_directory", type=Path)
    parser.add_argument("--index", type=Path, default=DEFAULT_PERSIST_DIRECTORY)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    try:
        documents = load_documents(args.documents_directory)
        chunks = split_documents(documents)
        create_vector_store(chunks, args.index)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"Indexing failed: {error}", file=sys.stderr)
        return 1
    print(f"Indexed {len(documents)} document pages/files into {len(chunks)} chunks at {args.index}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
