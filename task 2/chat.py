"""Interactive console Q&A over the Task 2 local index."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from rag import answer_question, open_vector_store


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    try:
        vector_store = open_vector_store()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Startup failed: {error}")
        return
    print("Ask a document question. Type 'exit' to quit.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        try:
            answer = answer_question(vector_store, question)
            print(f"Assistant: {answer.text}")
            if answer.sources:
                print("Sources: " + "; ".join(answer.sources))
        except (ValueError, RuntimeError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
