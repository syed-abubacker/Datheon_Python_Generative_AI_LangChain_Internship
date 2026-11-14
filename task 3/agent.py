"""Console LangChain agent with safe observable tool traces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import ToolInputError, read_allowed_file, safe_calculate, search_web

ALLOWED_DIRECTORY = Path(__file__).parent / "sample_files"


def build_tools(trace: list[dict[str, str]]) -> list[Any]:
    """Make LangChain tools and record only user-visible actions/results, not reasoning."""
    @tool
    def web_search(query: str) -> str:
        """Search the current public web for facts or recent information."""
        output = search_web(query)
        trace.append({"tool": "web_search", "input": query, "output": output})
        return output

    @tool
    def calculator(expression: str) -> str:
        """Calculate a basic arithmetic expression using numbers and + - * / % ** parentheses."""
        try:
            output = safe_calculate(expression)
        except ToolInputError as error:
            output = f"Calculator error: {error}"
        trace.append({"tool": "calculator", "input": expression, "output": output})
        return output

    @tool
    def local_file_reader(filename: str) -> str:
        """Read a .txt or .md file relative to the permitted sample_files directory only."""
        try:
            output = read_allowed_file(filename, ALLOWED_DIRECTORY)
        except (ToolInputError, OSError, UnicodeDecodeError) as error:
            output = f"File reader error: {error}"
        trace.append({"tool": "local_file_reader", "input": filename, "output": output})
        return output

    return [web_search, calculator, local_file_reader]


def ask_agent(question: str) -> tuple[str, list[dict[str, str]]]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Configure the root .env file.")
    trace: list[dict[str, str]] = []
    model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), temperature=0)
    agent = create_agent(
        model=model,
        tools=build_tools(trace),
        system_prompt=(
            "Use tools when useful. You may make multiple calls. Use local_file_reader only for files "
            "inside its permitted directory. Base the final answer on returned tool observations."
        ),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_message = result["messages"][-1]
    return str(final_message.content), trace


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    print("Tool-using agent ready. Type 'exit' to quit.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        try:
            answer, trace = ask_agent(question)
        except (ValueError, RuntimeError) as error:
            print(f"Error: {error}")
            continue
        for item in trace:
            print(f"[Tool] {item['tool']}\n  Input: {item['input']}\n  Observation: {item['output'][:500]}")
        print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
