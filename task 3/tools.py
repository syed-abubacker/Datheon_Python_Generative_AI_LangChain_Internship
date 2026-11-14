"""Safe, observable tools for the Task 3 agent."""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Callable


class ToolInputError(ValueError):
    """Raised when a tool receives an unsafe or invalid argument."""


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos, ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> str:
    """Evaluate basic arithmetic from a strict AST allow-list; never uses eval()."""
    if not expression.strip() or len(expression) > 200:
        raise ToolInputError("Enter a non-empty arithmetic expression under 200 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ToolInputError("The calculator expression is not valid syntax.") from error

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        raise ToolInputError("Only numbers and +, -, *, /, %, **, and parentheses are allowed.")

    try:
        result = visit(tree)
    except ZeroDivisionError as error:
        raise ToolInputError("Division by zero is not allowed.") from error
    if abs(result) == float("inf") or result != result:
        raise ToolInputError("The calculation did not produce a finite number.")
    return str(int(result) if result.is_integer() else result)


def read_allowed_file(filename: str, allowed_directory: str | Path) -> str:
    """Read only text/Markdown files physically located below allowed_directory."""
    root = Path(allowed_directory).resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ToolInputError("That file is outside the permitted sample_files directory.") from error
    if candidate.suffix.lower() not in {".txt", ".md"}:
        raise ToolInputError("Only .txt and .md files are permitted.")
    if not candidate.is_file():
        raise ToolInputError("The requested permitted file does not exist.")
    if candidate.stat().st_size > 100_000:
        raise ToolInputError("The permitted file is too large to read.")
    return candidate.read_text(encoding="utf-8")


def search_web(query: str) -> str:
    """Search DuckDuckGo and return a small, readable set of result snippets."""
    if not query.strip():
        raise ToolInputError("Search query cannot be empty.")
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=3))
    except Exception as error:
        return f"Web search failed: {error}"
    if not results:
        return "No web search results were found."
    return "\n\n".join(
        f"{item.get('title', 'Untitled')}\n{item.get('body', '')}\n{item.get('href', '')}"
        for item in results
    )
