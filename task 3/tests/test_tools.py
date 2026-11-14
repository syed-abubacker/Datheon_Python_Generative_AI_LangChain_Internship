import pytest

from tools import ToolInputError, read_allowed_file, safe_calculate


def test_safe_calculator_handles_arithmetic() -> None:
    assert safe_calculate("(2 + 3) * 4") == "20"


@pytest.mark.parametrize("expression", ["__import__('os')", "open('x')", "2 / 0"])
def test_safe_calculator_rejects_unsafe_or_invalid_input(expression: str) -> None:
    with pytest.raises(ToolInputError):
        safe_calculate(expression)


def test_file_reader_blocks_parent_escape(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(ToolInputError):
        read_allowed_file("../secret.txt", allowed)


def test_file_reader_reads_permitted_text_file(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "note.txt").write_text("safe content", encoding="utf-8")
    assert read_allowed_file("note.txt", allowed) == "safe content"
