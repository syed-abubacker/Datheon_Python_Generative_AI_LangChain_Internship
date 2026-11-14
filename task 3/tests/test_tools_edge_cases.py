import sys
import types

import pytest

from tools import ToolInputError, read_allowed_file, safe_calculate, search_web


@pytest.mark.parametrize("expression", ["9**9**9", "10**400", "(-8)**0.5", "1e308*10", "", "   ", "2 +"])
def test_calculator_rejects_overflow_complex_and_invalid_input(expression: str) -> None:
    with pytest.raises(ToolInputError):
        safe_calculate(expression)


def test_calculator_rejects_overlong_expression() -> None:
    with pytest.raises(ToolInputError):
        safe_calculate("1+" * 200 + "1")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("7 % 3", "1"), ("-3 + 5", "2"), ("2 ** 0.5", str(2**0.5)), ("10 / 4", "2.5"), ("+4", "4")],
)
def test_calculator_results(expression: str, expected: str) -> None:
    assert safe_calculate(expression) == expected


def test_file_reader_rejects_disallowed_extension(tmp_path) -> None:
    (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
    with pytest.raises(ToolInputError):
        read_allowed_file("data.csv", tmp_path)


def test_file_reader_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(ToolInputError):
        read_allowed_file("missing.txt", tmp_path)


def test_file_reader_rejects_absolute_path_outside_directory(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ToolInputError):
        read_allowed_file(str(outside), allowed)


def test_file_reader_rejects_symlink_escape(tmp_path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = allowed / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks are not available on this platform.")
    with pytest.raises(ToolInputError):
        read_allowed_file("link.txt", allowed)


def test_file_reader_rejects_oversized_file(tmp_path) -> None:
    (tmp_path / "big.txt").write_text("x" * 100_001, encoding="utf-8")
    with pytest.raises(ToolInputError):
        read_allowed_file("big.txt", tmp_path)


def _install_fake_ddgs(monkeypatch, behaviour) -> None:
    class FakeDDGS:
        def text(self, query: str, max_results: int):
            return behaviour(query, max_results)

    module = types.ModuleType("ddgs")
    module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)


def test_search_rejects_empty_query() -> None:
    with pytest.raises(ToolInputError):
        search_web("  ")


def test_search_formats_results(monkeypatch) -> None:
    _install_fake_ddgs(
        monkeypatch, lambda q, n: [{"title": "Title A", "body": "Body A", "href": "https://example.com/a"}]
    )
    output = search_web("anything")
    assert "Title A" in output and "Body A" in output and "https://example.com/a" in output


def test_search_reports_no_results(monkeypatch) -> None:
    _install_fake_ddgs(monkeypatch, lambda q, n: [])
    assert search_web("anything") == "No web search results were found."


def test_search_failure_is_returned_not_raised(monkeypatch) -> None:
    def boom(query: str, max_results: int):
        raise RuntimeError("network down")

    _install_fake_ddgs(monkeypatch, boom)
    assert search_web("anything").startswith("Web search failed:")
