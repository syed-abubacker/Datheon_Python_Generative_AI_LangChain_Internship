import pytest

import agent


def _tools_by_name(trace):
    return {tool.name: tool for tool in agent.build_tools(trace)}


def test_build_tools_exposes_three_named_tools() -> None:
    assert set(_tools_by_name([])) == {"web_search", "calculator", "local_file_reader"}


def test_calculator_tool_records_trace_and_reports_errors_without_raising() -> None:
    trace: list[dict[str, str]] = []
    tools = _tools_by_name(trace)
    assert tools["calculator"].invoke({"expression": "2 + 3"}) == "5"
    assert tools["calculator"].invoke({"expression": "10**400"}).startswith("Calculator error:")
    assert [item["tool"] for item in trace] == ["calculator", "calculator"]


def test_file_reader_tool_reads_sample_and_blocks_escape() -> None:
    trace: list[dict[str, str]] = []
    tools = _tools_by_name(trace)
    sample = tools["local_file_reader"].invoke({"filename": "project_notes.txt"})
    assert not sample.startswith("File reader error:")
    blocked = tools["local_file_reader"].invoke({"filename": "../agent.py"})
    assert blocked.startswith("File reader error:")
    assert "def build_tools" not in blocked


def test_web_search_tool_uses_search_function(monkeypatch) -> None:
    monkeypatch.setattr(agent, "search_web", lambda query: f"results for {query}")
    trace: list[dict[str, str]] = []
    assert _tools_by_name(trace)["web_search"].invoke({"query": "python"}) == "results for python"
    assert trace[0]["tool"] == "web_search"


def test_ask_agent_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        agent.ask_agent("   ")


def test_ask_agent_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        agent.ask_agent("hello")


def test_ask_agent_wires_model_tools_and_returns_final_answer(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    captured: dict = {}

    class FakeMessage:
        content = "final answer"

    class FakeAgent:
        def invoke(self, payload):
            captured["payload"] = payload
            return {"messages": [FakeMessage()]}

    def fake_create_agent(**kwargs):
        captured["tool_names"] = {tool.name for tool in kwargs["tools"]}
        return FakeAgent()

    monkeypatch.setattr(agent, "ChatGoogleGenerativeAI", lambda **kwargs: object())
    monkeypatch.setattr(agent, "create_agent", fake_create_agent)
    answer, trace = agent.ask_agent("What is 2+2?")
    assert answer == "final answer"
    assert trace == []
    assert captured["tool_names"] == {"web_search", "calculator", "local_file_reader"}
    assert captured["payload"]["messages"][0]["content"] == "What is 2+2?"
