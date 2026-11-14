import agent


def feed_inputs(monkeypatch, *values):
    """Replace input() with scripted values; an exception instance is raised instead of returned."""
    queue = list(values)

    def fake_input(prompt: str = "") -> str:
        value = queue.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", fake_input)


def test_trace_and_answer_are_printed_with_long_observations_truncated(monkeypatch, capsys) -> None:
    trace = [
        {"tool": "calculator", "input": "2+2", "output": "4"},
        {"tool": "web_search", "input": "q", "output": "x" * 900},
    ]
    monkeypatch.setattr(agent, "ask_agent", lambda question: ("The answer is 4.", trace))
    feed_inputs(monkeypatch, "what is 2+2?", "exit")
    agent.main()
    output = capsys.readouterr().out
    assert "[Tool] calculator" in output and "Input: 2+2" in output
    assert "Assistant: The answer is 4." in output
    assert "x" * 500 in output and "x" * 501 not in output


def test_validation_and_api_errors_do_not_end_the_session(monkeypatch, capsys) -> None:
    outcomes = iter([ValueError("Question cannot be empty."), ConnectionError("503 model overloaded"), ("Fine.", [])])

    def ask(question: str):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(agent, "ask_agent", ask)
    feed_inputs(monkeypatch, "", "second", "third", "exit")
    agent.main()
    output = capsys.readouterr().out
    assert "Error: Question cannot be empty." in output
    assert "Error: 503 model overloaded" in output
    assert "Assistant: Fine." in output


def test_ctrl_d_and_ctrl_c_exit_cleanly(monkeypatch) -> None:
    for interruption in (EOFError(), KeyboardInterrupt()):
        feed_inputs(monkeypatch, interruption)
        agent.main()  # must return without raising
