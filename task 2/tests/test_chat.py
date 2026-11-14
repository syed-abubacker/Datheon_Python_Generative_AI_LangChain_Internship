import chat
from rag import RAGAnswer


def feed_inputs(monkeypatch, *values):
    """Replace input() with scripted values; an exception instance is raised instead of returned."""
    queue = list(values)

    def fake_input(prompt: str = "") -> str:
        value = queue.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", fake_input)


def test_startup_failure_without_index_is_reported(monkeypatch, capsys) -> None:
    def no_index():
        raise FileNotFoundError("No local index found. Run ingest.py before asking questions.")

    monkeypatch.setattr(chat, "open_vector_store", no_index)
    chat.main()
    assert "Startup failed: No local index found" in capsys.readouterr().out


def test_answer_and_sources_are_printed(monkeypatch, capsys) -> None:
    monkeypatch.setattr(chat, "open_vector_store", lambda: object())
    monkeypatch.setattr(
        chat, "answer_question", lambda store, question: RAGAnswer("Ten to four.", ["company_policy.txt, page 1"], [])
    )
    feed_inputs(monkeypatch, "core hours?", "exit")
    chat.main()
    output = capsys.readouterr().out
    assert "Assistant: Ten to four." in output
    assert "Sources: company_policy.txt, page 1" in output


def test_api_failure_does_not_end_the_session(monkeypatch, capsys) -> None:
    calls = []

    def flaky(store, question):
        calls.append(question)
        if len(calls) == 1:
            raise ConnectionError("429 quota exceeded")
        return RAGAnswer("Recovered.", [], [])

    monkeypatch.setattr(chat, "open_vector_store", lambda: object())
    monkeypatch.setattr(chat, "answer_question", flaky)
    feed_inputs(monkeypatch, "first", "second", "quit")
    chat.main()
    output = capsys.readouterr().out
    assert "Error: 429 quota exceeded" in output
    assert "Assistant: Recovered." in output
    assert calls == ["first", "second"]


def test_ctrl_d_and_ctrl_c_exit_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(chat, "open_vector_store", lambda: object())
    for interruption in (EOFError(), KeyboardInterrupt()):
        feed_inputs(monkeypatch, interruption)
        chat.main()  # must return without raising
