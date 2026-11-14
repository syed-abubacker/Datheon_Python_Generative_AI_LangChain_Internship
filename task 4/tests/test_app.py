from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=60)


def test_app_starts_with_empty_state_and_no_errors() -> None:
    app = new_app().run()
    assert not app.exception
    assert app.title[0].value.endswith("Document RAG Chat")
    assert app.session_state["messages"] == []
    assert app.session_state["vector_store"] is None
    assert [button.label for button in app.sidebar.button] == ["Index documents", "Clear conversation"]


def test_question_before_indexing_warns_and_keeps_history() -> None:
    app = new_app().run()
    app.chat_input[0].set_value("What is the policy?").run()
    assert not app.exception
    assert any("Index a document folder" in warning.value for warning in app.warning)
    assert app.session_state["messages"] == [{"role": "user", "content": "What is the policy?"}]


def test_indexing_missing_folder_shows_error_instead_of_crashing(tmp_path) -> None:
    app = new_app().run()
    app.sidebar.text_input[0].set_value(str(tmp_path / "nope")).run()
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert any("Indexing failed" in error.value for error in app.error)
    assert app.session_state["vector_store"] is None


def test_full_flow_index_ask_stream_cite_and_clear(
    monkeypatch, tmp_path, docs_folder, offline_embeddings, fake_chat_model
) -> None:
    import backend

    monkeypatch.setattr(backend, "DEFAULT_INDEX_DIRECTORY", tmp_path / "app_index")
    app = new_app().run()
    app.sidebar.text_input[0].set_value(str(docs_folder)).run()
    app.sidebar.button[0].click().run()
    assert not app.exception, [e.value for e in app.error]
    assert any("Indexed 2 file pages into 2 chunks" in s.value for s in app.success)
    assert app.session_state["vector_store"] is not None

    app.chat_input[0].set_value("remotely three days per week").run()
    assert not app.exception, [e.value for e in app.error]
    messages = app.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "Three days per week."
    assert messages[1]["sources"] == ["remote.txt, page 1"]

    app.chat_input[0].set_value("submit travel receipts within thirty days for reimbursement").run()
    assert not app.exception
    assert "Three days per week." in fake_chat_model.prompts[-1]  # earlier turn is passed as history

    app.sidebar.button[1].click().run()
    assert app.session_state["messages"] == []


@pytest.mark.parametrize("failure", [RuntimeError("model unavailable")])
def test_generation_errors_are_shown_not_raised(
    monkeypatch, tmp_path, docs_folder, offline_embeddings, fake_chat_model, failure
) -> None:
    import backend

    def broken_stream(question, history, documents):
        raise failure
        yield  # pragma: no cover

    monkeypatch.setattr(backend, "DEFAULT_INDEX_DIRECTORY", tmp_path / "app_index")
    app = new_app().run()
    app.sidebar.text_input[0].set_value(str(docs_folder)).run()
    app.sidebar.button[0].click().run()
    monkeypatch.setattr(backend, "stream_answer", broken_stream)
    app.chat_input[0].set_value("remotely three days per week").run()
    assert not app.exception
    assert any("Unable to answer: model unavailable" in error.value for error in app.error)
