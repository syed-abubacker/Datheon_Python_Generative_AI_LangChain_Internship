import json
from pathlib import Path

import pytest

import main
from main import ExtractionError, extract_document, schema_for
from schemas import InvoiceData, ResumeData

SAMPLE_DIRECTORY = Path(__file__).resolve().parents[1] / "sample_inputs"


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return FakeResponse(self.text)


class FakeClient:
    def __init__(self, text=None, error=None):
        self.models = FakeModels(text, error)


def test_extracts_resume_and_sends_json_schema_config() -> None:
    payload = {"document_type": "resume", "name": "Ada Lovelace", "skills": ["math"]}
    client = FakeClient(json.dumps(payload))
    result = extract_document("Ada Lovelace, mathematician", "resume", client=client)
    assert isinstance(result, ResumeData)
    assert result.name == "Ada Lovelace" and result.skills == ["math"]
    call = client.models.calls[0]
    assert "Ada Lovelace, mathematician" in call["contents"]
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is ResumeData


def test_extracts_invoice_with_line_items() -> None:
    payload = {
        "document_type": "invoice",
        "invoice_number": "INV-1",
        "total": 120.5,
        "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 60.25, "amount": 120.5}],
    }
    result = extract_document("Invoice text", "invoice", client=FakeClient(json.dumps(payload)))
    assert isinstance(result, InvoiceData)
    assert result.line_items[0].description == "Widget"
    assert result.total == 120.5


@pytest.mark.parametrize(
    "bad_text",
    [
        "not json at all",
        '{"document_type": "invoice"}',  # wrong document type for a resume request
        '{"document_type": "resume", "skills": "python"}',  # wrong field type
        "[]",
    ],
)
def test_invalid_model_output_becomes_extraction_error(bad_text: str) -> None:
    with pytest.raises(ExtractionError, match="does not match the resume schema"):
        extract_document("some text", "resume", client=FakeClient(bad_text))


def test_empty_model_response_is_reported() -> None:
    with pytest.raises(ExtractionError, match="empty response"):
        extract_document("some text", "resume", client=FakeClient(""))
    with pytest.raises(ExtractionError, match="empty response"):
        extract_document("some text", "resume", client=FakeClient(None))


def test_api_failure_is_wrapped_without_leaking_a_traceback_type() -> None:
    with pytest.raises(ExtractionError, match="Gemini request failed: quota exceeded"):
        extract_document("some text", "resume", client=FakeClient(error=RuntimeError("quota exceeded")))


def test_empty_input_does_not_call_the_model() -> None:
    client = FakeClient("{}")
    with pytest.raises(ExtractionError, match="empty"):
        extract_document("   \n", "resume", client=client)
    assert client.models.calls == []


def test_missing_api_key_is_a_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="GOOGLE_API_KEY"):
        extract_document("some text", "resume")


def test_unknown_document_kind_is_rejected() -> None:
    with pytest.raises(ExtractionError, match="resume' or 'invoice"):
        schema_for("contract")
    with pytest.raises(ExtractionError):
        extract_document("some text", "contract", client=FakeClient("{}"))


def test_model_name_comes_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    client = FakeClient('{"document_type": "resume"}')
    extract_document("some text", "resume", client=client)
    assert client.models.calls[0]["model"] == "gemini-test-model"


def test_sample_inputs_exist_and_are_non_empty() -> None:
    for name in ("sample_resume.txt", "sample_invoice.txt"):
        assert (SAMPLE_DIRECTORY / name).read_text(encoding="utf-8").strip()


def test_cli_reports_missing_file_with_exit_code_1(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr("sys.argv", ["main.py", str(tmp_path / "nope.txt"), "--kind", "resume"])
    assert main.main() == 1
    assert "Error:" in capsys.readouterr().err
