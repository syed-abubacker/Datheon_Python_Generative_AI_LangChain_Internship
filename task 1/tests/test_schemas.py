from pydantic import ValidationError
import pytest

from prompts import build_prompt
from schemas import InvoiceData, ResumeData


def test_resume_allows_missing_optional_values() -> None:
    resume = ResumeData.model_validate({"document_type": "resume", "name": "Ada"})
    assert resume.skills == []
    assert resume.email is None


def test_invoice_rejects_wrong_document_type() -> None:
    with pytest.raises(ValidationError):
        InvoiceData.model_validate({"document_type": "resume"})


def test_prompt_includes_source_text() -> None:
    assert "Ada Lovelace" in build_prompt("resume", "Ada Lovelace")
