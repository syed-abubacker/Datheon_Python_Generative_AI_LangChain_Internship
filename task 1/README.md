# Task 1 — Structured LLM Prompt Pipeline

## Problem statement

Resumes and invoices arrive as free text. Downstream code needs predictable, typed data, not prose, and an LLM can return malformed or incomplete JSON.

## Solution overview

Gemini is instructed to produce JSON that matches a Pydantic schema, then the result is validated again locally before it is printed. Anything that does not fit the schema, or any network, authentication, empty-input or missing-file problem, becomes a short, understandable error instead of a traceback.

## Architecture / workflow

1. `main.py` reads a UTF-8 text file and the `--kind` (`resume` or `invoice`).
2. `prompts.py` builds a parameterized prompt that tells the model to use only facts in the text and to use `null` or `[]` for anything missing.
3. `main.py` calls Gemini once with `response_mime_type="application/json"` and the Pydantic model as `response_schema`, at temperature 0.
4. The reply is validated with `model_validate_json`; failures raise `ExtractionError`.
5. The validated model is printed as indented JSON (exit code 0), or the error goes to stderr (exit code 1).

## Files

- `main.py` — command-line entry point, Gemini call, error handling, and Pydantic validation.
- `prompts.py` — parameterized resume and invoice prompts.
- `schemas.py` — `ResumeData`, `InvoiceData` and their nested models.
- `sample_inputs/` — safe, fictional input examples.
- `tests/` — offline tests.

## Technologies

Python, `google-genai` (Gemini structured output), Pydantic v2, `python-dotenv`, pytest.

## Installation

From this directory, inside an activated Python 3.13 environment:

```text
pip install -r requirements.txt
```

## Environment variables

Copy the root `.env.example` to the root `.env` (one level above this folder) and set:

| Variable | Required | Default |
| --- | --- | --- |
| `GOOGLE_API_KEY` | yes | none |
| `GEMINI_MODEL` | no | `gemini-2.5-flash` |

## Example usage

```text
python main.py sample_inputs/sample_resume.txt --kind resume
python main.py sample_inputs/sample_invoice.txt --kind invoice
```

Expected output is formatted JSON: the resume should contain Priya Nair's name, email, phone, education, her Bright Labs internship and her skills; the invoice should contain `INV-1042`, two line items, and total `177.0`. Missing fields are `null` or `[]`, never invented. (This expected output follows from the prompt rules and the sample files; it has not been confirmed in a live Gemini run.)

## Testing

```text
pytest -q
```

The tests need no API key. They cover the schemas, prompt building, and `extract_document` with a fake Gemini client: valid resume and invoice replies, invalid JSON, wrong document type, wrong field types, empty replies, API failures, a missing key, an unknown document kind, and the command-line exit code. They also check offline that the Gemini SDK accepts both Pydantic models as a response schema.

Not covered: the quality of real Gemini output, which needs a live key.
