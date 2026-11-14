# Task 1 — Structured LLM Prompt Pipeline

## Objective

Convert an unstructured resume or invoice into validated JSON. Gemini is instructed to produce JSON that matches a Pydantic schema, then the result is validated again locally before it is printed.

## Main files

- `main.py` — command-line entry point, Gemini call, error handling, and Pydantic validation.
- `prompts.py` — parameterized resume and invoice prompts.
- `schemas.py` — explicit data contracts.
- `sample_inputs/` — safe, fictional input examples.

## Install and run

From this directory, after activating a Python 3.12/3.13 environment:

```powershell
pip install -r requirements.txt
python main.py sample_inputs/sample_resume.txt --kind resume
python main.py sample_inputs/sample_invoice.txt --kind invoice
pytest -q
```

Copy the root `.env.example` to the root `.env` and set `GOOGLE_API_KEY` first. Expected output is formatted JSON: the resume includes Priya's name and skills; the invoice includes `INV-1042`, two line items, and total `177.0`. Missing fields are `null` or `[]`, never invented.

## Concepts

Gemini structured output (`response_mime_type="application/json"` plus a Pydantic response schema), prompt rules, and local validation. Network, authentication, malformed JSON, empty input, and missing-file failures produce an understandable error.
