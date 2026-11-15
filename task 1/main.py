"""Command-line structured extractor using Gemini and Pydantic."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TypeAlias

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

from prompts import build_prompt
from schemas import InvoiceData, ResumeData

DocumentModel: TypeAlias = type[ResumeData] | type[InvoiceData]


class ExtractionError(RuntimeError):
    """A clear, user-facing failure while extracting a document."""


def schema_for(kind: str) -> DocumentModel:
    schemas: dict[str, DocumentModel] = {"resume": ResumeData, "invoice": InvoiceData}
    try:
        return schemas[kind]
    except KeyError as error:
        raise ExtractionError("Document kind must be 'resume' or 'invoice'.") from error


def extract_document(text: str, kind: str, client: genai.Client | None = None) -> BaseModel:
    """Call Gemini once and validate its JSON against the selected Pydantic model."""
    if not text.strip():
        raise ExtractionError("The input document is empty.")
    model_schema = schema_for(kind)
    api_key = os.getenv("GOOGLE_API_KEY")
    if client is None:
        if not api_key:
            raise ExtractionError("GOOGLE_API_KEY is missing. Copy .env.example to .env and set it.")
        client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            contents=build_prompt(kind, text),
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=model_schema, temperature=0
            ),
        )
        if not response.text:
            raise ExtractionError("Gemini returned an empty response.")
        return model_schema.model_validate_json(response.text)
    except ValidationError as error:
        raise ExtractionError(f"Gemini returned JSON that does not match the {kind} schema: {error}") from error
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError(f"Gemini request failed: {error}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract validated JSON from a resume or invoice.")
    parser.add_argument("input_file", type=Path, help="Path to a UTF-8 text file")
    parser.add_argument("--kind", choices=("resume", "invoice"), required=True)
    return parser.parse_args()


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = parse_args()
    try:
        text = args.input_file.read_text(encoding="utf-8")
        result = extract_document(text, args.kind)
    except (OSError, UnicodeDecodeError, ExtractionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
