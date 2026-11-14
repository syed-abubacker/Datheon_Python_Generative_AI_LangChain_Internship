"""Reusable prompts for Task 1."""

RESUME_PROMPT = """Extract a resume into the supplied JSON schema.
Rules: use only facts present in the text; use null for a missing scalar and [] for a missing list;
do not infer dates, employers, contact details, or skills.

Resume text:
{document_text}
"""

INVOICE_PROMPT = """Extract an invoice into the supplied JSON schema.
Rules: use only facts present in the text; use null for a missing scalar and [] for a missing list.
For each line item, preserve the description. Parse numeric amounts without currency symbols.
Do not calculate missing totals or invent an invoice number.

Invoice text:
{document_text}
"""


def build_prompt(kind: str, document_text: str) -> str:
    """Return a parameterized prompt for one supported document type."""
    prompts = {"resume": RESUME_PROMPT, "invoice": INVOICE_PROMPT}
    try:
        return prompts[kind].format(document_text=document_text)
    except KeyError as error:
        raise ValueError("kind must be 'resume' or 'invoice'") from error
