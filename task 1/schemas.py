"""Pydantic models used to validate Gemini structured output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    company: str | None = None
    title: str | None = None
    duration: str | None = None


class ResumeData(BaseModel):
    document_type: Literal["resume"] = "resume"
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)


class InvoiceLineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class InvoiceData(BaseModel):
    document_type: Literal["invoice"] = "invoice"
    invoice_number: str | None = None
    vendor: str | None = None
    customer: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
