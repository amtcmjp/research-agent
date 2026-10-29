"""Pydantic contracts for the research_agent OSINT batch flow."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CustomerInput(BaseModel):
    """Validated candidate parsed from one customers-JSON record."""

    model_config = ConfigDict(str_strip_whitespace=True)

    case_id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=200)
    company_name: str = Field(min_length=1, max_length=300)
    company_address: str = Field(default="", max_length=500)
    sns_url: str = Field(default="", max_length=1000)

    @field_validator("name", "company_name")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "CustomerInput":
        """Build a validated input from a raw customers-JSON record."""
        rep = record.get("representative") or {}
        company = record.get("company") or {}
        customer_id = (
            record.get("customer_id")
            or f"idx_{record.get('id', 'unknown')}"
        )
        address_parts = [
            company.get("zip") or "",
            company.get("prefecture") or "",
            company.get("address") or "",
        ]
        address = " ".join(p for p in address_parts if p).strip()
        return cls(
            case_id=f"cust_{customer_id}",
            name=rep.get("name") or "",
            company_name=company.get("name") or "",
            company_address=address,
            sns_url=rep.get("sns_url") or "",
        )

    @model_validator(mode="after")
    def _default_address(self) -> "CustomerInput":
        if not self.company_address:
            # Pydantic frozen=False, so direct assignment is fine.
            object.__setattr__(self, "company_address", "Unknown")
        return self


class Source(BaseModel):
    """One cited HIGH-RISK source for the bad-data report."""

    title: str = Field(min_length=1, max_length=400)
    url: str = Field(min_length=1, max_length=1000)
    description: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "One or two neutral sentences describing the adverse "
            "finding at this source. Must be grounded in the "
            "research notes, never invented."
        ),
    )
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Per-source adverse-risk score (0-100). Weight confirmed "
            "events and official findings higher than opinions. Only "
            "sources at or above the HIGH-RISK threshold should "
            "appear in the final BadDataReport."
        ),
    )


class BadDataReport(BaseModel):
    """Structured 'bad data' outcome for one customer."""

    score: int = Field(
        ge=0,
        le=100,
        description=(
            "0-100 overall adverse-information risk score. "
            "0 means limited or no documented concerns; "
            "100 means extensive severe well-supported concerns."
        ),
    )
    summary: str = Field(
        max_length=800,
        description=(
            "Two or three neutral sentences summarizing the risk "
            "picture. MUST stay under 700 characters."
        ),
    )
    sources: list[Source] = Field(
        default_factory=list,
        description=(
            "Every unique source actually cited in the research. "
            "Each entry has title + direct URL. Deduplicate URLs."
        ),
    )


class CustomerReport(BaseModel):
    """Per-customer envelope returned by the batch pipeline."""

    case_id: str
    name: str
    company_name: str
    company_address: str
    report: BadDataReport
    duration_ms: int = Field(
        default=0,
        ge=0,
        description="Wall-clock time to run this case, in milliseconds.",
    )
    agent_failed: bool = Field(
        default=False,
        description=(
            "True when the deep-agent invocation raised; the report "
            "falls back to a safe default and this flag warns the "
            "downstream consumer."
        ),
    )
    completed_at: str = Field(default_factory=_utcnow_iso)
