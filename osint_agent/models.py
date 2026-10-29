"""Pydantic contracts for the OSINT membership-vetting workflow.

 - CandidateInput: validated inputs from the membership intake form.
 - EvidenceEntry: a single search outcome, time-stamped.
 - FaceComparison / FaceVerification: AWS Rekognition identity signal.
 - Finding / OsintReport: scorer output; the risk score + findings.
 - OsintCaseRecord: the end-to-end audit record persisted on disk.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _case_id() -> str:
    return f"case_{uuid.uuid4().hex[:12]}"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


FindingCategory = Literal[
    "fraud",
    "legal",
    "insolvency",
    "complaints",
    "employment",
    "breach",
    "regulatory",
    "press",
    "identity",
    "other",
]

FindingClassification = Literal[
    "review_opinion",
    "allegation",
    "confirmed_event",
]

IdentityConfidence = Literal["high", "medium", "low"]

Recommendation = Literal[
    "approve_candidate",
    "needs_scrutiny",
    "reject_candidate",
]

FaceVerificationStatus = Literal[
    "verified",
    "face_only",
    "mismatch",
    "no_face",
    "error",
    "skipped",
]

QualityLabel = Literal["high", "medium", "low", "unknown"]

EvidenceSource = Literal["tavily", "apify", "serpapi", "rekognition"]


class CandidateInput(BaseModel):
    """Validated applicant data submitted by the intake form."""

    model_config = ConfigDict(str_strip_whitespace=True)

    case_id: str = Field(default_factory=_case_id)
    name: str = Field(min_length=1, max_length=200)
    company_name: str = Field(min_length=1, max_length=300)
    company_address: str = Field(min_length=1, max_length=500)
    id_front_url: str = Field(default="", max_length=1000)
    id_back_url: str = Field(default="", max_length=1000)
    sns_url: str = Field(default="", max_length=1000)
    submitted_at: str = Field(default_factory=_utcnow_iso)
    submitted_by: str = Field(
        default="intake",
        max_length=120,
        description="Account / system that submitted the application.",
    )

    @field_validator("name", "company_name", "company_address")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value


class EvidenceEntry(BaseModel):
    """One tool call (search / crawl / vision) and its raw result."""

    source: EvidenceSource = "tavily"
    query: str
    result: str
    searched_at: str = Field(default_factory=_utcnow_iso)


class FaceComparison(BaseModel):
    """One Rekognition CompareFaces result against a reference image."""

    reference_url: str
    similarity: float = Field(ge=0.0, le=100.0, default=0.0)
    matched: bool = False
    note: str = ""


class FaceVerification(BaseModel):
    """Rekognition-based identity signal for the candidate's ID photo."""

    status: FaceVerificationStatus
    id_face_detected: bool = False
    id_face_confidence: float = Field(ge=0.0, le=100.0, default=0.0)
    id_face_quality: QualityLabel = "unknown"
    sunglasses: bool = False
    comparisons: list[FaceComparison] = Field(default_factory=list)
    error: str = ""
    verified_at: str = Field(default_factory=_utcnow_iso)


class Finding(BaseModel):
    """A single adverse item extracted from evidence."""

    category: FindingCategory
    classification: FindingClassification
    summary: str = Field(description="One-sentence neutral summary.")
    source_url: str = Field(default="", description="Primary source URL if any.")
    source_date: str = Field(default="", description="Publication date if any.")
    identity_confidence: IdentityConfidence = Field(
        description="Confidence that this item refers to the target."
    )


class OsintReport(BaseModel):
    """Scorer output: 0-100 risk score plus structured findings."""

    risk_score: int = Field(ge=0, le=100, description="0-100 overall risk score.")
    short_description: str = Field(
        max_length=800,
        description=(
            "Two or three neutral sentences summarizing the risk picture. "
            "Keep well under 800 characters."
        ),
    )
    findings: list[Finding] = Field(default_factory=list)
    coverage_notes: str = Field(
        default="",
        description="Identity-match confidence and coverage gaps.",
    )
    scored_at: str = Field(default_factory=_utcnow_iso)


class OsintCaseRecord(BaseModel):
    """Persisted end-to-end audit record for a single vetting case."""

    case_id: str
    candidate: CandidateInput
    report: OsintReport
    system_recommendation: Recommendation
    face_verification: FaceVerification | None = None
    evidence: list[EvidenceEntry]
    query_history: list[str]
    search_count: int
    completed_at: str = Field(default_factory=_utcnow_iso)
