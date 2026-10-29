"""Environment-driven configuration for the OSINT agent."""

from __future__ import annotations

import os
from pathlib import Path

from osint_agent.models import Recommendation

MAX_SEARCHES = int(os.getenv("OSINT_MAX_SEARCHES", "12"))
MAX_FACE_COMPARES = int(os.getenv("OSINT_MAX_FACE_COMPARES", "3"))
RECURSION_LIMIT = int(os.getenv("OSINT_RECURSION_LIMIT", "60"))
INNER_AGENT_RECURSION = int(os.getenv("OSINT_INNER_RECURSION", "25"))

AUDIT_DIR = Path(os.getenv("OSINT_AUDIT_DIR", "output/osint_cases"))
DEFAULT_CUSTOMERS_PATH = Path(
    os.getenv(
        "OSINT_CUSTOMERS_PATH",
        str(Path(__file__).resolve().parent.parent / "customers260717123316.json"),
    )
)

APPROVE_MAX_SCORE = int(os.getenv("OSINT_APPROVE_MAX", "19"))
SCRUTINY_MAX_SCORE = int(os.getenv("OSINT_SCRUTINY_MAX", "59"))


def recommend_from_score(score: int) -> Recommendation:
    """Map a 0-100 risk score to a system recommendation band."""
    if score <= APPROVE_MAX_SCORE:
        return "approve_candidate"
    if score <= SCRUTINY_MAX_SCORE:
        return "needs_scrutiny"
    return "reject_candidate"
