"""Runtime configuration for the research_agent batch pipeline.

Everything the production entry point needs to know, pulled from
env in one place so tests can override by passing a Settings
instance instead of monkey-patching globals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class ResearchSettings:
    """Immutable settings for a batch run."""

    recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("RESEARCH_RECURSION_LIMIT", "150"))
    )
    high_risk_threshold: int = field(
        default_factory=lambda: int(os.getenv("RESEARCH_HIGH_RISK_THRESHOLD", "50"))
    )
    customers_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "RESEARCH_CUSTOMERS_PATH",
                str(_project_root() / "customers_coremobile.json"),
            )
        )
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("RESEARCH_LOG_LEVEL", "WARNING")
    )
