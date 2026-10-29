"""Deep research agent for OSINT customer batch processing."""

from research_agent.agent import build_agent
from research_agent.main import run_batch, run_customer
from research_agent.models import (
    BadDataReport,
    CustomerInput,
    CustomerReport,
    Source,
)
from research_agent.settings import ResearchSettings

__all__ = [
    "BadDataReport",
    "CustomerInput",
    "CustomerReport",
    "ResearchSettings",
    "Source",
    "build_agent",
    "run_batch",
    "run_customer",
]
