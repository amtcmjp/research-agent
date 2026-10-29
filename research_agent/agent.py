"""Deep research agent built on deepagents + Tavily + AWS Bedrock."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from botocore.config import Config as BotoConfig
from deepagents import SubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

from research_agent.prompts import (
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SOCIAL_PROFILE_FINDER_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from research_agent.tools import tavily_search, think_tool

load_dotenv()

MAX_CONCURRENT_RESEARCH_UNITS = int(os.getenv("RESEARCH_MAX_CONCURRENT_UNITS", "3"))
MAX_RESEARCHER_ITERATIONS = int(os.getenv("RESEARCH_MAX_ITERATIONS", "3"))


def _build_model() -> ChatBedrockConverse:
    boto_config = BotoConfig(
        read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "300")),
        connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "15")),
        retries={"max_attempts": 5, "mode": "adaptive"},
    )
    return ChatBedrockConverse(
        model=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.getenv("AWS_REGION", "ap-northeast-1"),
        temperature=0,
        config=boto_config,
    )


def _build_orchestrator_instructions() -> str:
    return (
        RESEARCH_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
            max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
        )
    )


def _build_research_sub_agent() -> SubAgent:
    current_date = datetime.now().strftime("%Y-%m-%d")
    sub_agent: SubAgent = {
        "name": "research-agent",
        "description": (
            "Delegate research to the sub-agent researcher. Only give "
            "this researcher one topic at a time."
        ),
        "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
        "tools": [tavily_search, think_tool],
    }
    return sub_agent


def _build_social_profile_sub_agent() -> SubAgent:
    current_date = datetime.now().strftime("%Y-%m-%d")
    sub_agent: SubAgent = {
        "name": "social-profile-agent",
        "description": (
            "Find social-media profiles that plausibly belong to the "
            "representative. Returns candidate profile URLs with "
            "identity-confidence labels. Give this sub-agent the "
            "person's full name + company name as the one-shot "
            "instruction."
        ),
        "system_prompt": SOCIAL_PROFILE_FINDER_INSTRUCTIONS.format(date=current_date),
        "tools": [tavily_search, think_tool],
    }
    return sub_agent


def build_agent() -> Any:
    """Build the deep research agent."""
    return create_deep_agent(
        model=_build_model(),
        tools=[tavily_search, think_tool],
        system_prompt=_build_orchestrator_instructions(),
        subagents=[_build_research_sub_agent()],
    )


__all__ = ["build_agent"]
