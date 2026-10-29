"""LangChain tools used by the deep-agent OSINT orchestrator + subagents.

Tools share a per-case `Runtime` closure that enforces hard
budget caps across the entire investigation (not per iteration —
the deep agent manages its own loop).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, TypedDict

from langchain.tools import tool

from osint_agent.config import MAX_FACE_COMPARES, MAX_SEARCHES
from osint_agent.models import FaceComparison
from shared.apify import run_apify_social_search
from shared.rekognition import compare_faces as rekognition_compare
from shared.search import run_tavily_search
from shared.serp import search_google_images

logger = logging.getLogger("osint_agent.tools")


class EvidenceEntryDict(TypedDict):
    source: str
    query: str
    result: str
    searched_at: str


class Runtime(TypedDict):
    total_searches: int
    face_compares: int
    known_queries: set[str]
    evidence: list[EvidenceEntryDict]
    comparisons: list[FaceComparison]
    id_front_url: str


def make_runtime(id_front_url: str = "") -> Runtime:
    return {
        "total_searches": 0,
        "face_compares": 0,
        "known_queries": set(),
        "evidence": [],
        "comparisons": [],
        "id_front_url": id_front_url,
    }


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_tools(runtime: Runtime) -> dict[str, Any]:
    """Return every tool keyed by name. The deep agent lists tools
    directly; subagents receive their relevant subset."""

    @tool
    def tavily_search(query: str) -> str:
        """Search public web sources for OSINT adverse findings."""
        normalized = " ".join(query.split()).strip()
        if not normalized:
            return "Search skipped: empty query."
        if normalized.casefold() in runtime["known_queries"]:
            return (
                "DUPLICATE_QUERY: This query was already searched. "
                "Pick a different angle."
            )
        if runtime["total_searches"] >= MAX_SEARCHES:
            return "SEARCH_BUDGET_EXHAUSTED: Do not search again."

        runtime["total_searches"] += 1
        runtime["known_queries"].add(normalized.casefold())

        logger.info(
            "tavily %d/%d: %s",
            runtime["total_searches"],
            MAX_SEARCHES,
            normalized,
        )

        try:
            result = str(run_tavily_search(normalized))
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            logger.warning("tavily failed: %s: %s", type(exc).__name__, exc)
            result = f"Search error: {type(exc).__name__}: {exc}"

        runtime["evidence"].append(
            {
                "source": "tavily",
                "query": normalized,
                "result": result,
                "searched_at": _utcnow_iso(),
            }
        )
        return result

    @tool
    def apify_social_search(query: str) -> str:
        """Crawl social media / search engines via Apify for OSINT signals."""
        normalized = " ".join(query.split()).strip()
        if not normalized:
            return "Search skipped: empty query."
        key = f"apify::{normalized}".casefold()
        if key in runtime["known_queries"]:
            return (
                "DUPLICATE_QUERY: This Apify query was already run. "
                "Pick a different angle."
            )
        if runtime["total_searches"] >= MAX_SEARCHES:
            return "SEARCH_BUDGET_EXHAUSTED: Do not search again."

        runtime["total_searches"] += 1
        runtime["known_queries"].add(key)

        logger.info(
            "apify %d/%d: %s",
            runtime["total_searches"],
            MAX_SEARCHES,
            normalized,
        )

        try:
            result = run_apify_social_search(normalized)
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            logger.warning("apify failed: %s: %s", type(exc).__name__, exc)
            result = f"Apify error: {type(exc).__name__}: {exc}"

        runtime["evidence"].append(
            {
                "source": "apify",
                "query": normalized,
                "result": result,
                "searched_at": _utcnow_iso(),
            }
        )
        return result

    @tool
    def serpapi_image_search(query: str) -> str:
        """Google Images via SerpAPI for reference-photo discovery.

        Returns candidate images; feed `original` URLs to
        compare_face_to_reference.
        """
        normalized = " ".join(query.split()).strip()
        if not normalized:
            return "Search skipped: empty query."
        key = f"serp::{normalized}".casefold()
        if key in runtime["known_queries"]:
            return (
                "DUPLICATE_QUERY: This image search was already run. "
                "Pick a different angle."
            )
        if runtime["total_searches"] >= MAX_SEARCHES:
            return "SEARCH_BUDGET_EXHAUSTED: Do not search again."

        runtime["total_searches"] += 1
        runtime["known_queries"].add(key)

        logger.info(
            "serpapi images %d/%d: %s",
            runtime["total_searches"],
            MAX_SEARCHES,
            normalized,
        )

        try:
            result = search_google_images(normalized)
        except Exception as exc:  # noqa: BLE001 - never crash the agent loop
            logger.warning("serpapi failed: %s: %s", type(exc).__name__, exc)
            result = f"SerpAPI error: {type(exc).__name__}: {exc}"

        runtime["evidence"].append(
            {
                "source": "serpapi",
                "query": normalized,
                "result": result,
                "searched_at": _utcnow_iso(),
            }
        )
        return result

    @tool
    def compare_face_to_reference(reference_image_url: str) -> str:
        """Compare the candidate's ID photo against a reference image URL.

        Pass a DIRECT image URL (jpg/png). The `original` field from
        serpapi_image_search is ideal.
        """
        reference = reference_image_url.strip()
        if not reference:
            return "FACE_COMPARE_SKIPPED: empty reference URL."
        if not runtime["id_front_url"]:
            return "FACE_COMPARE_UNAVAILABLE: no ID photo on file."
        if runtime["face_compares"] >= MAX_FACE_COMPARES:
            return "FACE_COMPARE_BUDGET_EXHAUSTED: no more compares allowed."

        runtime["face_compares"] += 1

        logger.info(
            "rekognition compare %d/%d: ref=%s",
            runtime["face_compares"],
            MAX_FACE_COMPARES,
            reference,
        )

        cmp = rekognition_compare(runtime["id_front_url"], reference)
        comparison = FaceComparison(
            reference_url=reference,
            similarity=cmp["similarity"],
            matched=cmp["matched"],
            note=cmp.get("error", ""),
        )
        runtime["comparisons"].append(comparison)

        runtime["evidence"].append(
            {
                "source": "rekognition",
                "query": f"compare_faces → {reference}",
                "result": json.dumps(cmp, ensure_ascii=False),
                "searched_at": _utcnow_iso(),
            }
        )
        return json.dumps(cmp, ensure_ascii=False)

    return {
        "tavily_search": tavily_search,
        "apify_social_search": apify_social_search,
        "serpapi_image_search": serpapi_image_search,
        "compare_face_to_reference": compare_face_to_reference,
    }
