"""Apify social-media crawler client used as an OSINT tool."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

APIFY_URL_TEMPLATE = (
    "https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
)
DEFAULT_ACTOR_ID = "apify/google-search-scraper"


def _build_payload(actor_id: str, query: str) -> dict[str, Any]:
    """Shape the payload for a few common Apify actors."""
    if "google-search-scraper" in actor_id:
        return {
            "queries": query,
            "maxPagesPerQuery": 1,
            "resultsPerPage": 10,
            "languageCode": "ja",
            "countryCode": "jp",
        }

    if "instagram-scraper" in actor_id:
        return {"search": query, "searchType": "user", "resultsLimit": 10}

    if "twitter-scraper" in actor_id or "x-scraper" in actor_id:
        return {"searchTerms": [query], "maxItems": 10}

    return {"query": query}


def _trim_item(item: Any) -> Any:
    """Keep only commonly useful fields; truncate large text blobs."""
    if not isinstance(item, dict):
        return item

    keep = {
        "title",
        "url",
        "displayedUrl",
        "description",
        "snippet",
        "username",
        "fullName",
        "biography",
        "profileUrl",
        "profilePicUrl",
        "profile_image_url",
        "externalUrl",
        "followersCount",
        "postsCount",
        "verified",
        "isPrivate",
        "text",
    }

    trimmed: dict[str, Any] = {}
    for key, value in item.items():
        if key in keep and value is not None:
            if isinstance(value, str) and len(value) > 500:
                value = value[:500] + "…"
            trimmed[key] = value

    if not trimmed:
        return item

    return trimmed


def run_apify_social_search(
    query: str,
    actor_id: str | None = None,
    max_items: int = 15,
    timeout: int = 180,
) -> str:
    """Run the configured Apify actor against `query` and return trimmed JSON."""
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        raise RuntimeError("APIFY_API_TOKEN is not configured.")

    resolved_actor = actor_id or os.getenv("APIFY_ACTOR_ID", DEFAULT_ACTOR_ID)
    actor_slug = resolved_actor.replace("/", "~")
    url = APIFY_URL_TEMPLATE.format(actor=actor_slug)

    print(f"\n[TOOL CALL] apify_social_search: {query} (actor={resolved_actor})")

    response = requests.post(
        url,
        params={"token": token},
        json=_build_payload(resolved_actor, query),
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    items = data if isinstance(data, list) else [data]
    trimmed = [_trim_item(item) for item in items[:max_items]]

    return json.dumps(
        {"query": query, "actor": resolved_actor, "items": trimmed},
        ensure_ascii=False,
    )
