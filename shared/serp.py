"""SerpAPI Google Images client used as an OSINT reference-photo tool."""

from __future__ import annotations

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

SERPAPI_URL = "https://serpapi.com/search.json"


def _trim_image(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": item.get("title", "") or "",
        "source": item.get("source", "") or "",
        "source_name": item.get("source_name", "") or "",
        "link": item.get("link", "") or "",
        "original": item.get("original", "") or "",
        "thumbnail": item.get("thumbnail", "") or "",
    }


def search_google_images(
    query: str,
    limit: int = 10,
    timeout: int = 30,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Search Google Images via SerpAPI; return trimmed JSON string."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not configured.")

    print(f"\n[TOOL CALL] serpapi_image_search: {query}")

    params: dict[str, Any] = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
        "ijn": 0,
    }
    if extra_params:
        params.update(extra_params)

    response = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    images = data.get("images_results") or []

    trimmed = [_trim_image(item) for item in images[:limit]]

    return json.dumps(
        {"query": query, "count": len(trimmed), "images": trimmed},
        ensure_ascii=False,
    )
