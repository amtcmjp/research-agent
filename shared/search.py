import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"


def run_tavily_search(query: str) -> str:
    """Search public web sources using the Tavily API."""

    api_key = os.environ.get("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    print(f"\n[TOOL CALL] tavily_search: {query}")

    response = requests.post(
        TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": 8,
            "include_answer": False,
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score"),
        }
        for item in data.get("results", [])
    ]

    return json.dumps(
        {
            "query": query,
            "results": results,
        },
        ensure_ascii=False,
    )
