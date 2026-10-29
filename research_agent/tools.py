"""Research tools.

Wraps the project's shared Tavily client plus a strategic
`think_tool` for reflective reasoning. Results are returned as
Markdown so the deep research agent can cite them directly.
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from shared.search import run_tavily_search


def _format_tavily_payload(raw: str, query: str) -> str:
    """Convert the shared Tavily JSON payload into Markdown blocks."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return f"🔍 Results for '{query}':\n\n{raw}"

    results = data.get("results") or []
    if not results:
        return f"🔍 No results found for '{query}'."

    blocks: list[str] = []
    for item in results:
        title = item.get("title") or "(no title)"
        url = item.get("url") or ""
        content = (item.get("content") or "").strip()
        blocks.append(
            f"## {title}\n"
            f"**URL:** {url}\n\n"
            f"{content}\n\n"
            "---"
        )

    return (
        f"🔍 Found {len(results)} result(s) for '{query}':\n\n"
        + "\n".join(blocks)
    )


@tool(parse_docstring=True)
def tavily_search(query: str) -> str:
    """Search the web for information on a given query via Tavily.

    Returns formatted Markdown with each result's title, URL, and
    content snippet. The underlying Tavily call uses advanced search
    depth and returns up to 8 results per query.

    Args:
        query: Search query to execute. Prefer specific, focused
            queries; broaden only if the first result set is empty.

    Returns:
        Markdown block with each result's title, URL, and content.
    """
    normalized = " ".join(query.split()).strip()
    if not normalized:
        return "Search skipped: empty query."

    try:
        raw = run_tavily_search(normalized)
    except Exception as exc:  # noqa: BLE001 - never crash the agent loop
        return f"Tavily error: {type(exc).__name__}: {exc}"

    return _format_tavily_payload(raw, normalized)


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps
    systematically. This creates a deliberate pause in the research
    workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence / examples?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress,
            findings, gaps, and next steps.

    Returns:
        Confirmation that reflection was recorded for decision-making.
    """
    return f"Reflection recorded: {reflection}"
