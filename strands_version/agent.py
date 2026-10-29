import os

from dotenv import load_dotenv
from strands import tool
from strands_harness import create_harness

from shared.instructions import RESEARCH_INSTRUCTIONS
from shared.search import run_tavily_search

load_dotenv()

os.environ.setdefault(
    "AWS_DEFAULT_REGION",
    os.getenv("AWS_REGION", "ap-northeast-1"),
)


@tool
def tavily_search(query: str) -> str:
    """Search public web sources for a specific company or issue.

    Args:
        query: A focused search query, including Japanese terms
            when relevant.
    """
    return run_tavily_search(query)


research_agent = create_harness(
    model=os.environ["BEDROCK_MODEL_ID"],
    instructions=RESEARCH_INSTRUCTIONS,
    tools=[tavily_search],
    builtin_tools=[],
    builtin_plugins=[],
    effort="low",
)
