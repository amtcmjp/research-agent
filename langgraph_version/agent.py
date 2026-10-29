import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse

from shared.instructions import RESEARCH_INSTRUCTIONS
from shared.search import run_tavily_search

load_dotenv()


@tool
def tavily_search(query: str) -> str:
    """Search public web sources for a specific company or issue.

    Args:
        query: A focused search query, including Japanese terms
            when relevant.
    """
    return run_tavily_search(query)


model = ChatBedrockConverse(
    model=os.environ["BEDROCK_MODEL_ID"],
    region_name=os.getenv("AWS_REGION", "ap-northeast-1"),
    temperature=0,
)

research_agent = create_agent(
    model=model,
    tools=[tavily_search],
    system_prompt=RESEARCH_INSTRUCTIONS,
)
