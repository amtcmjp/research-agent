from __future__ import annotations

import json
import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel, Field

from shared.search import run_tavily_search

load_dotenv()

MAX_SEARCHES = 12
RECURSION_LIMIT = 60

FindingCategory = Literal[
    "professional_association",
    "negative_review",
    "customer_complaint",
    "legal",
    "regulatory",
    "business_information",
    "other",
]

FindingClassification = Literal[
    "review_opinion",
    "allegation",
    "confirmed_event",
]

Confidence = Literal["high", "medium", "low"]


# ---------------------------------------------------------
# 1. Structured output models
# ---------------------------------------------------------


class ResearchFinding(BaseModel):
    subject_scope: Literal["person_professional", "company"]

    category: FindingCategory

    classification: FindingClassification

    summary: str = Field(
        description="Concise summary supported by the collected evidence."
    )

    source_url: str = Field(
        description="Exact source URL found in the collected search evidence."
    )

    source_date: str = Field(
        default="", description="Publication date, if available. Empty if unknown."
    )

    identity_confidence: Confidence = Field(
        description=(
            "Confidence that the source refers to the intended person or "
            "company, based on available identifying context."
        )
    )

    evidence_note: str = Field(
        default="",
        description="Short note explaining the evidence and its limitations.",
    )


class ResearchReport(BaseModel):
    person_name: str
    company_name: str
    submitted_company_address: str = ""

    person_company_association: str = Field(
        description=(
            "Summary of whether public professional sources support an "
            "association between the named person and the company."
        )
    )

    association_confidence: Confidence

    company_reputation_summary: str

    findings: list[ResearchFinding] = Field(default_factory=list)

    limitations: list[str] = Field(default_factory=list)

    searches_performed: int = 0


# ---------------------------------------------------------
# 2. Model setup
# ---------------------------------------------------------


def create_model() -> ChatBedrockConverse:
    model_id = os.getenv("BEDROCK_MODEL_ID")

    if not model_id:
        raise ValueError("BEDROCK_MODEL_ID is missing. Add it to your .env file.")

    return ChatBedrockConverse(
        model=model_id,
        region_name=os.getenv("AWS_REGION", "ap-northeast-1"),
        temperature=0,
    )


# ---------------------------------------------------------
# 3. Research function
# ---------------------------------------------------------


def research_company(
    person_name: str,
    company_name: str,
    company_address: str = "",
) -> ResearchReport:

    person_name = person_name.strip()
    company_name = company_name.strip()
    company_address = company_address.strip()

    if not person_name:
        raise ValueError("person_name cannot be empty.")

    if not company_name:
        raise ValueError("company_name cannot be empty.")

    # State is isolated to this research run.
    search_state: dict[str, object] = {
        "count": 0,
        "queries": set(),
        "evidence": [],
    }

    model = create_model()

    # -----------------------------------------------------
    # 3a. Tavily search tool
    # -----------------------------------------------------

    @tool
    def tavily_search(query: str) -> str:
        """Search public websites for company information, company reviews,
        documented business issues, and public professional information
        relevant to verifying a person's association with a company.

        Do not search for private personal information, personal ID numbers,
        home addresses, or individual creditworthiness.
        """

        normalized_query = " ".join(query.split()).strip()

        if not normalized_query:
            return "Search skipped: the query is empty."

        query_key = normalized_query.casefold()

        queries = search_state["queries"]
        evidence = search_state["evidence"]

        assert isinstance(queries, set)
        assert isinstance(evidence, list)

        if query_key in queries:
            return "Duplicate query. Choose a different, focused search query."

        count = search_state["count"]
        assert isinstance(count, int)

        if count >= MAX_SEARCHES:
            return "Search budget exhausted. Summarize the evidence collected."

        queries.add(query_key)
        search_state["count"] = count + 1

        current_count = count + 1

        print(f"\n[Search {current_count}/{MAX_SEARCHES}] " f"{normalized_query}")

        try:
            result = str(run_tavily_search(normalized_query))
        except Exception as exc:
            # Avoid exposing credentials or internal exception details.
            result = (
                f"Search failed: {type(exc).__name__}. "
                "No reliable evidence was collected for this query."
            )

        evidence.append(
            {
                "query": normalized_query,
                "result": result,
            }
        )

        return result

    # -----------------------------------------------------
    # 3b. Deep Agents research instructions
    # -----------------------------------------------------

    system_prompt = """
You are a careful public-source business research assistant.

Your task is to investigate a company's public reputation and verify
whether public professional sources support an association between a
named person and that company.

RESEARCH RULES

1. Use tavily_search to search the public web.
2. Use focused, distinct queries. Never repeat the same query.
3. Search the company name in its original language and spelling.
4. Search for company reviews, complaints, public legal or regulatory
   information, and public professional information relevant to the
   person's association with the company.
5. When relevant, use the supplied company address to distinguish
   companies with similar names. Do not search for a person's private
   home address.
6. Do not assume that two people with the same name are the same person.
7. Do not claim that a person works for or owns a company unless the
   collected sources support that association.
8. Do not infer misconduct from a person's name, nationality, address,
   employer, or lack of search results.
9. Distinguish subjective reviews, allegations, and confirmed events.
10. Treat an allegation as an allegation unless a reliable source
    establishes the underlying event.
11. Prefer original records and reputable sources over copied snippets.
12. Never invent a source, URL, publication date, quotation, or finding.
13. Do not investigate private personal information, government ID
    numbers, private financial records, or unrelated personal history.
14. Do not create an individual creditworthiness score, fraud probability,
    or credit approval/rejection recommendation.
15. Search no more than the tool's configured budget allows.

SEARCH STRATEGY

Choose one useful search query at a time. Cover relevant research angles
without repeating queries. Stop when the evidence is sufficient or the
search budget is exhausted.

When finished, provide a concise research summary based only on the
sources actually found. State clearly when the evidence is insufficient.
"""

    agent = create_deep_agent(
        model=model,
        tools=[tavily_search],
        system_prompt=system_prompt,
    )

    # -----------------------------------------------------
    # 3c. Run research
    # -----------------------------------------------------

    request = f"""
Research the company and its public professional association
with the named person.

PERSON NAME:
{person_name}

COMPANY NAME:
{company_name}

SUBMITTED COMPANY ADDRESS:
{company_address or "Not provided"}

Please investigate these areas:

A. PERSON-TO-COMPANY ASSOCIATION
- Look for public professional profiles, company pages, official
  announcements, or other reliable sources linking the person to
  the named company.
- Check whether the sources contain enough context to distinguish
  this person from other people with the same name.
- If the association cannot be established, say so.

B. COMPANY REPUTATION
- Search for public customer reviews and negative reviews.
- Search for customer complaints and documented business issues.
- Search for relevant public legal or regulatory announcements.
- Check the reliability, date, and relevance of each source.

C. EVIDENCE QUALITY
- Record exact source URLs when available.
- Distinguish opinions from allegations and confirmed events.
- Do not treat an absence of negative search results as proof that
  the company has no problems.
- Do not turn personal information into an individual risk score.

Use distinct, focused searches and stay within the search budget.
"""

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request,
                }
            ]
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )

    messages = result.get("messages", [])

    if not messages:
        raise RuntimeError("The research agent returned no messages.")

    # -----------------------------------------------------
    # 3d. Collect evidence for structured report generation
    # -----------------------------------------------------

    evidence = search_state["evidence"]
    assert isinstance(evidence, list)

    search_count = search_state["count"]
    assert isinstance(search_count, int)

    evidence_json = json.dumps(
        evidence,
        ensure_ascii=False,
        indent=2,
    )

    # -----------------------------------------------------
    # 3e. Produce a structured report
    # -----------------------------------------------------

    structured_model = model.with_structured_output(ResearchReport)

    report_instructions = """
Create a structured report from the supplied research evidence.

IMPORTANT EVIDENCE RULES

- Use only the supplied search evidence.
- Every finding's source_url must exactly match a URL present in that
  evidence. If no source URL can be verified, omit the finding.
- Do not invent events, URLs, dates, or quotations.
- If the evidence does not establish a person's association with the
  company, state that the association is unverified.
- Use low identity confidence when the person's identity is ambiguous.
- Do not combine information about different people who share a name.
- Distinguish review_opinion, allegation, and confirmed_event.
- A negative review is not automatically a confirmed event.
- Include material limitations, missing dates, and weak evidence.
- Do not produce a personal risk score or credit approval recommendation.
- Do not infer that a person or company is fraudulent merely because
  a search returned negative information.
- If evidence is insufficient, return fewer findings rather than
  guessing.
- searches_performed must equal the supplied search count.
"""

    report = structured_model.invoke(
        [
            ("system", report_instructions),
            (
                "human",
                f"""
Person name: {person_name}
Company name: {company_name}
Submitted company address: {company_address or "Not provided"}

Searches performed: {search_count}

Collected search evidence:
{evidence_json}
""",
            ),
        ]
    )

    report.searches_performed = search_count

    if search_count >= MAX_SEARCHES:
        budget_note = (
            "The maximum search budget was reached. "
            "Additional relevant sources may exist."
        )
        if budget_note not in report.limitations:
            report.limitations.append(budget_note)

    return report


# ---------------------------------------------------------
# 4. Run the program
# ---------------------------------------------------------

if __name__ == "__main__":
    report = research_company(
        person_name="Htet Wai Yan",
        company_name="株式会社コアモバイル",
        company_address=("〒111-0034 東京都台東区雷門2丁目17番2号 " "NEWNO浅草5階"),
    )

    print("\n" + "=" * 70)
    print("PUBLIC BUSINESS RESEARCH REPORT")
    print("=" * 70)

    print(
        json.dumps(
            report.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )
