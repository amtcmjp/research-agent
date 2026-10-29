from langchain.messages import AIMessage

from langgraph_version.agent import research_agent


def main() -> None:
    company = "稲川会"
    person = "内堀 和也"

    prompt = f"""
Research the following company using public sources.

Company: {company}
Associated person: {person}

Search for relevant Japanese and English sources.
Use tavily_search when more evidence is needed.

Return exactly one line in the required output format.
"""

    try:
        result = research_agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={
                "recursion_limit": 15,
            },
        )
    except Exception as exc:
        print(f"Research failed: {type(exc).__name__}: {exc}")
        return

    messages = result.get("messages", [])

    final_message = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AIMessage) and not message.tool_calls
        ),
        None,
    )

    if final_message is None:
        print("Research failed: No final answer returned.")
        return

    print("\n=== LangGraph Result ===\n")
    print(final_message.content)


if __name__ == "__main__":
    main()
