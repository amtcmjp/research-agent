from strands_version.agent import research_agent


def main() -> None:
    company = "稲川会"
    person = "内堀 和也"

    prompt = f"""
Research the following company using public sources.

Company: {company}
Associated person: {person}

Search for relevant Japanese and English sources.
Follow your research instructions and use tavily_search
when more evidence is needed.

Return exactly one line in the required output format.
"""

    try:
        result = research_agent(prompt)
    except Exception as exc:
        print(f"Research failed: {type(exc).__name__}: {exc}")
        return

    print("\n=== Strands Harness Result ===\n")

    message = getattr(result, "message", None)

    if isinstance(message, dict):
        text_blocks = [
            block["text"]
            for block in message.get("content", [])
            if isinstance(block, dict) and "text" in block
        ]
        print("\n".join(text_blocks) or str(result))
    else:
        print(str(result))


if __name__ == "__main__":
    main()
