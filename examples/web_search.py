"""Acervo chat with web search — demonstrates the full prepare/process + search flow.

Prerequisites:
    1. Start LM Studio with qwen2.5-3b-instruct loaded
    2. pip install acervo httpx
    3. Get a free Brave Search API key at https://brave.com/search/api/
    4. Set BRAVE_API_KEY environment variable

Run:
    BRAVE_API_KEY=your_key python examples/web_search.py

When Acervo has no data about a topic, it sets prep.needs_tool = True.
This example shows how to handle that by searching the web and feeding
results back to Acervo.
"""

import asyncio
import os

import httpx

from acervo import Acervo, OpenAIClient


async def brave_search(query: str, api_key: str, max_results: int = 5) -> str:
    """Search the web using Brave Search API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("web", {}).get("results", [])
    if not results:
        return ""

    import re
    sections = [f'Web results for: "{query}"\n']
    for i, r in enumerate(results[:max_results], 1):
        title = re.sub(r"<[^>]+>", "", r.get("title", ""))
        desc = re.sub(r"<[^>]+>", "", r.get("description", ""))
        section = f"{i}. {title}"
        if desc:
            section += f"\n   {desc}"
        sections.append(section)
    return "\n\n".join(sections)


async def main() -> None:
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        print("Set BRAVE_API_KEY environment variable to enable web search.")
        print("Get a free key at https://brave.com/search/api/\n")

    llm = OpenAIClient(
        base_url="http://localhost:1234/v1",
        model="qwen2.5-3b-instruct",
        api_key="lm-studio",
    )

    memory = Acervo(llm=llm, owner="User")
    history: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Respond in the same language as the user."},
    ]

    print("Acervo Chat + Web Search (type 'quit' to exit)")
    print(f"Graph: {memory.graph.node_count} nodes, {memory.graph.edge_count} edges")
    print(f"Web search: {'enabled' if api_key else 'disabled (no BRAVE_API_KEY)'}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        # Step 1: Acervo prepares context
        prep = await memory.prepare(user_input, history)

        # Step 2: Handle web search if needed
        web_results = ""
        if prep.needs_tool and api_key:
            search_query = prep.plan.entity or prep.topic
            print(f"  Searching web for: {search_query}...")
            web_results = await brave_search(search_query, api_key)
            if web_results:
                prep.add_web_results(web_results)
                print(f"  Found results!\n")

        # Step 3: Call LLM
        response = await llm.chat(
            prep.context_stack,
            temperature=0.7,
            max_tokens=500,
        )
        print(f"AI: {response}\n")

        history.append({"role": "assistant", "content": response})

        # Step 4: Acervo processes the response
        await memory.process(user_input, response, web_results=web_results)
        print(f"  Graph: {memory.graph.node_count} nodes, {memory.graph.edge_count} edges\n")


if __name__ == "__main__":
    asyncio.run(main())
