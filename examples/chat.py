"""Minimal Acervo chat — demonstrates the prepare/process pattern.

Prerequisites:
    1. Start LM Studio with qwen2.5-3b-instruct loaded
    2. pip install acervo

Run:
    python examples/chat.py

The knowledge graph is stored in data/graph/ and persists between sessions.
"""

import asyncio

from acervo import Acervo, OpenAIClient


async def main() -> None:
    # Connect to any OpenAI-compatible LLM server
    # LM Studio: load qwen2.5-3b-instruct, start server on port 1234
    llm = OpenAIClient(
        base_url="http://localhost:1234/v1",
        model="qwen2.5-3b-instruct",
        api_key="lm-studio",
    )

    # Create memory — persists to data/graph/nodes.json + edges.json
    memory = Acervo(llm=llm, owner="User")
    history: list[dict] = [
        {"role": "system", "content": "You are a helpful assistant. Respond in the same language as the user."},
    ]

    print("Acervo Chat (type 'quit' to exit)")
    print(f"Graph: {memory.graph.node_count} nodes, {memory.graph.edge_count} edges\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        history.append({"role": "user", "content": user_input})

        # Step 1: Acervo prepares context from knowledge graph
        prep = await memory.prepare(user_input, history)
        # prep.context_stack has the enriched messages (system + graph context + history + user)
        # prep.plan.tool tells you what Acervo recommends: "GRAPH_ALL", "WEB_SEARCH", or "READY"
        # prep.has_context is True if the graph had relevant data

        # Step 2: Call the LLM with enriched context
        response = await llm.chat(
            prep.context_stack,
            temperature=0.7,
            max_tokens=500,
        )
        print(f"AI: {response}\n")

        history.append({"role": "assistant", "content": response})

        # Step 3: Acervo extracts knowledge from the conversation
        result = await memory.process(user_input, response)

        # Show what was extracted
        if result.entities:
            names = ", ".join(f"{e.name}({e.type})" for e in result.entities)
            print(f"  Extracted: {names}")
        print(f"  Graph: {memory.graph.node_count} nodes, {memory.graph.edge_count} edges\n")


if __name__ == "__main__":
    asyncio.run(main())
