"""Local test script for the deepresearch environment.

Prerequisites:
1. Set EXA_API_KEY environment variable
2. Run the backend: uvicorn backend.server:app --port 8000
3. Run this script: python local_test.py
"""
import asyncio

import hud
from hud.agents import OpenAIChatAgent
from hud.settings import settings
from openai import AsyncOpenAI

from env import env

# Use HUD inference gateway - see all models at https://hud.ai/models
client = AsyncOpenAI(base_url="https://inference.hud.ai", api_key=settings.api_key)


async def test_tools_standalone():
    """Test environment tools directly."""
    print("=== Test 1: Standalone Tools ===")

    async with env:
        print(f"Tools: {[t.name for t in env.as_tools()]}")
        
        # Test search
        results = await env.call_tool("search", query="IEEE Frank Rosenblatt Award 2010")
        print(f"Search results: {results}")


async def test_research_manual():
    """Test research scenario with manual OpenAI calls."""
    print("\n=== Test 2: Research (Manual Agent Loop) ===")

    task = env("research", 
        question="Who received the IEEE Frank Rosenblatt Award in 2010?",
        answer_includes="Michio Sugeno"
    )

    async with hud.eval(task) as ctx:
        messages = [{"role": "user", "content": ctx.prompt}]

        while True:
            response = await client.chat.completions.create(
                model="gpt-4o",  # https://hud.ai/models
                messages=messages,
                tools=ctx.as_openai_chat_tools(),
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                break

            messages.append(msg)
            for tc in msg.tool_calls:
                result = await ctx.call_tool(tc)
                messages.append(result)


async def test_verify_claim_scenario():
    """Test verify-claim scenario with agent."""
    print("\n=== Test 3: Verify Claim Scenario ===")

    task = env("verify-claim",
        claim="The Eiffel Tower is located in London.",
        expected_verdict="false"
    )

    async with hud.eval(task) as ctx:
        agent = OpenAIChatAgent.create(model="gpt-4o")  # https://hud.ai/models
        await agent.run(ctx)


async def test_distribution():
    """Test multiple tasks with variants and groups for A/B testing."""
    print("\n=== Test 4: Distribution (Variants + Groups) ===")

    tasks = [
        env("research", question="Who won the Turing Award in 2020?", answer_includes="Lamport"),
        env("research", question="Who founded OpenAI?", answer_includes=["Altman", "Musk", "Brockman"]),
    ]
    variants = {"model": ["gpt-4o-mini", "gpt-4o"]}
    group = 2

    async with hud.eval(tasks, variants=variants, group=group) as ctx:
        agent = OpenAIChatAgent.create(model=ctx.variants["model"])
        await agent.run(ctx, max_steps=10)


async def main():
    await test_tools_standalone()
    # await test_research_manual()
    # await test_verify_claim_scenario()
    # await test_distribution()


if __name__ == "__main__":
    asyncio.run(main())
