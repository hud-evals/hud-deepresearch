"""Local test script for the deepresearch environment.

Usage:
    python local_test.py --list
    python local_test.py --task research_ieee_2010
    python local_test.py --task compare_renewable_nuclear --model gpt-4o
    python local_test.py --task verify_eiffel_paris --max-steps 10

Prerequisites:
    1. Set EXA_API_KEY environment variable
    2. Run the backend: uvicorn backend.server:app --port 8000
"""

import argparse
import asyncio

import hud
from hud.agents import OpenAIChatAgent

from tasks import ALL_TASKS


async def main() -> None:
    available = sorted(ALL_TASKS)

    parser = argparse.ArgumentParser(
        description="Run deepresearch tasks locally with an agent."
    )
    parser.add_argument(
        "--task",
        default=available[0],
        choices=available,
        help="Task to run (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use (default: %(default)s). See https://hud.ai/models",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum agent steps (default: %(default)s)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tasks and exit",
    )
    args = parser.parse_args()

    if args.list:
        for name in available:
            task = ALL_TASKS[name]
            print(f"  {name:35s} scenario={task.scenario}")
        return

    task = ALL_TASKS[args.task]
    print(f"=== {args.task} (scenario={task.scenario}, model={args.model}) ===")

    async with hud.eval(task, name=args.task) as ctx:
        agent = OpenAIChatAgent.create(model=args.model)
        await agent.run(ctx, max_steps=args.max_steps)
        print(f"Reward: {ctx.reward}")


if __name__ == "__main__":
    asyncio.run(main())
