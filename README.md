# Deep Research Environment

A web research environment using Exa for search and content extraction.

## 1. Deploy to Platform

If you haven't already, connect this repo to hud.ai:

1. Push to GitHub
2. Go to [hud.ai](https://hud.ai) → **New** → **Environment**
3. Connect your GitHub repo
4. Your environment builds automatically on each push

Once deployed, your environment is accessible by its slug (e.g., `my-org/deepresearch`).

**Required Environment Variable:** Set `EXA_API_KEY` in your environment settings.

## 2. Define Tools and Scenarios

Tools are functions agents can call. Scenarios define the evaluation lifecycle.

```python
from hud import Environment

env = Environment(name="deepresearch")

@env.tool()
async def search(query: str) -> list[dict[str, str]]:
    """Search the web using Exa."""
    resp = await http_client.post("/search", json={"query": query})
    return resp.json()

@env.tool()
async def fetch(url: str) -> str:
    """Fetch and extract content from a URL."""
    resp = await http_client.post("/fetch", json={"url": url})
    return resp.json().get("content", "")

@env.tool()
async def answer(final_answer: str) -> str:
    """Submit your final answer."""
    await http_client.post("/answer", json={"final_answer": final_answer})
    return f"Answer submitted: {final_answer}"

@env.scenario("research")
async def research(question: str, answer_includes: str | list[str]):
    await http_client.post("/setup")                    # Setup
    _ = yield f"{question}\n\nUse search and fetch..."  # Prompt
    # Evaluate: check if any answer_includes is in the submitted answer
    yield 1.0 if found else 0.0                         # Reward

@env.scenario("verify-claim")
async def verify_claim(claim: str, expected_verdict: str):
    await http_client.post("/setup")                    # Setup
    _ = yield f"Verify: {claim}"                        # Prompt
    yield 1.0 if expected_verdict in answer else 0.0    # Reward
```

## 3. Create Tasks from Scenarios

Tasks are scenario instances with specific arguments.

**In Code:**
```python
tasks = [
    env("research", question="Who won the Nobel Prize in Physics 2023?", answer_includes="Agostini"),
    env("verify-claim", claim="Water boils at 100°C at sea level.", expected_verdict="true"),
]
```

**From JSON:**
```json
[
  {"env": {"name": "my-org/deepresearch"}, "scenario": "research", "args": {"question": "...", "answer_includes": "..."}},
  {"env": {"name": "my-org/deepresearch"}, "scenario": "verify-claim", "args": {"claim": "...", "expected_verdict": "true"}}
]
```

**On Platform:**
After deploying, create tasks from your scenarios on hud.ai. Access them by slug:
```python
from hud.datasets import load_tasks
tasks = load_tasks("my-org/deepresearch-tasks")
```

## 4. Run Evaluations

Run tasks and see results on hud.ai. You have three options:

**On Platform:**
Run evaluations at scale directly on [hud.ai](https://hud.ai) with parallel execution and automatic tracing.

**CLI:**
```bash
hud eval ./remote_tasks.json --model gpt-4o --remote  # https://hud.ai/models
hud eval my-org/deepresearch --model gpt-4o --remote --group 5
```

**Python:**
```python
import hud
from hud.agents import OpenAIChatAgent  # See all models: https://hud.ai/models

tasks = [
    env("research", question="Who received the IEEE Frank Rosenblatt Award in 2010?", answer_includes="Michio Sugeno"),
]

async with hud.eval(tasks) as ctx:
    agent = OpenAIChatAgent.create(model="gpt-4o")  # Uses inference.hud.ai
    await agent.run(ctx)

# Results are automatically traced to hud.ai
```

**With Variants (A/B Testing):**

```python
tasks = [
    env("research", question="Who won the Nobel Prize in Physics 2023?", answer_includes="Agostini"),
    env("verify-claim", claim="Water boils at 100°C at sea level.", expected_verdict="true"),
]
variants = {"model": ["gpt-4o-mini", "gpt-4o"]}

async with hud.eval(tasks, variants=variants, group=2) as ctx:
    agent = OpenAIChatAgent.create(model=ctx.variants["model"])
    await agent.run(ctx)
```

## Local Development

```bash
# Set your Exa API key
export EXA_API_KEY=your-key-here

# Start the backend
uvicorn backend.server:app --port 8000 --reload

# Test locally
python local_test.py

# Test with remote tasks
python remote_test.py
```

## Structure

```
hud-deepresearch/
├── env.py                  # Environment + tools + scenarios
├── backend/
│   └── server.py           # FastAPI backend (Exa integration)
├── local_test.py           # Local testing examples
├── remote_test.py          # Platform integration examples
├── remote_tasks.json       # Task definitions
├── Dockerfile.hud
└── pyproject.toml
```

## Documentation

Full documentation: [docs.hud.ai](https://docs.hud.ai)
