# Deep Research Environment

A web research environment where agents use search, fetch, and answer tools to find information online. Powered by the Exa API for search and content extraction.

## Setup

```bash
uv sync
hud set HUD_API_KEY=your-key-here   # CLI auth, get one at hud.ai/project/api-keys
```

## Deploy & Run

```bash
hud deploy .                                        # deploy the environment (once)
hud sync tasks <taskset-name>                        # push tasks to a taskset (fast, re-run on every task change)
hud eval <taskset-name> --remote --full
```

**Iteration loop:** `hud deploy` is the slow step — run it once. After that, edit `tasks.py` and re-run `hud sync tasks` (takes seconds). Only redeploy when `env.py` or the Dockerfile changes.

See [Deploy & Go Remote](https://docs.hud.ai/building/running-at-scale) for deploy flags, secrets, and auto-deploy options.

## Scenarios

| Scenario | Key Args | Description |
|----------|----------|-------------|
| `research` | `question`, `answer_includes` | Research a question and submit an answer |
| `verify-claim` | `claim`, `expected_verdict` | Verify a claim as true/false/partially true |
| `multi-hop-research` | `question`, `answer_parts` | Multi-part question requiring chained research (partial credit) |

## Configuration

Requires `EXA_API_KEY` as a runtime environment variable for Exa web search and content extraction.

## Documentation

To learn more about tasks, evaluations, and running at scale see the [full docs](https://docs.hud.ai).
