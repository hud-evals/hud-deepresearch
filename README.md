# Deep Research Environment

A web research environment where agents use search, fetch, and answer tools to find information online. Powered by the Exa API for search and content extraction.

## Quick Start

```bash
uv sync                                                                     # install dependencies
hud deploy . --build-arg EXA_API_KEY=$EXA_API_KEY                           # build and deploy to HUD platform
hud sync tasks <name>                                                       # upload task definitions
```

## Scenarios

| Scenario | Key Args | Description |
|----------|----------|-------------|
| `research` | `question`, `answer_includes` | Research a question and submit an answer |
| `verify-claim` | `claim`, `expected_verdict` | Verify a claim as true/false/partially true |
| `multi-hop-research` | `question`, `answer_parts` | Multi-part question requiring chained research (partial credit) |

## Configuration

Requires `EXA_API_KEY` as a build argument for Exa web search and content extraction.

## Documentation

To learn more about tasks, evaluations, and running at scale see the [full docs](https://docs.hud.ai).
