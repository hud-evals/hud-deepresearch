# DeepResearch Environment (HUD v6)

A HUD v6 environment for **live deep research**, built around one loop:

> retrieve → inspect the source → answer with evidence.

The agent researches over real sources - the web via **Exa**, and people/companies via
**Sixtyfour** (the sponsor) - and its answer is graded by an LLM judge against ground truth.

## Tasks

| Task | Backend | What the agent does |
|------|---------|---------------------|
| `web-research-rust-1-0` | live web (Exa) | answer a factual question with a cited source |
| `research-jay-ram` | Sixtyfour + web | produce a sourced dossier on a person |

Each is graded by an LLM judge (`LLMJudgeGrader`) against ground truth, with partial credit.

## Tools

| Tool | What it does |
|------|--------------|
| `search(query)` | Web search via Exa; returns `{title, url, snippet}`. |
| `fetch(url)` | Return the full text of a web page by URL. |
| `enrich_person(name, company, linkedin)` | Deep-research a person via Sixtyfour. |
| `enrich_company(company, website)` | Deep-research a company via Sixtyfour. |

The tools are served over an `mcp` capability; the agent answers in its final message.

## Setup

```bash
uv sync
hud set HUD_API_KEY=your-key-here   # CLI auth + gateway, get one at hud.ai/project/api-keys
```

`HUD_API_KEY` is required (the agents and the LLM-judge grader both route through the HUD
gateway). Put the research keys in `.env`:

```
EXA_API_KEY=          # web search (web_research)
SIXTYFOUR_API_KEY=    # person/company research (research_person)
```

## Run

```bash
# local
hud eval tasks.py claude --task-ids research-jay-ram -y --runtime local

# deploy once (bakes EXA_API_KEY + SIXTYFOUR_API_KEY from .env), then run hosted
hud deploy . --env-file .env
hud eval tasks.py claude --runtime hud --full
```

## Web search (Exa)

`web-research-rust-1-0` (`web_research` template) runs over live web search. The env serves
`search`/`fetch` over **Exa** when `EXA_API_KEY` is set. Exa works both locally and on
`--runtime hud`, because the search runs **inside the environment** and survives hosted
execution. (Provider-native web search like `ClaudeWebSearchTool` works for local runs but is
stripped from hosted agent specs, which is why the env serves Exa instead.)

## People & company research (Sixtyfour)

[Sixtyfour](https://www.sixtyfour.ai/) (sponsor) is an agentic enrichment API: give it a name
(plus context to disambiguate) and it crawls LinkedIn, news, and the web to return a sourced
dossier. The env exposes it as `enrich_person` / `enrich_company`, backed by
`POST https://api.sixtyfour.ai/enrich-lead` and `/company-intelligence` (`x-api-key`).

`research-jay-ram` (`research_person` template) is a realistic prep-for-a-meeting brief:
deep-research HUD co-founder **Jay Ram** and produce a sourced dossier (role, what HUD does,
co-founders, background, prior companies). An LLM judge scores it against verified public facts
with partial credit, including a disambiguation check (the right Jay Ram, not a same-named other).

```bash
hud eval tasks.py claude --task-ids research-jay-ram -y --runtime local   # or --runtime hud
```

Latency note: Sixtyfour sync enrichment is slow at deeper tiers (5-10 min), so the tools force
`tier="micro"` (a fast ~30-60s call that still returns role, company, co-founders, prior
companies, and sources). The brief is scoped so one `enrich_person` call suffices, keeping the
rollout inside the default step budget. For richer output, change the hardcoded `tier="micro"`
in `enrich_person` (env.py) and run with `--max-steps 25`.

## Extending it

Each idea below is a prompt/scenario you can add on top of the same retrieve → inspect → answer
loop:

- **Multi-hop research** - chain two retrievals (find X, then a fact about X).
- **Citation audit** - given a report with citations, mark each claim supported/unsupported and
  propose corrected citations.
- **Contradictory sources** - a 2023 blog and a 2026 docs page disagree; decide the current
  answer and justify it by recency.
- **Company research** - given an ICP, find companies that match with one source-backed reason each.
- **Literature review** - summarize a small set of papers with citations.
- **Adversarial source pages** - a source that lies or buries a contradiction; reward the agent
  for catching it.

## Tests

```bash
uv run pytest tests/ -q
```

Offline tests stub the LLM judge and mock the Sixtyfour API; they cover the web/Sixtyfour tools,
the served `mcp` capability, and golden/baseline grading. No live keys are called.

## Documentation

See the [full docs](https://docs.hud.ai) for tasks, evaluation, and scaling.
