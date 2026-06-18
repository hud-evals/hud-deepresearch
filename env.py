"""DeepResearch v6 environment: live research tools over an `mcp` capability, LLM-judged.

Tools: search/fetch (live web via Exa) and Sixtyfour enrich_person/enrich_company.
Templates: web_research (a cited web answer) and research_person (a sourced dossier).
"""

# NOTE: do NOT add `from __future__ import annotations` here. Under it, a
# `@env.template` param annotated with Literal/alias/model crashes the
# sync/deploy manifest path (TypeAdapter on a string forward-ref). Keep
# annotations as real objects.
import asyncio
import contextlib
import logging
import os
import socket
import sys
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from dotenv import load_dotenv

from hud import Environment
from hud.capabilities import Capability
from hud.graders import LLMJudgeGrader, combine

load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[%(levelname)s] %(name)s | %(message)s")
for noisy in ("httpx", "httpcore", "FastMCP", "mcp"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("deepresearch")

env = Environment(name="deepresearch")

_MCP_PORT: int | None = None
_MCP_SERVER_TASK: asyncio.Task[None] | None = None


# ── web search (Exa) ──────────────────────────────────────────────────────────


async def _exa_search(query: str, k: int = 5) -> list[dict[str, str]]:
    key = os.getenv("EXA_API_KEY")
    if not key:
        return [{"message": "Live web search is not configured. Set EXA_API_KEY.", "query": query}]
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"query": query, "numResults": k, "contents": {"text": {"maxCharacters": 800}}},
        )
        r.raise_for_status()
        data = r.json()
    out = [
        {"title": it.get("title", ""), "url": it.get("url", ""), "snippet": (it.get("text") or "")[:200]}
        for it in data.get("results", [])
        if it.get("url")
    ]
    return out or [{"message": "No results found", "query": query}]


async def _exa_fetch(url: str, max_chars: int = 2500) -> str:
    key = os.getenv("EXA_API_KEY")
    if not key:
        return "Live fetch is not configured. Set EXA_API_KEY."
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.exa.ai/contents",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"urls": [url], "text": {"maxCharacters": max_chars}},
        )
        r.raise_for_status()
        data = r.json()
    results = data.get("results", [])
    if not results:
        return "No content available for this URL"
    return (results[0].get("text") or "")[:max_chars] or "No content available for this URL"


# ── tools (registered on the mcp server at serve time) ────────────────────────


async def search(query: str) -> list[dict[str, str]]:
    """Search the web for a query. Returns a list of {title, url, snippet}."""
    return await _exa_search(query)


async def fetch(url: str) -> str:
    """Fetch the full text of a web page by its URL (from a prior search result)."""
    return await _exa_fetch(url)


# ── Sixtyfour: deep research on people and companies ──────────────────────────
# Agentic enrichment API (sponsor: sixtyfour.ai). Sync calls take minutes, so the
# tools force the fast `micro` tier and use a long client timeout.

_SIXTYFOUR_BASE = "https://api.sixtyfour.ai"


async def _sixtyfour_post(path: str, payload: dict[str, Any], timeout: float = 900.0) -> dict[str, Any]:
    key = os.getenv("SIXTYFOUR_API_KEY")
    if not key:
        return {"error": "Sixtyfour is not configured. Set SIXTYFOUR_API_KEY to enable deep "
                         "person/company research."}
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{_SIXTYFOUR_BASE}{path}", headers=headers, json=payload)
        if r.status_code >= 500:  # one retry on a transient server error
            r = await client.post(f"{_SIXTYFOUR_BASE}{path}", headers=headers, json=payload)
        logger.info("sixtyfour %s -> %s in %.0fs", path, r.status_code, time.monotonic() - t0)
        if r.status_code >= 400:
            # Return a usable error instead of raising, so the agent can adapt and the
            # rollout doesn't crash on a Sixtyfour-side hiccup.
            try:
                detail = r.json().get("detail")
            except Exception:
                detail = r.text[:300]
            return {"error": f"Sixtyfour returned {r.status_code}", "detail": detail}
        return r.json()


async def enrich_person(name: str, company: str = "", linkedin: str = "") -> dict[str, Any]:
    """Deep-research a person and return a sourced dossier in one call.

    Pass ``company`` and/or ``linkedin`` to disambiguate common names. Returns
    ``structured_data`` (role, company, co-founders, prior companies, sources) plus a
    ``notes`` narrative and ``references``. This is the primary tool for a person dossier.
    """
    lead_info: dict[str, str] = {"name": name}
    if company:
        lead_info["company"] = company
    if linkedin:
        lead_info["linkedin"] = linkedin
    # Keep the struct lean: large structs 500 on Sixtyfour. The response also carries a
    # rich `notes` narrative + `references`, which cover the rest of the dossier.
    struct = {
        "current_role": "Current job title and company",
        "company_description": "What their current company does, in one sentence",
        "cofounders": "Names of their co-founders, if any",
        "prior_companies": "Notable companies or roles before the current one",
        "sources": "List of source URLs the research is based on",
    }
    # tier="micro" keeps the call fast enough for an interactive rollout (low+ can take
    # 5-10 min); the agent can't pick a slower tier.
    return await _sixtyfour_post("/enrich-lead", {"lead_info": lead_info, "struct": struct, "tier": "micro"})


async def enrich_company(company: str, website: str = "") -> dict[str, Any]:
    """Deep-research a company via Sixtyfour. Returns ``structured_data`` + ``confidence_score``."""
    target = f"{company} ({website})" if website else company
    struct = {
        "what_they_do": "One-sentence description of the company",
        "founded_year": "Year the company was founded",
        "headcount": "Approximate number of employees",
        "founders": "Names of the founders",
        "funding": "Funding stage and notable investors, if known",
        "sources": "List of source URLs the research is based on",
    }
    return await _sixtyfour_post(
        "/company-intelligence", {"target_company": target, "struct": struct, "tier": "micro"}
    )


# ── mcp capability lifecycle ──────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _listening(host: str, port: int, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"mcp server never came up on {host}:{port}")


@env.initialize
async def _up() -> None:
    # Import FastMCP lazily so `import tasks` (the task-collection path) stays
    # free of fastmcp/authlib import-time noise.
    from fastmcp import FastMCP

    global _MCP_PORT, _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is None:
        server = FastMCP(name="research-tools")
        server.tool(search)
        server.tool(fetch)
        server.tool(enrich_person)
        server.tool(enrich_company)
        _MCP_PORT = _free_port()
        _MCP_SERVER_TASK = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_MCP_PORT, show_banner=False)
        )
        await _listening("127.0.0.1", _MCP_PORT)
    env.add_capability(Capability.mcp(name="research", url=f"http://127.0.0.1:{_MCP_PORT}/mcp"))
    if not os.getenv("EXA_API_KEY"):
        logger.info("EXA_API_KEY not set; web_research needs it (search/fetch will say so).")


@env.shutdown
async def _down() -> None:
    global _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is not None:
        _MCP_SERVER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MCP_SERVER_TASK
        _MCP_SERVER_TASK = None


# ── tasks ─────────────────────────────────────────────────────────────────────


@env.template()
async def web_research(question: str, answer_should_include: str = "") -> AsyncGenerator[Any, Any]:
    """Answer a question from live web research (Exa); graded by an LLM judge."""
    answer = yield (
        f"{question}\n\nResearch this using web search, then give a direct, specific answer "
        "and cite the source URL you used."
    )

    criterion = (
        f"The answer correctly addresses the question and is consistent with: {answer_should_include}"
        if answer_should_include
        else "The answer correctly and specifically addresses the question, with a cited source."
    )
    result = await combine(
        LLMJudgeGrader.grade(
            weight=1.0, answer=str(answer or ""), criteria=[(criterion, 1.0)], question=question
        )
    )
    logger.info("web_research reward=%.3f", result.reward)
    yield result


@env.template()
async def research_person(
    brief: str, criteria: list[str], ground_truth: str = ""
) -> AsyncGenerator[Any, Any]:
    """Deep-research a person and produce a sourced dossier; graded by an LLM judge.

    The agent uses enrich_person (Sixtyfour) plus search/fetch to build the dossier.

    Args:
        brief: The research brief shown to the agent.
        criteria: Plain-English requirements the dossier must satisfy (one judge
            criterion each, partial credit).
        ground_truth: Verified facts handed to the judge so it can grade accurately.
    """
    answer = yield brief

    crit = [(c, 1.0) for c in criteria]
    question = brief + (
        f"\n\n=== VERIFIED GROUND TRUTH (for grading only) ===\n{ground_truth}" if ground_truth else ""
    )
    result = await combine(
        LLMJudgeGrader.grade(weight=1.0, answer=str(answer or ""), criteria=crit, question=question)
    )
    logger.info("research_person reward=%.3f", result.reward)
    yield result
