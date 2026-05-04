"""Deep Research Environment — Web research with Exa search and content fetching.

Tools (search/fetch/answer) call the Exa API directly. Per-scenario state lives
in a module-level instance, reset at the top of each scenario.
"""

import asyncio
import logging
import os
import subprocess
import sys
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from hud import Environment

load_dotenv()

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
for logger_name in ["httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

env = Environment(name="deepresearch")

T = TypeVar("T")


# =============================================================================
# STATE
# =============================================================================


@dataclass
class _EnvState:
    """Tracks tool usage and the agent's submitted answer for one scenario run."""

    search_count: int = 0
    fetch_count: int = 0
    submitted_answer: str | None = None

    def reset(self) -> None:
        self.search_count = 0
        self.fetch_count = 0
        self.submitted_answer = None


state = _EnvState()


# =============================================================================
# EXA API HELPERS
# =============================================================================


async def _call_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 4,
    initial_delay: float = 2.0,
    max_delay: float = 55.0,
    exponential_base: float = 2.0,
    **kwargs: Any,
) -> T:
    """Call an async function with exponential backoff on 429s and timeouts."""
    delay = initial_delay
    last: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 429 or attempt >= max_retries:
                raise
            last = e
            logger.warning(
                "Rate limit (429), retrying in %ss (attempt %d/%d)",
                delay, attempt + 1, max_retries,
            )
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            if attempt >= max_retries:
                raise
            last = e
            logger.warning(
                "Timeout, retrying in %ss (attempt %d/%d)",
                delay, attempt + 1, max_retries,
            )

        await asyncio.sleep(delay)
        delay = min(delay * exponential_base, max_delay)

    if last:
        raise last
    raise RuntimeError("unreachable")


def _require_api_key() -> str:
    key = os.getenv("EXA_API_KEY")
    if not key:
        raise RuntimeError("EXA_API_KEY is not set")
    return key


async def _exa_search(query: str, max_results: int = 1) -> list[dict[str, str]]:
    """Call Exa's search API. Returns a list of {title, url} dicts.

    Returns a [{"message": ..., "query": ..., "autopromptString": ...}] sentinel
    when Exa finds no results.
    """
    api_key = _require_api_key()

    async def _do() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "query": query,
                    "numResults": max_results,
                    "type": "keyword",
                    "userLocation": "us",
                    "contents": {"text": {"maxCharacters": 1000}},
                },
            )
            r.raise_for_status()
            return r.json()

    data = await _call_with_backoff(_do)
    results: list[dict[str, str]] = []
    for item in data.get("results", []):
        title, url = item.get("title", ""), item.get("url", "")
        if title and url:
            results.append({"title": title, "url": url})

    if not results:
        return [{
            "message": "No results found",
            "query": query,
            "autopromptString": data.get("autopromptString", query),
        }]
    return results


async def _exa_fetch(url: str, max_length: int = 2500) -> str:
    """Call Exa's contents API. Returns formatted summary + highlights + text."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    api_key = _require_api_key()

    async def _do() -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://api.exa.ai/contents",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "urls": [url],
                    "text": {"maxCharacters": max_length, "includeHtmlTags": False},
                    "highlights": {"numSentences": 5, "highlightsPerUrl": 3},
                    "summary": {"query": "main takeaways"},
                    "livecrawl": "fallback",
                },
            )
            r.raise_for_status()
            return r.json()

    data = await _call_with_backoff(_do)
    results = data.get("results", [])
    if not results:
        return "No content available for this URL"

    result = results[0]
    summary = result.get("summary", "")
    highlights = result.get("highlights", [])
    text = result.get("text", "")
    if text and len(text) > max_length:
        text = text[:max_length] + "...[truncated]"

    parts: list[str] = []
    if summary:
        parts.extend(["=== SUMMARY (Main Takeaways) ===", summary, ""])
    if highlights:
        parts.append("=== KEY HIGHLIGHTS ===")
        for idx, hl in enumerate(highlights[:3], 1):
            parts.extend([f"\nHighlight {idx}:", str(hl)])
        parts.append("")
    if text:
        parts.extend(["=== FULL CONTENT ===", text])

    return "\n".join(parts) if parts else "No content available"


# =============================================================================
# TOOLS
# =============================================================================


@env.tool()
async def search(query: str) -> list[dict[str, str]]:
    """Search the web using Exa. Returns a list of results with title and URL."""
    results = await _exa_search(query)
    state.search_count += 1
    return results


@env.tool()
async def fetch(url: str) -> str:
    """Fetch and extract content from a URL. Returns summary, highlights, and text."""
    content = await _exa_fetch(url)
    state.fetch_count += 1
    return content


@env.tool()
async def answer(final_answer: str) -> str:
    """Submit your final answer. Call this when you have completed your research."""
    state.submitted_answer = final_answer
    return f"Answer submitted: {final_answer}"


@env.tool()
async def hud_validate() -> str:
    """Run the test suite to validate the environment is working correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd="/app",
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output or f"pytest exited with code {result.returncode}")
    return output


# =============================================================================
# SCENARIOS
# =============================================================================


@env.scenario("research", exclude_tools=["hud_validate"])
async def research(
    question: str, answer_includes: str | list[str] | None = None
) -> AsyncGenerator[Any]:
    """Research a question and find the answer.

    Args:
        question: The research question to answer
        answer_includes: String or list of strings that must appear in the answer
    """
    state.reset()
    logger.info("Research scenario: %s", question)

    prompt = f"""{question}

Use the search and fetch tools to find the answer. When you have found the answer, call the answer tool with your final response.

Return just the answer, no other text."""

    response = yield prompt

    submitted = state.submitted_answer or response or ""
    if not submitted:
        logger.info("No answer submitted and no response from agent")
        yield 0.0
        return

    submitted_lower = submitted.strip().lower()
    candidates = [answer_includes] if isinstance(answer_includes, str) else (answer_includes or [])
    found = any(c.lower() in submitted_lower for c in candidates)
    reward = 1.0 if found else 0.0

    logger.info(
        "Research result: found=%s, candidates=%s, reward=%.2f, answer='%s'",
        found, candidates, reward, submitted[:100],
    )
    yield reward


@env.scenario("verify-claim", exclude_tools=["hud_validate"])
async def verify_claim(
    claim: str, expected_verdict: str | None = None
) -> AsyncGenerator[Any]:
    """Verify whether a claim is true or false.

    Args:
        claim: The claim to verify
        expected_verdict: Expected verdict ("true", "false", "partially true", etc.)
    """
    state.reset()
    logger.info("Verify claim scenario: %s", claim)

    prompt = f"""Verify the following claim:

"{claim}"

Use the search and fetch tools to find evidence. When you have determined whether the claim is true or false, call the answer tool with your verdict.

Your answer should be one of: "true", "false", or "partially true" followed by a brief explanation."""

    response = yield prompt

    submitted = state.submitted_answer or response or ""
    if not submitted:
        logger.info("No answer submitted and no response from agent")
        yield 0.0
        return

    if expected_verdict is None:
        logger.info("No expected verdict provided, defaulting to 0.0 reward")
        yield 0.0
        return

    is_correct = expected_verdict.strip().lower() in submitted.strip().lower()
    reward = 1.0 if is_correct else 0.0

    logger.info(
        "Verify claim result: expected='%s', got='%s', reward=%.2f",
        expected_verdict, submitted[:100], reward,
    )
    yield reward


@env.scenario("multi-hop-research")
async def multi_hop_research(
    question: str,
    answer_parts: list[str | list[str]] | None = None,
) -> AsyncGenerator[Any]:
    """Answer a question requiring chaining multiple research steps.

    Partial credit is awarded for each correct part found.

    Args:
        question: A multi-part question requiring chained research
        answer_parts: Each element is either a string or a list of acceptable alternatives.
    """
    state.reset()
    logger.info("Multi-hop research scenario: %s", question)

    prompt = f"""{question}

This question requires multiple steps of research. Break it down, search for each piece of information, and combine your findings. Call the answer tool when you have the complete answer.

Include ALL parts of the answer."""

    response = yield prompt

    submitted = state.submitted_answer or response or ""
    if not submitted:
        yield 0.0
        return

    if not answer_parts:
        yield 1.0
        return

    submitted_lower = submitted.strip().lower()

    def _part_found(part: str | list[str]) -> bool:
        candidates = [part] if isinstance(part, str) else part
        return any(c.lower() in submitted_lower for c in candidates)

    found_count = sum(1 for p in answer_parts if _part_found(p))
    reward = found_count / len(answer_parts)

    logger.info(
        "Multi-hop result: found %d/%d parts, reward=%.2f, answer='%s'",
        found_count, len(answer_parts), reward, submitted[:100],
    )
    yield round(reward, 4)


# =============================================================================
# LIFECYCLE
# =============================================================================


@env.initialize
async def init() -> None:
    """Fail fast if the Exa API key isn't configured."""
    _require_api_key()
    logger.info("EXA_API_KEY present; deepresearch env ready")


if __name__ == "__main__":
    env.run(transport="stdio")
