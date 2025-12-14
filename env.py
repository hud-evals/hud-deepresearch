"""Deep Research Environment - Web research with Exa search and content fetching.

This demonstrates:
- @env.tool() for search, fetch, and answer tools
- @env.scenario() for research and verify-claim evaluation flows
- Integration with Exa API for web search and content extraction
"""
import logging
import os
import sys
from typing import Any

import httpx
from hud import Environment

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    force=True,
)
for logger_name in ["httpx", "httpcore"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Backend connection
BACKEND_URL = os.getenv("ENV_SERVER_URL", "http://localhost:8000")
http_client = httpx.AsyncClient(base_url=BACKEND_URL, timeout=30.0)

env = Environment(name="deepresearch")


# =============================================================================
# TOOLS
# =============================================================================


@env.tool()
async def search(query: str) -> list[dict[str, str]]:
    """Search the web using Exa. Returns a list of results with title and URL."""
    resp = await http_client.post("/search", json={"query": query})
    resp.raise_for_status()
    return resp.json()


@env.tool()
async def fetch(url: str) -> str:
    """Fetch and extract content from a URL. Returns summary, highlights, and text."""
    resp = await http_client.post("/fetch", json={"url": url})
    resp.raise_for_status()
    return resp.json().get("content", "")


@env.tool()
async def answer(final_answer: str) -> str:
    """Submit your final answer. Call this when you have completed your research."""
    await http_client.post("/answer", json={"final_answer": final_answer})
    return f"Answer submitted: {final_answer}"


# =============================================================================
# SCENARIOS
# =============================================================================


@env.scenario("research")
async def research(question: str, answer_includes: str | list[str]) -> Any:
    """Research a question and find the answer.
    
    Args:
        question: The research question to answer
        answer_includes: String or list of strings that must appear in the answer
    """
    # Setup: reset state
    await http_client.post("/setup")
    logger.info("Research scenario: %s", question)
    
    # Yield prompt
    prompt = f"""{question}

Use the search and fetch tools to find the answer. When you have found the answer, call the answer tool with your final response.

Return just the answer, no other text."""
    
    _ = yield prompt
    
    # Evaluate: check if answer includes required strings
    resp = await http_client.get("/state")
    state = resp.json()
    submitted = state.get("submitted_answer", "")
    
    if not submitted:
        logger.info("No answer submitted")
        yield 0.0
        return
    
    # Normalize for comparison
    submitted_lower = submitted.strip().lower()
    
    # Handle both string and list - check if ANY match
    if isinstance(answer_includes, str):
        candidates = [answer_includes]
    else:
        candidates = answer_includes
    
    # Check if any of the candidate strings are present
    found = any(candidate.lower() in submitted_lower for candidate in candidates)
    reward = 1.0 if found else 0.0
    
    logger.info(
        "Research result: found=%s, candidates=%s, reward=%.2f, answer='%s'",
        found, candidates, reward, submitted[:100]
    )
    yield reward


@env.scenario("verify-claim")
async def verify_claim(claim: str, expected_verdict: str) -> Any:
    """Verify whether a claim is true or false.
    
    Args:
        claim: The claim to verify
        expected_verdict: Expected verdict ("true", "false", "partially true", etc.)
    """
    # Setup: reset state
    await http_client.post("/setup")
    logger.info("Verify claim scenario: %s", claim)
    
    # Yield prompt
    prompt = f"""Verify the following claim:

"{claim}"

Use the search and fetch tools to find evidence. When you have determined whether the claim is true or false, call the answer tool with your verdict.

Your answer should be one of: "true", "false", or "partially true" followed by a brief explanation."""
    
    _ = yield prompt
    
    # Evaluate: check if verdict matches
    resp = await http_client.get("/state")
    state = resp.json()
    submitted = state.get("submitted_answer", "")
    
    if not submitted:
        logger.info("No answer submitted")
        yield 0.0
        return
    
    # Normalize for comparison
    submitted_lower = submitted.strip().lower()
    expected_lower = expected_verdict.strip().lower()
    
    # Check if expected verdict is in the answer
    is_correct = expected_lower in submitted_lower
    reward = 1.0 if is_correct else 0.0
    
    logger.info(
        "Verify claim result: expected='%s', got='%s', reward=%.2f",
        expected_verdict, submitted[:100], reward
    )
    yield reward


# =============================================================================
# LIFECYCLE
# =============================================================================


@env.initialize
async def init() -> None:
    """Check backend health on startup."""
    resp = await http_client.get("/health")
    resp.raise_for_status()
    logger.info("Backend health check passed")


@env.shutdown
async def cleanup() -> None:
    """Close HTTP client on shutdown."""
    await http_client.aclose()


if __name__ == "__main__":
    env.run(transport="stdio")
