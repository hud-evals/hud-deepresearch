"""
FastAPI server for DeepResearch environment.
Holds EXA API key on the server side and exposes simple HTTP endpoints
that the controller calls. Mirrors the browser/blank environment pattern.
"""

import asyncio
import logging
import os
import socket
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

T = TypeVar("T")


# Set up logging
logger = logging.getLogger(__name__)


async def call_with_exponential_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 4,
    initial_delay: float = 2.0,
    max_delay: float = 55.0,
    exponential_base: float = 2.0,
    **kwargs: Any,
) -> T:
    """
    Call an async function with exponential backoff on rate limit or timeout errors.

    Args:
        func: The async function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay in seconds (default: 30.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception: Optional[Exception] = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        "Rate limit hit (429), retrying in %s seconds... (attempt %s/%s)",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)
                else:
                    raise
            else:
                raise
        except (httpx.TimeoutException, httpx.ReadTimeout) as e:
            # Retry on timeout errors
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    "Request timed out, retrying in %s seconds... (attempt %s/%s)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
                delay = min(delay * exponential_base, max_delay)
            else:
                raise
        except Exception:
            raise

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in exponential backoff")


class _EnvState:
    """In-memory environment state for tracking usage and agent answer."""

    def __init__(self) -> None:
        self.search_count: int = 0
        self.fetch_count: int = 0
        self.submitted_answer: Optional[str] = None

    def reset(self) -> None:
        self.search_count = 0
        self.fetch_count = 0
        self.submitted_answer = None


state = _EnvState()


class SearchRequest(BaseModel):
    query: str


class FetchRequest(BaseModel):
    url: str


class AnswerRequest(BaseModel):
    final_answer: str


class EvaluateRequest(BaseModel):
    expected_answer: str


app = FastAPI(title="DeepResearch Environment API", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy"}


async def _is_port_open(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.15)
    try:
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


@app.post("/setup")
async def setup() -> Dict[str, Any]:
    state.reset()
    return {"ok": True}


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Get current environment state."""
    return {
        "search_count": state.search_count,
        "fetch_count": state.fetch_count,
        "submitted_answer": state.submitted_answer,
    }


async def _execute_search(query: str, exa_api_key: str, max_results: int = 1) -> Dict[str, Any]:
    """Execute the actual Exa search API call."""
    search_url = "https://api.exa.ai/search"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            search_url,
            headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
            json={
                "query": query,
                "numResults": max_results,
                "type": "keyword",
                "userLocation": "us",
                "contents": {"text": {"maxCharacters": 1000}},
            },
        )
        response.raise_for_status()
        return response.json()


@app.post("/search")
async def search(req: SearchRequest) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    max_results: int = 1

    exa_api_key: Optional[str] = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set on environment")

    try:
        # Use exponential backoff for the API call
        data = await call_with_exponential_backoff(
            _execute_search,
            req.query,
            exa_api_key,
            max_results,
        )

        for item in data.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            if title and url:
                results.append({"title": title, "url": url})

        if not results:
            autoprompt = data.get("autopromptString", req.query)
            return [
                {
                    "message": "No results found",
                    "query": req.query,
                    "autopromptString": autoprompt,
                }
            ]

    except httpx.HTTPStatusError as e:  # pragma: no cover
        status_code = e.response.status_code
        try:
            error_body = e.response.text
        except Exception:
            error_body = str(e)
        logger.error("Exa API error %s: %s", status_code, error_body)
        if status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid EXA_API_KEY")
        if status_code == 429:
            raise HTTPException(status_code=429, detail="Exa API rate limit exceeded after retries. Try again later.")
        raise HTTPException(status_code=502, detail=f"Exa API error {status_code}: {error_body[:500]}")
    except (httpx.TimeoutException, httpx.ReadTimeout) as e:  # pragma: no cover
        logger.error("Exa API timeout after retries: %s", e)
        raise HTTPException(status_code=504, detail=f"Exa API request timed out after retries: {e}")
    except Exception as e:  # pragma: no cover
        logger.exception("Search failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"Search failed: {type(e).__name__}: {e}")

    state.search_count += 1
    return results


async def _execute_fetch(url: str, exa_api_key: str, max_length: int = 2500) -> Dict[str, Any]:
    """Execute the actual Exa contents API call."""
    contents_url = "https://api.exa.ai/contents"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            contents_url,
            headers={"x-api-key": exa_api_key, "Content-Type": "application/json"},
            json={
                "urls": [url],
                "text": {"maxCharacters": max_length, "includeHtmlTags": False},
                "highlights": {"numSentences": 5, "highlightsPerUrl": 3},
                "summary": {"query": "main takeaways"},
                "livecrawl": "fallback",
            },
        )
        response.raise_for_status()
        return response.json()


@app.post("/fetch")
async def fetch(req: FetchRequest) -> Dict[str, str]:
    from urllib.parse import urlparse

    max_length: int = 2500
    parsed = urlparse(req.url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {req.url}")

    exa_api_key: Optional[str] = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set on environment")

    try:
        # Use exponential backoff for the API call
        data = await call_with_exponential_backoff(
            _execute_fetch,
            req.url,
            exa_api_key,
            max_length,
        )

        results = data.get("results", [])
        if results:
            result = results[0]
            text = result.get("text", "")
            summary = result.get("summary", "")
            highlights = result.get("highlights", [])

            parts: List[str] = []
            if summary:
                parts.append("=== SUMMARY (Main Takeaways) ===")
                parts.append(summary)
                parts.append("")
            if highlights:
                parts.append("=== KEY HIGHLIGHTS ===")
                for idx, hl in enumerate(highlights[:3], 1):
                    parts.append(f"\nHighlight {idx}:")
                    parts.append(str(hl))
                parts.append("")
            if text:
                parts.append("=== FULL CONTENT ===")
                if len(text) > max_length:
                    text = text[:max_length] + "...[truncated]"
                parts.append(text)

            content = "\n".join(parts) if parts else "No content available"
        else:
            content = "No content available for this URL"

    except httpx.HTTPStatusError as e:  # pragma: no cover
        status_code = e.response.status_code
        try:
            error_body = e.response.text
        except Exception:
            error_body = str(e)
        logger.error("Exa API error %s: %s", status_code, error_body)
        if status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid EXA_API_KEY")
        if status_code == 429:
            raise HTTPException(status_code=429, detail="Exa API rate limit exceeded after retries. Try again later.")
        raise HTTPException(status_code=502, detail=f"Exa API error {status_code}: {error_body[:500]}")
    except (httpx.TimeoutException, httpx.ReadTimeout) as e:  # pragma: no cover
        logger.error("Exa API timeout after retries: %s", e)
        raise HTTPException(status_code=504, detail=f"Exa API request timed out after retries: {e}")
    except Exception as e:  # pragma: no cover
        logger.exception("Fetch failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"Fetch failed: {type(e).__name__}: {e}")

    state.fetch_count += 1
    return {"content": content}


@app.post("/answer")
async def answer(req: AnswerRequest) -> Dict[str, Any]:
    state.submitted_answer = req.final_answer
    return {"ok": True, "message": "Answer submitted"}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    submitted = state.submitted_answer
    if submitted is None:
        return {
            "reward": 0.0,
            "content": f"No answer submitted. Searches: {state.search_count}, Fetches: {state.fetch_count}",
            "done": False,
        }

    submitted_clean = submitted.strip().lower()
    expected_clean = req.expected_answer.strip().lower()
    is_correct = expected_clean in submitted_clean if expected_clean else False

    result_msg = (
        ("Correct! " if is_correct else "Incorrect. ")
        + f"Submitted: '{submitted}', Expected: '{req.expected_answer}'. "
        + f"Stats: {state.search_count} searches, {state.fetch_count} fetches, {state.search_count + state.fetch_count} total operations."
    )

    return {"reward": 1.0 if is_correct else 0.0, "content": result_msg, "done": True}


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not os.getenv("EXA_API_KEY"):
        raise ValueError("EXA_API_KEY is not set")

    uvicorn.run(app, host="0.0.0.0", port=8000)

