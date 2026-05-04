"""Tests for the deepresearch env: Exa helpers, tools, and scenarios."""

import pytest

from env import (
    _exa_fetch,
    _exa_search,
    answer,
    fetch,
    research,
    search,
    state,
)

pytestmark = pytest.mark.asyncio


class TestExaSearch:
    async def test_returns_results_with_title_and_url(self):
        results = await _exa_search("python programming language")
        assert isinstance(results, list)
        assert results
        first = results[0]
        # Either a real result with title+url, or the "no results" sentinel
        assert ("title" in first and "url" in first) or "message" in first


class TestExaFetch:
    async def test_returns_content_for_valid_url(self):
        content = await _exa_fetch("https://example.com")
        assert isinstance(content, str)
        assert content  # non-empty

    async def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            await _exa_fetch("not-a-valid-url")


class TestState:
    async def test_search_increments_count(self):
        assert state.search_count == 0
        await search("python")
        assert state.search_count == 1

    async def test_fetch_increments_count(self):
        assert state.fetch_count == 0
        await fetch("https://example.com")
        assert state.fetch_count == 1

    async def test_answer_stores_submission(self):
        assert state.submitted_answer is None
        await answer("The answer is Paris")
        assert state.submitted_answer == "The answer is Paris"

    async def test_reset_clears_everything(self):
        state.search_count = 3
        state.fetch_count = 5
        state.submitted_answer = "x"
        state.reset()
        assert state.search_count == 0
        assert state.fetch_count == 0
        assert state.submitted_answer is None


class TestResearchScenario:
    """Drive the research scenario as an async generator (no Exa calls)."""

    async def test_correct_answer_via_tool_scores_1(self):
        gen = research(question="Q", answer_includes="Paris")
        prompt = await gen.asend(None)
        assert "Q" in prompt

        # Simulate the agent calling answer(...)
        await answer("The answer is Paris")
        reward = await gen.asend(None)
        assert reward == 1.0

    async def test_correct_answer_via_response_fallback(self):
        gen = research(question="Q", answer_includes="Paris")
        await gen.asend(None)
        # No answer tool call — agent's response is used as fallback
        reward = await gen.asend("The capital of France is Paris")
        assert reward == 1.0

    async def test_incorrect_answer_scores_0(self):
        gen = research(question="Q", answer_includes="Paris")
        await gen.asend(None)
        await answer("The answer is London")
        reward = await gen.asend(None)
        assert reward == 0.0

    async def test_no_answer_scores_0(self):
        gen = research(question="Q", answer_includes="Paris")
        await gen.asend(None)
        reward = await gen.asend(None)
        assert reward == 0.0

    async def test_list_of_acceptable_answers(self):
        gen = research(
            question="Q", answer_includes=["Radcliffe College", "Radcliffe"]
        )
        await gen.asend(None)
        await answer("It was Radcliffe.")
        reward = await gen.asend(None)
        assert reward == 1.0
