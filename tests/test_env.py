"""Tests for the deepresearch env: web search, Sixtyfour tools, served mcp, and grading.

The grader is an LLM judge (HUD gateway). Offline tests stub it; the Sixtyfour API
is mocked (never called for real).
"""

import pytest

import env as M
from hud.graders import LLMJudgeGrader

WEB_GEN = M.web_research.func
PERSON_GEN = M.research_person.func


def _stub_judge(monkeypatch, predicate):
    """Replace the gateway LLM judge with a local predicate over the answer text."""

    async def stub(cls, answer="", criteria=None, question="", **kwargs):
        return (1.0 if predicate(answer) else 0.0, {})

    monkeypatch.setattr(LLMJudgeGrader, "compute_score", classmethod(stub))


class TestWebSearch:
    async def test_search_without_exa_key_is_graceful(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        results = await M.search("anything at all")
        assert "message" in results[0]  # graceful "set EXA_API_KEY" message, not a crash

    async def test_fetch_without_exa_key_is_graceful(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        assert "EXA_API_KEY" in await M.fetch("https://example.com")


class TestSixtyfour:
    async def test_enrich_person_not_configured(self, monkeypatch):
        monkeypatch.delenv("SIXTYFOUR_API_KEY", raising=False)
        out = await M.enrich_person("Jay Ram", company="HUD")
        assert "error" in out and "SIXTYFOUR_API_KEY" in out["error"]

    async def test_enrich_person_builds_request(self, monkeypatch):
        captured = {}

        async def fake_post(path, payload, timeout=900.0):
            captured["path"], captured["payload"] = path, payload
            return {"structured_data": {"current_role": "Co-founder & CEO, HUD"}, "confidence_score": 8}

        monkeypatch.setattr(M, "_sixtyfour_post", fake_post)
        out = await M.enrich_person("Jay Ram", company="HUD")
        assert captured["path"] == "/enrich-lead"
        assert captured["payload"]["lead_info"] == {"name": "Jay Ram", "company": "HUD"}
        assert captured["payload"]["tier"] == "micro"  # forced fast tier
        assert "struct" in captured["payload"]
        assert out["structured_data"]["current_role"].startswith("Co-founder")

    async def test_enrich_company_builds_request(self, monkeypatch):
        captured = {}

        async def fake_post(path, payload, timeout=900.0):
            captured["path"], captured["payload"] = path, payload
            return {"structured_data": {"founded_year": "2025"}, "confidence_score": 7}

        monkeypatch.setattr(M, "_sixtyfour_post", fake_post)
        await M.enrich_company("HUD", website="hud.so")
        assert captured["path"] == "/company-intelligence"
        assert "HUD" in captured["payload"]["target_company"]
        assert "struct" in captured["payload"]


class TestGrading:
    async def test_web_research_good_answer_scores_high(self, monkeypatch):
        _stub_judge(monkeypatch, lambda a: "2015" in a)
        gen = WEB_GEN(question="What year did Rust hit 1.0?", answer_should_include="2015")
        await gen.asend(None)
        r = await gen.asend("Rust reached 1.0 in 2015. Source: blog.rust-lang.org")
        assert r.reward == 1.0

    async def test_web_research_empty_answer_scores_zero(self, monkeypatch):
        _stub_judge(monkeypatch, lambda a: "2015" in a)
        gen = WEB_GEN(question="What year did Rust hit 1.0?", answer_should_include="2015")
        await gen.asend(None)
        r = await gen.asend("I'm not sure.")
        assert r.reward == 0.0

    async def test_research_person_good_dossier_scores_high(self, monkeypatch):
        _stub_judge(monkeypatch, lambda a: "hud" in a.lower())
        gen = PERSON_GEN(brief="b", criteria=["Names HUD"], ground_truth="HUD facts")
        await gen.asend(None)
        r = await gen.asend(
            "Jay Ram is co-founder & CEO of HUD, which builds RL environments. "
            "Source: ycombinator.com/companies/hud"
        )
        assert r.reward == 1.0

    async def test_research_person_empty_dossier_scores_zero(self, monkeypatch):
        _stub_judge(monkeypatch, lambda a: "hud" in a.lower())
        gen = PERSON_GEN(brief="b", criteria=["Names HUD"], ground_truth="HUD facts")
        await gen.asend(None)
        r = await gen.asend("Sorry, no idea who that is.")
        assert r.reward == 0.0


class TestServedCapability:
    async def test_mcp_capability_serves_all_tools(self):
        from hud.capabilities.mcp import MCPClient

        environment = M.env
        await environment.start()
        try:
            cap = environment.capability("research")
            assert cap.protocol == "mcp/2025-11-25"
            client = await MCPClient.connect(cap)
            try:
                names = sorted(t.name for t in await client.list_tools())
                assert names == ["enrich_company", "enrich_person", "fetch", "search"]
            finally:
                await client.close()
        finally:
            await environment.stop()
