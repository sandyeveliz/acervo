"""Layer 2: S1/S2/S3 Step Validation — test prepare() pipeline against indexed projects.

Each test calls prepare() with a specific question and validates that S1 intent
classification, S2 node activation, and S3 context assembly behave correctly.

Requires: LM Studio running (for S1 unified LLM calls)

Usage:
    pytest tests/integration/test_pipeline_steps.py -v -s
    pytest tests/integration/test_pipeline_steps.py -k "p1" -v -s
"""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════════
# P1 — TODO App (code project)
# ══════════════════════════════════════════════════════════════


class TestStepsP1:
    """S1/S2/S3 behavior for P1 code project."""

    @pytest.mark.asyncio
    async def test_s1_overview_intent(self, p1_acervo):
        """Overview keyword question → intent should be 'overview' (via fallback)."""
        # Use a question that triggers the keyword fallback ("how many", "list all", etc.)
        prep = await p1_acervo.prepare("How many files does this project have?", [])
        intent = prep.debug.get("s1_detection", {}).get("intent", "")
        assert intent == "overview", f"Expected overview intent, got '{intent}'"

    @pytest.mark.asyncio
    async def test_s1_specific_intent(self, p1_acervo):
        """Specific question → intent should be 'specific'."""
        prep = await p1_acervo.prepare("How does authentication work?", [])
        intent = prep.debug.get("s1_detection", {}).get("intent", "")
        assert intent == "specific", f"Expected specific intent, got '{intent}'"

    @pytest.mark.asyncio
    async def test_s2_overview_skips_vector(self, p1_acervo):
        """Overview intent should skip vector search."""
        prep = await p1_acervo.prepare("Give me an overview of the project", [])
        hits = prep.debug.get("s2_gathered", {}).get("vector_hits", [])
        assert len(hits) == 0, (
            f"Overview should skip vector search, got {len(hits)} hits"
        )

    @pytest.mark.asyncio
    async def test_s2_specific_activates_relevant_nodes(self, p1_acervo):
        """Specific question about auth should activate auth-related nodes."""
        prep = await p1_acervo.prepare("How does the authentication middleware work?", [])
        nodes = prep.debug.get("s2_gathered", {}).get("nodes", [])
        labels = [n.get("label", "").lower() for n in nodes]
        has_auth = any("auth" in l for l in labels)
        assert has_auth, (
            f"Expected auth-related nodes activated. Got: {labels[:10]}"
        )

    @pytest.mark.asyncio
    async def test_s3_has_context(self, p1_acervo):
        """Any question should produce non-zero warm context."""
        prep = await p1_acervo.prepare("What database does the app use?", [])
        assert prep.warm_tokens > 0, "Expected warm_tokens > 0"
        assert prep.has_context, "Expected has_context=True"

    @pytest.mark.asyncio
    async def test_s3_budget_respected(self, p1_acervo):
        """Warm tokens should stay within reasonable budget."""
        prep = await p1_acervo.prepare("Explain the app architecture", [])
        assert prep.warm_tokens <= 1200, (
            f"Warm tokens {prep.warm_tokens} exceeds budget"
        )

    @pytest.mark.asyncio
    async def test_s3_overview_has_project_structure(self, p1_acervo):
        """Overview context should mention project structure."""
        prep = await p1_acervo.prepare("What is this project?", [])
        content = prep.warm_content.lower()
        assert "src" in content or "backend" in content or "frontend" in content, (
            f"Overview missing project structure. Preview: {content[:200]}"
        )


# ══════════════════════════════════════════════════════════════
# P2 — Literature (Sherlock Holmes)
# ══════════════════════════════════════════════════════════════


class TestStepsP2:
    """S1/S2/S3 behavior for P2 literature project."""

    @pytest.mark.asyncio
    async def test_s1_overview_intent(self, p2_acervo):
        prep = await p2_acervo.prepare("How many stories do we have?", [])
        intent = prep.debug.get("s1_detection", {}).get("intent", "")
        assert intent == "overview", f"Expected overview, got '{intent}'"

    @pytest.mark.asyncio
    async def test_s1_specific_intent(self, p2_acervo):
        prep = await p2_acervo.prepare("Who is Irene Adler?", [])
        intent = prep.debug.get("s1_detection", {}).get("intent", "")
        assert intent == "specific", f"Expected specific, got '{intent}'"

    @pytest.mark.asyncio
    async def test_s2_overview_skips_vector(self, p2_acervo):
        prep = await p2_acervo.prepare("How many stories are there?", [])
        hits = prep.debug.get("s2_gathered", {}).get("vector_hits", [])
        assert len(hits) == 0, f"Overview should skip vector, got {len(hits)}"

    @pytest.mark.asyncio
    async def test_s3_has_context(self, p2_acervo):
        prep = await p2_acervo.prepare("Tell me about this book", [])
        assert prep.warm_tokens > 0
        assert prep.has_context

    @pytest.mark.asyncio
    async def test_s3_overview_mentions_sherlock(self, p2_acervo):
        prep = await p2_acervo.prepare("What is this collection about?", [])
        content = prep.warm_content.lower()
        has_sherlock = "sherlock" in content or "holmes" in content
        assert has_sherlock, (
            f"Overview should mention Sherlock Holmes. Preview: {content[:200]}"
        )


# ══════════════════════════════════════════════════════════════
# P3 — PM Docs (markdown project)
# ══════════════════════════════════════════════════════════════


class TestStepsP3:
    """S1/S2/S3 behavior for P3 PM docs project."""

    @pytest.mark.asyncio
    async def test_s1_overview_intent(self, p3_acervo):
        prep = await p3_acervo.prepare("Show all the project documents", [])
        intent = prep.debug.get("s1_detection", {}).get("intent", "")
        assert intent == "overview", f"Expected overview, got '{intent}'"

    @pytest.mark.asyncio
    async def test_s1_specific_intent(self, p3_acervo):
        prep = await p3_acervo.prepare("What issues are currently open?", [])
        intent = prep.debug.get("s1_detection", {}).get("intent", "")
        assert intent == "specific", f"Expected specific, got '{intent}'"

    @pytest.mark.asyncio
    async def test_s3_has_context(self, p3_acervo):
        prep = await p3_acervo.prepare("What tech stack does the project use?", [])
        assert prep.warm_tokens > 0
        assert prep.has_context

    @pytest.mark.asyncio
    async def test_s3_budget_respected(self, p3_acervo):
        prep = await p3_acervo.prepare("What are the deadlines?", [])
        assert prep.warm_tokens <= 1200

    @pytest.mark.asyncio
    async def test_s3_overview_has_project_info(self, p3_acervo):
        prep = await p3_acervo.prepare("Give me a project overview", [])
        content = prep.warm_content.lower()
        has_project_info = "todo" in content or "project" in content or "sprint" in content
        assert has_project_info, (
            f"Overview missing project info. Preview: {content[:200]}"
        )
