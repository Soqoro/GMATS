from __future__ import annotations
"""
Mock LLM
========
Deterministic, fast LLM replacement for offline tests and CI.
"""

from .base import Judge


class MockLLM(Judge):
    """A trivial LLM that truncates summaries and picks the shorter answer."""

    def summarize(self, prompt: str) -> str:
        return " ".join(prompt.split()[:40])

    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str:
        return "A" if len(a) <= len(b) else "B"
