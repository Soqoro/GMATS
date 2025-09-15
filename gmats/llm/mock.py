from __future__ import annotations
"""
Mock LLM
========
Deterministic, fast LLM replacement for offline tests and CI.
"""

from ..core.interfaces import LLM

class MockLLM(LLM):
    """A trivial LLM that truncates summaries and picks the shorter answer."""

    def summarize(self, prompt: str) -> str:
        """Return the first ~40 tokens as a 'summary'."""
        return " ".join(prompt.split()[:40])

    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str:
        """Pick 'A' if A is shorter or equal; otherwise 'B'."""
        return "A" if len(a) <= len(b) else "B"
