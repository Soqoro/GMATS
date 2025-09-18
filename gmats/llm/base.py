from __future__ import annotations
"""
Lightweight Judge protocol used by LLM backends (HF / llama.cpp).
Keep this minimal so backends can be swapped easily and tests can stub it.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    def summarize(self, prompt: str) -> str:
        """Summarize evidence into a short string."""
        ...

    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str:
        """
        Compare A vs B under the given criterion and return 'A' or 'B'.
        Implementations should be robust to extra whitespace / tokens.
        """
        ...
