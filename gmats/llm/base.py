from __future__ import annotations
"""
LLM base helpers (optional)
===========================
Defines a light Judge protocol used by coordinators.

Any class implementing:
    - summarize(prompt: str) -> str
    - choose(context: str, a: str, b: str, criterion: str = "...") -> str
is considered a valid Judge.
"""

from typing import Protocol


class Judge(Protocol):
    def summarize(self, prompt: str) -> str: ...
    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str: ...
