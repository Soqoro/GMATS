from __future__ import annotations
"""
Coordinator (Debate)
====================
Aggregates two opposing messages (bull vs bear) into a stance
using either a numeric fallback or an LLM judge.
"""

from typing import List
from ..core.interfaces import Coordinator, LLM, Message, Stance

class DebateCoordinator(Coordinator):
    """Debate coordinator that selects a winner between bull/bear arguments."""

    def __init__(self, llm: LLM | None = None):
        """
        Args:
            llm: Optional LLM judge. If None, a numeric fallback is used based on 'score'.
        """
        self.llm = llm

    def aggregate(self, messages: List[Message]) -> Stance:
        """Aggregate role messages into a winning stance and margin.

        Expects:
            messages: list with exactly one 'bull' and one 'bear' Message.
        """
        bull = next((m for m in messages if m.get("role") == "bull"), None)
        bear = next((m for m in messages if m.get("role") == "bear"), None)

        def _fallback() -> Stance:
            s_bull = float(bull.get("score", 0.0)) if bull else 0.0
            s_bear = float(bear.get("score", 0.0)) if bear else 0.0
            winner = "bull" if s_bull >= s_bear else "bear"
            return Stance(winner=winner, margin=abs(s_bull - s_bear))

        if not bull or not bear:
            return _fallback()

        if self.llm is None:
            return _fallback()

        # LLM judge
        pick = self.llm.choose(
            context=bull.get("context", "") or "",
            a=bull.get("speech", "") or "",
            b=bear.get("speech", "") or "",
            criterion="Which argument better predicts near-term performance?"
        )
        winner = "bull" if pick.strip().upper().startswith("A") else "bear"
        margin = abs(float(bull.get("score", 0.0)) - float(bear.get("score", 0.0)))
        return Stance(winner=winner, margin=margin)
