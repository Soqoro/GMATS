from __future__ import annotations
"""
Coordinator (Debate)
====================
Aggregates bull/bear messages into a stance using either a numeric fallback
or an LLM judge. Signature matches the Coordinator protocol exactly.
"""

from typing import Iterable, Optional, List
from ..core.interfaces import Coordinator, Message, CoordinationResult
from ..llm.base import Judge  # structural protocol: must have .choose(...)


class DebateCoordinator(Coordinator):
    """Debate coordinator that selects a winner between bull/bear arguments."""

    def __init__(self, llm: Optional[Judge] = None, criterion: Optional[str] = None):
        """
        Args:
            llm: Optional LLM judge. If None, fallback uses numeric 'score' fields.
            criterion: Optional text shown to the judge (may be pre-formatted).
        """
        self.llm = llm
        self.criterion = criterion or "Which argument better predicts near-term performance?"

    def _score(self, m: Optional[Message]) -> float:
        if not m:
            return 0.0
        try:
            return float((m.get("payload") or {}).get("score", 0.0))
        except Exception:
            return 0.0

    def coordinate(self, inbox: Iterable[Message]) -> CoordinationResult:
        """
        Args:
            inbox: Iterable of Message with exactly one 'bull' and one 'bear' ideally.
        Returns:
            CoordinationResult with signed stance and metadata.
        """
        msgs: List[Message] = list(inbox)

        # Extract bull/bear messages
        bull = next((m for m in msgs if (m.get("payload") or {}).get("role") == "bull"), None)
        bear = next((m for m in msgs if (m.get("payload") or {}).get("role") == "bear"), None)

        # Numeric fallback
        s_bull = self._score(bull)
        s_bear = self._score(bear)
        winner = "bull" if s_bull >= s_bear else "bear"
        margin = abs(s_bull - s_bear)

        # LLM judge if available (uses provided speeches/context if present)
        if self.llm and bull and bear:
            context = str((bull.get("payload") or {}).get("context", "") or (bear.get("payload") or {}).get("context", ""))
            a = str((bull.get("payload") or {}).get("speech", "") or "")
            b = str((bear.get("payload") or {}).get("speech", "") or "")
            pick = (self.llm.choose(context=context, a=a, b=b, criterion=self.criterion) or "").strip().upper()
            winner = "bull" if pick.startswith("A") else "bear"
            # keep numeric margin as simple strength proxy

        s = [+margin if winner == "bull" else -margin]
        rho = {"mode": "debate", "winner": winner, "margin": float(margin)}
        return CoordinationResult(s=s, rho=rho, log=msgs)
