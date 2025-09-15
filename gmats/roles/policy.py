from __future__ import annotations
"""
Threshold Policy
================
Maps a scalar score into BUY/HOLD/SELL decisions with margins.
"""

from typing import Final
from ..core.interfaces import Policy, Decision

class ThresholdPolicy(Policy):
    """BUY if score > buy_thr; SELL if score < sell_thr; else HOLD."""

    def __init__(self, buy_thr: float = 0.10, sell_thr: float = -0.10):
        self.buy_thr: Final[float] = float(buy_thr)
        self.sell_thr: Final[float] = float(sell_thr)

    def decide(self, score: float) -> Decision:
        """Return a strict Decision (TypedDict) to satisfy the Policy Protocol."""
        if score > self.buy_thr:
            return Decision(action="BUY", margin=float(score - self.buy_thr))
        if score < self.sell_thr:
            return Decision(action="SELL", margin=float(abs(score - self.sell_thr)))
        return Decision(action="HOLD", margin=0.0)
