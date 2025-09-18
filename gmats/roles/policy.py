from __future__ import annotations
"""
Threshold Policy
================
Maps a scalar score (from state `s` or factors `f`) into BUY/HOLD/SELL.

New interface (matches gmats.core.interfaces.Policy):
    decide(self, s: List[float], f: Optional[List[float]], state_t: Dict[str, Any])
        -> List[Action]  # e.g., ["BUY"] | ["SELL"] | ["HOLD"]
"""

from typing import Any, Dict, List, Optional, Sequence
from ..core.interfaces import Policy, Action


class ThresholdPolicy(Policy):
    """
    BUY if score > buy_thr; SELL if score < sell_thr; else HOLD.

    Args:
        buy_thr: threshold above which we BUY
        sell_thr: threshold below which we SELL
        source: which vector to read the score from: "state" or "factors"
        index: which coordinate to use as the score
    """

    def __init__(
        self,
        buy_thr: float = 0.10,
        sell_thr: float = -0.10,
        source: str = "state",   # "state" or "factors"
        index: int = 0,
    ):
        self.buy_thr = float(buy_thr)
        self.sell_thr = float(sell_thr)
        self.source = source
        self.index = int(index)

    def _pick_score(
        self,
        s: Sequence[float],
        f: Optional[Sequence[float]],
    ) -> Optional[float]:
        vec: Optional[Sequence[float]] = None
        if self.source == "factors" and f:
            vec = f
        elif self.source == "state" and s:
            vec = s
        elif f:  # fallback to factors if state is empty
            vec = f
        elif s:  # or fallback to state if factors empty
            vec = s

        if not vec:
            return None
        i = self.index if 0 <= self.index < len(vec) else len(vec) - 1
        try:
            return float(vec[i])
        except Exception:
            return None

    def decide(
        self,
        s: List[float],
        f: Optional[List[float]],
        state_t: Dict[str, Any]
    ) -> List[Action]:
        score = self._pick_score(s, f)
        if score is None:
            return ["HOLD"]

        if score > self.buy_thr:
            return ["BUY"]
        if score < self.sell_thr:
            return ["SELL"]
        return ["HOLD"]
