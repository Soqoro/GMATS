from __future__ import annotations
"""
Simple Alpha Miner
==================
Toy implementation that derives a scalar sentiment-ish factor from text.
"""

from typing import List, Dict, Any
from ..core.interfaces import AlphaMiner

class SimpleAlphaMiner(AlphaMiner):
    """Count a few positive/negative tokens and emit a small factor vector."""

    def __init__(self, k: int = 3):
        """
        Args:
            k: Number of factors to emit. Only the first carries signal in this toy.
        """
        self.k = int(k)

    def factors(self, evidence: List[Dict[str, Any]]) -> List[float]:
        """Produce k factors from evidence (first element carries weak sentiment)."""
        pos_words = ("beat","growth","up","gain","positive")
        neg_words = ("miss","down","loss","weak","negative")
        s = 0.0
        for e in evidence[:10]:
            t = (e.get("text","") or "").lower()
            s += sum(w in t for w in pos_words) * 0.05
            s -= sum(w in t for w in neg_words) * 0.05
        return [float(s)] + [0.0]*(self.k-1)

    def score(self, factors: List[float]) -> float:
        """Map factors to a single scalar via the mean (toy)."""
        return float(sum(factors) / max(1, len(factors)))
