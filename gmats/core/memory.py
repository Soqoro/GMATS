from __future__ import annotations
"""
Memory store for GMATS
======================
Implements a simple Memory with:
- admit(): intake filter
- promote(): optional deep/long-term promotion
- retrieve(): recency/relevance ranking with decay
- stats(): counters for instrumentation (p_adm, promotions, size)
"""

from typing import List, Dict, Any
import math, uuid
from dataclasses import dataclass, field
from ..core.interfaces import Memory as MemoryProtocol, MemoryItem

@dataclass
class MemItem:
    """Internal representation of a memory item."""
    id: str
    text: str
    ts: float
    cred: float = 0.5
    imp: float = 0.5
    deep: bool = False

@dataclass
class MemoryStore(MemoryProtocol):
    """A minimal in-process memory with decay-based retrieval."""
    deep_threshold: float = 0.75
    decay: float = 0.98
    items: List[MemItem] = field(default_factory=list)
    admitted: int = 0
    deep_promotions: int = 0

    def admit(self, item: Dict[str, Any]) -> bool:
        """Accept items that have enough content (toy policy)."""
        txt = item.get("text","")
        if not txt or len(txt) < 16:
            return False
        self.items.append(MemItem(
            id=str(uuid.uuid4()), text=txt, ts=float(item.get("ts",0.0)),
            cred=float(item.get("cred",0.5)), imp=float(item.get("imp",0.5))
        ))
        self.admitted += 1
        return True

    def promote(self, item_id: str, score: float) -> None:
        """Promote an item to deep memory if score >= threshold."""
        for it in self.items:
            if it.id == item_id and (not it.deep) and score >= self.deep_threshold:
                it.deep = True
                self.deep_promotions += 1
                return

    def retrieve(self, query: Dict[str, Any], k: int) -> List[MemoryItem]:
        """Rank items by decayed recency * relevance and return top-k."""
        tnow = float(query.get("ts", 0.0))
        ranked = []
        for it in self.items:
            age = max(0.0, tnow - it.ts)
            rec = self.decay ** age                         # recency decay
            rel = 0.5*it.cred + 0.5*it.imp + (0.1 if it.deep else 0.0)
            score = rec * rel
            ranked.append((score, it))
        ranked.sort(key=lambda x: x[0], reverse=True)
        out: List[MemoryItem] = [
            MemoryItem(id=it.id, text=it.text, deep=it.deep) for _, it in ranked[:k]
        ]
        return out

    def stats(self) -> Dict[str, float]:
        """Return simple counters for instrumentation/metrics."""
        return {
            "admitted": float(self.admitted),
            "deep_promotions": float(self.deep_promotions),
            "size": float(len(self.items))
        }
