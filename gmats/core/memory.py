from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import math, time

from gmats.core.interfaces import MemoryStore as MemoryStoreProto  

class MemoryStore(MemoryStoreProto): 
    """
    Minimal memory with (admit, promote, decay, retrieve).
    """
    def __init__(self):
        self._items: Dict[str, Dict[str, Any]] = {}

    def admit(self, item: Dict[str, Any]) -> bool:
        k = str(item.get("id") or f"k_{len(self._items)}")
        item = dict(item)
        item.setdefault("ts", time.time())
        item.setdefault("score", 1.0)
        self._items[k] = item
        return True

    def promote(self, key: str) -> None:
        if key in self._items:
            self._items[key]["score"] = float(self._items[key].get("score", 1.0)) + 1.0

    def decay(self, dt: float) -> None:
        if dt <= 0: return
        for it in self._items.values():
            it["score"] = float(it.get("score", 1.0)) * math.exp(-dt)

    def retrieve(self, query: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]:
        tag = query.get("tag"); sym = query.get("symbol")
        items = list(self._items.values())
        def _score(it: Dict[str, Any]) -> float:
            s = float(it.get("score", 1.0))
            if tag and tag in it.get("tags", []): s += 0.5
            if sym and sym == it.get("symbol"): s += 0.5
            return s
        items.sort(key=_score, reverse=True)
        return items[:k]
