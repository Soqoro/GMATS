from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

class AttackOverlay:
    """
    Private buffer of synthetic posts keyed by (date, asset).
    Shape of each post matches the SocialLoader output fields.
    """
    def __init__(self):
        self._buf: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    def inject(self, *, date: str, asset: str, post: dict) -> None:
        # post MUST include at least: {"date": date, "tweet": "...", "source": "synthetic://..."}
        self._buf[date][asset.upper()].append(post)

    def get_for(self, *, date: str, asset: str) -> List[dict]:
        return list(self._buf.get(date, {}).get(asset.upper(), []))

    def clear_day(self, date: str) -> None:
        self._buf.pop(date, None)
