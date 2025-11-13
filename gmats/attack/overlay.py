from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import datetime as dt

class AttackOverlay:
    """
    Private buffer of synthetic posts keyed by (date_str, ASSET).
    Each post should look like SocialLoader output: {"date", "tweet"/"text", "source", ...}
    """
    def __init__(self):
        # _buf[date_str][ASSET] -> [post, ...]
        self._buf: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
        # NEW: initialize bias map (you already call set_bias/get_bias)
        self._bias: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)

    def inject(self, *, date: str, asset: str, post: dict) -> None:
        """Insert a single synthetic post for (date, asset)."""
        self._buf[date][asset.upper()].append(post)

    def get_for(self, *, date: str, asset: str) -> List[dict]:
        """All posts exactly on `date` for `asset`."""
        return list(self._buf.get(date, {}).get(asset.upper(), []))
    
    def set_bias(self, *, date: str, asset: str, delta_social: float = 0.0, delta_news: float = 0.0) -> None:
        self._bias[date][asset.upper()] = (float(delta_social), float(delta_news))

    def get_bias(self, *, date: str, asset: str) -> Tuple[float, float]:
        return self._bias.get(date, {}).get(asset.upper(), (0.0, 0.0))
    
    def fetch_window(self, *, asset: str, start_date: str, end_date: str) -> List[dict]:
        """All posts for `asset` in inclusive [start_date, end_date]."""
        a = asset.upper()
        out: List[dict] = []
        d0 = dt.date.fromisoformat(start_date)
        d1 = dt.date.fromisoformat(end_date)
        cur = d0
        while cur <= d1:
            out.extend(self._buf.get(cur.isoformat(), {}).get(a, []))
            cur += dt.timedelta(days=1)
        return out

    def clear_day(self, date: str) -> None:
        """Remove all injected posts on `date` (optional utility)."""
        self._buf.pop(date, None)

    def get_ids_for_date(self, date: str) -> List[str]:
        out: List[str] = []
        for posts in self._buf.get(date, {}).values():
            for p in posts:
                pid = p.get("id")
                if isinstance(pid, str) and pid:
                    out.append(pid)
        return out

# ---------- Module-level global registry (so agents can find the overlay) ----------
_GLOBAL_OVERLAY = None

def set_global_overlay(ov):
    global _GLOBAL_OVERLAY
    _GLOBAL_OVERLAY = ov

def get_global_overlay():
    return _GLOBAL_OVERLAY
