# gmats/data/social_loader.py
import json
import datetime as dt
from typing import Any, Dict, List, Optional

class SocialLoader:
    def __init__(self, root, social_dir, assets: Optional[List[str]] = None, overlay=None):
        self.root, self.dir = root, social_dir
        self.assets = [s.upper() for s in (assets or [])]
        self.overlay = overlay  # AttackOverlay or None

    def slice(self, symbol: str, date: str, window_days: int, top_k: int):
        path = f"{self.root}/{self.dir}/{symbol}.jsonl"
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # normalize 'tweet' -> 'text'
                    if "text" not in obj and "tweet" in obj:
                        obj["text"] = obj["tweet"]
                    obj["_i"] = i
                    out.append(obj)
        except FileNotFoundError:
            return []
        D = dt.date.fromisoformat(date)
        start = D - dt.timedelta(days=window_days - 1)

        def _to_date(v: Any):
            try:
                return dt.date.fromisoformat(str(v))
            except Exception:
                return None

        out = [o for o in out if (d := _to_date(o.get("date"))) and (start <= d <= D)]
        if self.overlay is not None:
            out.extend(self.overlay.get_for(date=date, asset=symbol))
        return out[-top_k:] if top_k else out

    def observe(self, date: str, *, window_days: int = 3, top_k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        return {sym: self.slice(sym, date, window_days, top_k) for sym in self.assets}
