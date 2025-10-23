# gmats/data/market_loader.py
import csv
import datetime as dt
from typing import Any, Dict, List, Optional

class MarketLoader:
    def __init__(self, root, market_dir, assets: Optional[List[str]] = None):
        self.root, self.dir = root, market_dir
        self.assets = [s.upper() for s in (assets or [])]

    def slice(self, symbol: str, date: str, window_days: int, top_k: int):
        path = f"{self.root}/{self.dir}/{symbol}.csv"
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    rows.append(r)
        except FileNotFoundError:
            return []
        D = dt.date.fromisoformat(date)
        start = D - dt.timedelta(days=window_days - 1)
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                d = dt.date.fromisoformat(str(r.get("date")))
            except Exception:
                continue
            if start <= d <= D:
                out.append(r)
        return out[-top_k:] if top_k else out

    def observe(self, date: str, *, window_days: int = 60, top_k: int = 0) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for sym in self.assets:
            out[sym] = self.slice(sym, date, window_days, top_k)
        return out
