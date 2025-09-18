from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from .backtest import Backtester
from ..core.interfaces import DataFeed

class BacktesterFeed(DataFeed):
    def __init__(self, bt: Backtester, symbol: str):
        self.bt = bt
        self.symbol = symbol.upper()

    def observe(self, t: Any, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        df = self.bt.data.get(self.symbol)
        if df is None or df.empty:
            return []
        d = pd.to_datetime(t).normalize()
        limit = int(q.get("limit", 64))
        sub = df[df["date"] <= d].tail(limit)
        # Include provenance
        return [
            {"date": row["date"], "ret": float(row["ret"]),
             "prov": {"feed": "backtester", "symbol": self.symbol}}
            for _, row in sub.iterrows()
        ]
