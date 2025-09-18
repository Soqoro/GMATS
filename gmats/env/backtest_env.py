from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import pandas as pd

from .backtest import Backtester
from gmats.core.interfaces import Environment, Fills, Action, Weights

class BacktestEnvironment(Environment):
    """
    Environment wrapper over Backtester satisfying 𝓔: (a', x_t) -> (fills, x_{t+1}).

    x_t = {"symbol": str, "asof": "YYYY-MM-DD", "horizon": int}
    a'  = ["BUY"|"SELL"|"HOLD"] or weights (ignored here; we treat >0 -> BUY, <0 -> SELL)
    """

    def __init__(self, bt: Backtester):
        self.bt = bt

    def step(self, a_prime: Union[List[Action], Weights], x_t: Dict[str, Any]) -> Tuple[Fills, Dict[str, Any]]:
        sym = str(x_t.get("symbol"))
        asof = str(x_t.get("asof"))
        horizon = int(x_t.get("horizon", 1))

        # Interpret actions: single-name for now
        if isinstance(a_prime, list) and a_prime and isinstance(a_prime[0], str):
            a = a_prime[0].upper()
            qty = 1 if a == "BUY" else (-1 if a == "SELL" else 0)
        else:
            # weights → action
            w = float(a_prime[0]) if a_prime else 0.0
            qty = 1 if w > 0 else (-1 if w < 0 else 0)

        r = self.bt._next_ret(sym, asof, horizon=horizon)
        pnl = float(qty) * r - (self.bt.rf_daily if qty != 0 else 0.0)

        fills = Fills(
            details=[{"symbol": sym, "qty": qty, "ret": r, "pnl": pnl}],
            meta={"rf_daily": self.bt.rf_daily},
        )

        # Advance x_{t+1} to the next data row used for this horizon
        df = self.bt.data.get(sym, None)
        if df is None or df.empty:
            x_next = dict(x_t)  # unchanged
        else:
            d = pd.to_datetime(asof).normalize()
            idx = df.index[df["date"] >= d]
            if len(idx) == 0:
                x_next = dict(x_t)
            else:
                i = int(idx[0])
                j = min(i + int(horizon), len(df) - 1)
                next_date = df["date"].iloc[j]
                x_next = dict(x_t)
                x_next["asof"] = next_date.strftime("%Y-%m-%d")

        return (fills, x_next)
