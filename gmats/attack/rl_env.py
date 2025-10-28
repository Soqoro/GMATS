from __future__ import annotations
import datetime as dt
from typing import Dict, Any, Tuple, List
import numpy as np

from gmats.attack.overlay import AttackOverlay
from gmats.attack.quantizer import to_label
from gmats.attack.renderer import render_attack_post
from gmats.adapters.finsaber_dataset import GMATSDataset
from gmats.adapters.finsaber_strategy import GMATSLLMStrategy
from gmats.core.config import load_config

# ---- RL obs/action tuning knobs ----
RET_WINDOW = 20     # number of past daily returns per asset in observation vector
TRADE_GAIN = 1.0    # maps a_trade into a leverage multiplier: lev = clip(1 + TRADE_GAIN*a_trade, 0..2)

class RLAttackEnv:
    """
    One step = one trading day.
    Action = (trade_knob ∈ [-1,1], sentiment_knob ∈ [-1,1]).
    Reward = PnL delta (paper-faithful). Optionally add contagion shaping.
    """
    def __init__(self, config_path: str, data_root: str):
        self.cfg = load_config(config_path)
        self.data = GMATSDataset(data_root, market_dir=self.cfg.data["market_dir"])
        self.assets = [s.upper() for s in self.cfg.assets]
        self.overlay = AttackOverlay()
        self.strategy = GMATSLLMStrategy(
            config_path=config_path,
            data_root=data_root,
            log_dates=True  # ensure logs/ for contagion metrics
        )
        # Attach overlay to the strategy's social loader if it exposes it
        try:
            self.strategy.social.overlay = self.overlay  # easy path
        except Exception:
            pass
        self.dates = self._dates()
        self._t = 0
        self._equity = 1.0  # start NAV

    def _dates(self) -> List[str]:
        def _as_date(x):
            if isinstance(x, dt.date):
                return x
            return dt.date.fromisoformat(str(x))
        d0 = _as_date(self.cfg.schedule["date_from"])
        d1 = _as_date(self.cfg.schedule["date_to"])
        return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]

    # ---- API ----
    def reset(self) -> Dict[str, Any]:
        self.overlay = AttackOverlay()
        try:
            self.strategy.social.overlay = self.overlay
        except Exception:
            pass
        self._t = 0
        self._equity = 1.0
        return self._observe()

    def _observe(self) -> Dict[str, Any]:
        """Return dict with date/assets and a flat numeric vector 'obs_vec' for SB3."""
        cur_date = self.dates[self._t]
        obs_vec = self._obs_vector_for_date(cur_date)
        return {"date": cur_date, "assets": self.assets, "obs_vec": obs_vec}

    def _obs_vector_for_date(self, date: str) -> np.ndarray:
        """
        Build a stable numeric observation:
          concat over assets of last RET_WINDOW daily returns prior to `date`.
        Pads with zeros when insufficient history. Values ~[-1, +1].
        """
        out: List[float] = []
        d_end = dt.date.fromisoformat(date)
        d_start = (d_end - dt.timedelta(days=RET_WINDOW))
        s = d_start.isoformat()
        e = (d_end - dt.timedelta(days=1)).isoformat()  # strictly before `date`

        for sym in self.assets:
            rows = self.data.get_ticker_data_by_time_range(sym, s, e)
            closes: List[float] = []
            for r in rows:
                val = r.get("close")
                try:
                    closes.append(float(val))
                except Exception:
                    closes.append(np.nan)

            rets: List[float] = []
            for i in range(1, len(closes)):
                p0, p1 = closes[i-1], closes[i]
                if p0 and p1 and not np.isnan(p0) and not np.isnan(p1) and p0 != 0.0:
                    rets.append((p1 / p0) - 1.0)
                else:
                    rets.append(0.0)

            # pad/crop to fixed length
            if len(rets) < RET_WINDOW:
                rets = [0.0] * (RET_WINDOW - len(rets)) + rets
            else:
                rets = rets[-RET_WINDOW:]

            out.extend(rets)

        return np.asarray(out, dtype=np.float32)

    def step(self, a_trade: float, a_sent: float, *, seed_key: str = "ep0") -> Tuple[Dict[str,Any], float, bool, Dict[str,Any]]:
        date = self.dates[self._t]

        # 1) Inject ONE synthetic post per asset using the sentiment knob
        label = to_label(a_sent)
        for sym in self.assets:
            post = render_attack_post(date=date, asset=sym, label=label, seed_key=seed_key)
            self.overlay.inject(date=date, asset=sym, post=post)

        # 2) Run GMATS strategy for THIS day to get base orders (weights in [-1,1])
        orders = self.strategy.decide_for_date(date, self.assets)

        # 2b) Apply trade knob as a leverage multiplier on weights (paper-style), then clamp
        lev = float(np.clip(1.0 + TRADE_GAIN * float(a_trade), 0.0, 2.0))
        orders_eff: List[Dict[str, Any]] = []
        for o in orders:
            sym = o["symbol"].upper()
            w = float(o.get("weight", 0.0))
            w_eff = float(np.clip(w * lev, -1.0, 1.0))
            orders_eff.append({"symbol": sym, "weight": w_eff})

        # 3) Compute PnL on next-day returns (after-costs if available)
        pnl = 0.0
        for o in orders_eff:
            ret = self.data.get_next_day_return(o["symbol"], date)  # helper added in GMATSDataset
            pnl += o["weight"] * ret

        next_equity = self._equity * (1.0 + pnl)
        r = next_equity - self._equity  # delta equity (≈ PnL)
        self._equity = next_equity

        info = {"orders": orders_eff, "label": label, "equity": self._equity}

        # 4) Advance day
        self._t += 1
        done = (self._t >= len(self.dates))
        if done:
            return {}, r, True, info

        # 5) Next observation (numeric vector included)
        obs_next = self._observe()
        return obs_next, r, False, info
