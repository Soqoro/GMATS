# gmats/attack/rl_env.py
from __future__ import annotations
import datetime as dt
from typing import Dict, Any, Tuple, List
import numpy as np
import hashlib

from gmats.attack.overlay import AttackOverlay, set_global_overlay
from gmats.attack.quantizer import to_label
from gmats.attack.renderer import render_attack_post
from gmats.adapters.finsaber_dataset import GMATSDataset
from gmats.adapters.finsaber_strategy import GMATSLLMStrategy
from gmats.core.config import load_config

# ---- RL obs/action tuning knobs ----
RET_WINDOW = 20
TRADE_GAIN = 1.0

def _ingestion_id(sym: str, date: str, text: str) -> str:
    """
    Match llm_agent._hash_id('synthetic', sym, date, text) exactly, so
    poison_ids.jsonl lines align with analyst ranked_ids/consumed_ids.
    """
    h = hashlib.sha1(f"{sym}|{date}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"synthetic:{sym}:{date}:{h}"

class RLAttackEnv:
    def __init__(self, config_path: str, data_root: str):
        self.config_path = config_path  # keep for renderer
        self.cfg = load_config(config_path)
        self.data = GMATSDataset(data_root, market_dir=self.cfg.data["market_dir"])
        self.assets = [s.upper() for s in self.cfg.assets]

        self.overlay = AttackOverlay()
        set_global_overlay(self.overlay)

        self.strategy = GMATSLLMStrategy(
            config_path=config_path,
            data_root=data_root,
            log_dates=True
        )
        try:
            self.strategy.social.overlay = self.overlay  # type: ignore[attr-defined]
        except Exception:
            pass

        self.dates = self._dates_trading_only()
        self._t = 0
        self._equity = 1.0
        self._prev_orders: List[Dict[str, Any]] = []  # recent orders window

    def _dates_trading_only(self) -> List[str]:
        def _as_date(x):
            if isinstance(x, dt.date):
                return x
            return dt.date.fromisoformat(str(x))
        d0 = _as_date(self.cfg.schedule["date_from"])
        d1 = _as_date(self.cfg.schedule["date_to"])

        calendars: List[set] = []
        for sym in self.assets:
            df = self.data._load(sym)
            ds = [str(x) for x in df["date"].astype(str).tolist()]
            calendars.append(set(ds))

        common = sorted(list(set.intersection(*calendars))) if calendars else []
        common = [d for d in common if d0.isoformat() <= d <= d1.isoformat()]
        if common:
            common = common[:-1]
        return common

    # ---- API ----
    def reset(self) -> Dict[str, Any]:
        self.overlay = AttackOverlay()
        set_global_overlay(self.overlay)
        try:
            self.strategy.social.overlay = self.overlay  # type: ignore[attr-defined]
        except Exception:
            pass
        self._t = 0
        self._equity = 1.0
        self._prev_orders = []  # reset recent orders window
        return self._observe()

    def _observe(self) -> Dict[str, Any]:
        cur_date = self.dates[self._t]
        obs_vec = self._obs_vector_for_date(cur_date)
        return {"date": cur_date, "assets": self.assets, "obs_vec": obs_vec}

    def _obs_vector_for_date(self, date: str) -> np.ndarray:
        out: List[float] = []
        d_end = dt.date.fromisoformat(date)
        d_start = (d_end - dt.timedelta(days=RET_WINDOW))
        s = d_start.isoformat()
        e = (d_end - dt.timedelta(days=1)).isoformat()

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

            if len(rets) < RET_WINDOW:
                rets = [0.0] * (RET_WINDOW - len(rets)) + rets
            else:
                rets = rets[-RET_WINDOW:]

            out.extend(rets)

        return np.asarray(out, dtype=np.float32)

    def step(self, a_trade: float, a_sent: float, *, seed_key: str = "ep0") -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        date = self.dates[self._t]

        # 1) Inject ONE synthetic post per asset using the sentiment knob
        label = to_label(a_sent)
        injected: List[Dict[str, Any]] = []
        for sym in self.assets:
            post = render_attack_post(
                date=date,
                asset=sym,
                label=label,
                seed_key=seed_key,
                cfg_path=self.config_path,          # lets renderer resolve configs/prompts.yaml
                orders_window=self._prev_orders,    # populates {ORDERS_WINDOW}
            )

            # Ensure non-empty text (paranoid guard)
            txt = (post.get("text") or "").strip()
            if not txt:
                txt = f"Note: ${sym} mixed signals into close. #{label.replace(' ', '_')}"
                post["text"] = txt

            # Stable internal id (attk_...) for your own tracking
            pid = post.get("id")
            if not pid:
                pid = "atk_" + hashlib.md5(f"{date}|{sym}|{label}|{seed_key}".encode("utf-8")).hexdigest()[:12]
                post["id"] = pid

            # This is the ID that llm_agent will also compute when logging ingestion
            log_id = _ingestion_id(sym, date, post["text"])

            # Inject into overlay so the analyst can see it
            self.overlay.inject(date=date, asset=sym, post=post)

            # Record full manifest info; run_attack.py will append these rows to poison_ids.jsonl
            injected.append({
                "date": date,
                "symbol": sym,
                "id": log_id,               # <-- matches analyst ranked_ids/consumed_ids
                "attk_id": pid,             # <-- keep original deterministic id as well
                "label": label,
                "text": post["text"],
            })

        # 2) Run GMATS per asset
        lev = float(np.clip(1.0 + TRADE_GAIN * float(a_trade), 0.0, 2.0))
        orders_eff: List[Dict[str, Any]] = []
        pnl = 0.0

        for sym in self.assets:
            orders_sym = self.strategy.decide_for_date(date, [sym])
            for o in orders_sym:
                s = o["symbol"].upper()
                w = float(o.get("weight", 0.0))
                w_eff = float(np.clip(w * lev, -1.0, 1.0))
                orders_eff.append({"symbol": s, "weight": w_eff})
                ret = self.data.get_next_day_return(s, date)
                pnl += w_eff * ret

        # 3) Update equity / info
        next_equity = self._equity * (1.0 + pnl)
        r = next_equity - self._equity
        self._equity = next_equity
        info = {
            "orders": orders_eff,
            "label": label,
            "equity": self._equity,
            "injected": injected,
            "a_trade": float(a_trade),
            "a_sent": float(a_sent),
            "lev": lev,
        }

        # keep a short recent orders window for next prompt
        self._prev_orders = orders_eff[-10:]

        # 4) Advance day
        self._t += 1
        done = (self._t >= len(self.dates))
        if done:
            return {}, r, True, info

        # 5) Next observation
        obs_next = self._observe()
        return obs_next, r, False, info
