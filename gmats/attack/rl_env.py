# gmats/attack/rl_env.py
from __future__ import annotations
import os
import datetime as dt
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import hashlib

from gmats.attack.overlay import AttackOverlay, set_global_overlay
from gmats.attack.quantizer import to_label
from gmats.attack.renderer import render_attack_post
from gmats.adapters.finsaber_dataset import GMATSDataset
from gmats.adapters.finsaber_strategy import GMATSLLMStrategy
from gmats.core.config import load_config, AgentSpec
from gmats.llm.provider import make_llm_provider  # <-- IMPORTANT

# ---- RL obs/action tuning knobs ----
RET_WINDOW = 20
TRADE_GAIN = 1.0

def _ingestion_id(sym: str, date: str, text: str) -> str:
    h = hashlib.sha1(f"{sym}|{date}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"synthetic:{sym}:{date}:{h}"

class RLAttackEnv:
    def __init__(self, config_path: str, data_root: str):
        self.config_path = config_path
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

        # Build an attack LLM provider (env → cfg.attack.llm → first agent’s llm)
        self.attack_llm = self._resolve_attack_llm()
        if self.attack_llm is None:
            print("[ATTACK_LLM] No provider configured; using fallback text.", flush=True)

        self.dates = self._dates_trading_only()
        self._t = 0
        self._equity = 1.0
        self._prev_orders: List[Dict[str, Any]] = []

    # ---------------- LLM resolution ----------------
    def _resolve_attack_llm(self):
        # Pull API keys from config (now preserved) or env
        keys = dict(getattr(self.cfg, "llm_keys", {}) or {})

        # 1) Environment overrides (highest priority)
        env_provider = os.getenv("ATTACK_LLM_PROVIDER")
        if env_provider:
            acfg: Dict[str, Any] = {
                "provider": env_provider,
                "model": os.getenv("ATTACK_LLM_MODEL", ""),
                "temperature": float(os.getenv("ATTACK_LLM_TEMPERATURE", "0.0")),
            }
            # OpenAI-compatible base URL (vLLM, llama.cpp server, Ollama /v1, etc.)
            if os.getenv("ATTACK_LLM_BASE_URL"): acfg["base_url"] = os.getenv("ATTACK_LLM_BASE_URL")
            # Ollama python client host (if using provider="ollama")
            if os.getenv("ATTACK_LLM_HOST"): acfg["host"] = os.getenv("ATTACK_LLM_HOST")
            # HF options
            if os.getenv("ATTACK_LLM_DEVICE"): acfg["device"] = os.getenv("ATTACK_LLM_DEVICE")
            if os.getenv("ATTACK_LLM_DTYPE"): acfg["dtype"] = os.getenv("ATTACK_LLM_DTYPE")
            if os.getenv("ATTACK_LLM_TRUST_REMOTE_CODE"):
                acfg["trust_remote_code"] = os.getenv("ATTACK_LLM_TRUST_REMOTE_CODE", "false").lower() in ("1","true","yes")
            try:
                llm = make_llm_provider(acfg, keys)
                print(f"[ATTACK_LLM] Using env provider={acfg.get('provider')} model={acfg.get('model')}", flush=True)
                return llm
            except Exception as e:
                print(f"[ATTACK_LLM] Failed to init env provider: {e}", flush=True)

        # 2) YAML: attack.llm block
        attack_cfg = getattr(self.cfg, "attack", {}) or {}
        llm_cfg = attack_cfg.get("llm") if isinstance(attack_cfg, dict) else None
        if llm_cfg:
            try:
                llm = make_llm_provider(llm_cfg, keys)
                print(f"[ATTACK_LLM] Using yaml provider={llm_cfg.get('provider')} model={llm_cfg.get('model')}", flush=True)
                return llm
            except Exception as e:
                print(f"[ATTACK_LLM] Failed to init yaml provider: {e}", flush=True)

        # 3) Fallback: reuse first agent’s llm block
        try:
            agents = list(getattr(self.cfg, "agents", []) or [])
            if agents:
                # Agents are AgentSpec; llm block is preserved in params
                first = agents[0]
                if isinstance(first, AgentSpec):
                    first_llm = (first.params or {}).get("llm")
                elif isinstance(first, dict):
                    first_llm = first.get("llm")
                else:
                    first_llm = None
                if first_llm:
                    llm = make_llm_provider(first_llm, keys)
                    print(f"[ATTACK_LLM] Fallback to first agent provider={first_llm.get('provider')} model={first_llm.get('model')}", flush=True)
                    return llm
        except Exception as e:
            print(f"[ATTACK_LLM] Fallback init failed: {e}", flush=True)

        # 4) Nothing configured
        return None

    # ---------------- Dates ----------------
    def _dates_trading_only(self) -> List[str]:
        def _as_date(x):
            if isinstance(x, dt.date): return x
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

    # ---------------- API ----------------
    def reset(self) -> Dict[str, Any]:
        self.overlay = AttackOverlay()
        set_global_overlay(self.overlay)
        try:
            self.strategy.social.overlay = self.overlay  # type: ignore[attr-defined]
        except Exception:
            pass
        self._t = 0
        self._equity = 1.0
        self._prev_orders = []
        # Re-resolve in case env changed
        self.attack_llm = self._resolve_attack_llm()
        if self.attack_llm is None:
            print("[ATTACK_LLM] No provider configured; using fallback text.", flush=True)
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
                try: closes.append(float(val))
                except Exception: closes.append(np.nan)

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

        # 1) Inject ONE synthetic post per asset (use attack LLM when available)
        label = to_label(a_sent)
        injected: List[Dict[str, Any]] = []
        for sym in self.assets:
            post = render_attack_post(
                date=date,
                asset=sym,
                label=label,
                seed_key=seed_key,
                cfg_path=self.config_path,
                orders_window=self._prev_orders,
                llm=self.attack_llm,  # <-- THIS IS THE HOOK
            )

            txt = (post.get("text") or "").strip()
            if not txt:
                txt = f"Note: ${sym} mixed signals into close. #{label.replace(' ', '_')}"
                post["text"] = txt

            pid = post.get("id")
            if not pid:
                pid = "atk_" + hashlib.md5(f"{date}|{sym}|{label}|{seed_key}".encode("utf-8")).hexdigest()[:12]
                post["id"] = pid

            log_id = _ingestion_id(sym, date, post["text"])
            self.overlay.inject(date=date, asset=sym, post=post)

            injected.append({
                "date": date,
                "symbol": sym,
                "id": log_id,
                "attk_id": pid,
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

        self._prev_orders = orders_eff[-10:]

        # 4) Advance day
        self._t += 1
        done = (self._t >= len(self.dates))
        if done:
            return {}, r, True, info

        # 5) Next observation
        obs_next = self._observe()
        return obs_next, r, False, info
