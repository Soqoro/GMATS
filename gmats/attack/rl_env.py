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

        # Attack overlay (active in attack path only)
        self.overlay = AttackOverlay()
        set_global_overlay(self.overlay)

        # Attack strategy (uses overlay)
        self.strategy = GMATSLLMStrategy(
            config_path=config_path,
            data_root=data_root,
            log_dates=True
        )
        try:
            self.strategy.social.overlay = self.overlay  # type: ignore[attr-defined]
        except Exception:
            pass

        # Clean strategy (no overlay)
        self.strategy_clean = GMATSLLMStrategy(
            config_path=config_path,
            data_root=data_root,
            log_dates=True
        )
        try:
            self.strategy_clean.social.overlay = None  # type: ignore[attr-defined]
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
            if os.getenv("ATTACK_LLM_BASE_URL"): acfg["base_url"] = os.getenv("ATTACK_LLM_BASE_URL")
            if os.getenv("ATTACK_LLM_HOST"): acfg["host"] = os.getenv("ATTACK_LLM_HOST")
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

    # ---------------- Introspection helpers ----------------
    @staticmethod
    def _iter_events_from_strategy(strategy: Any):
        """
        Best-effort: find emitted events with payloads.
        Tries common attributes used by logging utilities.
        """
        # strategy.logger.buffer (preferred)
        try:
            buf = getattr(getattr(strategy, "logger", None), "buffer", None)
            if isinstance(buf, list):
                for ev in buf:
                    if isinstance(ev, dict):
                        yield ev
        except Exception:
            pass
        # fallbacks
        for name in ("_events", "events", "last_events", "emitted", "log_buffer"):
            try:
                evs = getattr(strategy, name, None)
                if isinstance(evs, list):
                    for ev in evs:
                        if isinstance(ev, dict):
                            yield ev
            except Exception:
                pass

    @classmethod
    def _latest_scorecard_for_agent(cls, strategy: Any, agent_id: str) -> Optional[List[dict]]:
        sc = None
        for ev in cls._iter_events_from_strategy(strategy):
            if ev.get("kind") == "agent_out" and ev.get("agent_id") == agent_id:
                payload = ev.get("payload_json")
                if isinstance(payload, dict) and isinstance(payload.get("scorecard"), list):
                    sc = payload["scorecard"]
        return sc

    @classmethod
    def _latest_ingested_ids(cls, strategy: Any) -> Optional[List[str]]:
        """
        Return last 'consumed_ids' (or 'ranked_ids') from social_analyst ingestion.
        """
        ids = None
        for ev in cls._iter_events_from_strategy(strategy):
            if ev.get("kind") == "ingestion" and ev.get("agent_id") == "social_analyst":
                # prefer consumed_ids
                lst = ev.get("consumed_ids")
                if isinstance(lst, list) and lst:
                    ids = [str(x) for x in lst]
                else:
                    lst = ev.get("ranked_ids")
                    if isinstance(lst, list) and lst:
                        ids = [str(x) for x in lst]
        return ids

    @staticmethod
    def _score_for_symbol(scorecard: Optional[List[dict]], sym: str) -> Optional[dict]:
        if not isinstance(scorecard, list):
            return None
        S = sym.upper()
        for r in scorecard:
            if isinstance(r, dict) and str(r.get("symbol", "")).upper() == S:
                return r
        return None

    # ---------------- API ----------------
    def reset(self) -> Dict[str, Any]:
        # reset overlays
        self.overlay = AttackOverlay()
        set_global_overlay(self.overlay)
        try:
            self.strategy.social.overlay = self.overlay  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.strategy_clean.social.overlay = None  # type: ignore[attr-defined]
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

        # 2) Run GMATS per asset — ATTACK path
        lev = float(np.clip(1.0 + TRADE_GAIN * float(a_trade), 0.0, 2.0))
        orders_eff: List[Dict[str, Any]] = []
        pnl_attack = 0.0

        for sym in self.assets:
            orders_sym = self.strategy.decide_for_date(date, [sym])
            for o in orders_sym:
                s = o["symbol"].upper()
                w = float(o.get("weight", 0.0))
                w_eff = float(np.clip(w * lev, -1.0, 1.0))
                orders_eff.append({"symbol": s, "weight": w_eff})
                ret = self.data.get_next_day_return(s, date)
                pnl_attack += w_eff * ret

        # Capture attack scorecards & ingested ids (best effort)
        analyst_sc_attack = self._latest_scorecard_for_agent(self.strategy, "social_analyst")
        coord_sc_attack   = self._latest_scorecard_for_agent(self.strategy, "coordinator")
        ingested_ids_attack = self._latest_ingested_ids(self.strategy)

        # 3) Run GMATS per asset — CLEAN path (no overlay, no leverage)
        # Temporarily disable global overlay to ensure clean run ignores injections
        set_global_overlay(None)
        try:
            pnl_clean = 0.0
            orders_clean_eff: List[Dict[str, Any]] = []
            for sym in self.assets:
                orders_sym = self.strategy_clean.decide_for_date(date, [sym])
                for o in orders_sym:
                    s = o["symbol"].upper()
                    w = float(o.get("weight", 0.0))
                    w_eff = float(np.clip(w * 1.0, -1.0, 1.0))  # lev_clean = 1.0 (attacker absent)
                    orders_clean_eff.append({"symbol": s, "weight": w_eff})
                    ret = self.data.get_next_day_return(s, date)
                    pnl_clean += w_eff * ret
            # Capture clean scorecards
            analyst_sc_clean = self._latest_scorecard_for_agent(self.strategy_clean, "social_analyst")
            coord_sc_clean   = self._latest_scorecard_for_agent(self.strategy_clean, "coordinator")
        finally:
            # Restore global overlay for future steps
            set_global_overlay(self.overlay)

        # 4) Update equity / info (attack path is the one actually “executed”)
        next_equity = self._equity * (1.0 + pnl_attack)
        r_attack = next_equity - self._equity
        self._equity = next_equity

        # Build info for reward wrapper (includes everything needed for IR@k + BSS)
        info: Dict[str, Any] = {
            "orders": orders_eff,
            "label": label,
            "equity": self._equity,
            "injected": injected,
            "a_trade": float(a_trade),
            "a_sent": float(a_sent),
            "lev": lev,
            # --- perf delta support ---
            "pnl_clean": float(pnl_clean),
            # --- symbols for BSS symbol disambiguation ---
            "symbols": self.assets[:],  # reward wrapper will pick the first if needed
            # --- IR@k support ---
            "ingested_topk_ids": ingested_ids_attack or [],
            # Expose attack post IDs for matching:
            "attack_posts": [{"id": x["attk_id"], "symbol": x["symbol"], "date": x["date"]} for x in injected],
            # --- scorecards for BSS (analyst + coordinator) ---
            "analyst_attack_scorecard": analyst_sc_attack or [],
            "analyst_clean_scorecard": analyst_sc_clean or [],
            "coordinator_attack_scorecard": coord_sc_attack or [],
            "coordinator_clean_scorecard": coord_sc_clean or [],
        }

        # Also include convenience top-level mus for the first symbol (if discoverable)
        try:
            sym0 = self.assets[0]
            ra = self._score_for_symbol(analyst_sc_attack, sym0) or {}
            rc = self._score_for_symbol(analyst_sc_clean, sym0) or {}
            ka = self._score_for_symbol(coord_sc_attack, sym0) or {}
            kc = self._score_for_symbol(coord_sc_clean, sym0) or {}
            if "mu" in ra and "mu" in rc:
                info["analyst_mu_attack"] = float(ra.get("mu"))
                info["analyst_mu_clean"] = float(rc.get("mu"))
            if "mu" in ka and "mu" in kc:
                info["coordinator_mu_attack"] = float(ka.get("mu"))
                info["coordinator_mu_clean"] = float(kc.get("mu"))
        except Exception:
            pass

        self._prev_orders = orders_eff[-10:]

        # 5) Advance day
        self._t += 1
        done = (self._t >= len(self.dates))
        if done:
            return {}, float(r_attack), True, info

        # 6) Next observation
        obs_next = self._observe()
        return obs_next, float(r_attack), False, info
