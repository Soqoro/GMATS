# gmats/attack/reward_wrappers.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List
from gmats.attack.overlay import get_global_overlay

class AttackRewardMixer(gym.Wrapper):
    """
    Reward:
        r = -(pnl_attack - pnl_clean)/scale + alpha*IR@k + beta_a*BSS_analyst + beta_c*BSS_coordinator

    Terms:
        pnl_attack: raw env reward (attacked PnL proxy)
        pnl_clean:  env-provided clean PnL (baseline); if missing, assumed 0
        scale: debiased EMA(|pnl_attack - pnl_clean|) with floor
        IR@k: overlap fraction between ingested_topk_ids and attack post IDs
        BSS_analyst: behavior shift of the social_analyst (|mu_attack - mu_clean|)
        BSS_coordinator: behavior shift of the coordinator (|mu_attack - mu_clean|)

    Notes:
        - If the environment supplies info['bss_analyst'] / info['bss_coordinator'] (or 'BSS_*'),
          those are used. Otherwise we attempt to infer from mu deltas using common info shapes.
        - BSS scale matches your metrics: mu ∈ [-1,1] → BSS ∈ [0,2]. Tune beta_a / beta_c accordingly.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 0.25,                 # backward-compat: if beta_a/beta_c not set, both use this
        beta_analyst: float | None = None,  # weight for analyst BSS
        beta_coordinator: float | None = None,  # weight for coordinator BSS
        pnl_scale_ema: float = 0.05,
        pnl_scale_floor: float = 1e-4,
        reward_clip: float | None = None,   # optional: clip final reward to [-clip, clip]
    ):
        super().__init__(env)
        self.k = int(topk)
        self.alpha = float(alpha)
        # handle beta weights
        if beta_analyst is None and beta_coordinator is None:
            self.beta_a = float(beta)
            self.beta_c = float(beta)
        else:
            self.beta_a = float(beta_analyst or 0.0)
            self.beta_c = float(beta_coordinator or 0.0)

        self._ema = float(pnl_scale_ema)
        self._pnl_abs_ema = 1.0
        self._pnl_floor = float(pnl_scale_floor)
        self._t = 0  # steps for EMA debiasing
        self._reward_clip = reward_clip

    # --------- helpers ---------
    @staticmethod
    def _safe_float(x: Any, default: float | None = 0.0) -> float | None:
        try:
            return float(x)
        except Exception:
            return None if default is None else float(default)

    @staticmethod
    def _extract_topk_ids(info: Dict[str, Any], k: int) -> List[str]:
        """
        Accepts a few common shapes:
          - info['ingested_topk_ids'] = ['id1', 'id2', ...]
          - info['ingested_topk']     = [{'id': ...}, ...]
          - info['topk'] / info['Top_k'] similarly
        """
        # Priority 1: explicit list of IDs
        ids = info.get("ingested_topk_ids")
        if isinstance(ids, list) and ids and isinstance(ids[0], (str, bytes)):
            return list(map(str, ids[:k]))

        # Priority 2: list of dicts with 'id'
        for key in ("ingested_topk", "topk", "Top_k"):
            lst = info.get(key)
            if isinstance(lst, list) and lst and isinstance(lst[0], dict):
                out = []
                for d in lst:
                    pid = d.get("id")
                    if isinstance(pid, (str, bytes)) and pid:
                        out.append(str(pid))
                    if len(out) >= k:
                        break
                if out:
                    return out

        return []  # fallback

    @staticmethod
    def _extract_attack_ids(info: Dict[str, Any]) -> List[str]:
        """Prefer explicit 'attack_posts'; otherwise ask the global overlay for today's date."""
        # From info['attack_posts']
        posts = info.get("attack_posts", [])
        out: List[str] = []
        if isinstance(posts, list):
            for p in posts:
                if isinstance(p, dict):
                    pid = p.get("id")
                    if isinstance(pid, (str, bytes)) and pid:
                        out.append(str(pid))
        if out:
            return out

        # Fallback: overlay by date
        date = info.get("date")
        ov = get_global_overlay()
        if ov and isinstance(date, str):
            try:
                return list(ov.get_ids_for_date(date))
            except Exception:
                pass
        return []

    # ---- BSS inference helpers ----
    @staticmethod
    def _first_symbol_from_info(info: Dict[str, Any]) -> str:
        syms = info.get("symbols")
        if isinstance(syms, list) and syms:
            return str(syms[0]).upper()
        sym = info.get("symbol")
        return str(sym).upper() if isinstance(sym, (str, bytes)) else ""

    @staticmethod
    def _find_mu_in_scorecard(rows: Any, symbol: str) -> float | None:
        if not isinstance(rows, list):
            return None
        for r in rows:
            if isinstance(r, dict) and str(r.get("symbol", "")).upper() == symbol:
                if "mu" in r:
                    try:
                        return float(r["mu"])
                    except Exception:
                        return None
        return None

    def _compute_bss_for_agent(self, info: Dict[str, Any], agent: str) -> float:
        """
        Priority:
          1) explicit info['bss_<agent>'] / info['BSS_<agent>'] (e.g., bss_analyst, bss_coordinator)
          2) |mu_attack - mu_clean| from top-level pairs (e.g., analyst_mu_attack/analyst_mu_clean)
          3) paired scorecards for the agent:
               - '<agent>_attack_scorecard' vs '<agent>_clean_scorecard'
               - generic 'attack_scorecard' vs 'clean_scorecard' as fallback
          4) 0.0
        """
        # Normalize agent key labels
        agent_key = agent.lower()
        explicit_keys = [
            f"bss_{agent_key}",
            f"BSS_{agent_key}",
            "bss" if agent_key in ("analyst", "social_analyst") else None,
            "BSS" if agent_key in ("analyst", "social_analyst") else None,
        ]
        for k in explicit_keys:
            if not k:
                continue
            if k in info:
                val = self._safe_float(info.get(k), 0.0)
                return 0.0 if val is None else val

        # 2) simple top-level mu pairs
        pairs = []
        if agent_key in ("analyst", "social_analyst"):
            pairs = [("analyst_mu_attack", "analyst_mu_clean"), ("mu_attack", "mu_clean")]
        elif agent_key == "coordinator":
            pairs = [("coordinator_mu_attack", "coordinator_mu_clean")]

        for a_key, c_key in pairs:
            if a_key in info and c_key in info:
                a = self._safe_float(info[a_key], None)
                c = self._safe_float(info[c_key], None)
                if a is not None and c is not None:
                    return abs(a - c)

        # 3) paired scorecards
        symbol = self._first_symbol_from_info(info)
        scorecard_pairs = [
            (f"{agent_key}_attack_scorecard", f"{agent_key}_clean_scorecard"),
        ]
        # generic fallback names
        if agent_key in ("analyst", "social_analyst"):
            scorecard_pairs += [("attack_scorecard", "clean_scorecard"),
                                ("analyst_attack_scorecard", "analyst_clean_scorecard")]
        if agent_key == "coordinator":
            scorecard_pairs += [("coordinator_attack_scorecard", "coordinator_clean_scorecard")]

        for atk_key, cln_key in scorecard_pairs:
            atk = info.get(atk_key)
            cln = info.get(cln_key)
            if isinstance(atk, list) and isinstance(cln, list):
                mu_a = self._find_mu_in_scorecard(atk, symbol)
                mu_c = self._find_mu_in_scorecard(cln, symbol)
                if (mu_a is not None) and (mu_c is not None):
                    return abs(mu_a - mu_c)

        return 0.0

    # --------- gym API ---------
    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r_attack, terminated, truncated, info = self.env.step(action)

        # --- PnL delta + debiased EMA scale ---
        pnl_attack = self._safe_float(r_attack, 0.0) or 0.0
        pnl_clean = self._safe_float(info.get("pnl_clean", 0.0), 0.0) or 0.0
        pnl_delta = pnl_attack - pnl_clean

        self._t += 1
        self._pnl_abs_ema = (1 - self._ema) * self._pnl_abs_ema + self._ema * abs(pnl_delta)
        # Debias the EMA early in training (avoids tiny scale on first steps)
        debias = 1.0 - (1.0 - self._ema) ** max(self._t, 1)
        scale_raw = self._pnl_abs_ema / max(debias, 1e-9)
        scale = max(scale_raw, self._pnl_floor)
        perf_term = -pnl_delta / scale

        # --- IR@k ---
        topk_ids = self._extract_topk_ids(info, self.k)
        poison_ids = self._extract_attack_ids(info)
        inter = len(set(topk_ids) & set(poison_ids))
        den = max(1, len(topk_ids))  # use actual ingested length
        ir_k = inter / float(den)

        # --- BSS (analyst & coordinator) ---
        bss_analyst = self._compute_bss_for_agent(info, "social_analyst")
        bss_coord   = self._compute_bss_for_agent(info, "coordinator")

        # --- Final reward ---
        shaped = perf_term + self.alpha * ir_k + self.beta_a * bss_analyst + self.beta_c * bss_coord
        if self._reward_clip is not None and self._reward_clip > 0:
            lim = float(self._reward_clip)
            shaped = float(np.clip(shaped, -lim, lim))

        # Diagnostics
        info.setdefault("attack_reward_terms", {})
        info["attack_reward_terms"].update({
            "perf": perf_term,
            "ir_k": ir_k,
            "ir_num": inter,
            "ir_den": den,
            "bss_analyst": bss_analyst,
            "bss_coordinator": bss_coord,
            "beta_a": self.beta_a,
            "beta_c": self.beta_c,
            "scale": scale,
            "pnl_attack": pnl_attack,
            "pnl_clean": pnl_clean,
            "pnl_delta": pnl_delta,
            "reward": shaped,
        })

        return obs, float(shaped), terminated, truncated, info
