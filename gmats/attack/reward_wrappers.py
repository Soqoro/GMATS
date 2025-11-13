# gmats/attack/reward_wrappers.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List
from gmats.attack.overlay import get_global_overlay

class AttackRewardMixer(gym.Wrapper):
    """
    Reward:
        r = -(pnl_attack - pnl_clean)/scale + alpha*IR@k + beta*BSS

    Terms:
        pnl_attack: raw env reward (attacked PnL proxy)
        pnl_clean:  env-provided clean PnL (baseline); if missing, assumed 0
        scale: debiased EMA(|pnl_attack - pnl_clean|) with floor
        IR@k: overlap fraction between ingested_topk_ids and attack post IDs
        BSS: behavior shift score (info['bss'] or info['BSS']); fallback 0.0
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 0.25,
        pnl_scale_ema: float = 0.05,
        pnl_scale_floor: float = 1e-4,
        reward_clip: float | None = None,  # optional: clip final reward to [-clip, clip]
    ):
        super().__init__(env)
        self.k = int(topk)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._ema = float(pnl_scale_ema)
        self._pnl_abs_ema = 1.0
        self._pnl_floor = float(pnl_scale_floor)
        self._t = 0  # steps for EMA debiasing
        self._reward_clip = reward_clip

    # --------- helpers ---------
    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

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

    # --------- gym API ---------
    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r_attack, terminated, truncated, info = self.env.step(action)

        # --- PnL delta + debiased EMA scale ---
        pnl_attack = self._safe_float(r_attack, 0.0)
        pnl_clean = self._safe_float(info.get("pnl_clean", 0.0), 0.0)
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
        ir_k = (inter / self.k) if self.k > 0 else 0.0

        # --- BSS ---
        bss = self._safe_float(info.get("bss", info.get("BSS", 0.0)), 0.0)

        # --- Final reward ---
        shaped = perf_term + self.alpha * ir_k + self.beta * bss
        if self._reward_clip is not None and self._reward_clip > 0:
            lim = float(self._reward_clip)
            shaped = float(np.clip(shaped, -lim, lim))

        # Diagnostics
        info.setdefault("attack_reward_terms", {})
        info["attack_reward_terms"].update({
            "perf": perf_term,
            "ir_k": ir_k,
            "bss": bss,
            "scale": scale,
            "pnl_attack": pnl_attack,
            "pnl_clean": pnl_clean,
            "pnl_delta": pnl_delta,
            "ir_num": inter,
            "ir_den": self.k,
            "reward": shaped,
        })

        return obs, float(shaped), terminated, truncated, info
