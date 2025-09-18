from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml  # pip install pyyaml

from gmats.core.interfaces import (
    Budgets, CoordinationResult, Fills,
    Coordinator, AlphaMiner, Policy, Constraints, Environment, Reward,
    DataFeed,
    MemoryStore as MemoryStoreProto,
    Message,
)
from gmats.core.memory import MemoryStore
from gmats.core.registry import COORDINATORS, ALPHAS, POLICIES, RISKS, REWARDS
from gmats.env.backtest import Backtester
from gmats.env.backtest_env import BacktestEnvironment
from gmats.env.backtester_feed import BacktesterFeed


# ----- Minimal defaults / fallbacks -----

class NullCoordinator(Coordinator):
    """
    Derives a simple stance from inbox:
      s = [ +margin ] if bull_score > bear_score
          [ -margin ] if bear_score > bull_score
      margin = |bull_score - bear_score|
    If no messages or no scores, falls back to neutral.
    """
    def _score(self, msgs: List[Message], role: str) -> float:
        for m in msgs:
            pl = (m.get("payload") or {})
            if str(pl.get("role", "")).lower() == role:
                try:
                    return float(pl.get("score", 0.0))
                except Exception:
                    return 0.0
        return 0.0

    def coordinate(self, inbox):
        msgs = list(inbox) if inbox else []
        bull = self._score(msgs, "bull")
        bear = self._score(msgs, "bear")
        if bull == 0.0 and bear == 0.0:
            return CoordinationResult(s=[0.0], rho={"mode": "null", "winner": "none", "margin": 0.0}, log=msgs)
        winner = "bull" if bull >= bear else "bear"
        margin = abs(bull - bear)
        sign = +margin if winner == "bull" else -margin
        rho = {"mode": "null-derived", "winner": winner, "margin": float(margin), "scores": {"bull": bull, "bear": bear}}
        return CoordinationResult(s=[sign], rho=rho, log=msgs)

class SimpleAlpha(AlphaMiner):
    def factors(self, data: DataFeed, memory: Optional[MemoryStoreProto] = None) -> List[float]:
        return [0.0]

class ThresholdPolicy(Policy):
    def __init__(self, buy_thr: float = 0.10, sell_thr: float = -0.10):
        self.buy_thr = float(buy_thr)
        self.sell_thr = float(sell_thr)
    def decide(self, s, f, state_t):
        if not s: return ["HOLD"]
        v = float(s[0])
        if v > self.buy_thr: return ["BUY"]
        if v < self.sell_thr: return ["SELL"]
        return ["HOLD"]

class PassConstraints(Constraints):
    def gate(self, a, state_t, history, budgets):
        return (True, a)

class PnLReward(Reward):
    def __call__(self, fills: Fills, x_next: Dict[str, Any]) -> float:
        return sum(float(d.get("pnl", 0.0)) for d in (fills.details or []))

# Register only if missing (so user code can pre-register custom plugins)
if "null" not in COORDINATORS: COORDINATORS.register("null", NullCoordinator())
if "simple" not in ALPHAS: ALPHAS.register("simple", SimpleAlpha())
if "threshold" not in POLICIES: POLICIES.register("threshold", ThresholdPolicy())
if "pass" not in RISKS: RISKS.register("pass", PassConstraints())
if "pnl" not in REWARDS: REWARDS.register("pnl", PnLReward())

# Try optional DebateCoordinator registration (user may have their own)
try:
    from gmats.roles.coordinator import DebateCoordinator  # optional
    if "debate" not in COORDINATORS:
        COORDINATORS.register("debate", DebateCoordinator())
    DEBATE_AVAILABLE = True
except Exception:
    DEBATE_AVAILABLE = False


# ----- Config & prompts loading / merge -----

def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """
    Auto-discover config in these locations (first hit wins) when --config is not provided:
      ./default.yaml
      ./configs/default.yaml
      ./gmats/configs/default.yaml
      <repo>/gmats/configs/default.yaml (relative to this file)
    """
    candidates: List[Path] = []
    if path:
        candidates = [Path(path)]
    else:
        cwd = Path.cwd()
        candidates.extend([
            cwd / "default.yaml",
            cwd / "configs" / "default.yaml",
            cwd / "gmats" / "configs" / "default.yaml",
        ])
        here = Path(__file__).resolve()
        candidates.append(here.parent.parent / "configs" / "default.yaml")

    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}

def load_prompts(path: Optional[str]) -> Dict[str, Any]:
    """
    Auto-discover prompts in these locations (first hit wins) when --prompts is not provided:
      ./prompts.yaml
      ./configs/prompts.yaml
      ./gmats/configs/prompts.yaml
      <repo>/gmats/configs/prompts.yaml (relative to this file)
    """
    candidates: List[Path] = []
    if path:
        candidates = [Path(path)]
    else:
        cwd = Path.cwd()
        candidates.extend([
            cwd / "prompts.yaml",
            cwd / "configs" / "prompts.yaml",
            cwd / "gmats" / "configs" / "prompts.yaml",
        ])
        here = Path(__file__).resolve()
        candidates.append(here.parent.parent / "configs" / "prompts.yaml")

    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}

def pick(val: Optional[Any], *fallbacks: Any) -> Any:
    for v in (val, *fallbacks):
        if v is not None:
            return v
    return None


# ----- Schedule -----

def rolling_schedule(
    bt: Backtester,
    symbol: str,
    start_asof: Optional[str],
    episodes: int,
    horizon: int = 1
) -> List[Dict[str, Any]]:
    """
    Produce orchestration states using *positional* start index via searchsorted.
    """
    df = bt.data.get(symbol.upper())
    if df is None or df.empty:
        return []
    if start_asof:
        start = pd.to_datetime(start_asof).normalize()
        start_pos = int(pd.Series(df["date"]).searchsorted(start, side="left"))
    else:
        start_pos = 0
    xs = []
    for k in range(episodes):
        i = min(start_pos + k, len(df) - 1)
        xs.append({"symbol": symbol.upper(), "asof": df["date"].iloc[i].strftime("%Y-%m-%d"), "horizon": horizon})
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config (auto-discovery enabled)")
    ap.add_argument("--prompts", type=str, default=None, help="Path to prompts.yaml (auto-discovery enabled)")
    ap.add_argument("--returns_dir", type=str, default=None)
    ap.add_argument("--symbol", type=str, default=None, help="If omitted, iterate all YAML assets")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--asof", type=str, default=None)
    ap.add_argument("--rf_daily", type=float, default=None)
    ap.add_argument("--coordinator", type=str, default=None)
    ap.add_argument("--alpha", type=str, default=None)
    ap.add_argument("--policy", type=str, default=None)
    ap.add_argument("--risk", type=str, default=None)
    ap.add_argument("--reward", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    prompts = load_prompts(args.prompts)
    coord_prompts = (prompts.get("coordinator") or {})
    llm_prompts = (prompts.get("llm") or {})

    # Seed (best-effort)
    try:
        seed = int(cfg.get("seed", 0))
        if seed:
            random.seed(seed)
            try:
                import numpy as np  # type: ignore
                np.random.seed(seed)
            except Exception:
                pass
            try:
                import torch  # type: ignore
                torch.manual_seed(seed)
            except Exception:
                pass
    except Exception:
        pass

    # --- Pull from YAML with CLI overrides ---
    assets = cfg.get("assets") or []
    symbols: List[str] = [pick(args.symbol, None)].copy() if args.symbol else [s for s in assets] or ["AAPL"]
    symbols = [str(s).upper() for s in symbols]

    data_cfg = cfg.get("data", {}) or {}
    returns_dir = pick(args.returns_dir, data_cfg.get("returns_dir"), "data/returns")
    rf_daily = float(pick(args.rf_daily, data_cfg.get("rf_daily"), 0.0))

    episodes = int(pick(args.episodes, cfg.get("episodes"), 20))
    horizon = int(pick(args.horizon, 1))
    asof = pick(args.asof, None)

    coord_name = pick(args.coordinator, (cfg.get("coordination", {}) or {}).get("type"), "null")
    alpha_name = pick(args.alpha, "simple")
    reward_name = pick(args.reward, "pnl")

    policy_name = args.policy
    yaml_policy = cfg.get("policy", {}) or {}
    use_yaml_thresholds = ("buy_thr" in yaml_policy) or ("sell_thr" in yaml_policy)
    risk_name = args.risk

    # 𝓔
    bt = Backtester(returns_dir, rf_daily=rf_daily)
    env: Environment = BacktestEnvironment(bt)

    # Build judge (once), then instantiate the coordinator per symbol with a formatted criterion
    judge = None
    criterion_template = None

    # If using debate, best-effort load judge + template
    if coord_name == "debate" and DEBATE_AVAILABLE:
        try:
            from gmats.llm.mock import MockLLM
            from gmats.llm.hf_generic import HFLLM

            judge_cfg = ((cfg.get("coordination", {}) or {}).get("judge", {}) or {})
            kind = (judge_cfg.get("llm") or "none").lower()

            # Prompt overrides
            judge_system = llm_prompts.get("judge_system")
            summarize_system = llm_prompts.get("summarize_system")
            judge_user_suffix = llm_prompts.get("judge_user_suffix")

            # Template: prompts.yaml wins over default.yaml if present
            criterion_template = coord_prompts.get("criterion") or (cfg.get("coordination", {}) or {}).get("criterion")

            if kind == "mock":
                judge = MockLLM()
            else:
                judge = HFLLM(
                    model_id=judge_cfg.get("model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                    max_new_tokens=int(judge_cfg.get("max_new_tokens", 96)),
                    temperature=float(judge_cfg.get("temperature", 0.2)),
                    top_p=float(judge_cfg.get("top_p", 0.9)),
                    dtype=str(judge_cfg.get("dtype", "auto")),
                    load_8bit=bool(judge_cfg.get("load_8bit", False)),
                    load_4bit=bool(judge_cfg.get("load_4bit", False)),
                    judge_system=judge_system,
                    summarize_system=summarize_system,
                    judge_user_suffix=judge_user_suffix,
                )
        except Exception:
            judge = None  # fall back below

    # Alpha
    alpha: AlphaMiner = ALPHAS.get(alpha_name)

    # Policy
    if policy_name is not None:
        policy: Policy = POLICIES.get(policy_name)
    elif use_yaml_thresholds:
        policy = ThresholdPolicy(
            buy_thr=float(yaml_policy.get("buy_thr", 0.10)),
            sell_thr=float(yaml_policy.get("sell_thr", -0.10)),
        )
    else:
        policy = POLICIES.get("threshold")

    # Risk
    if risk_name is not None:
        risk: Constraints = RISKS.get(risk_name)
    else:
        try:
            from gmats.roles.risk import RiskController  # optional user module
            risk = RiskController()  # type: ignore
        except Exception:
            risk = RISKS.get("pass")

    reward_fn: Reward = REWARDS.get(reward_name)

    # Output dir
    out_dir = Path((cfg.get("logging", {}) or {}).get("out_dir", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run per symbol ----
    for symbol in symbols:
        # Coordinator per symbol: format criterion (if template given)
        if coord_name == "debate" and DEBATE_AVAILABLE:
            try:
                # Note: if criterion_template is None, DebateCoordinator will use its default
                crit = None
                if criterion_template:
                    try:
                        crit = str(criterion_template).format(symbol=symbol, horizon=horizon)
                    except Exception:
                        crit = str(criterion_template)  # use literal if formatting fails
                coord: Coordinator = DebateCoordinator(llm=judge, criterion=crit)  # type: ignore
            except Exception:
                coord = COORDINATORS.get("null")
        else:
            coord: Coordinator = COORDINATORS.get(coord_name) if coord_name in COORDINATORS.list() else COORDINATORS.get("null")

        # 𝓜 + 𝓓 (per-asset instances)
        memory: MemoryStoreProto = MemoryStore()
        data_feed: DataFeed = BacktesterFeed(bt, symbol)

        xs = rolling_schedule(bt, symbol, asof, episodes, horizon)
        history: List[Dict[str, Any]] = []
        budgets = Budgets(capital=None).as_dict()

        results: List[Dict[str, Any]] = []
        cum_pnl = 0.0

        for ep, x_t in enumerate(xs, 1):
            # 𝓓 observe → toy momentum score
            obs = data_feed.observe(x_t["asof"], {"limit": 5})
            rets = [it["ret"] for it in obs if "ret" in it]
            m = (sum(rets) / len(rets)) if rets else 0.0

            # E routing → 𝓛 inbox
            inbox: List[Message] = [
                {
                    "id": f"bull:{symbol}:{ep}",
                    "sender": "analyst.momentum",
                    "t": "argument",
                    "payload": {"role": "bull", "score": max(m, 0.0), "speech": ""},
                    "schema": "gmats/argument@v1",
                    "prov": {"source": "momentum", "window": 5, "symbol": symbol},
                },
                {
                    "id": f"bear:{symbol}:{ep}",
                    "sender": "analyst.momentum",
                    "t": "argument",
                    "payload": {"role": "bear", "score": max(-m, 0.0), "speech": ""},
                    "schema": "gmats/argument@v1",
                    "prov": {"source": "momentum", "window": 5, "symbol": symbol},
                },
            ]

            # 𝓛
            coord_res: CoordinationResult = coord.coordinate(inbox=inbox)
            s = coord_res.s

            # Φ
            f = alpha.factors(data_feed, memory)

            # Π
            a = policy.decide(s, f, x_t)

            # Λ (pass coordination margin)
            x_t_for_risk = {**x_t, "coord_margin": float(coord_res.rho.get("margin", 0.0))}
            ok, a_prime = risk.gate(a, x_t_for_risk, history, budgets)

            # 𝓔
            fills, x_next = env.step(a_prime, x_t)

            # 𝓡
            r_t = reward_fn(fills, x_next)
            cum_pnl += r_t

            results.append({
                "episode": ep,
                "x_t": x_t,
                "inbox": inbox,
                "coord": {"s": s, "rho": coord_res.rho},
                "action": a,
                "a_prime": a_prime,
                "fills": fills.details,
                "reward": r_t,
                "cum_pnl": cum_pnl,
            })
            history.append({"x_t": x_t, "a_prime": a_prime, "fills": fills.details, "r": r_t, "rho": coord_res.rho})

            if args.debug:
                winner = coord_res.rho.get("winner", "n/a")
                margin = coord_res.rho.get("margin", 0.0)
                print(f"[{symbol}] [ep {ep}] {x_t['asof']} m={m:+.5f} winner={winner} margin={margin:.4f} "
                      f"a={a} a'={a_prime} r={r_t:.6f} cum={cum_pnl:.6f}")

            x_t = x_next  # optional advance

        out_file = out_dir / f"report_{symbol}.json"
        out_file.write_text(json.dumps({"symbol": symbol, "episodes": results}, indent=2))
        print(f"Wrote → {out_file}")


if __name__ == "__main__":
    main()
