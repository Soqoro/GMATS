from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from gmats.core.interfaces import (
    Budgets, CoordinationResult, Fills,
    Coordinator, AlphaMiner, Policy, Constraints, Environment, Reward,
    DataFeed,
    MemoryStore as MemoryStoreProto,  # type against the Protocol
    Message,
)
from gmats.core.memory import MemoryStore  # concrete impl
from gmats.core.registry import COORDINATORS, ALPHAS, POLICIES, RISKS, REWARDS
from gmats.env.backtest import Backtester
from gmats.env.backtest_env import BacktestEnvironment
from gmats.env.backtester_feed import BacktesterFeed  # <-- use your existing feed


# ----- Minimal default components (simple, deterministic) -----

class NullCoordinator(Coordinator):
    def coordinate(self, inbox):
        # Neutral 1-D state; pass through messages into log
        return CoordinationResult(s=[0.0], rho={"mode": "null", "margin": 0.0}, log=list(inbox) if inbox else [])

class SimpleAlpha(AlphaMiner):
    def factors(self, data: DataFeed, memory: Optional[MemoryStoreProto] = None) -> List[float]:
        # Placeholder α-miner
        return [0.0]

class ThresholdPolicy(Policy):
    """BUY if s[0] >= 0, SELL if s[0] < 0; HOLD if s is empty."""
    def decide(self, s, f, state_t):
        if not s:
            return ["HOLD"]
        return ["BUY" if s[0] >= 0 else "SELL"]

class PassConstraints(Constraints):
    def gate(self, a, state_t, history, budgets):
        return (True, a)

class PnLReward(Reward):
    def __call__(self, fills: Fills, x_next: Dict[str, Any]) -> float:
        return sum(float(d.get("pnl", 0.0)) for d in (fills.details or []))

# Register defaults if not already present
if "null" not in COORDINATORS:
    COORDINATORS.register("null", NullCoordinator())
if "simple" not in ALPHAS:
    ALPHAS.register("simple", SimpleAlpha())
if "threshold" not in POLICIES:
    POLICIES.register("threshold", ThresholdPolicy())
if "pass" not in RISKS:
    RISKS.register("pass", PassConstraints())
if "pnl" not in REWARDS:
    REWARDS.register("pnl", PnLReward())

# Optional: register debate coordinator if user has provided it elsewhere
try:
    from gmats.roles.coordinator import DebateCoordinator  # optional module
    if "debate" not in COORDINATORS:
        COORDINATORS.register("debate", DebateCoordinator())
except Exception:
    pass


# ----- Σ: simple rolling schedule -----

def rolling_schedule(
    bt: Backtester,
    symbol: str,
    start_asof: Optional[str],
    episodes: int,
    horizon: int = 1
) -> List[Dict[str, Any]]:
    """Produce a list of orchestration states x_t."""
    df = bt.data.get(symbol.upper())
    if df is None or df.empty:
        return []
    if start_asof:
        start = pd.to_datetime(start_asof).normalize()
        idx = df.index[df["date"] >= start]
        start_idx = int(idx[0]) if len(idx) else 0
    else:
        start_idx = 0
    xs = []
    for k in range(episodes):
        i = min(start_idx + k, len(df) - 1)
        x = {"symbol": symbol.upper(), "asof": df["date"].iloc[i].strftime("%Y-%m-%d"), "horizon": horizon}
        xs.append(x)
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returns_dir", type=str, default="data/returns")
    ap.add_argument("--symbol", type=str, default="AAPL")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD start (Σ)")
    ap.add_argument("--rf_daily", type=float, default=0.0)
    ap.add_argument("--coordinator", type=str, default="null", help="e.g., null | debate (if registered)")
    ap.add_argument("--alpha", type=str, default="simple")
    ap.add_argument("--policy", type=str, default="threshold")
    ap.add_argument("--risk", type=str, default="pass")
    ap.add_argument("--reward", type=str, default="pnl")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # 𝓔 Environment
    bt = Backtester(args.returns_dir, rf_daily=args.rf_daily)
    env: Environment = BacktestEnvironment(bt)

    # Components
    coord: Coordinator = COORDINATORS.get(args.coordinator)
    alpha: AlphaMiner = ALPHAS.get(args.alpha)
    policy: Policy = POLICIES.get(args.policy)
    risk: Constraints = RISKS.get(args.risk)
    reward_fn: Reward = REWARDS.get(args.reward)

    # 𝓜 Memory + 𝓓 DataFeed
    memory: MemoryStoreProto = MemoryStore()          # concrete implements the Protocol
    data_feed: DataFeed = BacktesterFeed(bt, args.symbol)

    # Σ schedule
    xs = rolling_schedule(bt, args.symbol, args.asof, args.episodes, args.horizon)
    history: List[Dict[str, Any]] = []
    budgets = Budgets(capital=None).as_dict()

    results: List[Dict[str, Any]] = []
    cum_pnl = 0.0

    for ep, x_t in enumerate(xs, 1):
        # --- 𝓓 observe (point-in-time) to build a toy momentum score for inbox ---
        obs = data_feed.observe(x_t["asof"], {"limit": 5})
        rets = [it["ret"] for it in obs if "ret" in it]
        m = (sum(rets) / len(rets)) if rets else 0.0  # simple 5-day momentum

        # --- E routing → 𝓛 coordinator inbox (bull vs bear messages) ---
        inbox: List[Message] = [
            {
                "id": f"bull:{ep}",
                "sender": "analyst.momentum",
                "t": "argument",
                "payload": {"role": "bull", "score": max(m, 0.0), "speech": ""},
                "schema": "gmats/argument@v1",
                "prov": {"source": "momentum", "window": 5},
            },
            {
                "id": f"bear:{ep}",
                "sender": "analyst.momentum",
                "t": "argument",
                "payload": {"role": "bear", "score": max(-m, 0.0), "speech": ""},
                "schema": "gmats/argument@v1",
                "prov": {"source": "momentum", "window": 5},
            },
        ]

        # --- 𝓛 (coordination) ---
        coord_res: CoordinationResult = coord.coordinate(inbox=inbox)
        s = coord_res.s  # 1-D state; sign encodes stance in the DebateCoordinator

        # --- Φ (alpha) ---
        f = alpha.factors(data_feed, memory)

        # --- Π (policy) ---
        a = policy.decide(s, f, x_t)

        # --- Λ (risk) --- pass coordination margin into state for gating
        x_t_for_risk = {**x_t, "coord_margin": float(coord_res.rho.get("margin", 0.0))}
        ok, a_prime = risk.gate(a, x_t_for_risk, history, budgets)

        # --- 𝓔 (environment) ---
        fills, x_next = env.step(a_prime, x_t)

        # --- 𝓡 (reward) ---
        r_t = reward_fn(fills, x_next)
        cum_pnl += r_t

        # --- Logging / results ---
        results.append({
            "episode": ep,
            "x_t": x_t,
            "inbox": inbox,
            "coord": {"s": s, "rho": coord_res.rho},
            "action": a,
            "a_prime": a_prime,
            "fills": fills.details,
            "reward": r_t,
            "cum_pnl": cum_pnl
        })
        history.append({
            "x_t": x_t,
            "a_prime": a_prime,
            "fills": fills.details,
            "r": r_t,
            "rho": coord_res.rho,
        })

        if args.debug:
            winner = coord_res.rho.get("winner", "n/a")
            margin = coord_res.rho.get("margin", 0.0)
            print(f"[ep {ep}] {x_t['asof']} m={m:+.5f} winner={winner} margin={margin:.4f} "
                  f"a={a} a'={a_prime} r={r_t:.6f} cum={cum_pnl:.6f}")

        # advance orchestration state (optional; schedule already advances across xs)
        x_t = x_next

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "report_minimal.json"
    out_file.write_text(json.dumps({"episodes": results}, indent=2))
    print(f"Wrote → {out_file}")


if __name__ == "__main__":
    main()
