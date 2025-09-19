from __future__ import annotations
import os
import json
import argparse
import pandas as pd
import yaml  # pip install pyyaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

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
from gmats.env.file_feeds import NewsFileFeed, SocialFileFeed, FundamentalsFileFeed
from gmats.attacks.pg_attacker import PGAttacker


# ----- Minimal defaults / fallbacks -----

class NullCoordinator(Coordinator):
    """
    Aggregate all 'bull' and all 'bear' scores across the inbox:
        s = [ +margin ] if sum(bull_scores) >= sum(bear_scores)
            [ -margin ] otherwise
        margin = |sum(bull) - sum(bear)|
    If no messages or no scores, falls back to neutral.
    """
    def _sum_role(self, msgs: Iterable[Message], role: str) -> float:
        total = 0.0
        for m in msgs or []:
            pl = (m.get("payload") or {})
            if str(pl.get("role", "")).lower() == role:
                try:
                    total += float(pl.get("score", 0.0))
                except Exception:
                    pass
        return float(total)

    def coordinate(self, inbox: Iterable[Message]) -> CoordinationResult:
        msgs = list(inbox) if inbox else []
        bull = self._sum_role(msgs, "bull")
        bear = self._sum_role(msgs, "bear")
        if bull == 0.0 and bear == 0.0:
            return CoordinationResult(s=[0.0], rho={"mode": "null", "winner": "none", "margin": 0.0}, log=msgs)
        winner = "bull" if bull >= bear else "bear"
        margin = abs(bull - bear)
        sign = +margin if winner == "bull" else -margin
        rho = {
            "mode": "null-agg",
            "winner": winner,
            "margin": float(margin),
            "scores": {"bull_sum": bull, "bear_sum": bear},
        }
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

def _load_yaml(path: Optional[str], *, fallbacks: Optional[List[Path]] = None) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    for c in (fallbacks or []):
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    return {}

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def pick(val: Optional[Any], *fallbacks: Any) -> Any:
    for x in (val, *fallbacks):
        if x is not None:
            return x
    return None

def rolling_schedule(bt, symbol: str, start_asof: Optional[str], episodes: int, horizon: int = 1) -> List[Dict[str, Any]]:
    df = (bt.data or {}).get(symbol.upper())
    if df is None or len(df) == 0:
        return []
    if start_asof:
        start = pd.to_datetime(start_asof).normalize()
        mask = (df["date"] >= start)
        start_pos = int(mask.to_numpy().argmax()) if mask.any() else 0
    else:
        start_pos = 0
    xs = []
    for k in range(episodes):
        i = min(start_pos + k, len(df) - 1)
        xs.append({"symbol": symbol.upper(), "asof": df["date"].iloc[i].strftime("%Y-%m-%d"), "horizon": horizon})
    return xs

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (auto-discovery enabled)")
    p.add_argument("--prompts", type=str, default=None, help="Path to prompts.yaml (auto-discovery enabled)")
    p.add_argument("--returns_dir", type=str, default=None)
    p.add_argument("--symbol", type=str, default=None, help="If omitted, iterate all YAML assets")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--asof", type=str, default=None)
    p.add_argument("--rf_daily", type=float, default=None)
    p.add_argument("--coordinator", type=str, default=None)
    p.add_argument("--alpha", type=str, default=None)
    p.add_argument("--policy", type=str, default=None)
    p.add_argument("--risk", type=str, default=None)
    p.add_argument("--reward", type=str, default=None)
    p.add_argument("--attacker", type=str, default=None, help="Path to attacker.yaml (optional)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    # Load main config (existing behavior)
    cfg = _load_yaml(
        getattr(args, "config", None),
        fallbacks=[
            Path.cwd() / "default.yaml",
            Path.cwd() / "configs" / "default.yaml",
            _repo_root() / "gmats" / "configs" / "default.yaml",
        ],
    )

    # Load prompts (fills llm_prompts/coord_prompts)
    prompts = _load_yaml(
        getattr(args, "prompts", None),
        fallbacks=[
            Path.cwd() / "prompts.yaml",
            Path.cwd() / "configs" / "prompts.yaml",
            _repo_root() / "gmats" / "configs" / "prompts.yaml",
        ],
    )
    coord_prompts = (prompts.get("coordinator") or {}) if isinstance(prompts, dict) else {}
    llm_prompts = (prompts.get("llm") or {}) if isinstance(prompts, dict) else {}

    # Load attacker config (separate file, with sensible fallbacks)
    attacker_cfg = _load_yaml(
        args.attacker,
        fallbacks=[
            Path.cwd() / "attacker.yaml",
            Path.cwd() / "configs" / "attacker.yaml",
            _repo_root() / "gmats" / "configs" / "attacker.yaml",
        ],
    )
    # Backward-compat: if separate file missing, fall back to attacks block in default.yaml
    if not attacker_cfg and isinstance(cfg.get("attacks"), dict):
        attacker_cfg = dict(cfg.get("attacks") or {})

    attacks_enabled = bool(attacker_cfg.get("enabled", False))
    attack_type = str(attacker_cfg.get("type", "pg")).lower()
    pg_cfg = (attacker_cfg.get("pg") or {})
    eps_cfg = (pg_cfg.get("eps") or {})
    lam_cfg = (pg_cfg.get("lambda") or {})

    # --- Pull from YAML with CLI overrides ---
    assets = cfg.get("assets") or []
    symbols: List[str] = [args.symbol] if args.symbol else (assets or ["AAPL"])
    symbols = [str(s).upper() for s in symbols]

    data_cfg = cfg.get("data", {}) or {}
    returns_dir = pick(args.returns_dir, data_cfg.get("returns_dir"), "data/returns")
    rf_daily = float(pick(args.rf_daily, data_cfg.get("rf_daily"), 0.0))
    # Offline sources config (news/social/fundamentals)
    sources = [str(s).lower() for s in (data_cfg.get("sources") or [])]
    offline_cfg = data_cfg.get("offline", {}) or {}
    news_dir = Path(offline_cfg.get("news_dir", "data/news"))
    social_dir = Path(offline_cfg.get("social_dir", "data/social"))
    fund_dir = Path(offline_cfg.get("fundamentals_dir", "data/fundamentals"))
    dayfirst = bool(offline_cfg.get("dayfirst", False))
    news_include_poison = str(offline_cfg.get("news_include_poison", "all"))
    news_weight = float(offline_cfg.get("news_weight", 1.0))
    social_weight = float(offline_cfg.get("social_weight", 1.0))
    news_limit = int(offline_cfg.get("news_limit", 5))
    social_limit = int(offline_cfg.get("social_limit", 5))

    episodes = int(pick(args.episodes, cfg.get("episodes"), 20))
    horizon = int(pick(args.horizon, cfg.get("horizon"), 1))
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

    # Filter symbols by available data
    available = set((bt.data or {}).keys())
    missing = [s for s in symbols if s not in available]
    if missing:
        print(f"Warning: missing returns data for {missing}. They will be skipped.")
    symbols = [s for s in symbols if s in available]
    if not symbols:
        print("No valid symbols to run.")
        return

    # Build judge (once) and criterion template
    judge = None
    criterion_template = None
    if coord_name == "debate" and DEBATE_AVAILABLE:
        try:
            from gmats.llm.mock import MockLLM
            from gmats.llm.hf_generic import HFLLM
            judge_cfg = ((cfg.get("coordination", {}) or {}).get("judge", {}) or {})
            kind = (judge_cfg.get("llm") or "none").lower()
            judge_system = llm_prompts.get("judge_system")
            summarize_system = llm_prompts.get("summarize_system")
            judge_user_suffix = llm_prompts.get("judge_user_suffix")
            criterion_template = coord_prompts.get("criterion") or (cfg.get("coordination", {}) or {}).get("criterion")
            if kind == "mock":
                judge = MockLLM()
            elif kind == "hf":
                judge = HFLLM(
                    model_id=judge_cfg.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3"),
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
            judge = None

    # Alpha / Policy / Risk / Reward
    alpha: AlphaMiner = ALPHAS.get(alpha_name)

    if policy_name is not None:
        policy: Policy = POLICIES.get(policy_name)
    elif use_yaml_thresholds:
        policy = ThresholdPolicy(
            buy_thr=float(yaml_policy.get("buy_thr", 0.10)),
            sell_thr=float(yaml_policy.get("sell_thr", -0.10)),
        )
    else:
        policy = POLICIES.get("threshold")

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
        # Coordinator per symbol
        if coord_name == "debate" and DEBATE_AVAILABLE:
            try:
                crit = None
                if criterion_template:
                    try:
                        crit = str(criterion_template).format(symbol=symbol, horizon=horizon)
                    except Exception:
                        crit = str(criterion_template)
                coord: Coordinator = DebateCoordinator(llm=judge, criterion=crit)  # type: ignore
            except Exception:
                coord = COORDINATORS.get("null")
        else:
            try:
                coord = COORDINATORS.get(coord_name)
            except Exception:
                coord = COORDINATORS.get("null")

        # 𝓜 + 𝓓 (per-asset instances)
        memory: MemoryStoreProto = MemoryStore()
        data_feed: DataFeed = BacktesterFeed(bt, symbol)

        # Offline feeds (optional)
        news_feed = NewsFileFeed(news_dir, default_symbol=symbol, dayfirst=dayfirst, include_poison=news_include_poison) if "news" in sources else None
        social_feed = SocialFileFeed(social_dir, default_symbol=symbol, dayfirst=dayfirst) if "social" in sources else None
        fund_feed = FundamentalsFileFeed(fund_dir, default_symbol=symbol, dayfirst=dayfirst) if "fundamentals" in sources else None

        # Per-asset attacker
        attacker = None
        if attacks_enabled and attack_type == "pg":
            attacker = PGAttacker(
                lr=float(pg_cfg.get("lr", 1e-3)),
                hidden=int(pg_cfg.get("hidden", 32)),
                eps_mom=float(eps_cfg.get("mom", 0.05)),
                eps_news=float(eps_cfg.get("news", 0.05)),
                eps_social=float(eps_cfg.get("social", 0.05)),
                lambda_lip=float(lam_cfg.get("lip", 1.0)),
                lambda_gen=float(lam_cfg.get("gen", 1e-4)),
                entropy_coef=float(pg_cfg.get("entropy_coef", 1e-3)),
                device=str(pg_cfg.get("device", "auto")),
            )

        xs = rolling_schedule(bt, symbol, asof, episodes, horizon)
        history: List[Dict[str, Any]] = []
        budgets = Budgets(capital=None).as_dict()

        results: List[Dict[str, Any]] = []
        cum_pnl = 0.0
        last_margin = 0.0

        for ep, x_t in enumerate(xs, 1):
            # Momentum from returns
            obs = data_feed.observe(x_t["asof"], {"limit": 5})
            rets = [it["ret"] for it in obs if "ret" in it]
            m_mom = (sum(rets) / len(rets)) if rets else 0.0

            # Aggregate offline sources (optional)
            context_str = ""
            s_news = None
            s_soc = None

            if news_feed is not None:
                news_items = news_feed.observe(x_t["asof"], {"limit": news_limit, "symbol": symbol})
                titles: List[str] = []
                s_vals: List[float] = []
                for it in news_items:
                    title_or_text = (it.get("title") or it.get("text") or "").strip()
                    if title_or_text:
                        titles.append(title_or_text)
                    v = it.get("score")
                    if v is not None:
                        try:
                            s_vals.append(float(v))
                        except Exception:
                            pass
                if titles:
                    context_str += "News:\n- " + "\n- ".join(titles[:5]) + "\n"
                if s_vals:
                    s_news = news_weight * (sum(s_vals) / max(1, len(s_vals)))

            if social_feed is not None:
                social_items = social_feed.observe(x_t["asof"], {"limit": social_limit, "symbol": symbol})
                texts: List[str] = []
                s_vals: List[float] = []
                for it in social_items:
                    text = (it.get("text") or it.get("title") or "").strip()
                    if text:
                        texts.append(text)
                    v = it.get("score")
                    if v is not None:
                        try:
                            s_vals.append(float(v))
                        except Exception:
                            pass
                if texts:
                    context_str += "Social:\n- " + "\n- ".join(texts[:5]) + "\n"
                if s_vals:
                    s_soc = social_weight * (sum(s_vals) / max(1, len(s_vals)))

            if fund_feed is not None:
                fund_snap = fund_feed.observe(x_t["asof"], {"symbol": symbol})
                if fund_snap:
                    snap = fund_snap[-1].get("metrics", {})
                    sample_keys = list(snap.keys())[:6]
                    sample = {k: snap.get(k) for k in sample_keys}
                    context_str += f"Fundamentals(sample): {sample}\n"

            # Attacker deltas (apply before building inbox)
            m_pert = m_mom
            if attacker is not None:
                obs_feats = {
                    "m_mom": float(m_mom),
                    "s_news": float(s_news) if s_news is not None else 0.0,
                    "s_social": float(s_soc) if s_soc is not None else 0.0,
                    "coord_margin": float(last_margin),
                }
                deltas = attacker.propose(x_t, obs_feats)
                m_pert = float(m_mom) + float(deltas.get("delta_mom", 0.0))
                # Optional: could perturb s_news/s_soc similarly if you later use them in policy
            else:
                deltas = {"delta_mom": 0.0, "delta_news": 0.0, "delta_social": 0.0}

            # Build inbox
            inbox: List[Message] = [
                {
                    "id": f"mom_bull:{symbol}:{ep}",
                    "sender": "analyst.momentum",
                    "t": "argument",
                    "payload": {"role": "bull", "score": max(m_pert, 0.0), "speech": "", "context": context_str},
                    "schema": "gmats/argument@v1",
                    "prov": {"source": "momentum", "window": 5, "symbol": symbol},
                },
                {
                    "id": f"mom_bear:{symbol}:{ep}",
                    "sender": "analyst.momentum",
                    "t": "argument",
                    "payload": {"role": "bear", "score": max(-m_pert, 0.0), "speech": "", "context": context_str},
                    "schema": "gmats/argument@v1",
                    "prov": {"source": "momentum", "window": 5, "symbol": symbol},
                },
            ]

            if s_news is not None:
                inbox.extend([
                    {"id": f"news_bull:{symbol}:{ep}", "sender": "analyst.news_offline", "t": "argument",
                     "payload": {"role": "bull", "score": max(float(s_news), 0.0), "speech": "", "context": context_str},
                     "schema": "gmats/argument@v1", "prov": {"source": "offline.news", "symbol": symbol}},
                    {"id": f"news_bear:{symbol}:{ep}", "sender": "analyst.news_offline", "t": "argument",
                     "payload": {"role": "bear", "score": max(-float(s_news), 0.0), "speech": "", "context": context_str},
                     "schema": "gmats/argument@v1", "prov": {"source": "offline.news", "symbol": symbol}},
                ])
            if s_soc is not None:
                inbox.extend([
                    {"id": f"social_bull:{symbol}:{ep}", "sender": "analyst.social_offline", "t": "argument",
                     "payload": {"role": "bull", "score": max(float(s_soc), 0.0), "speech": "", "context": context_str},
                     "schema": "gmats/argument@v1", "prov": {"source": "offline.social", "symbol": symbol}},
                    {"id": f"social_bear:{symbol}:{ep}", "sender": "analyst.social_offline", "t": "argument",
                     "payload": {"role": "bear", "score": max(-float(s_soc), 0.0), "speech": "", "context": context_str},
                     "schema": "gmats/argument@v1", "prov": {"source": "offline.social", "symbol": symbol}},
                ])

            # 𝓛 → Φ → Π → Λ → 𝓔 → 𝓡
            coord_res: CoordinationResult = coord.coordinate(inbox=inbox)
            last_margin = float(coord_res.rho.get("margin", 0.0)) if isinstance(coord_res.rho, dict) else 0.0
            s = coord_res.s
            f = alpha.factors(data_feed, memory)
            a = policy.decide(s, f, x_t)
            x_t_for_risk = {**x_t, "coord_margin": last_margin}
            ok, a_prime = risk.gate(a, x_t_for_risk, history, budgets)
            fills, x_next = env.step(a_prime, x_t)
            r_t = reward_fn(fills, x_next)
            cum_pnl += r_t

            # Train attacker (maximize -env reward)
            if attacker is not None:
                attacker.update({"reward": -float(r_t)})

            results.append({
                "episode": ep, "x_t": x_t, "inbox": inbox,
                "coord": {"s": s, "rho": coord_res.rho},
                "action": a, "a_prime": a_prime,
                "fills": getattr(fills, "details", []), "reward": r_t, "cum_pnl": cum_pnl,
                "deltas": deltas,
            })
            history.append({"x_t": x_t, "a_prime": a_prime, "fills": getattr(fills, "details", []), "r": r_t, "rho": coord_res.rho})

            if args.debug:
                winner = coord_res.rho.get("winner", "n/a")
                margin = coord_res.rho.get("margin", 0.0)
                print(f"[{symbol}] ep={ep} asof={x_t['asof']} m={m_mom:+.5f} m_pert={m_pert:+.5f} "
                      f"winner={winner} margin={margin:+.4f} a={a} a'={a_prime} r={r_t:+.6f} cum={cum_pnl:+.6f}")

        out_file = out_dir / f"report_{symbol}.json"
        out_file.write_text(json.dumps({"symbol": symbol, "episodes": results}, indent=2))
        print(f"Wrote → {out_file}")


if __name__ == "__main__":
    main()
