from __future__ import annotations
"""
Experiment runner
=================
Builds a GMATS instance from YAML config, runs paired clean/attack episodes,
and writes a JSON summary (contagion + performance).

CLI (recommended):
    python -m gmats.experiments.run --config gmats/configs/default.yaml
"""

import argparse, json, random, datetime as dt, sys, logging
from pathlib import Path
from typing import List, Dict, Any, cast
import yaml

from ..core.memory import MemoryStore
from ..core.interfaces import LLM, Message, Stance
from ..roles.coordinator import DebateCoordinator
from ..roles.alpha_miner import SimpleAlphaMiner
from ..roles.policy import ThresholdPolicy
from ..roles.risk import RiskController
from ..roles.trader import Trader
from ..env.backtest import Backtester
from .metrics import Metrics, EpisodeResult

LOG = logging.getLogger("gmats.run")

# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------

def build_llm(cfg) -> LLM | None:
    """Construct the judge LLM according to the YAML config."""
    kind = cfg["coordination"]["judge"]["llm"]
    LOG.info("Building LLM backend: %s", kind)
    if kind == "none":
        return None
    if kind == "mock":
        from ..llm.mock import MockLLM
        return MockLLM()
    if kind == "hf":
        from ..llm.hf_generic import HFLLM
        j = cfg["coordination"]["judge"]
        return HFLLM(model_id=j["model_id"], max_new_tokens=j["max_new_tokens"],
                     temperature=j["temperature"], top_p=j["top_p"],
                     dtype=j["dtype"], load_8bit=j["load_8bit"], load_4bit=j["load_4bit"])
    if kind == "llama_cpp":
        from ..llm.llama_cpp_backend import LlamaCppLLM
        j = cfg["coordination"]["judge"]
        return LlamaCppLLM(model_path=j["model_id"])
    raise ValueError(f"Unknown LLM kind: {kind}")

def load_cfg(path: str):
    """Load YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Episode simulation
# ---------------------------------------------------------------------------

def simulate_episode(symbol: str, now: str, cfg, llm, attack: bool = False, seed: int = 0) -> EpisodeResult:
    """Run a single (toy) episode and return debate/action/perf summaries."""
    random.seed(seed)

    # --- memory intake (toy): admit two synthetic items ---
    mem = MemoryStore(deep_threshold=cfg["memory"]["deep_threshold"], decay=cfg["memory"]["decay"])
    t0 = dt.datetime.fromisoformat(now).timestamp()
    mem.admit({"text": f"{symbol} beats guidance; growth outlook positive", "ts": t0 - 1, "cred": 0.6, "imp": 0.7})
    mem.admit({"text": f"{symbol} faces probe; risk of delay", "ts": t0 - 2, "cred": 0.5, "imp": 0.6})

    # Retrieve evidence from memory
    ev = mem.retrieve({"ts": t0}, k=cfg["retrieval"]["top_k"])

    # Convert MemoryItem -> plain evidence dicts expected by AlphaMiner.factors
    ev_texts: List[str] = []
    for e in ev:
        t = cast(str, e.get("text", ""))
        if t:
            ev_texts.append(t)

    evidence: List[Dict[str, Any]] = [{"text": t} for t in ev_texts]

    # attack A3 example: bias evidence (very lightweight demonstration)
    if attack and cfg["attacks"]["A3_debate"]["enabled"]:
        if cfg["attacks"]["A3_debate"]["bull_bias"] > 0:
            injected = f"{symbol} strong demand uptrend"
            evidence.append({"text": injected})
            ev_texts.append(injected)

    # --- build bull/bear arguments ---
    context = f"Symbol={symbol}. Evidence: " + " | ".join(ev_texts[:4])
    bull_speech = (llm.summarize("Write a concise bullish argument: " + context) if llm
                   else "Positive catalysts likely outweigh risks.")
    bear_speech = (llm.summarize("Write a concise bearish argument: " + context) if llm
                   else "Risks outweigh positives in near term.")

    # crude scores based on simple alpha factors
    alpha = SimpleAlphaMiner(k=cfg["alpha_path"]["k"])
    factors = alpha.factors(evidence)
    score = alpha.score(factors)

    bull_msg: Message = {"role": "bull", "speech": bull_speech, "score": max(0.0, score),  "context": context}
    bear_msg: Message = {"role": "bear", "speech": bear_speech, "score": max(0.0, -score), "context": context}

    coord = DebateCoordinator(llm=llm)
    stance: Stance = coord.aggregate([bull_msg, bear_msg])

    # risk + policy
    risk = RiskController(stop_loss=cfg["risk"]["stop_loss"], take_profit=cfg["risk"]["take_profit"])
    # If your RiskController.gate is typed as gate(Stance), you can pass 'stance' directly.
    gate = risk.gate(cast(Dict[str, Any], stance))
    policy = ThresholdPolicy(buy_thr=cfg["policy"]["buy_thr"], sell_thr=cfg["policy"]["sell_thr"])
    decision = policy.decide(score)["action"] if gate.get("ok", False) else "HOLD"

    # trader + environment
    trader = Trader()
    _ = trader.execute(decision)
    bt = Backtester(cfg["data"]["returns_dir"], rf_daily=cfg["data"]["rf_daily"])
    ret = bt.trade_return(symbol, now[:10], decision, horizon=1)

    episode: EpisodeResult = {
        "debate": {"winner": stance["winner"], "margin": float(stance["margin"])},
        "final_action": decision,
        "perf": {"ret": float(ret)}
    }
    return episode

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point: build components, run episodes, write JSON report."""
    ap = argparse.ArgumentParser(description="GMATS experiment runner")
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    ap.add_argument("--episodes", type=int, default=None, help="Override episodes (for quick tests)")
    ap.add_argument("--llm", type=str, choices=["config", "none", "mock", "hf", "llama_cpp"],
                    default="config", help="Override judge LLM backend")
    ap.add_argument("--debug", action="store_true", help="Verbose progress logging")
    args = ap.parse_args()

    # Logging
    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    LOG.info("Loading config: %s", args.config)
    cfg = load_cfg(args.config)

    # Quick overrides for fast smoke tests
    if args.episodes is not None:
        cfg["episodes"] = int(args.episodes)
    if args.llm != "config":
        cfg["coordination"]["judge"]["llm"] = args.llm

    # Validate data dir
    returns_dir = Path(cfg["data"]["returns_dir"])
    if not returns_dir.exists():
        LOG.error("Returns directory does not exist: %s", returns_dir)
        sys.exit(2)
    n_csv = len(list(returns_dir.glob("*.csv")))
    if n_csv == 0:
        LOG.error("No CSV files found in %s (need at least one like AAPL.csv)", returns_dir)
        sys.exit(2)
    LOG.info("Found %d CSV files under %s", n_csv, returns_dir)

    random.seed(cfg["seed"])
    llm = build_llm(cfg)
    metrics = Metrics()

    # Fixed timestamp (toy). Replace with rolling calendar for realism.
    now = "2024-01-05T15:00:00"

    LOG.info("Starting simulation: episodes=%d, assets=%s", cfg["episodes"], cfg["assets"])
    for e in range(cfg["episodes"]):
        symbol = random.choice(cfg["assets"])
        LOG.info("Episode %d/%d symbol=%s [clean]", e + 1, cfg["episodes"], symbol)
        clean = simulate_episode(symbol, now, cfg, llm, attack=False, seed=e)
        LOG.info("Episode %d/%d symbol=%s [attack]", e + 1, cfg["episodes"], symbol)
        att = simulate_episode(symbol, now, cfg, llm, attack=True, seed=e)
        metrics.update(clean, att)

    out = metrics.report()
    out["model"] = cfg["coordination"]["judge"]
    out_dir = Path(cfg["logging"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"report_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    LOG.info("Wrote report → %s", out_path)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Always print a traceback to help diagnose "nothing happens" cases
        import traceback
        traceback.print_exc()
        sys.exit(1)
