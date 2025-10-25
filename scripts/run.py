"""CLI launcher: reads config, runs FINSABER with GMATS LLM strategy."""
from __future__ import annotations
import datetime as dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtest.finsaber import FINSABER
from backtest.toolkit.trade_config import TradeConfig
from gmats.adapters.finsaber_dataset import GMATSDataset
from gmats.adapters.finsaber_strategy import GMATSLLMStrategy
from gmats.core.config import load_config


def _to_date(v):
    if v is None:
        return None
    if isinstance(v, dt.date):
        return v
    y, m, d = [int(x) for x in str(v).strip().split("-")]
    return dt.date(y, m, d)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_root")
    p.add_argument("--date_from")
    p.add_argument("--date_to")
    p.add_argument("--log_dates", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--delist_check", action="store_true")
    # RESTORED flags
    p.add_argument(
        "--log_agents", action="store_true", help="Log agent prompts/outputs"
    )
    p.add_argument(
        "--log_by_asset", action="store_true", help="Write per-asset JSON logs"
    )
    p.add_argument(
        "--log_dir", default="./logs", help="Logs output directory"
    )
    p.add_argument("--log_reset", action="store_true", help="Truncate per-asset logs at start")  # NEW
    args = p.parse_args()

    # Make flags available to all components
    os.environ["GMATS_LOG_AGENTS"] = "1" if args.log_agents else "0"
    os.environ["GMATS_LOG_BY_ASSET"] = "1" if args.log_by_asset else "0"
    os.environ["GMATS_LOG_DIR"] = args.log_dir
    os.environ["GMATS_LOG_RESET"] = "1" if args.log_reset else "0"  # NEW

    if args.debug:
        if args.date_from and args.date_to:
            print(f"[GMATS] Running with date range: {args.date_from} -> {args.date_to}")
        if args.log_agents:
            mode = "per-asset JSON only (console muted)" if args.log_by_asset else "global"
            print(f"[GMATS] Agent I/O logging: {mode}")
            print(f"[GMATS] Per-asset log dir: {os.path.abspath(args.log_dir)}")

    cfg = load_config(args.config)

    dataset = GMATSDataset(
        cfg.data.get("root", args.data_root),
        cfg.data.get("market_dir", "market"),
    )

    date_from = _to_date(args.date_from or cfg.schedule.get("date_from"))
    date_to = _to_date(args.date_to or cfg.schedule.get("date_to"))
    if date_from is None or date_to is None:
        ds_min, ds_max = dataset.get_date_range()
        date_from = _to_date(ds_min)
        date_to = _to_date(ds_max)

    # Build raw desired config (our intent)
    intent = {
        "setup_name": "gmats_local",
        "date_from": date_from,
        "date_to": date_to,
        "training_years": 0,  # minimal warm-up
        "rolling_window_size": 1,  # we’ll map to the actual field name below
        "rolling_window_step": 1,
        "tickers": cfg.assets,
        "data_loader": dataset,
    }

    # Map to actual TradeConfig fields
    allowed = list(getattr(TradeConfig, "__dataclass_fields__", {}).keys())
    intent_mapped = {k: v for k, v in intent.items() if k in allowed}

    # Rolling window name varies across versions; set whichever exists
    for candidate in ("rolling_window_size", "rolling_window", "window_size"):
        if candidate in allowed:
            intent_mapped[candidate] = 1
            chosen_rw = candidate
            break
    else:
        chosen_rw = None

    # Step key can also vary (optional)
    for candidate in ("rolling_window_step", "window_step", "step"):
        if candidate in allowed:
            intent_mapped[candidate] = 1
            chosen_step = candidate
            break
    else:
        chosen_step = None

    # Training years key may vary (fallbacks)
    if "training_years" not in allowed:
        for candidate in ("train_years", "n_training_years", "warmup_years"):
            if candidate in allowed:
                intent_mapped[candidate] = 0
                break

    # Print the resolved config knobs so we know what the framework will see
    print("[GMATS][TradeConfig] fields:", allowed)
    print(
        "[GMATS][TradeConfig] applied:",
        {k: intent_mapped[k] for k in ("date_from", "date_to") if k in intent_mapped},
        {"training_years": intent_mapped.get("training_years", intent_mapped.get("train_years", intent_mapped.get("warmup_years")))},
        {chosen_rw: intent_mapped.get(chosen_rw)} if chosen_rw else {},
        {chosen_step: intent_mapped.get(chosen_step)} if chosen_step else {},
    )

    engine = FINSABER(intent_mapped)
    engine.run_iterative_tickers(
        strategy_class=GMATSLLMStrategy,
        strat_params={
            "config_path": args.config,
            "data_root": cfg.data.get("root", args.data_root),
            "log_dates": args.log_dates,
        },
        tickers=cfg.assets,
        delist_check=args.delist_check,  # default False
    )
