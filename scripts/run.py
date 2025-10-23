"""CLI launcher: reads config, runs FINSABER with GMATS LLM strategy."""
from __future__ import annotations
import argparse, datetime as dt, os
from backtest.finsaber import FINSABER
from backtest.toolkit.trade_config import TradeConfig
from gmats.adapters.finsaber_dataset import GMATSDataset
from gmats.adapters.finsaber_strategy import GMATSLLMStrategy
from gmats.core.config import load_config

def _to_date(v):
    if v is None: return None
    if isinstance(v, dt.date): return v
    y, m, d = [int(x) for x in str(v).strip().split("-")]
    return dt.date(y, m, d)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="configs/gmats.yaml")
parser.add_argument("--data_root", default="./data")
parser.add_argument("--date_from")
parser.add_argument("--date_to")
parser.add_argument("--log_dates", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--delist_check", action="store_true", help="Enable delist check (default: off)")
args = parser.parse_args()

if args.log_dates or args.debug:
    os.environ["GMATS_DEBUG"] = "1"

cfg = load_config(args.config)

dataset = GMATSDataset(
    cfg.data.get("root", args.data_root),
    cfg.data.get("market_dir", "market"),
)

date_from = _to_date(args.date_from or cfg.schedule.get("date_from"))
date_to   = _to_date(args.date_to   or cfg.schedule.get("date_to"))
if date_from is None or date_to is None:
    ds_min, ds_max = dataset.get_date_range()
    date_from = _to_date(ds_min); date_to = _to_date(ds_max)

print(f"[GMATS] Running with date range: {date_from} -> {date_to}")

# Build raw desired config (our intent)
intent = {
    "setup_name": "gmats_local",
    "date_from": date_from,
    "date_to":   date_to,
    "training_years": 0,          # minimal warm-up
    "rolling_window_size": 1,     # we’ll map to the actual field name below
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
print("[GMATS][TradeConfig] applied:",
      {k: intent_mapped[k] for k in ("date_from","date_to") if k in intent_mapped},
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
