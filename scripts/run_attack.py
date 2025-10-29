#!/usr/bin/env python3
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse, json
import numpy as np
from stable_baselines3 import TD3

from gmats.attack.rl_env import RLAttackEnv
from gmats.attack.gym_env import GymAttackEnv  # wrapper we added earlier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gmats.yaml")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--model", default="td3_attacker.zip")
    ap.add_argument("--out_dir", default="results/attack")
    args = ap.parse_args()

    # --- ensure output + GMATS per-asset logging mirror clean run ---
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["GMATS_LOG_AGENTS"] = "1"
    os.environ["GMATS_LOG_BY_ASSET"] = "1"
    os.environ["GMATS_LOG_DIR"] = log_dir
    os.environ["GMATS_LOG_RESET"] = "1"

    # poison manifest path (deterministic IDs emitted by env->info["injected"])
    manifest_path = os.path.join(args.out_dir, "poison_ids.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as _mf:
        pass  # truncate

    # optional: CSV results (mirrors clean run convenience)
    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "attack_results.csv")
    if not os.path.exists(results_csv):
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("date,equity,pnl,label,lev,a_sent,orders_json\n")

    core = RLAttackEnv(args.config, args.data_root)
    env = GymAttackEnv(core)

    model = TD3.load(args.model, env=env, print_system_info=False)

    obs, _ = env.reset()
    logs = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)

        # resolve current day for logging
        cur_date = core.dates[core._t - 1] if core._t > 0 else core.dates[0]

        # per-day JSON log
        row = {
            "date": cur_date,
            "reward": float(r),
            "equity": float(info.get("equity", np.nan)),
            "label": info.get("label"),
            "orders": info.get("orders", []),
            "a_trade": info.get("a_trade"),
            "a_sent": info.get("a_sent"),
            "lev": info.get("lev"),
        }
        logs.append(row)

        # append to poison manifest
        injected = info.get("injected") or []
        if injected:
            with open(manifest_path, "a", encoding="utf-8") as mf:
                for prow in injected:
                    mf.write(json.dumps(prow) + "\n")

        # append to CSV (date,equity,pnl,label,lev,a_sent,orders_json)
        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{cur_date},{row['equity']},{float(r)},{row['label']},"
                f"{row['lev']},{row['a_sent']},{json.dumps(row['orders'])}\n"
            )

        if terminated or truncated:
            break

    # Save per-day log + a tiny summary
    with open(os.path.join(args.out_dir, "attack_log.jsonl"), "w", encoding="utf-8") as f:
        for rrow in logs:
            f.write(json.dumps(rrow) + "\n")

    final_equity = logs[-1]["equity"] if logs else 1.0
    print(f"[ATTACK] Episode equity: {final_equity:.6f}; days={len(logs)}")
    print(f"[ATTACK] Logs dir: {log_dir}")
    print(f"[ATTACK] Poison manifest: {manifest_path}")
    print(f"[ATTACK] Results CSV: {results_csv}")


if __name__ == "__main__":
    main()
