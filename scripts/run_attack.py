#!/usr/bin/env python3
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse, json, os
import numpy as np
from stable_baselines3 import TD3

from gmats.attack.rl_env import RLAttackEnv
from gmats.attack.gym_env import GymAttackEnv  # wrapper we added earlier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gmats.yaml")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--model", default="td3_attacker.zip")
    ap.add_argument("--out_dir", default="runs/attack_run")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    core = RLAttackEnv(args.config, args.data_root)
    env = GymAttackEnv(core)

    model = TD3.load(args.model, env=env, print_system_info=False)

    obs, _ = env.reset()
    done = False
    logs = []
    equity0 = 1.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        logs.append({
            "date": core.dates[core._t-1] if core._t > 0 else core.dates[0],
            "reward": float(r),
            "equity": float(info.get("equity", np.nan)),
            "label": info.get("label"),
            "orders": info.get("orders", []),
        })
        if terminated or truncated:
            break

    # Save per-day log + a tiny summary
    with open(os.path.join(args.out_dir, "attack_log.jsonl"), "w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")

    final_equity = logs[-1]["equity"] if logs else equity0
    print(f"[ATTACK] Episode equity: {final_equity:.6f}; days={len(logs)}")

if __name__ == "__main__":
    main()
