#!/usr/bin/env python3
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse, json, csv
import numpy as np

from gmats.attack.rl_env import RLAttackEnv
from gmats.attack.gym_env import GymAttackEnv
from gmats.attack.reward_wrappers import AttackRewardMixer


def make_wrapped_env(config_path, data_root, *, topk, alpha, beta_a, beta_c, reward_clip, posts_per_day: int = 1):
    core = RLAttackEnv(config_path, data_root, posts_per_day=posts_per_day)
    base_env = GymAttackEnv(core)
    env = AttackRewardMixer(
        base_env,
        topk=topk,                 # IR@k
        alpha=alpha,               # weight for IR@k
        beta_analyst=beta_a,       # weight for BSS (analyst)
        beta_coordinator=beta_c,   # weight for BSS (coordinator)
        reward_clip=reward_clip,   # optional: None or a positive float
    )
    return env, core


def choose_action(attacker, state, rng, amp_threshold: float = 0.3):
    """
    Map attacker type + state to a continuous action [a_trade, a_sent] in [-1, 1]^2.

    state fields (updated after each env.step):
        has_prev: bool
        prev_exposure: float   # approximate signed exposure to target asset
        prev_mu: float         # approximate sentiment mu in [-1, 1]
        prev_volume: float     # optional volume proxy (unused if not set)
    """
    # Fallback: pure random policy
    def _random_action():
        # a_trade=0 → no extra leverage; env trades at base size
        a_trade = 0.0
        a_sent  = float(rng.uniform(-1.0, 1.0))
        return np.array([a_trade, a_sent], dtype=np.float32)

    if attacker == "random":
        return _random_action()

    has_prev = state.get("has_prev", False)
    exposure = float(state.get("prev_exposure", 0.0))
    mu       = float(state.get("prev_mu", 0.0))
    volume   = float(state.get("prev_volume", 1.0))

    # If we have no previous info yet, fall back to random once
    if not has_prev:
        return _random_action()

    if attacker == "greedy":
        # Greedy victim-aware: lean against previous day's exposure.
        if abs(exposure) < 1e-6:
            # No clear exposure → stay neutral.
            return np.array([0.0, 0.0], dtype=np.float32)
        direction = -1.0 if exposure > 0.0 else 1.0  # push against long/short
        a_trade = 0
        a_sent  = direction  # let RLAttackEnv + quantizer map this to strong bull/bear labels
        return np.array([a_trade, a_sent], dtype=np.float32)

    if attacker == "amplify":
        # Volume-amplifying: push in the direction of strong, high-volume sentiment.
        strength = abs(mu) * max(volume, 1.0)
        if strength < amp_threshold:
            # Weak narrative → remain neutral.
            return np.array([0.0, 0.0], dtype=np.float32)
        direction = 1.0 if mu > 0.0 else -1.0
        a_trade = 0
        a_sent  = direction
        return np.array([a_trade, a_sent], dtype=np.float32)

    # Unknown attacker string → default to random
    return _random_action()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gmats.yaml")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="results/attack")

    # attacker configuration
    ap.add_argument(
        "--attacker",
        choices=["random", "greedy", "amplify"],
        default="random",
        help="scripted attacker policy: random | greedy (victim-aware) | amplify (volume-amplifying)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for scripted attackers")
    ap.add_argument(
        "--amp_threshold",
        type=float,
        default=0.3,
        help="minimum |mu|*volume needed for the amplify attacker to post",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=1,
        help="integer number of attack posts per asset per day (B)",
    )

    # reward shaping knobs (still used for logging / diagnostics)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta_analyst", type=float, default=0.25)
    ap.add_argument("--beta_coordinator", type=float, default=0.25)
    ap.add_argument("--reward_clip", type=float, default=0.0, help="0 or negative to disable")

    args = ap.parse_args()
    reward_clip = args.reward_clip if args.reward_clip > 0 else None

    rng = np.random.default_rng(args.seed)

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
    with open(manifest_path, "w", encoding="utf-8"):
        pass  # truncate

    # results dir
    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # daily telemetry CSV (kept for backward-compat; note: 'pnl' column stores SHAPED reward)
    results_csv = os.path.join(results_dir, "attack_results.csv")
    if not os.path.exists(results_csv):
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("date,equity,pnl,label,lev,a_sent,orders_json\n")

    # reward-term breakdown JSONL (new, helpful for debugging)
    terms_jsonl = os.path.join(results_dir, "attack_reward_terms.jsonl")
    with open(terms_jsonl, "w", encoding="utf-8"):
        pass  # truncate

    env, core = make_wrapped_env(
        args.config, args.data_root,
        topk=args.topk,
        alpha=args.alpha,
        beta_a=args.beta_analyst,
        beta_c=args.beta_coordinator,
        reward_clip=reward_clip,
        posts_per_day=max(1, args.budget),
    )

    obs, _ = env.reset()
    logs = []

    # --- per-ticker equity tracking (to write baseline-style rows per ticker)
    # state: {ticker: {"eq": float, "eq_series": [floats], "rets": [floats], "dates": [str]}}
    per_ticker: dict = {}

    # --- attacker state (drives scripted policies) ---
    attacker_state = {
        "has_prev": False,
        "prev_exposure": 0.0,
        "prev_mu": 0.0,
        "prev_volume": 1.0,
    }

    while True:
        # Choose scripted action based on previous day's state
        action = choose_action(
            args.attacker,
            attacker_state,
            rng,
            amp_threshold=args.amp_threshold,
        )

        obs, r, terminated, truncated, info = env.step(action)

        # resolve current day for logging
        cur_date = core.dates[core._t - 1] if core._t > 0 else core.dates[0]

        # per-day JSON log entry
        row = {
            "date": cur_date,
            "reward": float(r),                           # shaped reward
            "equity": float(info.get("equity", np.nan)),
            "label": info.get("label"),
            "orders": info.get("orders", []),
            "a_trade": info.get("a_trade"),
            "a_sent": info.get("a_sent"),
            "lev": info.get("lev"),
        }
        logs.append(row)

        # ---- reward terms (optional but useful)
        terms = info.get("attack_reward_terms", {})
        if terms:
            terms_row = {"date": cur_date, **terms}
            with open(terms_jsonl, "a", encoding="utf-8") as tf:
                tf.write(json.dumps(terms_row) + "\n")

        # ---- update per-ticker equity from today's orders
        for od in row["orders"]:
            sym = str(od.get("symbol", "")).upper()
            if not sym:
                continue
            w_eff = float(od.get("weight", 0.0))
            # next-day asset return as used by the env
            ret_sym = core.data.get_next_day_return(sym, cur_date)

            st = per_ticker.setdefault(sym, {"eq": 1.0, "eq_series": [1.0], "rets": [], "dates": []})
            prev_eq = st["eq"]
            new_eq = prev_eq * (1.0 + w_eff * ret_sym)
            st["eq"] = new_eq
            st["eq_series"].append(new_eq)
            st["rets"].append((new_eq / prev_eq) - 1.0 if prev_eq != 0 else w_eff * ret_sym)
            st["dates"].append(cur_date)

        # ---- update attacker_state for next day ----
        # Approximate exposure from today's orders on the first symbol (single-asset experiments).
        target_sym = None
        syms = info.get("symbols")
        if isinstance(syms, list) and syms:
            target_sym = str(syms[0]).upper()
        elif isinstance(info.get("symbol"), (str, bytes)):
            target_sym = str(info["symbol"]).upper()

        net_w = 0.0
        if target_sym:
            for od in row["orders"]:
                if str(od.get("symbol", "")).upper() == target_sym:
                    try:
                        net_w += float(od.get("weight", 0.0))
                    except Exception:
                        continue
        attacker_state["prev_exposure"] = net_w

        # Estimate sentiment mu for the target asset from any available scorecard.
        # Prefer attack-side analyst scorecard; fall back to clean keys if present.
        mu = None
        if target_sym:
            for key in (
                "analyst_attack_scorecard",   # preferred: attack-side analyst μ
                "social_clean_scorecard",
                "analyst_clean_scorecard",
                "clean_scorecard",
            ):
                rows = info.get(key)
                if isinstance(rows, list):
                    for rrow in rows:
                        try:
                            if str(rrow.get("symbol", "")).upper() != target_sym:
                                continue
                            if "mu" in rrow:
                                mu = float(rrow["mu"])
                                break
                        except Exception:
                            continue
                    if mu is not None:
                        break
        attacker_state["prev_mu"] = mu if mu is not None else 0.0

        # Volume proxy: if scorecard exposes 'n' or 'count', use that; otherwise 1.0
        vol = 1.0
        if target_sym:
            for key in (
                "analyst_attack_scorecard",   # preferred: attack-side analyst volume proxy
                "social_clean_scorecard",
                "analyst_clean_scorecard",
                "clean_scorecard",
            ):
                rows = info.get(key)
                if isinstance(rows, list):
                    for rrow in rows:
                        try:
                            if str(rrow.get("symbol", "")).upper() != target_sym:
                                continue
                            if "n" in rrow:
                                vol = max(vol, float(rrow["n"]))
                            elif "count" in rrow:
                                vol = max(vol, float(rrow["count"]))
                        except Exception:
                            continue
        attacker_state["prev_volume"] = vol
        attacker_state["has_prev"] = True

        # append to poison manifest
        injected = info.get("injected") or []
        if injected:
            with open(manifest_path, "a", encoding="utf-8") as mf:
                for prow in injected:
                    mf.write(json.dumps(prow) + "\n")

        # append daily telemetry row (keeps legacy header)
        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{cur_date},{row['equity']},{float(r)},{row['label']},"
                f"{row['lev']},{row['a_sent']},{json.dumps(row['orders'])}\n"
            )

        if terminated or truncated:
            break

    # Save per-day log (compact JSONL of a few fields)
    with open(os.path.join(args.out_dir, "attack_log.jsonl"), "w", encoding="utf-8") as f:
        for rrow in logs:
            f.write(json.dumps(rrow) + "\n")

    # === Portfolio-wide summary (single row) ===
    def _std(x):
        return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    equity = [row["equity"] for row in logs]
    dates  = [row["date"]   for row in logs]

    if equity:
        # build daily returns with first step from 1.0 -> equity[0]
        daily_rets = [equity[0] - 1.0] + [(equity[i] / equity[i - 1]) - 1.0 for i in range(1, len(equity))]
        N = len(daily_rets)
        ann = 252

        total_return = equity[-1] - 1.0
        annual_return = (np.prod([1.0 + r for r in daily_rets]) ** (ann / max(N, 1))) - 1.0 if N > 0 else 0.0
        daily_std = _std(daily_rets)
        annual_vol = daily_std * np.sqrt(ann)
        mean_ret = float(np.mean(daily_rets)) if N > 0 else 0.0
        sharpe = (mean_ret / daily_std * np.sqrt(ann)) if daily_std > 0 else 0.0
        downside = [min(r, 0.0) for r in daily_rets]
        lpsd_daily = float(np.sqrt(np.mean([d * d for d in downside]))) if N > 0 else 0.0
        sortino = (mean_ret / (lpsd_daily * np.sqrt(ann))) if lpsd_daily > 0 else 0.0
        eq_curve = np.array([1.0] + equity, dtype=float)
        peaks = np.maximum.accumulate(eq_curve)
        drawdowns = (eq_curve / peaks) - 1.0
        max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0

        summary_csv = os.path.join(results_dir, "attack_summary.csv")
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="", encoding="utf-8") as fsum:
            w = csv.writer(fsum)
            if write_header:
                w.writerow([
                    "setup","strategy","ticker","date_from","date_to",
                    "final_value","total_return","annual_return","annual_volatility",
                    "sharpe_ratio","sortino_ratio","max_drawdown","total_commission"
                ])
            setup    = "gmats_local"
            strategy = "GMATSLLMStrategy"
            tickers  = "+".join(sorted(core.assets))
            ticker   = f"PORTFOLIO({tickers})"
            date_from, date_to = (dates[0] if dates else ""), (dates[-1] if dates else "")
            initial_capital = 100000.0
            final_value = equity[-1] * initial_capital
            total_commission = 0.0
            w.writerow([
                setup, strategy, ticker, date_from, date_to,
                f"{final_value:.2f}", f"{total_return:.6f}", f"{annual_return:.6f}",
                f"{annual_vol:.6f}", f"{sharpe:.6f}", f"{sortino:.6f}",
                f"{max_drawdown:.6f}", f"{total_commission:.2f}"
            ])
        print(f"[ATTACK] Portfolio summary written: {summary_csv}")
    else:
        print("[ATTACK] No equity rows logged; skipped portfolio summary export.")

    # === Per-ticker summaries (one row per ticker, baseline-compatible) ===
    by_ticker_csv = os.path.join(results_dir, "attack_summary_by_ticker.csv")
    write_header_bt = not os.path.exists(by_ticker_csv)
    with open(by_ticker_csv, "a", newline="", encoding="utf-8") as fbt:
        w = csv.writer(fbt)
        if write_header_bt:
            w.writerow([
                "setup","strategy","ticker","date_from","date_to",
                "final_value","total_return","annual_return","annual_volatility",
                "sharpe_ratio","sortino_ratio","max_drawdown","total_commission"
            ])

        for sym in sorted(per_ticker.keys()):
            st = per_ticker[sym]
            eq_series = st["eq_series"]          # includes initial 1.0
            rets      = st["rets"]
            dates_sym = st["dates"]
            if len(eq_series) < 2:
                continue

            N = len(rets)
            ann = 252
            total_return = eq_series[-1] - 1.0
            annual_return = (np.prod([1.0 + r for r in rets]) ** (ann / max(N, 1))) - 1.0 if N > 0 else 0.0
            daily_std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
            annual_vol = daily_std * np.sqrt(ann)
            mean_ret = float(np.mean(rets)) if N > 0 else 0.0
            sharpe = (mean_ret / daily_std * np.sqrt(ann)) if daily_std > 0 else 0.0
            downside = [min(r, 0.0) for r in rets]
            lpsd_daily = float(np.sqrt(np.mean([d * d for d in downside]))) if N > 0 else 0.0
            sortino = (mean_ret / (lpsd_daily * np.sqrt(ann))) if lpsd_daily > 0 else 0.0
            eq_curve = np.array(eq_series, dtype=float)
            peaks = np.maximum.accumulate(eq_curve)
            drawdowns = (eq_curve / peaks) - 1.0
            max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0

            setup    = "gmats_local"
            strategy = "GMATSLLMStrategy"
            date_from = dates_sym[0] if dates_sym else ""
            date_to   = dates_sym[-1] if dates_sym else ""
            initial_capital = 100000.0
            final_value = eq_series[-1] * initial_capital
            total_commission = 0.0

            w.writerow([
                setup, strategy, sym, date_from, date_to,
                f"{final_value:.2f}", f"{total_return:.6f}", f"{annual_return:.6f}",
                f"{annual_vol:.6f}", f"{sharpe:.6f}", f"{sortino:.6f}",
                f"{max_drawdown:.6f}", f"{total_commission:.2f}"
            ])
    print(f"[ATTACK] Per-ticker summaries written: {by_ticker_csv}")

    final_equity = logs[-1]["equity"] if logs else 1.0
    print(f"[ATTACK] Episode equity: {final_equity:.6f}; days={len(logs)}")
    print(f"[ATTACK] Logs dir: {log_dir}")
    print(f"[ATTACK] Poison manifest: {manifest_path}")
    print(f"[ATTACK] Results CSV: {results_csv}")
    print(f"[ATTACK] Reward-term breakdown: {terms_jsonl}")


if __name__ == "__main__":
    main()
