#!/usr/bin/env python3
import argparse
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import gymnasium as gym  # noqa: F401 (keep for gym.Env typing / wrappers)

from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from gmats.attack.rl_env import RLAttackEnv
from gmats.attack.gym_env import GymAttackEnv
from gmats.attack.reward_wrappers import AttackRewardMixer


def make_wrapped_env(config_path, data_root, *, topk, alpha, beta_a, beta_c, reward_clip):
    """
    Construct the RL attack environment with reward shaping.

    Pipeline:
        RLAttackEnv (core, GMATS-specific)
        -> GymAttackEnv (Gymnasium adapter)
        -> AttackRewardMixer (shaped reward with IR@k, BSS, etc.)
    """
    core = RLAttackEnv(config_path, data_root)
    base_env = GymAttackEnv(core)
    env = AttackRewardMixer(
        base_env,
        topk=topk,                 # IR@k
        alpha=alpha,               # weight for IR@k
        beta_analyst=beta_a,       # weight for BSS (analyst)
        beta_coordinator=beta_c,   # weight for BSS (coordinator)
        reward_clip=reward_clip,   # optional: None or a positive float
    )
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gmats.yaml")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--timesteps", type=int, default=200_000)

    # reward shaping knobs
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta_analyst", type=float, default=0.25)
    ap.add_argument("--beta_coordinator", type=float, default=0.25)
    ap.add_argument("--reward_clip", type=float, default=0.0, help="0 or negative to disable")

    # logging / checkpointing
    ap.add_argument("--logdir", type=str, default="logs/td3_attacker")
    ap.add_argument("--checkpoint_dir", type=str, default="td3_attacker_ckpts")
    ap.add_argument("--best_model_dir", type=str, default="td3_attacker_best")
    ap.add_argument("--eval_logdir", type=str, default="td3_attacker_eval")
    ap.add_argument("--eval_freq", type=int, default=100)
    ap.add_argument("--save_freq", type=int, default=100)

    # reproducibility
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    reward_clip = args.reward_clip if args.reward_clip > 0 else None

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.best_model_dir, exist_ok=True)
    os.makedirs(args.eval_logdir, exist_ok=True)

    # 1) Training env
    env = make_wrapped_env(
        args.config, args.data_root,
        topk=args.topk,
        alpha=args.alpha,
        beta_a=args.beta_analyst,
        beta_c=args.beta_coordinator,
        reward_clip=reward_clip,
    )
    env = Monitor(env)  # logs episode returns/lengths

    # 2) TD3 config
    n_actions = env.action_space.shape[0]  # expecting 2 (trade, sentiment)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.20 * np.ones(n_actions))
    policy_kwargs = dict(net_arch=[128, 128])

    model = TD3(
        MlpPolicy,
        env,
        learning_rate=1e-3,
        buffer_size=200_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.02,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.4,
        target_noise_clip=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=args.logdir,
        seed=args.seed,
        device="auto",
    )

    # 3) Evaluation & checkpoint callbacks
    eval_env = make_wrapped_env(
        args.config, args.data_root,
        topk=args.topk,
        alpha=args.alpha,
        beta_a=args.beta_analyst,
        beta_c=args.beta_coordinator,
        reward_clip=reward_clip,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.best_model_dir,
        log_path=args.eval_logdir,
        eval_freq=args.eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_dir,
        name_prefix="td3_attacker",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 4) Train
    model.learn(
        total_timesteps=args.timesteps,
        log_interval=10,
        callback=[eval_callback, checkpoint_callback],
    )
    model.save("td3_attacker")

    # 5) Deterministic evaluation (use the SAME wrapper so rewards are comparable)
    eval_env = make_wrapped_env(
        args.config, args.data_root,
        topk=args.topk,
        alpha=args.alpha,
        beta_a=args.beta_analyst,
        beta_c=args.beta_coordinator,
        reward_clip=reward_clip,
    )
    obs, _ = eval_env.reset()
    ret = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = eval_env.step(action)
        ret += r
        if terminated or truncated:
            break

    equity = info.get("equity", np.nan)
    label = info.get("label", None)
    if isinstance(equity, (int, float, np.floating)):
        equity_str = f"{equity:.6f}"
    else:
        equity_str = str(equity)

    print(
        f"[Eval] Shaped return: {ret:.6f} | "
        f"equity={equity_str} | "
        f"last_label={label}"
    )


if __name__ == "__main__":
    main()
