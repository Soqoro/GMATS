#!/usr/bin/env python3
import argparse
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from gmats.attack.rl_env import RLAttackEnv
from gmats.attack.gym_env import GymAttackEnv
from gmats.attack.reward_wrappers import AttackRewardMixer

def make_wrapped_env(config_path, data_root, *, topk, alpha, beta_a, beta_c, reward_clip):
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

    args = ap.parse_args()
    reward_clip = args.reward_clip if args.reward_clip > 0 else None

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
    )

    # 3) Train
    model.learn(total_timesteps=args.timesteps, log_interval=10)
    model.save("td3_attacker")

    # 4) Deterministic evaluation (use the SAME wrapper so rewards are comparable)
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
    print(f"[Eval] Shaped return: {ret:.6f} | equity={info.get('equity'):.6f} | last_label={info.get('label')}")

if __name__ == "__main__":
    main()
