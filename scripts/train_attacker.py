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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gmats.yaml")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--timesteps", type=int, default=200_000)
    args = ap.parse_args()

    # 1) Core env + Gym wrapper
    core = RLAttackEnv(args.config, args.data_root)
    base_env = GymAttackEnv(core)

    env = AttackRewardMixer(
        base_env,
        topk=10,            # IR@10
        alpha=0.5,          # weight for IR@k
        beta=0.25,          # weight for IACR (if provided)
        c_post=1e-3,        # cost per synthetic post
        eta=0.1,            # penalty weight for risk beyond delta
        risk_tol=0.5,       # tolerated detection risk
    )
    env = Monitor(env)  # logs episode returns/lengths

    # 2) TD3 config
    n_actions = env.action_space.shape[0]  # 2
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.20 * np.ones(n_actions))
    policy_kwargs = dict(net_arch=[128, 128])  # small net is fine for tabular obs

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
        policy_delay=2,      # delayed actor updates
        target_policy_noise=0.4,
        target_noise_clip=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    # 3) Train
    model.learn(total_timesteps=args.timesteps, log_interval=10)
    model.save("td3_attacker")

    # 4) Deterministic evaluation
    eval_core = RLAttackEnv(args.config, args.data_root)
    eval_env = GymAttackEnv(eval_core)
    obs, _ = eval_env.reset()
    done = False
    ret = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = eval_env.step(action)
        ret += r
        if terminated or truncated:
            break
    print(f"[Eval] Episode return (PnL delta sum): {ret:.6f} | equity={info.get('equity'):.6f} | last_label={info.get('label')}")

if __name__ == "__main__":
    main()
