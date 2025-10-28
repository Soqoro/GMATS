# GMATS: A Minimal, Reproducible Framework for LLM‑Augmented Market Agents

Commands:

Run clean baseline:
```bash
python -m scripts.run --config configs/baseline.yaml --debug --log_by_asset --log_agents
```

Train RL agent:
```bash
python scripts/train_attacker.py --config configs/gmats.yaml --timesteps 14
```

Run Attack:
```bash
python scripts/run_attack.py --config configs/gmats.yaml --model td3_attacker.zip --out_dir runs/attack_oct1_8
```

Evalute Metrics:
```bash
