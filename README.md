# GMATS: A Minimal, Reproducible Framework for LLM‑Augmented Market Agents

Commands:

Run clean baseline:
```bash
python scripts/run.py --config configs/gmats.yaml --debug --log_by_asset --log_agents
```

Train RL agent:
```bash
python scripts/train_attacker.py --config configs/gmats.yaml --timesteps 14
```

Run Attack:
```bash
python scripts/run_attack.py --config configs/gmats.yaml --model td3_attacker.zip
```

Evalute Metrics:
```bash
python scripts/metrics.py   --attack_logs results/attack/logs   --clean_logs  results/baseline/logs   --poison_ids  results/attack/poison_ids.jsonl --attack_results_csv results/attack/results/attack_summary_by_ticker.csv --clean_results_csv results/baseline/results/gmats_local_GMATSLLMStrategy.csv
``