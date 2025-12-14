# GMATS: A Minimal, Reproducible Framework for LLM‑Augmented Market Agents

## Data

### Reddit

We release a curated subset of Reddit posts and comments that mention  
{AAPL, AMZN, NFLX, MSFT, GOOG, META, TSLA} between 2021-09-30 and 2022-09-30.

Source subreddits:
- r/investing
- r/StockMarket
- r/stocks
- r/wallstreetbets

The raw data were obtained from publicly available Pushshift Reddit dumps
(Baumgartner et al., 2020). For each post/comment we keep text and basic
metadata (id, subreddit, created_utc, score, num_comments, link_id/parent_id,
etc.). Reddit usernames and profile information are not included.

Reddit data used in our experiments can be downloaded from:

> https://drive.google.com/drive/folders/1zgphtMdGT9ZR2le_24EkprXdtRYnJcq8

Use of this data must comply with Reddit’s User Agreement, Developer Terms,
and Data API Terms. This repository does not grant any additional rights to
Reddit content beyond what Reddit itself permits.

To reproduce our experiments, download the Reddit data and convert it into the
same tabular format as the example files in the data/ directory
(e.g., JSONL/CSV with matching column names). You are free to implement your
own preprocessing pipeline; we only require that the final format matches
the examples.

---

### Twitter

For the Twitter side, we do not redistribute tweet text in this
repository. Instead, we rely on the public Kaggle dataset:

> equinxx, “Stock Tweets for Sentiment Analysis and Prediction”  
> https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction

Please download the dataset directly from Kaggle (subject to Twitter/X and
Kaggle terms of use), then manually process it into the same format as the
example Twitter files in data/ (e.g., one row per tweet with the expected
columns).

## Commands

### Run clean baseline
python scripts/run.py --config configs/gmats.yaml --debug --log_by_asset --log_agents

### Train RL agent:
python scripts/train_attacker.py --config configs/gmats.yaml --timesteps 14

### Run Attack:
python scripts/run_attack.py --config configs/gmats.yaml --budget 1 --attacker random

### Evalute Metrics:
python scripts/metrics.py   --attack_logs results/attack/logs   --clean_logs  results/baseline/logs   --poison_ids  results/attack/poison_ids.jsonl --attack_results_csv results/attack/results/attack_summary_by_ticker.csv --clean_results_csv results/baseline/results/gmats_local_GMATSLLMStrategy.csv
