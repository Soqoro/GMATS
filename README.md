# GMATS: A Minimal, Reproducible Framework for LLM‑Augmented Market Agents

Abstract
GMATS is a lightweight, modular research framework for studying market-facing agent systems that combine symbolic modules (alpha mining, policy, risk) with LLM-based coordination. It emphasizes reproducibility, ablation-friendly design, and offline evaluation via a toy backtester. GMATS exposes a clean set of protocols and registries that make it easy to swap components, test adversarial robustness, and measure contagion and performance metrics under different debate/coordination strategies. The framework is intentionally concise to enable rapid iteration, transparent experiments, and fair comparisons.

Keywords: multi-agent systems, LLM-augmented decision making, financial backtesting, robustness, reproducibility.

Highlights and Contributions
- Minimal, well-typed protocols for each component: data feeds, memory, alpha miner Φ, coordinator/judge 𝓛, policy Π, constraints/risk Λ, environment 𝓔, reward 𝓡.
- Pluggable registries to swap components at runtime without changing orchestration code.
- Offline, deterministic backtester with simple return-based evaluation and risk-free rate support.
- LLM-judge interface for debate-style coordination with a mock backend for CI and an HF Transformers backend for research models.
- Contagion metrics (action/consensus flip rates) plus standard performance stats (ann. return, vol, Sharpe, max drawdown).
- Optional policy-gradient attacker to study poisoning of pre-policy signals; safe no-op fallback when PyTorch is unavailable.

System Definition (Interfaces)
GMATS = (V, E, D, T, M, L, Φ, Π, Λ, 𝓔 | 𝓤, Σ, 𝓑, 𝓡)
- Messages (V, E): gmats.core.interfaces.Message
- Data 𝓓: DataFeed.observe(t, q) -> List[Dict]
- Tools 𝓣: optional; protocol for tool-augmented pipelines
- Memory 𝓜: MemoryStore with admit/promote/decay/retrieve
- Judge/Coordinator 𝓛: Coordinator.coordinate(inbox) -> CoordinationResult
- Alpha Φ: AlphaMiner.factors(...)
- Policy Π: Policy.decide(s, f, state_t) -> actions or weights
- Constraints Λ: Constraints.gate(a, state, history, budgets) -> (ok, a′)
- Environment 𝓔: Environment.step(a′, x_t) -> (fills, x_{t+1})
- Reward 𝓡: callable on (fills, x_{t+1})
- Optional Update 𝓤, Schedule Σ, Budgets 𝓑

Repository Structure
- gmats/core: type-safe interfaces, memory, registries
- gmats/env: backtester and environment wrapper
- gmats/roles: reference role implementations (alpha miner, coordinator, policy, risk, trader)
- gmats/llm: LLM judge protocol, mock backend, HF backend
- gmats/experiments: metrics and experimental runner (minimal scaffold)
- gmats/configs: default settings, prompts, attacker config
- data/: example directories for returns/news/social/fundamentals
- runs/: example JSON reports (outputs)

Placeholders for Figures
- Architecture Diagram (high-level)
  - docs/figures/architecture.png
  - Placeholder: A layered diagram showing Data/Memory -> Alpha Φ -> Debate 𝓛 -> Policy Π -> Risk Λ -> Environment 𝓔 -> Reward 𝓡 with feedback edges to Memory/Attacker.
- Debate Flow (LLM coordination)
  - docs/figures/debate_flow.png
  - Placeholder: Two advocates (bull/bear) generate arguments; a Judge selects A/B; margin yields signed state s ∈ R.
- Evaluation Loop (Backtest)
  - docs/figures/backtest_loop.png
  - Placeholder: For each episode date, observe -> factors -> debate -> action -> env.step -> reward -> metrics update.
- Contagion Metrics
  - docs/figures/contagion_metrics.png
  - Placeholder: Side-by-side clean vs. attacked outcomes with action/consensus flip rates and performance curves.

Quick Start
1) Environment
- Python 3.10+
- CPU-only works for core; GPU recommended for LLMs.

2) Install
- pip install -r requirements.txt
- Optional for HF LLMs: pip install transformers accelerate bitsandbytes
- Optional for attacker: pip install torch

3) Data
- Place CSVs under data/returns with either:
  - date,ret
  - date,close (daily percentage returns are computed)
- Example (date,ret):
  2020-01-02,0.012
  2020-01-03,-0.006

Minimal Usage (Pure Python)
- Construct a backtester and environment, then submit an action:
  - from gmats.env.backtest import Backtester
  - from gmats.env.backtest_env import BacktestEnvironment
  - from gmats.roles.trader import Trader
  - bt = Backtester("data/returns", rf_daily=0.0)
  - env = BacktestEnvironment(bt)
  - trader = Trader(env)
  - state = {"symbol":"AAPL","asof":"2020-01-02","horizon":1}
  - result = trader.execute(["BUY"], state)
  - print(result)

Debate Coordinator
- Numeric fallback (no LLM): gmats.roles.coordinator.DebateCoordinator infers winner by score if LLM is None.
- Mock LLM (deterministic, CI-friendly):
  - from gmats.llm.mock import MockLLM
  - from gmats.roles.coordinator import DebateCoordinator
  - judge = MockLLM()
  - coord = DebateCoordinator(llm=judge, criterion="Which side better forecasts 1-day return?")
- HF LLM (research models):
  - from gmats.llm.hf_generic import HFLLM
  - judge = HFLLM("mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=96, temperature=0.2, load_4bit=True)
  - coord = DebateCoordinator(llm=judge)
- Prompts (ReAct-ready) are configurable in gmats/configs/prompts.yaml.

Configuration
- Edit gmats/configs/default.yaml:
  - assets: ["AAPL","MSFT"]
  - data.returns_dir: "data/returns"
  - coordination.judge.llm: "mock" | "hf"
  - coordination.judge.model_id: e.g., mistralai/Mistral-7B-Instruct-v0.3
  - prompts in gmats/configs/prompts.yaml
- Attacker settings in gmats/configs/attacker.yaml (pg: lr, eps, regularizers).

Metrics
- gmats/experiments/metrics.py computes:
  - Contagion: action flip rate, consensus flip rate (ASR_action, ASR_consensus)
  - Performance: cumulative/annual return, annualized vol, Sharpe, max drawdown
- Output: runs/*.json for quick inspection and plotting.

Attacker (Optional)
- REINFORCE-style policy gradient attacker perturbs momentum/news/social signals within bounded ε.
- If torch is unavailable, the attacker degrades to a safe no-op.
- Use to probe robustness of coordination and policy stages.

Extending GMATS (Plugins)
- Register new components via registries (gmats.core.registry):
  - from gmats.core.registry import COORDINATORS
  - COORDINATORS.register("my_coord", MyCoordinator())
- Conform to the corresponding Protocol in gmats.core.interfaces.

Reproducibility Notes
- Use fixed seeds where applicable.
- Prefer deterministic HF settings and explicit dtype.
- Backtester uses normalized timestamps and robust date parsing.

LLM Choices and Resource Notes
- Recommended open-source judges commonly used in research:
  - mistralai/Mistral-7B-Instruct-v0.3 (solid quality, accessible VRAM)
  - meta-llama/Meta-Llama-3.1-8B-Instruct (license-dependent)
- 4-bit loading (bitsandbytes) reduces memory; quality vs. cost trade-offs apply.

Ethical and Safety Considerations
- The framework is for research on decision-making and robustness. Avoid using it for real-money trading without proper risk controls, compliance review, and extensive validation.
- When using external LLMs, ensure data privacy and terms-of-use compliance.

Limitations
- The experimental runner is intentionally minimal and subject to change.
- The backtester is a toy and does not model slippage, fees, or multi-asset portfolios.

Citation
- Please cite this project if it helps your research:
  @misc{gmats2025,
    title  = {GMATS: A Minimal, Reproducible Framework for LLM-Augmented Market Agents},
    author = {Your Name and Contributors},
    year   = {2025},
    url    = {https://github.com/your-org/GMATS}
  }

License
- Add your license file and statement here (e.g., Apache-2.0/MIT).

Contact
- For questions and contributions, open an issue or pull request.

Appendix: File Map (Selected)
- gmats/core/interfaces.py: Protocols and data classes
- gmats/core/registry.py: Global registries for plugins
- gmats/core/memory.py: Minimal memory store
- gmats/env/backtest.py: Offline returns loader and PnL logic
- gmats/env/backtest_env.py: Environment wrapper 𝓔
- gmats/roles/coordinator.py: DebateCoordinator (numeric fallback + LLM judge)
- gmats/roles/policy.py: ThresholdPolicy baseline
- gmats/roles/risk.py: Risk gate
- gmats/roles/trader.py: Execution shim over 𝓔
- gmats/llm/base.py: Judge protocol
- gmats/llm/mock.py: Deterministic mock LLM
- gmats/llm/hf_generic.py: HF Transformers backend (judge/summarize)
- gmats/experiments/metrics.py: Contagion and performance metrics
- gmats/configs/default.yaml, prompts.yaml, attacker.yaml: Configs