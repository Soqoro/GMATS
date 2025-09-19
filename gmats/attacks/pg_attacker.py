from __future__ import annotations
"""
Policy-Gradient Attacker (REINFORCE)
====================================
Learns to perturb pre-policy signals (momentum/news/social scores) to *hurt* GMATS.

- Action space: a_t = [Δ_mom, Δ_news, Δ_social] bounded via tanh * eps
- Objective: maximize attack reward R_att = - r_env  (minimize PnL)
- Regularizers:
    * Lipschitz (temporal smoothness):  λ_lip * ||a_t - a_{t-1}||^2
    * Generalization (weight decay):    λ_gen * sum ||θ||^2

Safe fallback: if torch isn't available, the attacker becomes a no-op.
"""

from typing import Any, Dict, Optional
import math  # NEW

# ---- Safe pre-binding to avoid "possibly unbound" warnings ----
torch: Any = None
nn: Any = None
optim: Any = None
TORCH_OK: bool = False
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    TORCH_OK = True
except Exception:
    # keep the pre-bound stubs above; attacker will degrade to no-op
    TORCH_OK = False


class _NoOpAttacker:
    """Fallback when PyTorch is unavailable."""
    def __init__(self, *_, **__):
        self.last_action = [0.0, 0.0, 0.0]

    def propose(self, x_t: Dict[str, Any], obs: Dict[str, float]) -> Dict[str, float]:
        return {"delta_mom": 0.0, "delta_news": 0.0, "delta_social": 0.0}

    def update(self, feedback: Dict[str, Any]) -> None:
        return


class PGAttacker:
    """
    Gaussian policy with REINFORCE + moving baseline.

    Inputs (features): [m_mom, s_news, s_social, coord_margin]
    Outputs: mean/log_std for 3 dims -> tanh -> scale by epsilons
    """

    # NOTE: avoid torch types in annotations to keep file importable without torch
    def __init__(
        self,
        *,
        lr: float = 1e-3,
        hidden: int = 32,
        eps_mom: float = 0.05,
        eps_news: float = 0.05,
        eps_social: float = 0.05,
        lambda_lip: float = 1.0,
        lambda_gen: float = 1e-4,
        entropy_coef: float = 1e-3,
        device: str = "auto",
    ):
        # always define attributes to satisfy static analyzers
        self._impl: Optional[_NoOpAttacker] = None
        self.device: Any = "cpu"
        self.eps: Any = None
        self.net: Any = None
        self.opt: Any = None
        self.baseline: Any = None
        self._last_action: Any = None
        self._curr_action: Any = None
        self._last_logprob: Any = None
        self._last_log_std: Any = None        # NEW: keep log-std for entropy

        self.lambda_lip = float(lambda_lip)
        self.lambda_gen = float(lambda_gen)
        self.entropy_coef = float(entropy_coef)

        if not TORCH_OK:
            # degrade to no-op gracefully
            self._impl = _NoOpAttacker()
            # keep shapes consistent for callers
            self._last_action = [0.0, 0.0, 0.0]
            return

        # Select device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Bounds for each action component
        self.eps = torch.tensor([eps_mom, eps_news, eps_social], dtype=torch.float32, device=self.device)

        # Small MLP policy: 4 -> hidden -> hidden -> 6 (3 means + 3 log_stds)
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 6),
        ).to(self.device)

        self.opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=0.0)

        # Moving baseline for variance reduction
        self.baseline = torch.zeros(1, device=self.device)

        # Temporal state
        self._last_action = torch.zeros(3, device=self.device)
        self._curr_action = torch.zeros(3, device=self.device)
        self._last_logprob = None

    # --- helpers ---
    def _feat(self, obs: Dict[str, float]) -> Any:
        if not TORCH_OK:
            return None
        m = float(obs.get("m_mom", 0.0))
        n = float(obs.get("s_news", 0.0))
        s = float(obs.get("s_social", 0.0))
        c = float(obs.get("coord_margin", 0.0))
        x = torch.tensor([m, n, s, c], dtype=torch.float32, device=self.device)
        return x

    # --- public API ---
    def propose(self, x_t: Dict[str, Any], obs: Dict[str, float]) -> Dict[str, float]:
        if self._impl is not None:
            return self._impl.propose(x_t, obs)

        x = self._feat(obs)
        out = self.net(x)
        mean, log_std = out[:3], out[3:]
        log_std = torch.clamp(log_std, -5.0, 2.0)          # numerical safety
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()                        # reparameterized sample
        logprob = dist.log_prob(raw_action).sum()

        action = torch.tanh(raw_action) * self.eps         # bound to [-eps, +eps]

        # Keep for update() with gradient path intact
        self._last_logprob = logprob
        self._curr_action = action                         # NOTE: no .detach() here
        self._last_log_std = log_std                       # NEW

        return {
            "delta_mom": float(action[0].detach().cpu().item()),
            "delta_news": float(action[1].detach().cpu().item()),
            "delta_social": float(action[2].detach().cpu().item()),
        }

    def update(self, feedback: Dict[str, Any]) -> None:
        if self._impl is not None:
            self._impl.update(feedback)
            return
        if self._last_logprob is None:
            return

        R_att = float(feedback.get("reward", 0.0))

        # Lipschitz (temporal smoothness) penalty — now has gradients via _curr_action
        lip = torch.sum((self._curr_action - self._last_action) ** 2)

        # Manual L2 weight decay (generalization)
        wd = torch.tensor(0.0, device=self.device)
        for p in self.net.parameters():
            wd = wd + p.pow(2).sum()

        # Moving baseline (no grad)
        with torch.no_grad():
            self.baseline = 0.9 * self.baseline + 0.1 * torch.tensor([R_att], device=self.device)

        advantage = torch.tensor(R_att, device=self.device) - self.baseline

        # Entropy of diagonal Normal: 0.5*log(2πe) + log_std per dim
        if self._last_log_std is not None:
            const = 0.5 * math.log(2.0 * math.pi * math.e)
            entropy = (const + self._last_log_std).sum()
        else:
            entropy = torch.tensor(0.0, device=self.device)

        loss = -(self._last_logprob * advantage) \
               + self.lambda_lip * lip \
               + self.lambda_gen * wd \
               - self.entropy_coef * entropy

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        except Exception:
            pass
        self.opt.step()

        # Roll temporal state (store detached copy for next Lipschitz term)
        self._last_action = self._curr_action.detach()
        self._last_logprob = None
        self._last_log_std = None
