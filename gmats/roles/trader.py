from __future__ import annotations
"""
Trader (adapter over Environment)
================================
Light execution shim that delegates to 𝓔 (Environment.step).
Accepts either discrete actions (List[Action]) or weight vectors (Weights).
"""

from typing import Any, Dict, List, TypedDict, Union

from ..core.interfaces import Environment, Action, Weights


class ExecutionResult(TypedDict, total=False):
    a_prime: Union[List[Action], Weights]
    fills: List[Dict[str, Any]]     # flattened from Fills.details
    x_next: Dict[str, Any]
    status: str                     # "submitted" | "ok" | "error"
    meta: Dict[str, Any]


class Trader:
    """Minimal executor that calls env.step(a', x_t) and returns a simple result."""

    def __init__(self, env: Environment):
        self.env = env

    def _coerce_actions(self, a: Union[str, List[Action], Weights]) -> Union[List[Action], Weights]:
        # Allow legacy single-string actions.
        if isinstance(a, str):
            return [a.upper()]
        # If it's a list and first element is a string, normalize to upper-case actions.
        if isinstance(a, list) and (len(a) == 0 or isinstance(a[0], str)):
            return [str(x).upper() for x in a]  # type: ignore[list-item]
        # Otherwise treat as weights (List[float])
        return a  # assume Weights

    def execute(
        self,
        a_prime: Union[str, List[Action], Weights],
        x_t: Dict[str, Any],
    ) -> ExecutionResult:
        a_norm = self._coerce_actions(a_prime)
        fills, x_next = self.env.step(a_norm, x_t)
        return {
            "a_prime": a_norm,
            "fills": fills.details or [],
            "x_next": x_next,
            "status": "submitted",
            "meta": getattr(fills, "meta", {}) or {},
        }
