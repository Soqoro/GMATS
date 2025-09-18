from __future__ import annotations
"""
Risk Controller (toy)
=====================
Implements gmats.core.interfaces.Constraints:

    gate(
        a: List[Action] | Weights,
        state_t: Dict[str, Any],
        history: List[Dict[str, Any]],
        budgets: Dict[str, Any],
    ) -> tuple[bool, List[Action] | Weights]

Behavior:
- If a stance margin is available (e.g., from the coordinator), veto BUY/SELL when
  margin < min_margin.
- Optionally shape actions to HOLD (or zero weights) when vetoed.
- Budget checks are permissive by default (capital=None → OK).
"""

from typing import Any, Dict, List, Tuple, Union, Optional

from ..core.interfaces import Constraints, Action, Weights


class RiskController(Constraints):
    """Toy risk gate: veto actions if stance margin is too small."""

    def __init__(self, min_margin: float = 0.02, shape_on_veto: bool = True):
        """
        Args:
            min_margin: Minimum acceptable debate/coordination margin to allow risk-on.
            shape_on_veto: If True, replace action with HOLD / zero weights when vetoed.
        """
        self.min_margin = float(min_margin)
        self.shape_on_veto = bool(shape_on_veto)

    # --- helpers -------------------------------------------------------------

    def _is_action_list(self, a: Union[List[Action], Weights]) -> bool:
        return isinstance(a, list) and (len(a) == 0 or isinstance(a[0], str))

    def _extract_margin(self, state_t: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
        """
        Try multiple places for a coordination margin:
        - state_t["coord_margin"] or state_t["margin"] or state_t["stance_margin"]
        - history[-1]["rho"]["margin"] if caller logged coordinator rho
        Defaults to 1.0 (permissive) if not found.
        """
        for key in ("coord_margin", "margin", "stance_margin"):
            if key in state_t:
                try:
                    return float(state_t[key])
                except Exception:
                    pass
        try:
            if history:
                rho = history[-1].get("rho", {}) or {}
                m = rho.get("margin", None)
                if m is not None:
                    return float(m)
        except Exception:
            pass
        return 1.0  # permissive default so lack of margin info doesn't block trades

    def _budget_ok(self, budgets: Dict[str, Any]) -> bool:
        """
        Very light budget check. If capital is provided and <= 0, block.
        Extend with tokens/api/time as needed.
        """
        capital = budgets.get("capital", None)
        if capital is None:
            return True
        try:
            return float(capital) > 0.0
        except Exception:
            return True

    # --- interface ----------------------------------------------------------

    def gate(
        self,
        a: Union[List[Action], Weights],
        state_t: Dict[str, Any],
        history: List[Dict[str, Any]],
        budgets: Dict[str, Any],
    ) -> Tuple[bool, Union[List[Action], Weights]]:
        margin = self._extract_margin(state_t, history)
        budgets_ok = self._budget_ok(budgets)

        ok = (margin >= self.min_margin) and budgets_ok

        if ok or not self.shape_on_veto:
            return ok, a

        # Shape action on veto
        if self._is_action_list(a):
            return False, ["HOLD"]
        # weights → zero out
        try:
            return False, [0.0 for _ in a]  # type: ignore[reportGeneralTypeIssues]
        except Exception:
            return False, a
