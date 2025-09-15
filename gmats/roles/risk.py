from __future__ import annotations
"""
Risk Controller (toy)
=====================
Gates actions using a simple proxy (debate margin). Placeholder for real limits.
"""

from typing import Dict, Any

class RiskController:
    """Toy risk gate: veto actions if stance margin is too small."""

    def __init__(self, stop_loss: float = 0.08, take_profit: float = 0.12):
        """
        Args:
            stop_loss: Placeholder for future risk logic (not used in toy).
            take_profit: Placeholder for future risk logic (not used in toy).
        """
        self.sl = float(stop_loss)
        self.tp = float(take_profit)

    def gate(self, stance: Dict[str, Any]) -> Dict[str, Any]:
        """Allow action if stance margin >= 0.02; else veto."""
        ok = bool(stance.get("margin", 0.0) >= 0.02)
        return {"ok": ok}
