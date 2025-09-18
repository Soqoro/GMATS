from __future__ import annotations
"""
Coordinator (Debate)
====================
Aggregates opposing messages (bull vs bear) into a stance using either:
- numeric fallback (compare payload.score), or
- an LLM-style chooser with a .choose(...) API.

Fits gmats.core.interfaces.Coordinator:
    coordinate(self, inbox: Iterable[Message]) -> CoordinationResult
"""

from typing import Any, Dict, Iterable, List, Optional, Protocol

from ..core.interfaces import Coordinator, CoordinationResult, Message


class Chooser(Protocol):
    """Minimal judge interface expected from an LLM wrapper."""
    def choose(self, *, context: str, a: str, b: str, criterion: str) -> str: ...


class DebateCoordinator(Coordinator):
    """Debate coordinator selecting a winner between bull/bear arguments."""

    def __init__(self, llm: Optional[Chooser] = None, criterion: str | None = None):
        """
        Args:
            llm: Optional judge exposing .choose(context=, a=, b=, criterion=) -> str
            criterion: Optional string prompt for the judge.
        """
        self.llm: Optional[Chooser] = llm
        self.criterion: str = (
            criterion
            or "Which argument better predicts near-term performance?"
        )

    def _get_role(self, m: Message) -> Optional[str]:
        # Prefer payload.role; fall back to top-level 'role' if present.
        payload = m.get("payload", {}) or {}
        role = payload.get("role") or m.get("role")
        if isinstance(role, str):
            r = role.strip().lower()
            if r in ("bull", "bear"):
                return r
        return None

    def _get_score(self, m: Message) -> float:
        payload = m.get("payload", {}) or {}
        try:
            return float(payload.get("score", 0.0))
        except Exception:
            return 0.0

    def _get_text(self, m: Message) -> str:
        payload = m.get("payload", {}) or {}
        # Common fields: 'speech' or 'text'; include brief provenance if available.
        text = str(payload.get("speech") or payload.get("text") or "")
        return text

    def _get_context(self, bull: Optional[Message], bear: Optional[Message]) -> str:
        ctx_parts: List[str] = []
        for tag, m in (("BULL", bull), ("BEAR", bear)):
            if not m:
                continue
            prov = m.get("prov", {}) or {}
            src = prov.get("source") or prov.get("tool") or prov.get("feed") or ""
            t = m.get("t") or ""
            if src or t:
                ctx_parts.append(f"[{tag} src={src} t={t}]")
        return "\n".join(ctx_parts)

    def coordinate(self, inbox: Iterable[Message]) -> CoordinationResult:
        msgs = list(inbox) if inbox else []
        bull = next((m for m in msgs if self._get_role(m) == "bull"), None)
        bear = next((m for m in msgs if self._get_role(m) == "bear"), None)

        # Fallback: numeric score comparison
        def fallback() -> tuple[str, float, str]:
            s_bull = self._get_score(bull) if bull else 0.0
            s_bear = self._get_score(bear) if bear else 0.0
            if s_bull == s_bear:
                return ("bull", 0.0, "fallback")  # tie → neutral margin
            return ("bull", abs(s_bull - s_bear), "fallback") if s_bull > s_bear else ("bear", abs(s_bull - s_bear), "fallback")

        method = "fallback"
        if not bull or not bear:
            winner, margin, method = fallback()
        elif self.llm is None:
            winner, margin, method = fallback()
        else:
            # LLM judge path
            context = self._get_context(bull, bear)
            a_text = self._get_text(bull)
            b_text = self._get_text(bear)
            try:
                pick = str(self.llm.choose(context=context, a=a_text, b=b_text, criterion=self.criterion)).strip()
                winner = "bull" if pick.upper().startswith("A") else "bear"
                # use numeric gap as margin if available, else a small default
                margin = abs(self._get_score(bull) - self._get_score(bear)) or 1.0
                method = "llm"
            except Exception:
                winner, margin, method = fallback()

        # Map stance to 1-D state vector s in R^1 (sign encodes stance)
        s_value = float(margin if winner == "bull" else -margin)
        s = [s_value]

        rho: Dict[str, Any] = {
            "winner": winner,
            "margin": float(margin),
            "method": method,
            "scores": {
                "bull": self._get_score(bull) if bull else 0.0,
                "bear": self._get_score(bear) if bear else 0.0,
            },
        }

        # Append a compact result message to the log
        result_msg: Message = {
            "id": "debate:result",
            "sender": "coordinator.debate",
            "t": "stance",
            "payload": {"winner": winner, "margin": float(margin)},
            "schema": "gmats/stance@v1",
            "prov": {"method": method},
        }
        log = msgs + [result_msg]

        return CoordinationResult(s=s, rho=rho, log=log)
