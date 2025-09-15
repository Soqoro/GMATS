from __future__ import annotations
"""
GMATS Interfaces
================
Protocols (PEP 544) and TypedDicts that define the pluggable contracts in GMATS.
"""

from typing import Protocol, List, Dict, Any, Optional, TypedDict, Literal

# ---------------------------------------------------------------------------
# Common payload types
# ---------------------------------------------------------------------------

class Message(TypedDict, total=False):
    """Message for coordination; in debate we expect roles 'bull' and 'bear'."""
    role: Literal["bull", "bear"]
    speech: str
    score: float
    context: str

class Stance(TypedDict):
    """Coordinator output."""
    winner: Literal["bull", "bear"]
    margin: float

class Decision(TypedDict):
    """Policy decision."""
    action: Literal["BUY", "HOLD", "SELL"]
    margin: float

class ExecutionResult(TypedDict, total=False):
    """Trader result."""
    action: str
    status: Literal["submitted", "filled", "rejected", "error"]
    info: str

class MemoryItem(TypedDict, total=False):
    """Memory item (minimal)."""
    id: str
    text: str
    deep: bool
    ts: float
    cred: float
    imp: float


# ---------------------------------------------------------------------------
# LLM protocol
# ---------------------------------------------------------------------------

class LLM(Protocol):
    """Minimal interface for an LLM used by analysts/judges."""

    def summarize(self, prompt: str) -> str:
        """Compress evidence into a concise argument (≤ a few sentences)."""
        ...

    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str:
        """Choose 'A' or 'B' given shared context and a criterion."""
        ...


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Memory(Protocol):
    """Persistent store for notes/summaries/signals with admission and retrieval."""

    def admit(self, item: Dict[str, Any]) -> bool:
        """Attempt to add an item; return True iff accepted."""
        ...

    def promote(self, item_id: str, score: float) -> None:
        """Optionally promote an item (e.g., to long-term memory)."""
        ...

    def retrieve(self, query: Dict[str, Any], k: int) -> List[MemoryItem]:
        """Return the top-k items relevant to the query."""
        ...

    def stats(self) -> Dict[str, float]:
        """Return counters/ratios (admitted, promotions, size, etc.)."""
        ...


# ---------------------------------------------------------------------------
# Coordinator / Aggregator
# ---------------------------------------------------------------------------

class Coordinator(Protocol):
    """Resolve conflicting messages into a single stance."""

    def aggregate(self, messages: List[Message]) -> Stance:
        """Aggregate role messages into a stance with a winner and margin."""
        ...


# ---------------------------------------------------------------------------
# Alpha Miner
# ---------------------------------------------------------------------------

class AlphaMiner(Protocol):
    """Map heterogeneous evidence to factor vectors and a scalar score."""

    def factors(self, evidence: List[Dict[str, Any]]) -> List[float]:
        """Produce a k-dimensional factor vector from evidence."""
        ...

    def score(self, factors: List[float]) -> float:
        """Map factors to a single scalar (e.g., weighted sum)."""
        ...


# ---------------------------------------------------------------------------
# Policy / Optimizer (PM)
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """Translate scores/stance into discrete actions or weights."""

    def decide(self, score: float) -> Decision:
        """Return {'action': BUY|HOLD|SELL, 'margin': float}."""
        ...


# ---------------------------------------------------------------------------
# Risk / Constraints
# ---------------------------------------------------------------------------

class RiskGate(Protocol):
    """Enforce risk, liquidity, and compliance constraints."""

    def gate(self, stance: Stance) -> Dict[str, Any]:
        """Return at least {'ok': bool}; may include overrides/notes."""
        ...


# ---------------------------------------------------------------------------
# Trader / Execution
# ---------------------------------------------------------------------------

class Trader(Protocol):
    """Submit actions to execution (sim/live)."""

    def execute(self, action: str) -> ExecutionResult:
        """Attempt to execute an action; return status/info."""
        ...


# ---------------------------------------------------------------------------
# Environment / Backtester
# ---------------------------------------------------------------------------

class Environment(Protocol):
    """Convert actions into realized returns (evaluation loop)."""

    def trade_return(self, symbol: str, asof_date: str, action: str, horizon: int = 1) -> float:
        """Return realized (signed) return over the horizon for the action."""
        ...
