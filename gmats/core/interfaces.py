from __future__ import annotations
"""
GMATS Core Interfaces aligned to the formal definition:

GMATS = (V, E, D, T, M, L, Φ, Π, Λ, 𝓔 | 𝓤, Σ, 𝓑, 𝓡)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, TypedDict, Union

# ---------- Graph: Roles & Wiring (V, E) ----------

class Message(TypedDict, total=False):
    id: str
    sender: str         # v \in V
    recipient: str      # v \in V (optional; broadcast if omitted)
    t: str              # message type
    payload: Dict[str, Any]
    schema: Union[str, Dict[str, Any]]
    prov: Dict[str, Any]  # provenance (timestamps, sources, tools used, etc.)

# ---------- Data (𝓓) ----------

class DataFeed(Protocol):
    """Point-in-time observable data with provenance."""
    def observe(self, t: Any, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return items with timestamp <= t and provenance attached."""
        ...

# ---------- Tools (𝓣) ----------

class Tool(Protocol):
    name: str
    schema_in: Dict[str, Any]
    schema_out: Dict[str, Any]
    def validate(self, q: Dict[str, Any]) -> None: ...
    def __call__(self, q: Dict[str, Any]) -> Dict[str, Any]: ...

# ---------- Memory (𝓜) ----------

class MemoryStore(Protocol):
    def admit(self, item: Dict[str, Any]) -> bool: ...
    def promote(self, key: str) -> None: ...
    def decay(self, dt: float) -> None: ...
    def retrieve(self, query: Dict[str, Any], k: int = 10) -> List[Dict[str, Any]]: ...

# ---------- Coordination / Judge (𝓛) ----------

@dataclass
class CoordinationResult:
    s: List[float]                  # state vector in R^n
    rho: Dict[str, Any]             # e.g., routing/weights/consensus meta
    log: List[Message]              # audit log / rationale

class Coordinator(Protocol):
    def coordinate(self, inbox: Iterable[Message]) -> CoordinationResult: ...

# ---------- Alpha-miner (Φ) ----------

class AlphaMiner(Protocol):
    def factors(self, data: DataFeed, memory: Optional[MemoryStore] = None) -> List[float]: ...

# ---------- Policy (Π) ----------

Action = str  # "BUY", "SELL", "HOLD"
Weights = List[float]

class Policy(Protocol):
    def decide(
        self,
        s: List[float],
        f: Optional[List[float]],
        state_t: Dict[str, Any]
    ) -> Union[List[Action], Weights]:
        ...

# ---------- Constraints / Risk (Λ) ----------

class Constraints(Protocol):
    def gate(
        self,
        a: Union[List[Action], Weights],
        state_t: Dict[str, Any],
        history: List[Dict[str, Any]],
        budgets: Dict[str, Any]
    ) -> Tuple[bool, Union[List[Action], Weights]]:
        """Return (ok, possibly-shaped action a')."""
        ...

# ---------- Environment (𝓔) ----------

@dataclass
class Fills:
    details: List[Dict[str, Any]]   # [{symbol, qty, price?, ret, pnl, ...}]
    meta: Dict[str, Any]            # slippage, fees, etc.

class Environment(Protocol):
    def step(self, a_prime: Union[List[Action], Weights], x_t: Dict[str, Any]) -> Tuple[Fills, Dict[str, Any]]:
        """Maps (a', x_t) -> (fills, x_{t+1})."""
        ...

# ---------- Optional: Reward (𝓡), Update (𝓤), Schedule (Σ), Budgets (𝓑) ----------

class Reward(Protocol):
    def __call__(self, fills: Fills, x_next: Dict[str, Any]) -> float: ...

class Updater(Protocol):
    def update(
        self,
        components: Dict[str, Any],   # e.g., {"memory": M, "policy": Π, ...}
        logs: List[Message],
        reward: float
    ) -> None: ...

class Schedule(Protocol):
    def next_tick(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Returns next orchestration state or None to stop."""
        ...

# Budgets 𝓑 can be a plain dict, but we offer a tiny helper dataclass
@dataclass
class Budgets:
    tokens: Optional[int] = None
    api_calls: Optional[int] = None
    time_sec: Optional[float] = None
    capital: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens,
            "api_calls": self.api_calls,
            "time_sec": self.time_sec,
            "capital": self.capital,
        }
