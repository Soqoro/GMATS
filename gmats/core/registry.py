"""Registries for agents and policies + default registrations."""
from __future__ import annotations
from typing import Callable, Dict, Any, List

# Registry mapping role -> factory (class or callable returning an agent)
_REGISTRY: Dict[str, Callable[..., Any]] = {}

def _get_llm_agent_factory() -> Callable[..., Any]:
    from gmats.agents.llm_agent import LLMAgent
    return LLMAgent

def _get_executor_factory() -> Callable[..., Any]:
    try:
        from gmats.agents.executor import Executor  # adjust if your class is named differently
        return Executor
    except Exception:
        # Fallback to LLMAgent if executor class is unavailable
        return _get_llm_agent_factory()

# Pre-register common roles
_REGISTRY.setdefault("llm", _get_llm_agent_factory())
_REGISTRY.setdefault("analyst", _get_llm_agent_factory())
_REGISTRY.setdefault("executor", _get_executor_factory())
# Any custom role (e.g., "social_analyst") will default to LLMAgent

def register_role(role: str, factory: Callable[..., Any]) -> None:
    """Register or override a role -> factory mapping."""
    _REGISTRY[role] = factory

def get_agent_factory(role: str) -> Callable[..., Any]:
    """
    Return a factory (class/callable) for the given role.
    Defaults to LLMAgent for unknown roles.
    """
    role_key = (role or "llm").lower()
    factory = _REGISTRY.get(role_key)
    if factory is None:
        # Default to LLMAgent for unrecognized roles
        factory = _get_llm_agent_factory()
        _REGISTRY[role_key] = factory
    return factory

class Registry:
    def __init__(self):
        self._d: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]):
        self._d[name] = factory

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._d:
            raise KeyError(f"Factory not registered: {name}")
        return self._d[name]

AGENTS = Registry()      # agent type → factory(id, assets, prompt)
POLICIES = Registry()    # policy type → callable(params)

# ---- decision policy (kept programmatic for execution determinism) ----
def decision_sign(**p):
    mode = str(p.get("mode", "discrete"))
    thr = float(p.get("threshold", 0.10))
    def _f(vector: List[float]) -> List[float]:
        if mode == "weights":
            return [max(-1.0, min(1.0, v)) for v in vector]
        # discrete: {-1,0,1} by threshold
        out: List[int] = []
        for v in vector:
            out.append(1 if v >= thr else (-1 if v <= -thr else 0))
        return out
    return _f

POLICIES.register("decision.sign", decision_sign)

# Back-compat: export dict alias for legacy imports
REGISTRY = _REGISTRY

__all__ = [
    "register_role",
    "get_agent_factory",
    "AGENTS",
    "POLICIES",
    "decision_sign",
    "REGISTRY",
]
