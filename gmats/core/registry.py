from __future__ import annotations
from typing import Any, Dict, TypeVar, Callable

T = TypeVar("T")

class Registry:
    def __init__(self):
        self._reg: Dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        key = name.lower()
        if key in self._reg:
            raise KeyError(f"Duplicate plugin: {name}")
        self._reg[key] = obj

    def get(self, name: str) -> Any:
        key = name.lower()
        if key not in self._reg:
            raise KeyError(f"Plugin not found: {name}")
        return self._reg[key]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._reg

    def list(self) -> Dict[str, Any]:
        return dict(self._reg)

# Global registries for convenience
COORDINATORS = Registry()
ALPHAS = Registry()
POLICIES = Registry()
RISKS = Registry()
TOOLS = Registry()
ENVIRONMENTS = Registry()
REWARDS = Registry()
SCHEDULES = Registry()
