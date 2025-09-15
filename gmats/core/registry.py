from __future__ import annotations
"""
GMATS Registry
==============
A tiny plugin registry for building components by name (from config files).

Usage
-----
1) Register a factory:

    from gmats.core.registry import register

    @register("policy.threshold")
    def make_policy(buy_thr: float = 0.10, sell_thr: float = -0.10):
        from gmats.roles.policy import ThresholdPolicy
        return ThresholdPolicy(buy_thr=buy_thr, sell_thr=sell_thr)

2) Build it from a config:

    from gmats.core.registry import build
    policy = build("policy.threshold", buy_thr=0.12, sell_thr=-0.12)

Notes
-----
- Ensure the module that performs the @register calls is imported before you
  call build(...), otherwise the entry won't exist.
"""

from typing import Dict, Any, Callable

_REG: Dict[str, Callable[..., Any]] = {}

def register(name: str):
    """Decorator to register a factory under a string name."""
    def deco(fn: Callable[..., Any]):
        _REG[name] = fn
        return fn
    return deco

def build(name: str, **kwargs):
    """Instantiate a registered factory by name, forwarding kwargs."""
    if name not in _REG:
        raise ValueError(f"Unknown component '{name}'. Registered: {list(_REG)}")
    return _REG[name](**kwargs)

def list_components():
    """List all registered component names."""
    return list(_REG)
