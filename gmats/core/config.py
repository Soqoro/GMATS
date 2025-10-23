"""YAML config models + loader (prompt-only agents)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os, yaml

@dataclass
class AgentSpec:
    id: str
    role: str
    prompt_key: Optional[str] = None      # preferred key
    prompt_id: Optional[str] = None       # compat alias
    prompt: Optional[Any] = None
    params: Dict[str, Any] = field(default_factory=dict)
    # Accept graph specified inline per agent
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

@dataclass
class RouteSpec:
    src: str
    dst: str

@dataclass
class Config:
    assets: List[str]
    agents: List[AgentSpec]
    routing: List[RouteSpec]
    data: Dict[str, Any]
    schedule: Dict[str, str]
    llm: Dict[str, Any]
    prompts_file: Optional[str] = None

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_prompts(path: str) -> Dict[str, str]:
    raw = load_yaml(path)
    return (raw.get("prompts") or {}) if isinstance(raw, dict) else {}

def load_config(path: str) -> Config:
    raw = load_yaml(path)

    # Build AgentSpec, mapping prompt_id -> prompt_key and folding unknown keys into params
    agents: List[AgentSpec] = []
    allowed = {"id","role","prompt_key","prompt_id","prompt","params","inputs","outputs"}
    for a in raw.get("agents", []):
        a = dict(a)  # copy
        if "prompt_key" not in a and "prompt_id" in a:
            a["prompt_key"] = a["prompt_id"]
        extras = {k: v for k, v in a.items() if k not in allowed}
        base = {k: v for k, v in a.items() if k in allowed}
        # merge extras into params
        params = dict(base.get("params") or {})
        params.update(extras)
        base["params"] = params
        agents.append(AgentSpec(**base))

    # Start with explicit routing if provided
    routing: List[RouteSpec] = [RouteSpec(**r) for r in raw.get("routing", [])]

    # Also derive routing from per-agent inputs/outputs if present
    for ag in agents:
        for s in ag.inputs:
            routing.append(RouteSpec(src=s, dst=ag.id))
        for d in ag.outputs:
            routing.append(RouteSpec(src=ag.id, dst=d))

    prompts_file = raw.get("prompts_file") or os.path.join(os.path.dirname(path), "prompts.yaml")
    return Config(
        assets=[s.upper() for s in raw.get("assets", [])],
        agents=agents,
        routing=routing,
        data=raw.get("data", {}),
        schedule=raw.get("schedule", {}),
        llm=raw.get("llm", {}),
        prompts_file=prompts_file,
    )
