# gmats/core/engine.py
from __future__ import annotations
import datetime as dt
from collections import defaultdict
from typing import Dict, Tuple, List, Any
import yaml

from gmats.core.registry import get_agent_factory
from gmats.llm.provider import make_llm_provider

def load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_agents(cfg: Dict[str, Any], prompts: Dict[str, Any]):
    agents: Dict[str, Any] = {}
    keys = cfg.get("llm_keys", {}) or {}
    for a in cfg.get("agents", []):
        aid = a["id"]
        role = a.get("role", "analyst")
        prompt_id = a.get("prompt_id")
        prompt = prompts.get(prompt_id, a.get("prompt", "")) if isinstance(prompts, dict) else a.get("prompt", "")
        params = {k: v for k, v in a.items() if k not in ("id", "role", "prompt_id", "prompt", "llm")}
        llm_cfg = a.get("llm", {}) or {}
        llm = make_llm_provider(llm_cfg, keys)
        factory = get_agent_factory(role)
        agents[aid] = factory(id=aid, role=role, prompt=prompt, llm=llm, **params)
    return agents

def compile_graph(cfg: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    outs: Dict[str, List[str]] = defaultdict(list)
    ins: Dict[str, List[str]] = defaultdict(list)
    for e in cfg.get("routing", []):
        s, d = e["src"], e["dst"]
        outs[s].append(d)
        ins[d].append(s)
    return outs, ins

def topo(agents: Dict[str, Any]) -> List[str]:
    order: List[str] = []
    role_order = ["analyst", "coordinator", "controller", "executor"]
    for role in role_order:
        for aid, a in agents.items():
            if getattr(a, "role", None) == role:
                order.append(aid)
    return order

def date_range(sched: Dict[str, str]):
    d0 = dt.date.fromisoformat(sched["date_from"])
    d1 = dt.date.fromisoformat(sched["date_to"])
    cur = d0
    while cur <= d1:
        yield cur.isoformat()
        cur = cur + dt.timedelta(days=1)

def run(config_path="configs/gmats.yaml", prompts_path="configs/prompts.yaml"):
    cfg = load_yaml(config_path)
    prompts = load_yaml(prompts_path)
    agents = build_agents(cfg, prompts)
    outs, ins = compile_graph(cfg)
    order = topo(agents)
    assets = cfg["assets"]

    mailbox = defaultdict(list)  # agent_id -> list[message]

    for date in date_range(cfg["schedule"]):
        for aid in order:
            agent = agents[aid]
            upstream = [m for m in mailbox[aid] if m["ts"] == date]
            downstream_ids = outs.get(aid, [])
            msg = agent.run(date=date, assets=assets, inbox=upstream, downstream_ids=downstream_ids)
            # deliver
            for dst in downstream_ids:
                mailbox[dst].append(msg)
            mailbox[aid].append(msg)
    return mailbox
