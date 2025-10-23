# gmats/agents/llm_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import json
import datetime as dt

# ---------- light-weight LLM shim ----------
class LLMClient:
    def __init__(self, keyring: Dict[str, str], cfg: Dict[str, Any]):
        self.keyring, self.cfg = keyring, cfg  # {provider, model, temperature}
    def generate(self, system: str, user: str) -> str:
        # TODO: wire to OpenAI/HF/vLLM. For now, return "{}" so pipeline runs.
        return "{}"

# ---------- PIT data adapters (use your loaders) ----------
from gmats.data.market_loader import MarketLoader
from gmats.data.social_loader import SocialLoader

@dataclass
class LLMAgent:
    id: str
    role: str                 # analyst | coordinator | controller
    prompt: Dict[str, str]    # {system, template} from prompts.yaml
    inputs: List[Dict[str, Any]]  # e.g., [{"source":"market","window":{"days":7},"select":{"by":"recency","top_k":30}}]
    llm: LLMClient
    data_cfg: Dict[str, str]  # {root, market_dir, social_dir, ...}

    def __init__(
        self,
        id: str,
        role: str,
        prompt: str,
        llm: Any = None,
        inputs: Optional[Dict[str, Any]] = None,
        data_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.id = id
        self.role = role
        self.prompt = prompt
        self.llm = llm
        self.inputs: Dict[str, Any] = inputs or {}
        # safe defaults
        self.data_cfg: Dict[str, Any] = {
            "root": os.getenv("GMATS_DATA_ROOT", "./data"),
            "market_dir": "market",
            "social_dir": "social",
            "fundamentals_dir": "fundamentals",
        }
        if data_cfg:
            self.data_cfg.update(data_cfg)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _prompt_text(self) -> str:
        """Return a string template regardless of how prompt is provided."""
        p = self.prompt
        if isinstance(p, str):
            return p
        if isinstance(p, dict):
            # common shapes in YAML prompts
            if "template" in p:
                return str(p["template"])
            parts = []
            if p.get("system"):
                parts.append(str(p["system"]))
            if p.get("user"):
                parts.append(str(p["user"]))
            if parts:
                return "\n\n".join(parts)
            # fallback: stringify dict
            return json.dumps(p, ensure_ascii=False)
        if isinstance(p, list):
            return "\n".join(map(str, p))
        return str(p)

    def _rag_vars(self, date, assets, inbox, downstream_ids):
        root = self.data_cfg.get("root", os.getenv("GMATS_DATA_ROOT", "./data"))
        market_dir = self.data_cfg.get("market_dir", "market")
        social_dir = self.data_cfg.get("social_dir", "social")
        fundamentals_dir = self.data_cfg.get("fundamentals_dir", "fundamentals")

        date_iso = date.isoformat() if hasattr(date, "isoformat") else str(date)
        paths = {
            "market": os.path.join(root, market_dir),
            "social": os.path.join(root, social_dir),
            "fundamentals": os.path.join(root, fundamentals_dir),
        }

        inbox_map = inbox or {}
        # Flatten inbox messages to a list[str]
        inbox_msgs: list[str] = []
        if isinstance(inbox_map, dict):
            for _, v in inbox_map.items():
                if isinstance(v, list):
                    for m in v:
                        if isinstance(m, dict):
                            inbox_msgs.append(str(m.get("text") or m.get("message") or m))
                        else:
                            inbox_msgs.append(str(m))
                elif v is not None:
                    inbox_msgs.append(str(v))
        # Uppercase aliases for prompt placeholders
        vars: Dict[str, Any] = {
            "date": date,
            "date_iso": date_iso,
            "assets": assets or [],
            "inbox": inbox_map,
            "inbox_messages": inbox_msgs,
            "downstream": downstream_ids or [],
            "paths": paths,
        }
        vars.update({
            "DATE": date_iso,
            "ASSETS": vars["assets"],
            "INBOX": vars["inbox"],
            "INBOX_MESSAGES": vars["inbox_messages"],
            "DOWNSTREAM": vars["downstream"],
            "PATHS": paths,
        })
        return vars

    def run(self, date, assets=None, inbox=None, downstream_ids=None):
        vars = self._rag_vars(date, assets, inbox, downstream_ids)
        tpl = self._prompt_text()
        prompt = (
            tpl
            .replace("{DATE}", str(vars.get("DATE", "")))
            .replace("{ASSETS}", json.dumps(vars.get("ASSETS", [])))
            .replace("{INBOX_MESSAGES}", json.dumps(vars.get("INBOX_MESSAGES", [])))
        )
        system = self.prompt.get("system", "")
        text = self.llm.generate(system=system, user=prompt)
        payload_json = None
        try:
            if text and text.strip() and text.strip()[0] in "[{":
                payload_json = json.loads(text)
        except Exception:
            pass
        return {
            "id": f"{self.id}:{date}",
            "src": self.id,
            "role": self.role,
            "ts": date,
            "payload_text": text,
            "payload_json": payload_json,
        }
