# gmats/agents/llm_agent.py
from __future__ import annotations
import os, json, logging, datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gmats.llm import provider as LLM_PROVIDER
from gmats.llm.provider import _redact

LOGGER = logging.getLogger("gmats")

# ---------- light-weight LLM shim ----------
class LLMClient:
    def __init__(self, keyring: Dict[str, str] | None, cfg: Dict[str, Any] | None):
        self.keyring = keyring or {}
        self.cfg = cfg or {}
    def generate(self, system: str, user: str) -> str:
        try:
            out = LLM_PROVIDER.generate(system=system, user=user)
            if isinstance(out, dict):
                # prefer "text" field if present; else dump JSON
                return out.get("text") if isinstance(out.get("text"), str) else json.dumps(out)
            return str(out)
        except Exception:
            LOGGER.exception("LLM generate failed")
            return "{}"

@dataclass
class LLMAgent:
    id: str
    role: str                          # analyst | coordinator | controller | executor
    prompt: Dict[str, str]             # {system, template}
    inputs: Optional[List[Dict[str, Any]]] = None
    llm: Optional[LLMClient] = None

    def _prompt_text(self) -> str:
        return (self.prompt.get("template") or "") if isinstance(self.prompt, dict) else ""

    def _load_social_posts(self, date_str: str, assets: List[str]) -> List[Dict[str, Any]]:
        root = os.getenv("GMATS_DATA_ROOT", "./data")
        social_dir = os.getenv("GMATS_SOCIAL_DIR", "social")
        out: List[Dict[str, Any]] = []
        for sym in assets or []:
            p = Path(root) / social_dir / f"{sym}.jsonl"
            if not p.exists():
                continue
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if str(rec.get("date")) == date_str:
                            out.append({
                                "symbol": sym,
                                "text": rec.get("tweet") or rec.get("text") or "",
                            })
            except Exception:
                LOGGER.debug("Failed reading social for %s on %s", sym, date_str)
        max_items = int(os.getenv("GMATS_SOCIAL_TOPK", "50"))
        return out[:max_items]

    def _rag_vars(self, date, assets, inbox, downstream_ids) -> Dict[str, Any]:
        date_str = str(date)
        assets = assets or []
        inbox = inbox or []
        social = self._load_social_posts(date_str, assets)
        return {
            "DATE": date_str,
            "ASSETS": assets,
            "INBOX_MESSAGES": inbox,
            "SOCIAL_POSTS": social,
            "DOWNSTREAM": downstream_ids or [],
        }

    def run(self, date, assets=None, inbox=None, downstream_ids=None):
        vars = self._rag_vars(date, assets, inbox, downstream_ids)
        tpl = self._prompt_text()
        # Render known placeholders (safe even if not present in template)
        prompt = (
            tpl
            .replace("{DATE}", str(vars.get("DATE", "")))
            .replace("{ASSETS}", json.dumps(vars.get("ASSETS", [])))
            .replace("{INBOX_MESSAGES}", json.dumps(vars.get("INBOX_MESSAGES", [])))
            .replace("{SOCIAL_POSTS}", json.dumps(vars.get("SOCIAL_POSTS", []), ensure_ascii=False))
        )
        system = self.prompt.get("system", "") if isinstance(self.prompt, dict) else ""

        # Log the exact variables and rendered prompt
        if os.getenv("GMATS_LOG_AGENTS") == "1":
            record = {
                "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "kind": "agent_prompt",
                "agent_id": self.id,
                "role": self.role,
                "date": str(date),
                "symbols": [s.upper() for s in (assets or [])],
                "vars": {k: vars.get(k) for k in ("DATE", "ASSETS", "INBOX_MESSAGES", "SOCIAL_POSTS")},
                "system": _redact(system),
                "prompt": _redact(prompt),
                "inbox_preview": json.dumps((vars.get("INBOX_MESSAGES") or [])[:10], ensure_ascii=False),
            }
            try:
                LLM_PROVIDER._write_jsonl(record)
            except Exception:
                LOGGER.exception("Failed to buffer agent_prompt for %s", self.id)

        # Call LLM
        text = self.llm.generate(system=system, user=prompt) if self.llm else "{}"

        payload_json = None
        try:
            if text and text.strip() and text.strip()[0] in "[{":
                payload_json = json.loads(text)
        except Exception:
            LOGGER.debug("Non-JSON output for %s on %s", self.id, date)

        if os.getenv("GMATS_LOG_AGENTS") == "1":
            out_record = {
                "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "kind": "agent_out",
                "agent_id": self.id,
                "role": self.role,
                "date": str(date),
                "symbols": [s.upper() for s in (assets or [])],
                "payload_text": _redact(text),
                "payload_json": payload_json,
            }
            try:
                LLM_PROVIDER._write_jsonl(out_record)
            except Exception:
                LOGGER.exception("Failed to buffer agent_out for %s", self.id)

        return {
            "id": f"{self.id}:{date}",
            "src": self.id,
            "role": self.role,
            "ts": date,
            "payload_text": text,
            "payload_json": payload_json,
        }
