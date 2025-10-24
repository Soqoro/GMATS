# gmats/agents/llm_agent.py
from __future__ import annotations
import os, json, logging, datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from gmats.llm import provider as LLM_PROVIDER
from gmats.llm.provider import _redact

LOGGER = logging.getLogger("gmats")

def _hash_id(prefix: str, sym: str, date: str, text: str) -> str:
    import hashlib
    h = hashlib.sha1(f"{sym}|{date}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{sym}:{date}:{h}"

@dataclass
class LLMClient:
    keyring: Dict[str, str] | None = None
    cfg: Dict[str, Any] | None = None
    def generate(self, system: str, user: str) -> str:
        out = LLM_PROVIDER.generate(system=system, user=user)
        if isinstance(out, dict):
            return out.get("text") if isinstance(out.get("text"), str) else json.dumps(out)
        return str(out)

@dataclass
class LLMAgent:
    id: str
    role: str                 # analyst | coordinator | controller | executor
    prompt: Dict[str, str]    # {system, template}
    inputs: Optional[List[Dict[str, Any]]] = None
    llm: Optional[LLMClient] = None

    # --- RAG spec parsing: window and selection policy ---
    def _parse_rag_spec(self):
        spec = {}
        for inp in (self.inputs or []):
            if (inp or {}).get("source") == "social":
                spec = inp or {}
                break
        win_days = int(((spec.get("window") or {}).get("days") or os.getenv("GMATS_RAG_DAYS", 1)))
        sel = spec.get("select") or {}
        by = (sel.get("by") or os.getenv("GMATS_RAG_BY", "recency")).lower()
        top_k = int(sel.get("top_k") or os.getenv("GMATS_RAG_TOPK", 50))
        mode = (sel.get("mode") or os.getenv("GMATS_RAG_MODE", "global")).lower()  # global|per_asset
        return win_days, by, top_k, mode

    # --- Load posts within [DATE-(D-1), DATE] for all assets ---
    def _load_social_window(self, date_str: str, assets: List[str]) -> List[Dict[str, Any]]:
        root = os.getenv("GMATS_DATA_ROOT", "./data")
        social_dir = os.getenv("GMATS_SOCIAL_DIR", "social")
        d = dt.date.fromisoformat(date_str)
        win_days, _, _, _ = self._parse_rag_spec()
        start = d - dt.timedelta(days=max(0, int(win_days) - 1))

        out: List[Dict[str, Any]] = []
        for sym in assets or []:
            p = Path(root) / social_dir / f"{sym}.jsonl"
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    rdate = str(rec.get("date") or "")
                    try:
                        rd = dt.date.fromisoformat(rdate)
                    except Exception:
                        continue
                    if start <= rd <= d:
                        text = rec.get("tweet") or rec.get("text") or ""
                        out.append({
                            "id": _hash_id("social", sym, rdate, text),
                            "symbol": sym,
                            "date": rdate,
                            "text": text,
                        })
        return out

    # --- Simple rankers; add more as needed ---
    def _rank_social(self, posts: List[Dict[str, Any]], by: str, top_k: int, mode: str, assets: List[str]) -> tuple[list[dict], list[dict]]:
        ranked_all: List[Dict[str, Any]] = []
        if by == "recency":
            ranked_all = sorted(posts, key=lambda r: (r["date"]), reverse=True)
        elif by == "length_desc":
            ranked_all = sorted(posts, key=lambda r: len(r.get("text","")), reverse=True)
        else:
            ranked_all = sorted(posts, key=lambda r: (r["date"]), reverse=True)  # default

        if mode == "per_asset":
            selected = []
            per = max(1, top_k // max(1, len(assets or [])))
            for a in assets or []:
                aa = [p for p in ranked_all if p["symbol"] == a][:per]
                selected.extend(aa)
            selected = selected[:top_k]
        else:
            selected = ranked_all[:top_k]
        return selected, ranked_all

    def _prompt_text(self) -> str:
        return (self.prompt.get("template") or "") if isinstance(self.prompt, dict) else ""

    def _rag_vars(self, date, assets, inbox) -> Dict[str, Any]:
        date_str = str(date)
        assets = assets or []
        inbox = inbox or []
        # RAG retrieval
        win_days, by, top_k, mode = self._parse_rag_spec()
        window_posts = self._load_social_window(date_str, assets)
        selected, ranked_all = self._rank_social(window_posts, by, int(top_k), mode, assets)
        return {
            "DATE": date_str,
            "ASSETS": assets,
            "INBOX_MESSAGES": inbox,
            "SOCIAL_POSTS": selected,         # Top-k actually consumed
            "_RAG_WINDOW_ALL": ranked_all,    # For logging only
            "_RAG_SPEC": {"days": win_days, "by": by, "top_k": top_k, "mode": mode},
        }

    def run(self, date, assets=None, inbox=None, downstream_ids=None):
        vars = self._rag_vars(date, assets, inbox)
        tpl = self._prompt_text()
        prompt = (
            tpl
            .replace("{DATE}", str(vars.get("DATE", "")))
            .replace("{ASSETS}", json.dumps(vars.get("ASSETS", [])))
            .replace("{INBOX_MESSAGES}", json.dumps(vars.get("INBOX_MESSAGES", [])))
            .replace("{SOCIAL_POSTS}", json.dumps(vars.get("SOCIAL_POSTS", []), ensure_ascii=False))
        )
        system = self.prompt.get("system", "") if isinstance(self.prompt, dict) else ""

        # Ingestion log for IR@k
        try:
            LLM_PROVIDER._write_jsonl({
                "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "kind": "ingestion",
                "agent_id": self.id, "role": self.role,
                "date": str(date), "symbols": [s.upper() for s in (assets or [])],
                "k": len(vars["SOCIAL_POSTS"]),
                "spec": vars.get("_RAG_SPEC"),
                "ranked_ids": [r.get("id") for r in (vars.get("_RAG_WINDOW_ALL") or []) if r.get("id")],
                "consumed_ids": [r.get("id") for r in (vars.get("SOCIAL_POSTS") or []) if r.get("id")],
            })
        except Exception:
            LOGGER.debug("Failed to write ingestion log")

        # Snapshot of prompt inputs
        if os.getenv("GMATS_LOG_AGENTS") == "1":
            record = {
                "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                "kind": "agent_prompt",
                "agent_id": self.id,
                "role": self.role,
                "date": str(date),
                "symbols": [s.upper() for s in (assets or [])],
                "vars": {k: vars.get(k) for k in ("DATE","ASSETS","INBOX_MESSAGES","SOCIAL_POSTS")},
                "system": _redact(system),
                "prompt": _redact(prompt),
            }
            try:
                LLM_PROVIDER._write_jsonl(record)
            except Exception:
                LOGGER.debug("Failed to buffer agent_prompt for %s", self.id)

        # Call LLM
        text = self.llm.generate(system=system, user=prompt) if self.llm else "[]"

        payload_json = None
        try:
            if text and text.strip() and text.strip()[0] in "[{":
                payload_json = json.loads(text)
        except Exception:
            LOGGER.debug("Non-JSON output for %s on %s", self.id, date)

        if os.getenv("GMATS_LOG_AGENTS") == "1":
            try:
                LLM_PROVIDER._write_jsonl({
                    "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                    "kind": "agent_out",
                    "agent_id": self.id, "role": self.role,
                    "date": str(date), "symbols": [s.upper() for s in (assets or [])],
                    "payload_text": _redact(text),
                    "payload_json": payload_json,
                })
            except Exception:
                LOGGER.debug("Failed to buffer agent_out for %s", self.id)

        return {
            "id": f"{self.id}:{date}",
            "src": self.id,
            "role": self.role,
            "ts": date,
            "payload_text": text,
            "payload_json": payload_json,
        }
