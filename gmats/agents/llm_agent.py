# gmats/agents/llm_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import os, json, datetime as dt, logging, re
from pathlib import Path
from gmats.llm import provider as LLM_PROVIDER
from gmats.llm.provider import _redact
from gmats.attack.overlay import get_global_overlay  # merge synthetic posts

LOGGER = logging.getLogger(__name__)

# Strict routing: ONLY the analyst may ingest social unless explicitly allowed
ALLOW_NONANALYST_SOCIAL = os.getenv("GMATS_ALLOW_NONANALYST_SOCIAL", "0").lower() in ("1", "true", "yes")

# ---------------------------
# Utilities
# ---------------------------
def _hash_id(prefix: str, sym: str, date: str, text: str) -> str:
    import hashlib
    h = hashlib.sha1(f"{sym}|{date}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{sym}:{date}:{h}"

# ---------------------------
# Sentiment backend (lazy)
# ---------------------------
_SENTIMENT_BACKEND: Dict[str, Any] = {"loaded": False, "kind": "none", "fn": None}

def _load_sentiment_backend() -> Tuple[str, Any]:
    """Try VADER → HF transformers → lexicon fallback. Returns (kind, fn)."""
    if _SENTIMENT_BACKEND["loaded"]:
        return _SENTIMENT_BACKEND["kind"], _SENTIMENT_BACKEND["fn"]

    # 1) VADER (pip install vaderSentiment)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
        analyzer = SentimentIntensityAnalyzer()
        def _vader(text: str) -> float:
            return float(analyzer.polarity_scores(text or "").get("compound", 0.0))  # [-1, 1]
        _SENTIMENT_BACKEND.update(loaded=True, kind="vader", fn=_vader)
        return "vader", _vader
    except Exception:
        pass

    # 2) Hugging Face transformers (pip install transformers torch accelerate)
    try:
        from transformers import pipeline  # type: ignore
        nlp = pipeline("sentiment-analysis")
        def _hf(text: str) -> float:
            if not (text and text.strip()):
                return 0.0
            out = nlp(text[:512])[0]  # truncate very long inputs
            label = str(out.get("label", "NEUTRAL")).upper()
            score = float(out.get("score", 0.0))
            if "POS" in label:
                return score
            if "NEG" in label:
                return -score
            return 0.0
        _SENTIMENT_BACKEND.update(loaded=True, kind="hf", fn=_hf)
        return "hf", _hf
    except Exception:
        pass

    # 3) Tiny lexicon fallback (no deps)
    POS = {
        "beat","beats","beating","surge","surges","soar","soars","bull","bullish","upgrade","upgrades",
        "breakout","strong","outperform","profit","profits","win","wins","winning","record","raise","raises",
    }
    NEG = {
        "miss","misses","missed","fall","falls","plunge","plunges","bear","bearish","downgrade","downgrades",
        "selloff","weak","underperform","loss","losses","lawsuit","fraud","probe","cut","cuts","cuts guidance",
    }
    tok = re.compile(r"[A-Za-z']+")

    def _lex(text: str) -> float:
        words = [w.lower() for w in tok.findall(text or "")]
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in POS)
        neg = sum(1 for w in words if w in NEG)
        raw = pos - neg
        # normalize to [-1, 1]
        return max(-1.0, min(1.0, raw / max(3.0, len(words) ** 0.5)))

    _SENTIMENT_BACKEND.update(loaded=True, kind="lex", fn=_lex)
    return "lex", _lex

def _sentiment_score(text: str) -> float:
    _, fn = _load_sentiment_backend()
    try:
        return float(fn(text or "")) if fn else 0.0
    except Exception:
        return 0.0

# ---------------------------
# LLM client wrappers
# ---------------------------
@dataclass
class LLMClient:
    keyring: Dict[str, str] | None = None
    cfg: Dict[str, Any] | None = None
    def generate(self, system: str, user: str) -> str:
        out = LLM_PROVIDER.generate(system=system, user=user)
        if isinstance(out, dict):
            return out.get("text") if isinstance(out.get("text"), str) else json.dumps(out, ensure_ascii=False)
        return str(out)

# ---------------------------
# Agent
# ---------------------------
@dataclass
class LLMAgent:
    id: str
    role: str                 # analyst | coordinator | controller | executor
    prompt: Dict[str, str]    # {system, template}
    inputs: Optional[List[Dict[str, Any]]] = None
    llm: Optional[LLMClient] = None

    # --- RAG spec parsing: window and selection policy ---
    def _parse_rag_spec(self) -> Tuple[int, str, int, str]:
        spec: Dict[str, Any] = {}
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

    # --- Load posts within [DATE-(D-1), DATE] for assets ---
    def _load_social_window(self, date_str: str, assets: List[str]) -> List[Dict[str, Any]]:
        """
        Loads historical posts from disk and MERGES synthetic attack posts
        from the global AttackOverlay across the same time window.
        """
        # prefer agent-scoped data_cfg, else env, else defaults
        data_cfg = getattr(self, "data_cfg", {}) or {}
        root = os.getenv("GMATS_DATA_ROOT") or data_cfg.get("root") or "./data"
        social_dir = os.getenv("GMATS_SOCIAL_DIR") or data_cfg.get("social_dir") or "social"

        d = dt.date.fromisoformat(date_str)
        win_days, _, _, _ = self._parse_rag_spec()
        start = d - dt.timedelta(days=max(0, int(win_days) - 1))

        out: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        # 1) Real posts from dataset
        for sym in assets or []:
            p = Path(root) / social_dir / f"{sym}.jsonl"
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rdate = str(rec.get("date") or "")
                    try:
                        rd = dt.date.fromisoformat(rdate)
                    except Exception:
                        continue
                    if start <= rd <= d:
                        text = rec.get("tweet") or rec.get("text") or ""
                        rid = _hash_id("social", sym, rdate, text)
                        if rid in seen_ids:
                            continue
                        seen_ids.add(rid)
                        out.append({
                            "id": rid,
                            "symbol": sym,
                            "date": rdate,
                            "text": text,
                        })

        # 2) Synthetic posts from RL overlay (if any) across the window
        ov = None
        try:
            ov = get_global_overlay()
        except Exception:
            pass

        if ov is not None:
            num_days = (d - start).days + 1
            for sym in assets or []:
                for i in range(num_days):
                    cur = (start + dt.timedelta(days=i)).isoformat()
                    extras = ov.get_for(date=cur, asset=sym) or []
                    for rec in extras:
                        rdate = str(rec.get("date") or cur)
                        text = rec.get("tweet") or rec.get("text") or ""
                        # >>> changed: preserve overlay-provided id if present; else fall back to hashed synthetic id
                        rid = str(rec.get("id")) if rec.get("id") else _hash_id("synthetic", sym, rdate, text)
                        if rid in seen_ids:
                            continue
                        seen_ids.add(rid)
                        out.append({
                            "id": rid,
                            "symbol": sym,
                            "date": rdate,
                            "text": text,
                            "source": rec.get("source") or "synthetic://attack",
                            "label": rec.get("label"),  # may be None
                        })

        return out

    # --- Rankers: recency, length, sentiment(+/-/abs) ---
    def _rank_social(
        self,
        posts: List[Dict[str, Any]],
        by: str,
        top_k: int,
        mode: str,
        assets: List[str],
    ) -> tuple[list[dict], list[dict]]:
        by = (by or "recency").lower()
        ranked_all: List[Dict[str, Any]] = []

        if by in ("sentiment", "sentiment_desc", "sent", "sent_desc"):
            scored = []
            for p in posts:
                s = _sentiment_score(p.get("text", ""))
                q = dict(p); q["sentiment"] = s
                scored.append(q)
            ranked_all = sorted(scored, key=lambda r: r.get("sentiment", 0.0), reverse=True)

        elif by in ("sentiment_neg", "sent_neg", "sentiment_asc"):
            scored = []
            for p in posts:
                s = _sentiment_score(p.get("text", ""))
                q = dict(p); q["sentiment"] = s
                scored.append(q)
            ranked_all = sorted(scored, key=lambda r: r.get("sentiment", 0.0))

        elif by in ("sentiment_abs", "sent_abs", "sentiment_extreme"):
            scored = []
            for p in posts:
                s = _sentiment_score(p.get("text", ""))
                q = dict(p); q["sentiment"] = s
                scored.append(q)
            ranked_all = sorted(scored, key=lambda r: abs(r.get("sentiment", 0.0)), reverse=True)

        elif by == "length_desc":
            ranked_all = sorted(posts, key=lambda r: len(r.get("text", "")), reverse=True)

        else:
            # default: recency
            ranked_all = sorted(posts, key=lambda r: (r["date"]), reverse=True)

        if mode == "per_asset":
            selected: List[Dict[str, Any]] = []
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
        """Kept for compatibility; respects strict routing like run()."""
        date_str = str(date)
        assets = assets or []
        inbox = inbox or []
        win_days, by, top_k, mode = self._parse_rag_spec()
        # Strict routing gate here too
        if (self.role == "analyst") or (ALLOW_NONANALYST_SOCIAL and self._social_input_enabled()):
            window_posts = self._load_social_window(date_str, assets)
            selected, ranked_all = self._rank_social(window_posts, by, int(top_k), mode, assets)
        else:
            selected, ranked_all = [], []
        return {
            "DATE": date_str,
            "ASSETS": assets,
            "INBOX_MESSAGES": inbox,
            "SOCIAL_POSTS": selected,         # Top-k actually consumed (analyst only)
            "_RAG_WINDOW_ALL": ranked_all,    # For logging only
            "_RAG_SPEC": {"days": win_days, "by": by, "top_k": top_k, "mode": mode},
        }

    # Helper: does this agent declare a SOCIAL input?
    def _social_input_enabled(self) -> bool:
        for spec in (self.inputs or []):
            try:
                if str(spec.get("source", "")).lower() == "social":
                    return True
            except Exception:
                continue
        return False

    def run(self, date, assets=None, inbox=None, downstream_ids=None):
        date_str = str(getattr(date, "isoformat", lambda: date)())
        assets = [str(a).upper() for a in (assets or [])]
        inbox_msgs = inbox or []

        # STRICT: only analyst may ingest social unless explicitly allowed
        social_enabled = (
            (self.role == "analyst") or
            (ALLOW_NONANALYST_SOCIAL and self._social_input_enabled())
        )

        # Build prompt vars
        vars: Dict[str, Any] = {
            "DATE": date_str,
            "ASSETS": assets,
            "INBOX_MESSAGES": inbox_msgs,
        }

        # Load and rank SOCIAL only if enabled (analyst by default)
        consumed: List[Dict[str, Any]] = []
        ranked_all: List[Dict[str, Any]] = []
        if social_enabled:
            try:
                D, by, top_k, mode = self._parse_rag_spec()
                posts = self._load_social_window(date_str, assets)
                consumed, ranked_all = self._rank_social(posts, by=by, top_k=top_k, mode=mode, assets=assets)
                vars["SOCIAL_POSTS"] = consumed
                vars["_RAG_SPEC"] = {"days": D, "by": by, "top_k": top_k, "mode": mode}
            except Exception:
                LOGGER.debug("Failed SOCIAL load/rank for %s on %s", self.id, date_str)
        else:
            # Ensure templates with {SOCIAL_POSTS} don’t break
            vars["SOCIAL_POSTS"] = []

        # Ingestion log: ANALYST ONLY
        if os.getenv("GMATS_LOG_AGENTS") == "1" and social_enabled and self.role == "analyst":
            try:
                LLM_PROVIDER._write_jsonl({
                    "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                    "kind": "ingestion",
                    "agent_id": self.id, "role": self.role,
                    "date": str(date_str), "symbols": assets,
                    "k": len(consumed),
                    "spec": vars.get("_RAG_SPEC"),
                    "ranked_ids": [r.get("id") for r in ranked_all if r.get("id")],
                    "consumed_ids": [r.get("id") for r in consumed if r.get("id")],
                    "ranked_meta": [
                        {"id": r.get("id"), "sentiment": r.get("sentiment")}
                        for r in ranked_all if r.get("id") is not None and "sentiment" in r
                    ],
                })
            except Exception:
                LOGGER.debug("Failed to write ingestion log for %s", self.id)

        # Render prompt from template (SOCIAL omitted for non-analyst)
        tpl = self.prompt.get("template", "") or ""
        prompt = (
            tpl
            .replace("{DATE}", vars["DATE"])
            .replace("{ASSETS}", json.dumps(vars["ASSETS"]))
            .replace("{INBOX_MESSAGES}", json.dumps(vars["INBOX_MESSAGES"]))
            .replace("{SOCIAL_POSTS}", json.dumps(vars.get("SOCIAL_POSTS", [])))
        )
        system = self.prompt.get("system", "")
        text = self.llm.generate(system=system, user=prompt) if self.llm else "[]"

        payload_json = None
        try:
            s = text.strip() if text else ""
            if s and s[0] in "[{":
                payload_json = json.loads(s)
        except Exception:
            LOGGER.debug("Non-JSON output for %s on %s", self.id, date_str)

        if os.getenv("GMATS_LOG_AGENTS") == "1":
            try:
                LLM_PROVIDER._write_jsonl({
                    "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                    "kind": "agent_prompt",
                    "agent_id": self.id, "role": self.role,
                    "date": str(date_str), "symbols": assets,
                    "vars": {
                        "DATE": vars["DATE"],
                        "ASSETS": vars["ASSETS"],
                        "INBOX_MESSAGES": vars["INBOX_MESSAGES"],
                        # Only include SOCIAL_POSTS when enabled (analyst by default)
                        **({"SOCIAL_POSTS": vars["SOCIAL_POSTS"]} if social_enabled else {}),
                    },
                    "system": system,
                    "prompt": prompt,
                })
            except Exception:
                LOGGER.debug("Failed to buffer agent_prompt for %s", self.id)

            try:
                LLM_PROVIDER._write_jsonl({
                    "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
                    "kind": "agent_out",
                    "agent_id": self.id, "role": self.role,
                    "date": str(date_str), "symbols": assets,
                    "payload_text": _redact(text),
                    "payload_json": payload_json,
                })
            except Exception:
                LOGGER.debug("Failed to buffer agent_out for %s", self.id)

        return {
            "id": f"{self.id}:{date_str}",
            "src": self.id,
            "role": self.role,
            "ts": date_str,
            "payload_text": text,
            "payload_json": payload_json,
        }
