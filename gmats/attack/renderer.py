# gmats/attack/renderer.py
from __future__ import annotations
import hashlib, random, json, os, re, logging, datetime as dt
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from gmats.llm import provider as LLM_PROVIDER
from gmats.llm.provider import _redact  # for safe console output

LOGGER = logging.getLogger(__name__)

LEX = {
    "openers": ["Heads up", "Note", "Watch", "Alert"],
    "neg": ["bearish momentum", "supply overhang", "weak tape", "distribution"],
    "pos": ["bullish momentum", "accumulation", "buyers in control", "follow-through"],
    "neu": ["range-bound", "indecisive", "two-way flows", "mixed signals"],
}

def _seed_from(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)

def _make_id(date: str, asset: str, label: str, seed_key: str, text: str) -> str:
    raw = f"{date}|{asset}|{label}|{seed_key}|{text}"
    return "attk_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

# ---------------------------
# Structured debug logging
# ---------------------------
def _log_struct(level: str, event: str, **fields: Any) -> None:
    """
    Best-effort JSONL log for attack rendering.
    Enabled when GMATS_LOG_ATTACK=1 or GMATS_LOG_AGENTS=1.
    """
    if os.getenv("GMATS_LOG_ATTACK", "0") != "1" and os.getenv("GMATS_LOG_AGENTS", "0") != "1":
        return
    try:
        rec = {
            "ts": dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "kind": "attack_render_debug",
            "level": level,
            "event": event,
            **fields,
        }
        LLM_PROVIDER._write_jsonl(rec)  # type: ignore[attr-defined]
    except Exception:
        pass

def _dbg(event: str, **fields: Any) -> None:
    _log_struct("DEBUG", event, **fields)
    LOGGER.debug("%s | %s", event, {k: v for k, v in fields.items() if k != "prompt_user"})

def _warn(event: str, **fields: Any) -> None:
    _log_struct("WARN", event, **fields)
    LOGGER.warning("%s | %s", event, fields)

# ---------------------------
# prompts.yaml loader (cached)
# ---------------------------
_PROMPTS_CACHE: Dict[str, Any] = {}

def _best_prompts_path(cfg_path: Optional[str]) -> Path:
    envp = os.getenv("GMATS_PROMPTS_FILE")
    if envp and Path(envp).exists():
        _dbg("prompts_path_env", path=envp)
        return Path(envp)
    if cfg_path:
        p2 = Path(cfg_path).resolve().parent / "prompts.yaml"
        if p2.exists():
            _dbg("prompts_path_cfg_sibling", path=str(p2))
            return p2
    try:
        repo_root = Path(__file__).resolve().parents[2]
        p3 = repo_root / "configs" / "prompts.yaml"
        if p3.exists():
            _dbg("prompts_path_repo_configs", path=str(p3))
            return p3
    except Exception:
        pass
    fallback = Path("configs/prompts.yaml")
    _dbg("prompts_path_fallback", path=str(fallback))
    return fallback

def _load_prompts(prompts_path: Optional[str] = None, cfg_path: Optional[str] = None) -> Dict[str, Any]:
    pp = Path(prompts_path) if prompts_path else _best_prompts_path(cfg_path)
    key = str(pp.resolve())
    if key in _PROMPTS_CACHE:
        _dbg("prompts_cache_hit", path=key)
        return _PROMPTS_CACHE[key]
    try:
        with pp.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _dbg("prompts_loaded", path=key, keys=list(data.keys()))
    except Exception as e:
        _warn("prompts_load_failed", path=key, error=str(e))
        data = {}
    _PROMPTS_CACHE[key] = data
    return data

# ---------------------------
# LLM rendering helpers
# ---------------------------
_WS = re.compile(r"\s+")
def _one_line(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    s = _WS.sub(" ", s)
    return s

def _clamp_words(s: str, max_words: int = 40) -> str:
    toks = s.split()
    if len(toks) <= max_words:
        return s
    return " ".join(toks[:max_words])

def _orders_window_to_str(orders_window: Optional[List[Dict[str, Any]]]) -> str:
    if not orders_window:
        return "[]"
    buf = []
    for o in orders_window[:10]:
        try:
            sym = str(o.get("symbol", "")).upper()
            w = float(o.get("weight", 0.0))
            buf.append(f"{sym}:{w:+.2f}")
        except Exception:
            continue
    return "[" + ", ".join(buf) + "]" if buf else "[]"

def _render_with_llm(
    date: str,
    asset: str,
    label: str,
    orders_window_str: str,
    *,
    cfg_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
    llm: Optional[Any] = None,   # accept a provider instance
) -> Optional[str]:
    prompts = _load_prompts(prompts_path, cfg_path)
    spec = (prompts or {}).get("attack_post") or {}
    system = (spec.get("system") or "").strip()
    template = (spec.get("template") or "").strip()

    if not system or not template:
        _warn(
            "llm_skipped_no_prompt",
            reason="missing_system_or_template",
            has_system=bool(system),
            has_template=bool(template),
        )
        # Console print (always)
        print(f"[LLM/attack_post] SKIP: missing prompt parts "
              f"(has_system={bool(system)} has_template={bool(template)})", flush=True)
        return None

    user = (
        template
        .replace("{DATE}", str(date))
        .replace("{TICKER}", str(asset))
        .replace("{LABEL}", str(label))
        .replace("{ORDERS_WINDOW}", orders_window_str)
    )

    prov = "injected" if (llm and hasattr(llm, "generate")) else "module"
    # Console print (always)
    print(f"[LLM/attack_post] provider={prov} system_len={len(system)} user_len={len(user)} "
          f"asset={asset} label={label}", flush=True)

    try:
        # Prefer injected provider (OpenAIProvider/Ollama/HF). Fall back to module function.
        raw = llm.generate(system=system, user=user) if (llm and hasattr(llm, "generate")) \
              else LLM_PROVIDER.generate(system=system, user=user)

        # Normalize to string
        if isinstance(raw, dict):
            text = raw.get("text", json.dumps(raw, ensure_ascii=False))
            kind = "dict"
        else:
            text = str(raw)
            kind = "str"

        s = _clamp_words(_one_line(text), 40)
        accepted = bool(s.strip())

        # ---- ALWAYS PRINT RAW AND NORMALIZED ----
        print(f"[LLM/attack_post] raw_kind={kind} asset={asset} label={label} raw='{_redact(text)}'", flush=True)
        print(f"[LLM/attack_post] normalized='{s}' accepted={accepted}", flush=True)

        _dbg(
            "llm_result",
            raw_kind=kind,
            raw_preview=(text[:160] if isinstance(text, str) else str(text)[:160]),
            normalized_preview=s[:160],
            accepted=accepted,
        )

        return s if accepted else None
    except Exception as e:
        _warn("llm_exception", error=str(e))
        # ALWAYS print exceptions too
        print(f"[LLM/attack_post] EXCEPTION: {e}", flush=True)
        return None

# ---------------------------
# Public API
# ---------------------------
def render_attack_post(
    *,
    date: str,
    asset: str,
    label: str,
    seed_key: str,
    cfg_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
    orders_window: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Config-driven renderer for synthetic attack posts using prompts.yaml:attack_post.
    Uses the provided LLM instance when available; otherwise falls back to a deterministic template.
    Always returns a non-empty 'text'.
    """
    # Deterministic fallback composed first (so ID stays stable if LLM fails)
    rng = random.Random(_seed_from(f"{seed_key}|{date}|{asset}|{label}"))
    opener = rng.choice(LEX["openers"])
    lb = label.lower()
    bucket = "neg" if ("neg" in lb or "bear" in lb) else ("pos" if ("pos" in lb or "bull" in lb) else "neu")
    phrase = rng.choice(LEX[bucket])
    fallback = f"{opener}: ${asset} {phrase} into close. #{label.replace(' ', '_')}"

    # Try LLM
    ow_str = _orders_window_to_str(orders_window)
    llm_text = _render_with_llm(
        date, asset, label, ow_str,
        cfg_path=cfg_path, prompts_path=prompts_path, llm=llm
    )

    # Accept any non-empty LLM text; otherwise use fallback
    final_text = (llm_text or "").strip() or fallback

    _dbg(
        "render_attack_post_done",
        used_llm=bool(llm_text),
        final_preview=final_text[:160],
        asset=asset,
        label=label,
    )

    # ALWAYS print the final line as well
    print(f"[LLM/attack_post] FINAL asset={asset} label={label} text='{final_text}'", flush=True)

    return {
        "id": _make_id(date, asset, label, seed_key, final_text),
        "date": date,
        "symbol": asset,
        "text": final_text,   # guaranteed non-empty
        "source": "synthetic://attack/rl",
        "label": label,
    }
