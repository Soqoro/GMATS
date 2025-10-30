# gmats/attack/renderer.py
from __future__ import annotations
import hashlib, random, json, os, re
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from gmats.llm import provider as LLM_PROVIDER

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

_PROMPTS_CACHE: Dict[str, Any] = {}

def _best_prompts_path(cfg_path: Optional[str]) -> Path:
    envp = os.getenv("GMATS_PROMPTS_FILE")
    if envp and Path(envp).exists():
        return Path(envp)
    if cfg_path:
        p2 = Path(cfg_path).resolve().parent / "prompts.yaml"
        if p2.exists():
            return p2
    try:
        repo_root = Path(__file__).resolve().parents[2]
        p3 = repo_root / "configs" / "prompts.yaml"
        if p3.exists():
            return p3
    except Exception:
        pass
    return Path("configs/prompts.yaml")

def _load_prompts(prompts_path: Optional[str] = None, cfg_path: Optional[str] = None) -> Dict[str, Any]:
    pp = Path(prompts_path) if prompts_path else _best_prompts_path(cfg_path)
    key = str(pp.resolve())
    if key in _PROMPTS_CACHE:
        return _PROMPTS_CACHE[key]
    try:
        with pp.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        data = {}
    _PROMPTS_CACHE[key] = data
    return data

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
) -> Optional[str]:
    prompts = _load_prompts(prompts_path, cfg_path)
    spec = (prompts or {}).get("attack_post") or {}
    system = (spec.get("system") or "").strip()
    template = (spec.get("template") or "").strip()
    if not (system and template):
        return None

    user = (
        template
        .replace("{DATE}", str(date))
        .replace("{TICKER}", str(asset))
        .replace("{LABEL}", str(label))
        .replace("{ORDERS_WINDOW}", orders_window_str)
    )

    try:
        text = LLM_PROVIDER.generate(system=system, user=user)
        if isinstance(text, dict):
            text = text.get("text", json.dumps(text, ensure_ascii=False))
        text = _one_line(str(text))
        text = _clamp_words(text, 40)
        if not text or not text.strip():
            return None
        return text
    except Exception:
        return None

def render_attack_post(
    *,
    date: str,
    asset: str,
    label: str,
    seed_key: str,
    cfg_path: Optional[str] = None,
    prompts_path: Optional[str] = None,
    orders_window: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    # Try LLM first
    ow_str = _orders_window_to_str(orders_window)
    text = _render_with_llm(date, asset, label, ow_str, cfg_path=cfg_path, prompts_path=prompts_path)

    # Deterministic lexical fallback
    if not text or not str(text).strip():
        rng = random.Random(_seed_from(f"{seed_key}|{date}|{asset}|{label}"))
        opener = rng.choice(LEX["openers"])
        bucket = "neg" if "neg" in label.lower() else ("pos" if "pos" in label.lower() else "neu")
        phrase = rng.choice(LEX[bucket])
        text = f"{opener}: ${asset} {phrase} into close. #{label.replace(' ', '_')}"

    text = str(text).strip()

    return {
        "id": _make_id(date, asset, label, seed_key, text),
        "date": date,
        "symbol": asset,
        "text": text,
        "source": "synthetic://attack/rl",
        "label": label,
    }
