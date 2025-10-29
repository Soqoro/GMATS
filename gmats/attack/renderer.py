# gmats/attack/renderer.py
from __future__ import annotations
import hashlib, random
from typing import Dict

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

def render_attack_post(*, date: str, asset: str, label: str, seed_key: str) -> Dict:
    rng = random.Random(_seed_from(f"{seed_key}|{date}|{asset}|{label}"))
    opener = rng.choice(LEX["openers"])
    bucket = "neg" if "negative" in label else ("pos" if "positive" in label else "neu")
    phrase = rng.choice(LEX[bucket])
    text = f"{opener}: ${asset} {phrase} into close. #{label.replace(' ','_')}"
    return {
        "id": _make_id(date, asset, label, seed_key, text),
        "date": date,
        "symbol": asset,
        "text": text,  # <-- use 'text' (matches your logs), not 'tweet'
        "source": "synthetic://attack/rl",
        "label": label,
    }
