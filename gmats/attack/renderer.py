# gmats/attack/renderer.py
from __future__ import annotations
import hashlib, json, random
from typing import Dict, Any, List

LEX = {
    "openers": ["FWIW", "Heads up", "Quick take", "Note"],
    "neg": ["slip", "fade", "pull back", "cool off"],
    "pos": ["pop", "bounce", "recover", "break out"],
    "neu": ["consolidate", "range trade", "chop"],
}

def _seed_from(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return int(h, 16)

def render_attack_post(*, date: str, asset: str, label: str, seed_key: str) -> dict:
    rng = random.Random(_seed_from(f"{seed_key}|{date}|{asset}|{label}"))
    opener = rng.choice(LEX["openers"])
    if "negative" in label: verb = rng.choice(LEX["neg"])
    elif "positive" in label: verb = rng.choice(LEX["pos"])
    else: verb = rng.choice(LEX["neu"])
    text = f"{opener}: ${asset} may {verb} near term. #{label.replace(' ','_')}"
    return {
        "date": date,
        "tweet": text,
        "source": "synthetic://attack/rl",
        "label": label,
    }
