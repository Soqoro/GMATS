# gmats/tools/sentiment.py
from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional heavy deps
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except Exception:  # pragma: no cover
        BitsAndBytesConfig = None  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore


def _resolve_dtype(name: str) -> Any:
    n = (name or "auto").lower()
    if n == "auto" or torch is None:
        return "auto"
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float16", "fp16", "half"):
        return torch.float16
    if n in ("float32", "fp32"):
        return torch.float32
    return "auto"


@dataclass
class SentimentConfig:
    # HF model knobs
    llm: str = "hf"                       # "hf" expected
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_new_tokens: int = 48
    temperature: float = 0.0
    top_p: float = 0.9
    dtype: str = "auto"
    load_8bit: bool = False
    load_4bit: bool = False
    device_map: str = "auto"

    # Prompt knobs
    system: Optional[str] = None          # system prefix
    user_template: Optional[str] = None   # can use {symbol} and {text}

    # Scoring knobs
    neutral_band: float = 0.0             # |score| < band -> 0
    cache_path: Optional[Path] = None     # JSONL cache


class SentimentScorer:
    """
    LLM-backed sentiment scorer producing a numeric score in [-1, 1].

    If HF/torch aren't available, falls back to a tiny lexicon heuristic.
    """

    def __init__(self, cfg: SentimentConfig):
        self.cfg = cfg
        self._tok = None
        self._model = None
        self._has_chat = False
        self._cache: Dict[str, float] = {}
        self._cache_path = cfg.cache_path
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        # Build model if deps are present
        if AutoTokenizer and AutoModelForCausalLM:
            try:
                self._tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
                if self._tok.pad_token is None:
                    # use EOS as pad
                    self._tok.pad_token = self._tok.eos_token

                dtype_val = _resolve_dtype(cfg.dtype)
                model_kwargs: Dict[str, Any] = {"device_map": cfg.device_map}

                if BitsAndBytesConfig is not None and (cfg.load_4bit or cfg.load_8bit):
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=bool(cfg.load_4bit),
                        load_in_8bit=bool(cfg.load_8bit) and not cfg.load_4bit,
                        bnb_4bit_compute_dtype=(torch.bfloat16 if torch is not None else None),
                    )
                else:
                    if cfg.load_8bit:
                        model_kwargs["load_in_8bit"] = True
                    if cfg.load_4bit:
                        model_kwargs["load_in_4bit"] = True

                if dtype_val != "auto":
                    model_kwargs["torch_dtype"] = dtype_val

                self._model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
                try:
                    self._model.eval()
                except Exception:
                    pass

                # chat template check must be robust
                self._has_chat = bool(getattr(self._tok, "chat_template", None)) and hasattr(self._tok, "apply_chat_template")
            except Exception:
                # fall back to heuristic if model load fails
                self._tok = None
                self._model = None
                self._has_chat = False

    # ---------------- Cache ----------------
    def _key(self, symbol: Optional[str], text: str) -> str:
        h = hashlib.sha1()
        h.update((symbol or "").encode("utf-8"))
        h.update(b"\n")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _load_cache(self) -> None:
        try:
            if self._cache_path and self._cache_path.exists():
                with self._cache_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            k = obj.get("key")
                            v = obj.get("score")
                            if isinstance(k, str) and isinstance(v, (int, float)):
                                self._cache[k] = float(v)
                        except Exception:
                            continue
        except Exception:
            pass

    def _cache_put(self, key: str, score: float, symbol: Optional[str], text: str) -> None:
        self._cache[key] = score
        if not self._cache_path:
            return
        try:
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "symbol": symbol, "score": score}) + "\n")
        except Exception:
            pass

    # ---------------- Heuristic fallback ----------------
    @staticmethod
    def _heuristic_score(text: str) -> float:
        text_l = text.lower()
        pos = sum(text_l.count(w) for w in ["beat", "beats", "surge", "surges", "bullish", "upside", "strong", "profit", "gain"])
        neg = sum(text_l.count(w) for w in ["miss", "misses", "fall", "falls", "bearish", "downside", "weak", "loss", "drop"])
        total = pos + neg
        if total == 0:
            return 0.0
        raw = (pos - neg) / total
        # clamp to [-1, 1]
        return max(-1.0, min(1.0, float(raw)))

    # ---------------- LLM plumbing ----------------
    def _build_prompt(self, system: Optional[str], user_template: Optional[str], symbol: Optional[str], text: str) -> Tuple[str, str]:
        sys = (system or "You are a financial sentiment rater. Output a SINGLE number in [-1, 1] reflecting near-term equity return sentiment. No words, just the number.")
        tmpl = user_template or "Text about {symbol}:\n\n{text}\n\nReturn a single number in [-1,1]."
        user = tmpl.format(symbol=(symbol or "the company"), text=text)
        return sys, user

    def _encode(self, system: str, user: str) -> Tuple[Dict[str, Any], int]:
        assert self._tok is not None
        if self._has_chat:
            try:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"{system}\n\n### Instruction:\n{user}\n\n### Response:\n"
        else:
            prompt = f"{system}\n\n### Instruction:\n{user}\n\n### Response:\n"
        ids = self._tok(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            ids = {k: v.to(self._model.device) for k, v in ids.items()}  # type: ignore
        return ids, int(ids["input_ids"].shape[1])

    def _generate(self, ids: Dict[str, Any]) -> Any:
        assert self._tok is not None and self._model is not None
        eos_id = self._tok.eos_token_id
        pad_id = self._tok.pad_token_id or eos_id
        out = self._model.generate(
            **ids,
            max_new_tokens=int(self.cfg.max_new_tokens),
            do_sample=float(self.cfg.temperature) > 0.0,
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        return out

    def _decode_new(self, ids: Dict[str, Any], out: Any) -> str:
        assert self._tok is not None
        gen = out[0][ids["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_score(text: str) -> Optional[float]:
        # find first float-like token; clamp to [-1,1]
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not m:
            return None
        try:
            v = float(m.group(0))
            return max(-1.0, min(1.0, v))
        except Exception:
            return None

    # ---------------- Public API ----------------
    def score(self, text: str, *, symbol: Optional[str] = None) -> float:
        """Return sentiment in [-1, 1]; apply neutral band if configured."""
        text = (text or "").strip()
        if not text:
            return 0.0

        key = self._key(symbol, text)
        if key in self._cache:
            s = self._cache[key]
            if abs(s) < float(self.cfg.neutral_band):
                return 0.0
            return s

        # Heuristic fallback when HF/torch not available
        if self._tok is None or self._model is None:
            s = self._heuristic_score(text)
            if abs(s) < float(self.cfg.neutral_band):
                s = 0.0
            self._cache_put(key, s, symbol, text)
            return s

        # LLM pass
        system, user = self._build_prompt(self.cfg.system, self.cfg.user_template, symbol, text)
        ids, _ = self._encode(system, user)
        with torch.no_grad():  # type: ignore
            out = self._generate(ids)
        completion = self._decode_new(ids, out)
        s = self._extract_score(completion)
        if s is None:
            s = 0.0
        if abs(s) < float(self.cfg.neutral_band):
            s = 0.0
        self._cache_put(key, s, symbol, text)
        return float(s)
