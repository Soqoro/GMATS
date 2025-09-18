from __future__ import annotations
"""
HFLLM: Hugging Face backend (dtype-compatible)
==============================================
Uses `dtype=` (new API). Falls back to `torch_dtype=` for older Transformers.
"""

from typing import Any
import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from .base import Judge


def _resolve_dtype(name: str) -> Any:
    """Map YAML/CLI string -> torch dtype or 'auto'."""
    n = (name or "auto").lower()
    if n in ("auto",):
        return "auto"
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float16", "fp16", "half"):
        return torch.float16
    if n in ("float32", "fp32"):
        return torch.float32
    return "auto"


class HFLLM(Judge):
    """Generic HuggingFace CausalLM wrapper implementing the Judge protocol."""

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        top_p: float = 0.9,
        dtype: str = "auto",
        load_8bit: bool = False,
        load_4bit: bool = False,
    ):
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Build kwargs with quantization + dtype
        kwargs: dict[str, Any] = {}
        if load_8bit:
            kwargs["load_in_8bit"] = True
        if load_4bit:
            kwargs["load_in_4bit"] = True

        resolved = _resolve_dtype(dtype)

        # Prefer new API: dtype=...
        try:
            kwargs["dtype"] = resolved
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", **kwargs
            )
        except TypeError:
            # Older transformers: fall back to torch_dtype=
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = resolved
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", **kwargs
            )

        self.has_chat = hasattr(self.tokenizer, "apply_chat_template")

        # Ensure we have a pad token (some chat models lack one)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---- prompt helpers ----
    def _format(self, system: str, user: str):
        """Build a chat or instruction-formatted prompt."""
        if self.has_chat:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"{system}\n\n### Instruction:\n{user}\n\n### Response:\n"
        ids = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in ids.items()}

    def _gen(self, ids):
        """Run generation with configured decoding parameters."""
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id
        return self.model.generate(
            **ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    def _decode_new(self, ids, out):
        """Decode only the newly generated tokens."""
        gen = out[0][ids["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ---- Judge methods ----
    def summarize(self, prompt: str) -> str:
        ids = self._format(
            "You are a concise financial analyst. Summarize in two sentences.",
            prompt,
        )
        with torch.no_grad():
            out = self._gen(ids)
        return self._decode_new(ids, out)

    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str:
        user = (
            f"{criterion}\n\nCONTEXT:\n{context}\n\nA:\n{a}\n\nB:\n{b}\n\n"
            "Answer with A or B."
        )
        ids = self._format("You are a fair debate judge. Reply 'A' or 'B' only.", user)
        with torch.no_grad():
            out = self._gen(ids)
        t = self._decode_new(ids, out).upper()
        if "A" in t[:3] and "B" not in t[:3]:
            return "A"
        if "B" in t[:3] and "A" not in t[:3]:
            return "B"
        return "A"  # conservative fallback
