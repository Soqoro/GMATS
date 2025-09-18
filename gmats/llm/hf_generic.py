from __future__ import annotations
"""
HFLLM: Hugging Face backend (GPU/CPU-capable) implementing the Judge protocol.

Key points:
- Uses chat templates when available (tokenizer.apply_chat_template).
- Falls back to instruction-style plain prompts otherwise.
- Decodes only NEW tokens (token-level slice), not by string subtraction.
- Graceful if torch isn't available (dtype='auto' only).
"""

from typing import Any, Optional, Dict, Tuple
import warnings

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
try:
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:
    BitsAndBytesConfig = None  # type: ignore
from .base import Judge


def _resolve_dtype(name: str) -> Any:
    n = (name or "auto").lower()
    if n in ("auto",):
        return "auto"
    if torch is None:
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
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.2,
        top_p: float = 0.9,
        dtype: str = "auto",
        load_8bit: bool = False,
        load_4bit: bool = False,
        judge_system: Optional[str] = None,
        summarize_system: Optional[str] = None,
        judge_user_suffix: Optional[str] = None,
        device_map: Optional[str] = "auto",
    ):
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        # Defaults preserve earlier behavior if not provided via prompts.yaml
        self.system_judge: str = (judge_system or "You are a fair debate judge. Reply 'A' or 'B' only.").strip()
        self.system_summarize: str = (summarize_system or "You are a concise financial analyst. Summarize in two sentences.").strip()
        self.judge_user_suffix: str = (judge_user_suffix or "Answer with A or B.").strip()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_val = _resolve_dtype(dtype)
        model_kwargs: Dict[str, Any] = {"device_map": device_map}

        # Prefer BitsAndBytesConfig when available
        if BitsAndBytesConfig is not None and (load_4bit or load_8bit):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=bool(load_4bit),
                load_in_8bit=bool(load_8bit) and not load_4bit,
                bnb_4bit_compute_dtype=(torch.bfloat16 if (torch is not None) else None),
            )
        else:
            if load_8bit:
                model_kwargs["load_in_8bit"] = True
            if load_4bit:
                model_kwargs["load_in_4bit"] = True

        if dtype_val != "auto":
            model_kwargs["torch_dtype"] = dtype_val

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        # eval mode
        try:
            self.model.eval()
        except Exception:
            pass

        # Use chat template only if present and non-empty
        self.has_chat = bool(getattr(self.tokenizer, "chat_template", None)) and hasattr(self.tokenizer, "apply_chat_template")

    # ---------- prompt formatting ----------

    def _build_inputs(self, system: str, user: str) -> Tuple[Dict[str, Any], int]:
        """
        Returns tokenized inputs (moved to model device if possible) and
        the input length (for new-token slicing).
        """
        if self.has_chat:
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": user})
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Fallback if chat template exists but fails
                sys_part = (system + "\n\n") if system else ""
                text = f"{sys_part}### Instruction:\n{user}\n\n### Response:\n"
        else:
            # Simple instruction style fallback
            sys_part = (system + "\n\n") if system else ""
            text = f"{sys_part}### Instruction:\n{user}\n\n### Response:\n"

        ids = self.tokenizer(text, return_tensors="pt")
        # Move to device if determinable
        try:
            device = getattr(self.model, "device", None)
            if device is None:
                first_param = next(self.model.parameters(), None)
                device = getattr(first_param, "device", None)
            if device is not None:
                ids = {k: v.to(device) for k, v in ids.items()}
        except Exception:
            pass

        input_len = int(ids["input_ids"].shape[1])
        return ids, input_len

    def _generate(self, ids: Dict[str, Any]) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if torch is not None:
                with torch.no_grad():
                    return self.model.generate(
                        **ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=self.temperature > 0,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                return self.model.generate(
                    **ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

    def _decode_new(self, ids: Dict[str, Any], out: Any, input_len: int) -> str:
        # slice newly generated tokens
        gen_ids = out[0][input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # ---------- Judge protocol ----------

    def summarize(self, prompt: str) -> str:
        ids, input_len = self._build_inputs(self.system_summarize, prompt)
        out = self._generate(ids)
        return self._decode_new(ids, out, input_len)

    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str:
        user = (
            f"{criterion}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"A:\n{a}\n\n"
            f"B:\n{b}\n\n"
            f"{self.judge_user_suffix}"
        )
        ids, input_len = self._build_inputs(self.system_judge, user)
        out = self._generate(ids)
        t = self._decode_new(ids, out, input_len).strip().upper()

        # Robust parse: scan head then fallback
        head = t[:5]
        if "A" in head and "B" not in head:
            return "A"
        if "B" in head and "A" not in head:
            return "B"
        for ch in t:
            if ch == "A":
                return "A"
            if ch == "B":
                return "B"
        return "A"
