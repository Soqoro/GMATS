from __future__ import annotations
"""
LlamaCppLLM
===========
Local GGUF-backed LLM using llama-cpp-python. Point `model_path` to your .gguf.
"""

from ..core.interfaces import LLM
from llama_cpp import Llama

class LlamaCppLLM(LLM):
    """llama.cpp-backed implementation of the LLM Protocol."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

    def summarize(self, prompt: str) -> str:
        """Summarize evidence using a short completion."""
        out = self.llm(prompt, max_tokens=96, temperature=0.2, top_p=0.9)
        return out["choices"][0]["text"].strip()

    def choose(self, context: str, a: str, b: str, criterion: str = "Which is better?") -> str:
        """Return 'A' or 'B' based on a direct prompt."""
        q = f"{criterion}\n\nCONTEXT:\n{context}\n\nA:\n{a}\n\nB:\n{b}\n\nAnswer A or B:"
        out = self.llm(q, max_tokens=4, temperature=0.0, top_p=1.0)
        t = out["choices"][0]["text"].strip().upper()
        return "A" if t.startswith("A") else "B"
