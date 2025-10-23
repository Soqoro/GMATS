"""LLM provider: choose 'mock' (no API) or 'openai' via config."""
from __future__ import annotations
from typing import Any, Dict, Optional
import json, re, os

_CLIENT = None
_CFG: Dict[str, Any] = {}

def init(cfg: Dict[str, Any]):
    global _CLIENT, _CFG
    _CFG = cfg or {}
    provider = str(_CFG.get("provider", "mock")).lower()
    if provider == "openai":
        try:
            # Lazy import to avoid dependency if unused
            import openai  # type: ignore
            _CLIENT = openai
        except Exception as e:
            _CLIENT = "mock"
    else:
        _CLIENT = "mock"

def _extract_json(text: str) -> Dict[str, Any]:
    # grab first {...} block
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}

def generate(system: str, user: str) -> Dict[str, Any]:
    """Return parsed JSON dict from the model output; mock returns heuristic."""
    provider = str((_CFG.get("provider") or "mock")).lower()
    if provider == "openai":
        # Example scaffold; replace with actual API call shape if needed
        model = _CFG.get("model", "gpt-4o-mini")
        temperature = float(_CFG.get("temperature", 0.0))
        # Fallback to mock until wired to your API
        return _extract_json(user)

    # mock: attempt simple extraction/heuristics from user content
    try:
        js = _extract_json(user)
        if js:
            return js
        # Heuristic: if user mentions symbols and BUY/SELL, synthesize orders
        orders = []
        for sym in re.findall(r"\b[A-Z]{2,6}\b", user):
            if re.search(rf"\bBUY\b", user, re.I):
                orders.append({"symbol": sym, "side": 1, "weight": 1.0})
            elif re.search(rf"\bSELL\b", user, re.I):
                orders.append({"symbol": sym, "side": -1, "weight": -1.0})
        return {"orders": orders} if orders else {}
    except Exception:
        return {}

class LLMProvider:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, temperature: float = 0.0, base_url: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)

    def generate(self, system: str, user: str) -> str:
        msg = [{"role": "system", "content": system or ""}, {"role": "user", "content": user}]
        resp = self.client.chat.completions.create(model=self.model, temperature=self.temperature, messages=msg)
        return resp.choices[0].message.content or ""

class OllamaProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.0, host: Optional[str] = None):
        try:
            import ollama  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install ollama") from e
        self.ollama = ollama
        if host:
            os.environ["OLLAMA_HOST"] = host
        self.model = model
        self.temperature = float(temperature)

    def generate(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}" if system else user
        resp = self.ollama.generate(model=self.model, prompt=prompt, options={"temperature": self.temperature})
        return resp.get("response", "")

class HFProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.0, device: Optional[str] = None, dtype: Optional[str] = None, trust_remote_code: bool = False):
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError("pip install transformers accelerate torch") from e
        kwargs: Dict[str, Any] = {"model": model}
        if device:
            kwargs["device_map"] = device
        if dtype:
            from torch import float16, bfloat16
            kwargs["torch_dtype"] = {"float16": float16, "bfloat16": bfloat16}.get(dtype, None)
        kwargs["trust_remote_code"] = trust_remote_code
        self.pipe = pipeline("text-generation", **kwargs)
        self.temperature = float(temperature)

    def generate(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}" if system else user
        out = self.pipe(prompt, do_sample=self.temperature > 0, temperature=max(1e-6, self.temperature), max_new_tokens=512)
        if isinstance(out, list) and out:
            return out[0].get("generated_text", "")
        return str(out)

def make_llm_provider(agent_llm_cfg: Dict[str, Any], keys: Dict[str, str]) -> Optional[LLMProvider]:
    """
    agent_llm_cfg: e.g. {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0, ...}
    keys: from config["llm_keys"]
    """
    if not agent_llm_cfg:
        return None
    prov = (agent_llm_cfg.get("provider") or "").lower()
    model = agent_llm_cfg.get("model") or ""
    temp = agent_llm_cfg.get("temperature", 0.0)

    # OpenAI (official)
    if prov == "openai":
        api_key = agent_llm_cfg.get("api_key") or keys.get("openai") or os.getenv("OPENAI_API_KEY", "")
        base_url = agent_llm_cfg.get("base_url")  # allow Azure/OpenAI-compatible hosts
        if not api_key and not base_url:
            # If using openai-compatible with no key, prefer compat block below
            raise RuntimeError("OpenAI provider requires an API key or base_url.")
        return OpenAIProvider(api_key=api_key, model=model, temperature=temp, base_url=base_url)

    # OpenAI-compatible (local servers like vLLM, llama.cpp server, Ollama /v1)
    if prov in ("openai_compat", "openai-compatible", "compat"):
        base_url = agent_llm_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        api_key = agent_llm_cfg.get("api_key") or keys.get("openai") or os.getenv("OPENAI_API_KEY", "sk-local")
        return OpenAIProvider(api_key=api_key, model=model, temperature=temp, base_url=base_url)

    # Ollama (Python package)
    if prov == "ollama":
        host = agent_llm_cfg.get("host") or os.getenv("OLLAMA_HOST")
        return OllamaProvider(model=model, temperature=temp, host=host)

    # Hugging Face Transformers
    if prov in ("hf", "transformers"):
        device = agent_llm_cfg.get("device")  # e.g., "auto"
        dtype = agent_llm_cfg.get("dtype")    # "float16" or "bfloat16"
        trust = bool(agent_llm_cfg.get("trust_remote_code", False))
        return HFProvider(model=model, temperature=temp, device=device, dtype=dtype, trust_remote_code=trust)

    return None
