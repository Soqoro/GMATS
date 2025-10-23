# gmats/agents/executor.py
from dataclasses import dataclass
from typing import Any, Dict, List
import json

@dataclass
class TradeExecutor:
    id: str
    role: str
    prompt: Dict[str, str]
    llm: Any  # LLMClient (we won't actually use for now)

    def _extract_orders(self, inbox: List[Dict]) -> List[Dict]:
        # prefer the newest upstream message with a JSON orders object
        for m in reversed(inbox):
            js = m.get("payload_json")
            if js and isinstance(js, dict) and "orders" in js and isinstance(js["orders"], list):
                return self._clean(js["orders"])
            if isinstance(js, list):
                return self._clean(js)
            txt = m.get("payload_text")
            if isinstance(txt, str) and txt.strip().startswith(("{", "[")):
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, dict) and isinstance(parsed.get("orders"), list):
                        return self._clean(parsed["orders"])
                    if isinstance(parsed, list):
                        return self._clean(parsed)
                except Exception:
                    pass
        return []

    @staticmethod
    def _clean(arr: List[Dict]) -> List[Dict]:
        cleaned: List[Dict] = []
        for o in arr:
            try:
                sym = str(o["symbol"]).upper()
                side = int(o.get("side", 0))
                side = -1 if side < 0 else (1 if side > 0 else 0)
                w = float(o.get("weight", side))
                w = max(-1.0, min(1.0, w))
                cleaned.append({"symbol": sym, "side": side, "weight": w})
            except Exception:
                continue
        return cleaned

    def run(self, date: str, assets: List[str], inbox: List[Dict], downstream_ids: List[str]) -> Dict[str, Any]:
        orders = self._extract_orders(inbox)
        return {
            "id": f"{self.id}:{date}",
            "src": self.id,
            "role": self.role,
            "ts": date,
            "payload_json": {"orders": orders},
            "payload_text": json.dumps({"orders": orders})
        }
