"""FundamentalsLoader: reads data/fundamentals/<TICKER>.jsonl with {date, metrics:{...}}."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Any

class FundamentalsLoader:
    def __init__(self, root: str, subdir: str, assets: List[str]):
        self.dir = Path(root) / subdir
        self.assets = [s.upper() for s in assets]

    def observe(self, date: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for s in self.assets:
            p = self.dir / f"{s}.jsonl"
            if not p.exists(): continue
            metrics = None
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if str(obj.get("date")) == date:
                            metrics = obj.get("metrics", {})
                    except Exception:
                        pass
            if metrics is not None:
                out[s] = {"metrics": metrics}
        return out
