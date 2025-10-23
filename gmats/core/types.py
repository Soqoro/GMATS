"""Core datatypes: Message, Stance, Order."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Schema = Literal["argument", "stance", "order", "risk_alert"]

@dataclass
class Message:
    id: str
    src: str
    dst: str
    schema: Schema
    payload: Dict[str, Any]
    ts: str  # ISO date
    prov: Dict[str, Any] = field(default_factory=dict)

    def validate(self, allowed: Optional[List[Schema]] = None) -> None:
        if allowed is not None and self.schema not in allowed:
            raise ValueError(f"Schema {self.schema} not allowed here")

@dataclass
class Stance:
    vector: List[float]  # aligned to engine.assets order
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Order:
    values: List[float]  # weights or {-1,0,1}
    kind: Literal["discrete", "weights"] = "discrete"
    meta: Dict[str, Any] = field(default_factory=dict)
