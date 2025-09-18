from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
from pathlib import Path
import json
import pandas as pd

from gmats.core.interfaces import DataFeed


# ---------- safe converters ----------

def _to_int(x: Any) -> Optional[int]:
    """Best-effort int conversion; returns None on None/NaN/invalid."""
    try:
        if x is None:
            return None
        # pandas/NumPy NaN handling
        if isinstance(x, float) and pd.isna(x):
            return None
        return int(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

def _to_float(x: Any) -> Optional[float]:
    """Best-effort float conversion; returns None on None/NaN/invalid."""
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        # strings like "" should become None
        if isinstance(x, str) and not x.strip():
            return None
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ---------- date parsing ----------

def _parse_date(v: Any, *, dayfirst: Optional[bool] = None) -> Optional[pd.Timestamp]:
    if v is None:
        return None
    try:
        ts = pd.to_datetime(v, dayfirst=bool(dayfirst), utc=False, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts).normalize()
    except Exception:
        return None


# ---------- file readers ----------

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_i"] = i  # plain int
                obj["_file"] = str(path)
                yield obj
            except Exception:
                continue

def _iter_csv(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    # ensure 0..N-1 positional indices
    df = df.reset_index(drop=True)

    # enumerate gives us an int position regardless of original index dtype
    for pos, row in enumerate(df.itertuples(index=False, name=None)):
        d: Dict[str, Any] = dict(zip(df.columns, row))
        d["_i"] = pos            # plain int, type-checker friendly
        d["_file"] = str(path)
        yield d


# ---------- base feed ----------

class _BaseFileFeed(DataFeed):
    """
    Base for offline file-backed feeds. Supports JSONL/CSV per-symbol files.

    Directory structure (any of these per SYMBOL):
      <base>/SYMBOL.jsonl
      <base>/SYMBOL.csv
      <base>/SYMBOL/data.jsonl
      <base>/SYMBOL/data.csv
    """
    def __init__(self, base_dir: Path, default_symbol: Optional[str] = None, *, dayfirst: Optional[bool] = None):
        self.base_dir = base_dir
        self.default_symbol = default_symbol.upper() if default_symbol else None
        self.dayfirst = dayfirst
        self._cache: Dict[str, List[Dict[str, Any]]] = {}  # cache per symbol

    def _file_candidates(self, sym: str) -> List[Path]:
        return [
            self.base_dir / f"{sym}.jsonl",
            self.base_dir / f"{sym}.csv",
            self.base_dir / sym / "data.jsonl",
            self.base_dir / sym / "data.csv",
        ]

    def _load_items(self, sym: str) -> List[Dict[str, Any]]:
        if sym in self._cache:
            return self._cache[sym]
        for p in self._file_candidates(sym):
            if p.suffix == ".jsonl" and p.exists():
                items = list(_iter_jsonl(p))
                self._cache[sym] = items
                return items
            if p.suffix == ".csv" and p.exists():
                items = list(_iter_csv(p))
                self._cache[sym] = items
                return items
        self._cache[sym] = []
        return []

    def _normalize_common(self, sym: str, raw: Dict[str, Any], source: str) -> Dict[str, Any]:
        dt = _parse_date(
            raw.get("date") or raw.get("timestamp") or raw.get("time") or raw.get("published_at"),
            dayfirst=self.dayfirst
        )
        title = raw.get("title") or raw.get("headline") or ""
        text = raw.get("text") or raw.get("body") or raw.get("summary") or ""
        row_idx = _to_int(raw.get("_i"))
        return {
            "symbol": sym,
            "date": dt,
            "title": str(title) if title is not None else "",
            "text": str(text) if text is not None else "",
            "prov": {
                "source": source,
                "file": str(raw.get("_file")) if raw.get("_file") is not None else None,
                "row": row_idx,  # Optional[int]
            },
            "_raw": raw,
        }

    def _cutoff_and_limit(self, items: List[Dict[str, Any]], *, t: Any, limit: int) -> List[Dict[str, Any]]:
        cutoff = _parse_date(t, dayfirst=self.dayfirst) if t is not None else None
        if cutoff is not None:
            items = [x for x in items if (x.get("date") is not None and x["date"] <= cutoff)]
        items.sort(key=lambda x: (x.get("date") or pd.Timestamp.min))
        return items[-limit:]


# ---------- concrete feeds ----------

class NewsFileFeed(_BaseFileFeed):
    """Read offline news headlines/articles for a symbol from files."""
    def __init__(
        self,
        base_dir: Path,
        default_symbol: Optional[str] = None,
        *,
        dayfirst: Optional[bool] = None,
        include_poison: str = "all",  # 'all' | 'clean_only' | 'poison_only'
    ):
        super().__init__(base_dir, default_symbol, dayfirst=dayfirst)
        self.include_poison = include_poison

    def observe(self, t: Any, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        sym_in = q.get("symbol")
        sym = (str(sym_in).upper() if sym_in else (self.default_symbol or "")).upper()
        if not sym:
            return []
        limit = int(q.get("limit", 5))
        rows = self._load_items(sym)
        items: List[Dict[str, Any]] = []
        for raw in rows:
            norm = self._normalize_common(sym, raw, "offline.news")
            poison = bool(raw.get("poison", False))
            if self.include_poison == "clean_only" and poison:
                continue
            if self.include_poison == "poison_only" and not poison:
                continue
            norm["score"] = _to_float(raw.get("score"))
            stance = raw.get("stance")
            norm["stance"] = str(stance) if stance is not None else None
            items.append(norm)
        return self._cutoff_and_limit(items, t=t, limit=limit)


class SocialFileFeed(_BaseFileFeed):
    """Read offline social posts for a symbol (e.g., tweets/reddit)."""
    def observe(self, t: Any, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        sym_in = q.get("symbol")
        sym = (str(sym_in).upper() if sym_in else (self.default_symbol or "")).upper()
        if not sym:
            return []
        limit = int(q.get("limit", 5))
        rows = self._load_items(sym)
        items: List[Dict[str, Any]] = []
        for raw in rows:
            norm = self._normalize_common(sym, raw, "offline.social")
            norm["score"] = _to_float(raw.get("score"))
            stance = raw.get("stance")
            norm["stance"] = str(stance) if stance is not None else None
            items.append(norm)
        return self._cutoff_and_limit(items, t=t, limit=limit)


class FundamentalsFileFeed(_BaseFileFeed):
    """Read offline fundamentals snapshots keyed by date (e.g., ratios, earnings)."""
    def observe(self, t: Any, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        sym_in = q.get("symbol")
        sym = (str(sym_in).upper() if sym_in else (self.default_symbol or "")).upper()
        if not sym:
            return []
        rows = self._load_items(sym)
        items: List[Dict[str, Any]] = []
        for raw in rows:
            dt = _parse_date(raw.get("date") or raw.get("asof") or raw.get("period_end"), dayfirst=self.dayfirst)
            if dt is None:
                continue
            items.append({
                "symbol": sym,
                "date": dt,
                "metrics": raw,
                "prov": {
                    "source": "offline.fundamentals",
                    "file": str(raw.get("_file")) if raw.get("_file") is not None else None,
                    "row": _to_int(raw.get("_i")),
                },
            })
        # fundamentals are sparse; return most recent snapshot <= t
        return self._cutoff_and_limit(items, t=t, limit=1)
