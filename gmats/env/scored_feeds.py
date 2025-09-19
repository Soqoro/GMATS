# gmats/env/scored_feeds.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from .file_feeds import NewsFileFeed, SocialFileFeed
from ..tools.sentiment import SentimentScorer


class ScoredNewsFeed(NewsFileFeed):
    """Wrap NewsFileFeed and attach a 'score' via SentimentScorer when missing."""

    def __init__(
        self,
        base_dir: Path,
        *,
        default_symbol: Optional[str] = None,
        dayfirst: bool = False,
        include_poison: str = "all",
        scorer: Optional[SentimentScorer] = None,
    ):
        super().__init__(base_dir, default_symbol=default_symbol, dayfirst=dayfirst, include_poison=include_poison)
        self.scorer = scorer

    def observe(self, t: Any, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = super().observe(t, q)
        if self.scorer is None:
            return items
        sym = (q.get("symbol") or self.default_symbol or "").upper()
        out: List[Dict[str, Any]] = []
        for it in items:
            if it.get("score") is None:
                text = (it.get("title") or "") + "\n" + (it.get("text") or "")
                s = self.scorer.score(text, symbol=sym)
                it = {**it, "score": float(s)}
                prov = dict(it.get("prov") or {})
                prov["score_source"] = "llm"
                it["prov"] = prov
            out.append(it)
        return out


class ScoredSocialFeed(SocialFileFeed):
    """Wrap SocialFileFeed and attach a 'score' via SentimentScorer when missing."""

    def __init__(
        self,
        base_dir: Path,
        *,
        default_symbol: Optional[str] = None,
        dayfirst: bool = False,
        scorer: Optional[SentimentScorer] = None,
    ):
        super().__init__(base_dir, default_symbol=default_symbol, dayfirst=dayfirst)
        self.scorer = scorer

    def observe(self, t: Any, q: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = super().observe(t, q)
        if self.scorer is None:
            return items
        sym = (q.get("symbol") or self.default_symbol or "").upper()
        out: List[Dict[str, Any]] = []
        for it in items:
            if it.get("score") is None:
                text = (it.get("text") or "") or (it.get("title") or "")
                s = self.scorer.score(text, symbol=sym)
                it = {**it, "score": float(s)}
                prov = dict(it.get("prov") or {})
                prov["score_source"] = "llm"
                it["prov"] = prov
            out.append(it)
        return out
