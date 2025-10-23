"""DataHub: unified snapshot per date."""
from __future__ import annotations
from typing import Dict, Any, List
from gmats.data.market_loader import MarketLoader
from gmats.data.social_loader import SocialLoader
from gmats.data.fundamentals_loader import FundamentalsLoader

class DataHub:
    def __init__(self, root: str, market_dir: str, news_dir: str, social_dir: str, fundamentals_dir: str, assets: List[str]):
        self.assets = [s.upper() for s in assets]
        self.market = MarketLoader(root, market_dir, self.assets)
        self.social = SocialLoader(root, social_dir, self.assets)
        self.fund = FundamentalsLoader(root, fundamentals_dir, self.assets)

    def observe(self, date: str) -> Dict[str, Any]:
        return {
            "market": self.market.observe(date, window_days=60),
            "social": self.social.observe(date),
            "fundamentals": self.fund.observe(date),
        }
