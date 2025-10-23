from __future__ import annotations
from typing import Any, Optional
import logging

class BaseStrategyIso:
    def __init__(self):
        self.trades = []
        self.trade_returns = []
        self.buys = []
        self.sells = []
        self.peak_equity = 0.0
        self.equity: list[float] = []
        self.equity_date: list[Any] = []

        # Optional Backtrader-style handles (unused in this framework)
        self.broker: Optional[Any] = None
        self.data: Optional[Any] = None

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

    def on_data(self, date, data_loader, framework):
        raise NotImplementedError("The on_data method must be implemented by the strategy.")

    def update_info(self, date, data_loader, framework):
        # defensive: keep running equity, even if any single price is missing
        self.equity_date.append(date)
        try:
            total = float(getattr(framework, "cash", 0.0))
            for ticker, pos in getattr(framework, "portfolio", {}).items():
                price = data_loader.get_ticker_price_by_date(ticker, date)  # adapter API
                if price is None:
                    continue
                qty = float(pos.get("quantity", 0))
                total += qty * float(price)
            self.equity.append(total)
        except Exception as e:
            self.logger.debug(f"update_info failed: {e}")

    def disable_logger(self):
        self.logger.disabled = True

    def _adjust_size_for_commission(self, max_size: int) -> int:
        # Backtrader-only helper. If broker/data are not set, return 0 safely.
        if self.broker is None or self.data is None:
            return 0
        commission_info = self.broker.getcommissioninfo(self.data)
        cash = self.broker.get_cash()
        price = float(self.data.close[0]) if hasattr(self.data, "close") else 0.0
        if price <= 1e-8:
            return 0
        size = int(max_size)
        while size > 0:
            commission_cost = commission_info._getcommission(size=size, price=price, pseudoexec=True)
            if cash >= (size * price + commission_cost):
                return size
            size -= 1
        return 0