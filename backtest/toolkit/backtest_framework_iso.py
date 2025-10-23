import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, datetime as dt
from typing import Any, Optional, Tuple, List, Protocol, runtime_checkable, cast
from .custom_exceptions import InsufficientTrainingDataException

@runtime_checkable
class DataLoaderProtocol(Protocol):
    def get_date_range(self) -> tuple[dt.date, dt.date]: ...
    def get_ticker_data_by_time_range(self, ticker: str, start_date: dt.date, end_date: dt.date) -> List[dict]: ...
    def get_ticker_price_by_date(self, ticker: str, date: dt.date, field: str = "close") -> Optional[float]: ...

class FINSABERFrameworkHelper:
    def __init__(self, config):
        from .trade_config import TradeConfig
        self.config = config if isinstance(config, TradeConfig) else TradeConfig.from_dict(config)

        # knobs
        self.rolling_window_size: int = getattr(self.config, "rolling_window_size", 20)
        self.rolling_window_step: int = getattr(self.config, "rolling_window_step", 1)
        self.training_years: int = getattr(self.config, "training_years", 2)

        # portfolio state
        self.initial_cash: float = float(getattr(self.config, "cash", 100000.0))
        self.risk_free_rate: float = float(getattr(self.config, "risk_free_rate", 0.0))
        self.cash: float = float(self.initial_cash)
        self.portfolio: dict[str, dict[str, float]] = {}
        self.history: List[dict] = []

        # commission model
        self.min_commission: float = 1.0
        self.commission_per_share: float = 0.005
        self.max_commission_rate: float = 0.01

        # data handle
        self.data_loader: Optional[DataLoaderProtocol] = None

        if os.getenv("GMATS_DEBUG") == "1":
            print(f"[GMATS][framework:init] rw={self.rolling_window_size} "
                  f"step={self.rolling_window_step} train_years={self.training_years}")

    def _require_loader(self) -> DataLoaderProtocol:
        """Type-safe accessor for the active single-ticker view."""
        if self.data_loader is None:
            raise RuntimeError("No data loaded.")
        return self.data_loader

    def load_backtest_data(
        self,
        data,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        if start_date is not None and end_date is not None:
            self.data_loader = cast(DataLoaderProtocol, data.get_subset_by_time_range(start_date, end_date))
        else:
            self.data_loader = cast(DataLoaderProtocol, data)
        return True

    def _to_date(self, x: Any) -> dt.date:
        if isinstance(x, dt.date):
            return x
        y, m, d = [int(v) for v in str(x).split("-")]
        return dt.date(y, m, d)

    def load_backtest_data_single_ticker(
        self,
        data,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        delist_check: bool = False
    ) -> bool:
        # Normalize to concrete dates
        s_req = self._to_date(start_date) if start_date is not None else None
        e_req = self._to_date(end_date) if end_date is not None else None

        # Fill from dataset range if any is None
        if s_req is None or e_req is None:
            ds_start, ds_end = cast(DataLoaderProtocol, data).get_date_range()
            s_req = s_req or ds_start
            e_req = e_req or ds_end
        # Assert for the type checker
        assert isinstance(s_req, dt.date) and isinstance(e_req, dt.date)

        if os.getenv("GMATS_DEBUG") == "1":
            print(f"[GMATS][load] {ticker} requested {s_req} -> {e_req}")

        # Optional delist check uses concrete dates
        if delist_check:
            raw_rows = cast(DataLoaderProtocol, data).get_ticker_data_by_time_range(ticker, s_req, e_req)
            if raw_rows:
                last_raw = dt.date.fromisoformat(str(raw_rows[-1]["date"]))
                gap = (e_req - last_raw).days
                if gap >= 7:
                    print(f"Current symbol appears to be delisted on {last_raw}, adjust the end date for 7 days ahead announcement.")
                    e_req = last_raw + dt.timedelta(days=7)

        # Build view with concrete dates
        self.data_loader = cast(DataLoaderProtocol, data.get_ticker_subset_by_time_range(ticker, s_req, e_req))

        # Count slice days and apply warmup
        loader = self._require_loader()
        rows = loader.get_ticker_data_by_time_range(ticker, s_req, e_req)
        n_days = len(rows)
        warmup_days = max(0, int(self.training_years * 252))
        window_warmup = max(0, int(self.rolling_window_size) - 1)
        available_days = max(0, n_days - (warmup_days + window_warmup))

        if os.getenv("GMATS_DEBUG") == "1":
            print(f"[GMATS][framework] n_days={n_days} train_years={self.training_years} rw={self.rolling_window_size} "
                  f"step={self.rolling_window_step} warmup_days={warmup_days} window_warmup={window_warmup} available_days={available_days}")

        if available_days < 3:
            raise InsufficientTrainingDataException(
                f"Not enough data for backtesting. Slice days={n_days}, warmup={warmup_days + window_warmup}, "
                f"available={available_days} (train_years={self.training_years}, rw={self.rolling_window_size})."
            )
        return True

    def calculate_commission(self, quantity: int, price: float) -> float:
        commission = abs(quantity) * self.commission_per_share
        txn_amount = abs(quantity * price)
        return max(self.min_commission, min(commission, txn_amount * self.max_commission_rate))

    def buy(self, date: dt.date, ticker: str, price: float, quantity: int):
        total_cost = 0.0
        commission = 0.0
        if quantity >= 0:
            cost = price * quantity
            commission = self.calculate_commission(quantity, price)
            total_cost = cost + commission
        elif quantity == -1:
            # buy all cash
            total_cost = self.cash
            est_qty = int(total_cost / max(price, 1e-8))
            commission = self.calculate_commission(est_qty, price)
            total_cost -= commission
            quantity = int(total_cost / max(price, 1e-8))
            total_cost = price * quantity + commission
        else:
            print(f"Invalid buy quantity {quantity} for {ticker} on {date}")
            return

        if self.cash >= total_cost and quantity > 0:
            self.cash -= total_cost
            if ticker in self.portfolio:
                self.portfolio[ticker]['quantity'] += quantity
            else:
                self.portfolio[ticker] = {'quantity': float(quantity), 'price': float(price)}
            self.history.append({'date': date, 'ticker': ticker, 'type': 'buy', 'price': float(price), 'quantity': int(quantity), 'commission': float(commission)})
        else:
            print(f"Insufficient cash to buy {quantity} of {ticker} on {date}")

    def sell(self, date: dt.date, ticker: str, price: float, quantity: int):
        if ticker in self.portfolio and self.portfolio[ticker]['quantity'] >= quantity > 0:
            revenue = price * quantity
            commission = self.calculate_commission(quantity, price)
            net_revenue = revenue - commission
            self.cash += net_revenue
            self.portfolio[ticker]['quantity'] -= quantity
            if self.portfolio[ticker]['quantity'] <= 0:
                del self.portfolio[ticker]
            self.history.append({'date': date, 'ticker': ticker, 'type': 'sell', 'price': float(price), 'quantity': int(quantity), 'commission': float(commission)})
        else:
            print(f"Insufficient holdings to sell {quantity} of {ticker} on {date}")

    def run(self, strategy, delist_check: bool = False) -> bool:
        try:
            loader = self._require_loader()
        except RuntimeError as _:
            print("No data loaded.")
            return False

        start_d, end_d = loader.get_date_range()
        # Build calendar dates; adapter already ffilled missing trading days
        dates = [d.date() if isinstance(d, pd.Timestamp) else d for d in pd.date_range(start=start_d, end=end_d, freq="D")]

        if len(dates) < 3:
            print(f"Not enough data for backtesting. Only {len(dates)} days available.")
            return False

        for date in dates:
            status = strategy.on_data(date, loader, self)
            strategy.update_info(date, loader, self)
            if status == "done":
                break

        # liquidate remaining holdings at the last date
        last_date = dates[-1]
        for ticker in list(self.portfolio.keys()):
            price = loader.get_ticker_price_by_date(ticker, last_date)
            if price is None:
                continue
            qty = int(self.portfolio[ticker]['quantity'])
            if qty > 0:
                self.sell(last_date, ticker, float(price), qty)

        return True

    def _price(self, loader: DataLoaderProtocol, ticker: str, date: dt.date, field: str = "close") -> float:
        """Return a float price; coalesce None to 0.0 for safe math."""
        p = loader.get_ticker_price_by_date(ticker, date, field=field)
        return float(p) if p is not None else 0.0

    def evaluate(self, strategy) -> dict:
        if self.data_loader is None:
            print("No data loaded for evaluation.")
            return {'final_value': 0.0, 'total_return': 0.0, 'annual_return': 0.0, 'annual_volatility': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'total_commission': 0.0, 'max_drawdown': 0.0}

        loader = self._require_loader()
        start_d, end_d = loader.get_date_range()
        final_value = self.cash + sum(
            float(self.portfolio[t]['quantity']) * self._price(loader, t, end_d)
            for t in self.portfolio
        )

        if len(strategy.equity) <= 1:
            print("No equity data available for evaluation.")
            return {
                'final_value': final_value, 'total_return': 0.0, 'annual_return': 0.0, 'annual_volatility': 0.0,
                'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'total_commission': sum([tr['commission'] for tr in self.history]),
                'max_drawdown': 0.0
            }

        daily_returns = pd.Series([strategy.equity[i] / strategy.equity[i - 1] - 1 for i in range(1, len(strategy.equity))])
        total_return = (final_value / self.initial_cash) - 1
        n_days = max(1, len(pd.date_range(start=start_d, end=end_d, freq="B")))
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        annual_volatility = daily_returns.std() * np.sqrt(252)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0.0
        sharpe_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0
        sortino_ratio = (daily_returns.mean() * 252 - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        total_commission = float(sum([trade['commission'] for trade in self.history]))

        equity_series = pd.Series(strategy.equity)
        running_max = equity_series.cummax()
        drawdowns = (equity_series - running_max) / running_max
        max_drawdown = abs(float(drawdowns.min())) * 100 if not drawdowns.empty else 0.0

        return {
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_commission': total_commission,
            'max_drawdown': max_drawdown
        }

    def reset(self):
        self.cash = float(self.initial_cash)
        self.portfolio = {}
        self.history = []
        self.data_loader = None