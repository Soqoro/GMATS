from dataclasses import dataclass, field, asdict
from typing import Union, List, Optional, Any
import os
import datetime as dt
from backtest.strategy import BaseSelector

@dataclass
class TradeConfig:
    tickers: Union[List[str], str]
    date_from: dt.date = dt.date(2004, 1, 1)
    date_to: dt.date = dt.date(2024, 1, 1)
    cash: float = 100000.0
    risk_free_rate: float = 0.03
    print_trades_table: bool = False
    silence: bool = False
    rolling_window_size: int = 2
    rolling_window_step: int = 1
    training_years: int = 0
    selection_strategy: Optional[BaseSelector] = None
    setup_name: Optional[str] = None
    result_filename: Optional[str] = None
    save_results: bool = True
    log_base_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    data_loader: Any = None

    def __post_init__(self):
        # tickers validation
        if isinstance(self.tickers, str):
            if self.tickers.lower() != "all":
                raise ValueError("tickers can either be a list of tickers or the string 'all'")
        elif not isinstance(self.tickers, list) or not all(isinstance(t, str) for t in self.tickers):
            raise ValueError("tickers must be a list of strings")

        # coerce date_from/date_to if passed as strings
        if isinstance(self.date_from, str):
            self.date_from = dt.date.fromisoformat(self.date_from)
        if isinstance(self.date_to, str):
            self.date_to = dt.date.fromisoformat(self.date_to)

        if self.date_from > self.date_to:
            raise ValueError("date_from must be earlier than date_to")

    @classmethod
    def from_dict(cls, config_dict):
        # Accept strings for dates and coerce via __post_init__
        return cls(**config_dict)

    def to_dict(self):
        d = asdict(self)
        # make dates serializable
        d["date_from"] = self.date_from.isoformat()
        d["date_to"] = self.date_to.isoformat()
        return d


if __name__ == "__main__":
    # test the TradeConfig class
    # Initialize with a dictionary that includes "all" tickers
    config_dict = {
        "tickers": "all",
        "date_from": "2010-01-01",
        "date_to": "2023-12-31",
        "cash": 150000.0,
        "commission": 0.0002,
        "slippage_perc": 0.0001
    }

    try:
        config = TradeConfig.from_dict(config_dict)
        print(config)
    except ValueError as e:
        print(e)

    # Initialize with a list of tickers
    config_dict_list = {
        "tickers": ["AAPL", "GOOGL"],
        "date_from": "2010-01-01",
        "date_to": "2023-12-31",
        "cash": 150000.0,
        "commission": 0.0002,
        "slippage_perc": 0.0001
    }

    config_list = TradeConfig.from_dict(config_dict_list)
    print(config_list)

    # Convert back to dictionary
    print(config_list.to_dict())