"""FINSABER BacktestDataset adapter (price-only)."""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import os
import pandas as pd
import datetime as dt
from backtest.data_util.backtest_dataset import BacktestDataset

def _to_datestr(v: Any) -> str:
    # Accept datetime.date, datetime.datetime, or flexible strings (YYYY-M-D)
    if isinstance(v, dt.datetime):
        return v.date().isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, str):
        s = v.strip()
        try:
            y, m, d = [int(x) for x in s.split("-")]
            return dt.date(y, m, d).isoformat()
        except Exception:
            return pd.to_datetime(s).date().isoformat()
    raise TypeError(f"Unsupported date type: {type(v)}")

class GMATSDataset(BacktestDataset):
    def __init__(self, data_root: str, market_dir: str = "market"):
        self.dir = Path(data_root) / market_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load(self, ticker: str) -> pd.DataFrame:
        t = ticker.upper()
        if t in self._cache:
            return self._cache[t]
        p = self.dir / f"{t}.csv"
        df = pd.read_csv(p)
        # normalize dates to string day granularity
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        self._cache[t] = df
        return df

    # --- framework API ---
    def get_tickers_list(self) -> List[str]:
        return [p.stem.upper() for p in self.dir.glob("*.csv")]

    def get_ticker_subset_by_time_range(self, ticker: str, start_date: Any, end_date: Any) -> "GMATSDatasetView":
        view = GMATSDatasetView(self, ticker.upper(), start_date, end_date)
        if os.getenv("GMATS_DEBUG") == "1":
            df = view.df
            print(f"[GMATS][view] {ticker.upper()} window {_to_datestr(start_date)} -> {_to_datestr(end_date)} | rows={len(df)} | range={df['date'].min() if len(df) else None}..{df['date'].max() if len(df) else None}")
            if len(df):
                head = ", ".join(list(df["date"].head(3)))
                tail = ", ".join(list(df["date"].tail(3)))
                print(f"[GMATS][view] {ticker.upper()} head: {head} | tail: {tail}")
        return view

    def get_ticker_price_by_date(self, ticker: str, date: Any, field: str = "close") -> Optional[float]:
        # not used directly by runner in our path, but keep robust
        return GMATSDatasetView(self, ticker, date, date).get_ticker_price_by_date(ticker, date, field)

    def get_ticker_data_by_date(self, ticker: str, date: Any):
        df = self._load(ticker)
        ds = _to_datestr(date)
        row = df.loc[df["date"] == ds]
        return None if row.empty else row.iloc[0].to_dict()

    def get_ticker_data_by_time_range(self, ticker: str, start_date: Any, end_date: Any):
        df = self._load(ticker)
        s = _to_datestr(start_date); e = _to_datestr(end_date)
        m = (df["date"] >= s) & (df["date"] <= e)
        return df.loc[m].to_dict(orient="records")

    def get_subset_by_time_range(self, start_date: Any, end_date: Any) -> "GMATSDataset":
        return self

    def get_date_range(self) -> Tuple[dt.date, dt.date]:
        tickers = self.get_tickers_list()
        if not tickers:
            return (dt.date(1970, 1, 1), dt.date(1970, 1, 1))
        df = self._load(tickers[0])
        start = dt.date.fromisoformat(str(df["date"].min()))
        end   = dt.date.fromisoformat(str(df["date"].max()))
        return (start, end)

class GMATSDatasetView(BacktestDataset):
    """Single-ticker, time-windowed view; calendarized daily and ffilled."""
    def __init__(self, parent: GMATSDataset, ticker: str, start_date: Any, end_date: Any):
        self.parent = parent
        self.ticker = ticker.upper()
        df = parent._load(self.ticker).copy()
        s = _to_datestr(start_date); e = _to_datestr(end_date)

        # save last observed trading date BEFORE calendarization
        if not df.empty:
            self._last_observed_date: dt.date = pd.to_datetime(df["date"]).max().date()
        else:
            self._last_observed_date = dt.date(1970, 1, 1)

        m = (df["date"] >= s) & (df["date"] <= e)
        df = df.loc[m].copy()
        # calendarize + ffill as before
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            full = pd.date_range(start=s, end=e, freq="D")
            df = df.reindex(full).ffill().bfill()
            df.index = df.index.date.astype(str)
            df = df.reset_index().rename(columns={"index": "date"})
        else:
            full = pd.date_range(start=s, end=e, freq="D")
            df = pd.DataFrame({"date": full.date.astype(str)})

        self.df = df.reset_index(drop=True)

    # new helper for framework delist check
    def get_last_observed_data_date(self) -> dt.date:
        return self._last_observed_date

    def get_tickers_list(self) -> List[str]:
        return [self.ticker]

    def get_date_range(self) -> Tuple[dt.date, dt.date]:
        if self.df.empty:
            return (dt.date(1970, 1, 1), dt.date(1970, 1, 1))
        start = dt.date.fromisoformat(str(self.df["date"].min()))
        end   = dt.date.fromisoformat(str(self.df["date"].max()))
        return (start, end)

    def get_ticker_price_by_date(self, ticker: str, date: Any, field: str = "close") -> Optional[float]:
        ds = _to_datestr(date)
        row = self.df.loc[self.df["date"] == ds]
        if row.empty:
            if os.getenv("GMATS_DEBUG") == "1":
                print(f"[GMATS][price-miss] {self.ticker} @ {ds} (no row)")
            return None
        val = row.iloc[0].get(field)
        try:
            val = float(val)
        except Exception:
            if os.getenv("GMATS_DEBUG") == "1":
                print(f"[GMATS][price-miss] {self.ticker} @ {ds} field={field} val={val}")
            return None
        if pd.isna(val):
            if os.getenv("GMATS_DEBUG") == "1":
                print(f"[GMATS][price-miss] {self.ticker} @ {ds} NaN")
            return None
        return val

    def get_ticker_data_by_date(self, ticker: str, date: Any):
        ds = _to_datestr(date)
        row = self.df.loc[self.df["date"] == ds]
        return None if row.empty else row.iloc[0].to_dict()

    def get_ticker_data_by_time_range(self, ticker: str, start_date: Any, end_date: Any):
        s = _to_datestr(start_date); e = _to_datestr(end_date)
        m = (self.df["date"] >= s) & (self.df["date"] <= e)
        return self.df.loc[m].to_dict(orient="records")
