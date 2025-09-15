from __future__ import annotations
"""
Backtester
==========
Loads per-symbol CSVs and provides a toy 'trade_return' API for evaluation.

Accepted CSV formats per symbol (filename = SYMBOL.csv):
- date,ret
- date,close   (daily % change is computed)
"""

from pathlib import Path
from typing import Dict
import re
import pandas as pd


class Backtester:
    """Lightweight backtester reading daily returns from CSV files."""

    def __init__(self, returns_dir: str, rf_daily: float = 0.0):
        """
        Args:
            returns_dir: Folder with files like AAPL.csv, MSFT.csv, ...
            rf_daily: Daily risk-free rate to subtract from realized returns.
        """
        self.rf_daily: float = float(rf_daily)
        self.data: Dict[str, pd.DataFrame] = {}

        p = Path(returns_dir)
        p.mkdir(parents=True, exist_ok=True)

        for f in p.glob("*.csv"):
            sym = f.stem.upper()
            df = pd.read_csv(f)

            if "date" not in df.columns:
                raise ValueError(f"{f} missing 'date' column")

            # --- Robust date parsing with explicit formats (no warnings) ---
            df = df.copy()
            df["date"] = self._parse_dates_no_warning(df["date"])

            # Fail fast if any unparseable
            bad = df["date"].isna()
            if bad.any():
                sample_raw = df.loc[bad].index[:3]
                sample_vals = [str(v) for v in df.loc[sample_raw, "date"]]
                raise ValueError(
                    f"{f} has {bad.sum()} rows with unparseable dates. "
                    f"Ensure ISO (YYYY-MM-DD) or common slash/dash formats. Sample: {sample_vals}"
                )

            # Sort before computing returns
            df = df.sort_values("date").reset_index(drop=True)

            if "ret" in df.columns:
                df = df[["date", "ret"]].copy()
                df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0).astype(float)

            elif "close" in df.columns:
                df = df[["date", "close"]].copy()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                if df["close"].isna().any():
                    raise ValueError(f"{f} has non-numeric 'close' values after coercion.")
                df["ret"] = df["close"].pct_change().fillna(0.0).astype(float)
                df = df.drop(columns=["close"])
            else:
                raise ValueError(f"{f} must have either 'ret' or 'close' column")

            self.data[sym] = df

    @staticmethod
    def _parse_dates_no_warning(series: pd.Series) -> pd.Series:
        """
        Parse common date formats deterministically with explicit strptime formats
        to avoid pandas' dayfirst warnings.

        Supported (auto-detected) formats:
          - YYYY-MM-DD
          - YYYY/MM/DD
          - DD/MM/YYYY, MM/DD/YYYY
          - DD-MM-YYYY, MM-DD-YYYY
          - Fallback: generic parser (e.g., 'Jan 02 2024')
        """
        s = series.astype(str).str.strip()

        # Fast path: strict ISO YYYY-MM-DD
        iso_mask = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
        if iso_mask.all():
            dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
            return dt.dt.normalize()

        # Detect separator
        # Prefer slash or dash; ignore other punctuation.
        has_slash = s.str.contains("/")
        has_dash = s.str.contains("-")

        # YYYY/MM/DD
        if has_slash.any():
            # If *all* look like YYYY/... and first token is 4-digit → year-first
            parts = s.str.split("/", n=2, expand=True)
            if parts.shape[1] == 3:
                first_len4 = parts[0].str.fullmatch(r"\d{4}", na=False)
                third_len4 = parts[2].str.fullmatch(r"\d{4}", na=False)

                if first_len4.mean() > 0.9:
                    fmt = "%Y/%m/%d"
                    dt = pd.to_datetime(s, format=fmt, errors="coerce")
                    if not dt.isna().all():
                        return dt.dt.normalize()

                # Decide DD/MM/YYYY vs MM/DD/YYYY by majority on first field
                if third_len4.mean() > 0.9:
                    first_num = pd.to_numeric(parts[0], errors="coerce")
                    # If most first tokens are >12 → likely DD/MM/YYYY
                    dayfirst = (first_num > 12).mean() > 0.5
                    fmt = "%d/%m/%Y" if dayfirst else "%m/%d/%Y"
                    dt = pd.to_datetime(s, format=fmt, errors="coerce")
                    if dt.isna().any():
                        # Try the opposite for any leftovers (no warnings in either case)
                        alt_fmt = "%m/%d/%Y" if dayfirst else "%d/%m/%Y"
                        dt2 = pd.to_datetime(s, format=alt_fmt, errors="coerce")
                        dt = dt.fillna(dt2)
                    return dt.dt.normalize()

        # Dash variants
        if has_dash.any():
            parts = s.str.split("-", n=2, expand=True)
            if parts.shape[1] == 3:
                # YYYY-MM-DD
                first_len4 = parts[0].str.fullmatch(r"\d{4}", na=False)
                if first_len4.mean() > 0.9:
                    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
                    if not dt.isna().all():
                        return dt.dt.normalize()

                # Decide DD-MM-YYYY vs MM-DD-YYYY
                third_len4 = parts[2].str.fullmatch(r"\d{4}", na=False)
                if third_len4.mean() > 0.9:
                    first_num = pd.to_numeric(parts[0], errors="coerce")
                    dayfirst = (first_num > 12).mean() > 0.5
                    fmt = "%d-%m-%Y" if dayfirst else "%m-%d-%Y"
                    dt = pd.to_datetime(s, format=fmt, errors="coerce")
                    if dt.isna().any():
                        alt_fmt = "%m-%d-%Y" if dayfirst else "%d-%m-%Y"
                        dt2 = pd.to_datetime(s, format=alt_fmt, errors="coerce")
                        dt = dt.fillna(dt2)
                    return dt.dt.normalize()

        # Final fallback: let pandas guess (rare strings), then normalize
        dt = pd.to_datetime(s, errors="coerce")
        return dt.dt.normalize()

    def _next_ret(self, symbol: str, asof_date: str, horizon: int = 1) -> float:
        """Return the next-day (or next-horizon) return from the as-of date."""
        df = self.data.get(symbol.upper())
        if df is None or df.empty:
            return 0.0
        d = pd.to_datetime(asof_date).normalize()
        idx = df.index[df["date"] >= d]
        if len(idx) == 0:
            return 0.0
        i = int(idx[0])
        j = i + int(horizon)
        if j >= len(df):
            return 0.0
        return float(df["ret"].iloc[j])

    def trade_return(self, symbol: str, asof_date: str, action: str, horizon: int = 1) -> float:
        """Compute signed PnL contribution of an action over the horizon."""
        r = self._next_ret(symbol, asof_date, horizon)
        if action == "BUY":
            pnl = r - self.rf_daily
        elif action == "SELL":
            pnl = -r - self.rf_daily
        else:
            pnl = 0.0
        return float(pnl)
