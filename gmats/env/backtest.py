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
                sample_idx = df.index[bad][:3]
                sample_vals = [str(v) for v in df.loc[sample_idx, "date"]]
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
        """
        s = series.astype(str).str.strip()

        # Fast path: strict ISO YYYY-MM-DD
        iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        if iso.notna().all():
            return iso.dt.normalize()

        # Try YYYY/MM/DD
        ymd_slash = pd.to_datetime(s, format="%Y/%m/%d", errors="coerce")
        if ymd_slash.notna().all():
            return ymd_slash.dt.normalize()

        # Try DD/MM/YYYY then MM/DD/YYYY (fill-na strategy; no warnings)
        dmy_slash = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
        if dmy_slash.notna().any():
            mdy_slash = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
            out = dmy_slash.fillna(mdy_slash)
            if out.notna().all():
                return out.dt.normalize()

        # Try dash variants
        ymd_dash = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        if ymd_dash.notna().all():
            return ymd_dash.dt.normalize()

        dmy_dash = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
        if dmy_dash.notna().any():
            mdy_dash = pd.to_datetime(s, format="%m-%d-%Y", errors="coerce")
            out = dmy_dash.fillna(mdy_dash)
            if out.notna().all():
                return out.dt.normalize()

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
