# FINSABER/backtest/finsaber.py
# Lean backtest runner for GMATS: rich progress + plotting + aggregation, no LLM deps.
from rich.progress import Progress
import os, csv
from typing import List, Dict, Any

from backtest.toolkit.trade_config import TradeConfig
from backtest.toolkit.backtest_framework_iso import FINSABERFrameworkHelper
from backtest.toolkit.custom_exceptions import InsufficientTrainingDataException
from backtest.toolkit.operation_utils import aggregate_results_one_strategy


class FINSABER:
    def __init__(self, trade_config: dict):
        # Normalize to TradeConfig and keep it
        self.trade_config = TradeConfig.from_dict(trade_config)
        if os.getenv("GMATS_DEBUG") == "1":
            print(
                f"[GMATS][FINSABER:init] rw={self.trade_config.rolling_window_size} "
                f"step={self.trade_config.rolling_window_step} train_years={self.trade_config.training_years}"
            )

        self.framework: FINSABERFrameworkHelper | None = None

    def _save_result_row(self, path: str, row: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def run_iterative_tickers(self, strategy_class, strat_params=None, tickers=None, delist_check=False, aggregate=False):
        strat_params = strat_params or {}
        data = getattr(self.trade_config, "data_loader", None)
        if data is None:
            raise RuntimeError("TradeConfig.data_loader is None")

        tickers = tickers or getattr(self.trade_config, "tickers", None) or getattr(data, "get_tickers_list", lambda: [])()
        if not tickers:
            print("No tickers to backtest.")
            return

        self.framework = FINSABERFrameworkHelper(self.trade_config)

        ran_any = False
        results: List[Dict[str, Any]] = []
        with Progress() as progress:
            task = progress.add_task("Backtesting", total=len(tickers))
            for t in tickers:
                try:
                    # pass the delist_check flag here
                    self.framework.load_backtest_data_single_ticker(
                        data, t, self.trade_config.date_from, self.trade_config.date_to, delist_check=delist_check # type: ignore
                    )
                except InsufficientTrainingDataException as e:
                    print(str(e))
                    print(f"Run failed for {t}. Skipping.")
                    progress.advance(task, 1)
                    continue

                # Instantiate strategy for this run
                strategy = strategy_class(**strat_params)

                # Run and evaluate
                self.framework.run(strategy)
                metrics = self.framework.evaluate(strategy)
                ran_any = True

                # Print a compact summary
                print(f"[GMATS][result] {t}: final={metrics['final_value']:.2f} "
                      f"ret={metrics['total_return']:.2%} sharpe={metrics['sharpe_ratio']:.2f} "
                      f"mdd={metrics['max_drawdown']:.2f}%")

                # Save optional CSV row
                if self.trade_config.save_results:
                    out_dir = os.path.join(self.trade_config.log_base_dir, "results")
                    setup = self.trade_config.setup_name or "default"
                    out_path = os.path.join(out_dir, f"{setup}_{strategy_class.__name__}.csv")
                    row = {
                        "setup": setup,
                        "strategy": strategy_class.__name__,
                        "ticker": t,
                        "date_from": self.trade_config.date_from.isoformat(),
                        "date_to": self.trade_config.date_to.isoformat(),
                        "final_value": f"{metrics['final_value']:.2f}",
                        "total_return": f"{metrics['total_return']:.6f}",
                        "annual_return": f"{metrics['annual_return']:.6f}",
                        "annual_volatility": f"{metrics['annual_volatility']:.6f}",
                        "sharpe_ratio": f"{metrics['sharpe_ratio']:.6f}",
                        "sortino_ratio": f"{metrics['sortino_ratio']:.6f}",
                        "max_drawdown": f"{metrics['max_drawdown']:.6f}",
                        "total_commission": f"{metrics['total_commission']:.6f}",
                    }
                    self._save_result_row(out_path, row)

                progress.advance(task, 1)

        # Only aggregate if explicitly requested (rolling windows mode)
        if ran_any and aggregate:
            try:
                aggregate_results_one_strategy(
                    setup_name=self.trade_config.setup_name, # type: ignore
                    trading_strategy=strategy_class.__name__,
                )
            except ZeroDivisionError:
                print("No valid backtest windows to aggregate. Skipping aggregation.")
        elif not ran_any:
            print("No valid backtest windows across selected tickers. Skipping aggregation.")
