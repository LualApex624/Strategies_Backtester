import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, Any, List, Tuple
from copy import deepcopy

from strategies.base import BaseStrategy
from backtester.engine import BacktestEngine
from metrics.performance import compute_metrics


class WalkForwardOptimizer:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        param_grid: Dict[str, list],
        n_splits: int = 5,
        train_ratio: float = 0.7,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.001,
        fee_pct: float = 0.001,
    ):
        self.data = data
        self.strategy = strategy
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct

    def run(self) -> Dict[str, Any]:
        splits = self._create_splits()
        fold_results = []

        for i, (train_idx, test_idx) in enumerate(splits):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            best_params, train_sharpe = self._grid_search(train_data)

            strat = deepcopy(self.strategy)
            strat.set_params(best_params)
            engine = BacktestEngine(
                test_data, strat, self.initial_cash, self.slippage_pct, self.fee_pct
            )
            results = engine.run()
            test_metrics = compute_metrics(results["Equity_Realistic"])

            fold_results.append(
                {
                    "fold": i,
                    "best_params": best_params,
                    "train_sharpe": train_sharpe,
                    "test_sharpe": test_metrics["sharpe"],
                    "test_metrics": test_metrics,
                }
            )

        return self._aggregate_results(fold_results)

    def _create_splits(self) -> List[Tuple[range, range]]:
        n = len(self.data)
        window = n // self.n_splits
        splits = []
        for i in range(self.n_splits):
            start = i * window
            end = min(start + window, n)
            if end - start < 20:
                continue
            split_point = start + int((end - start) * self.train_ratio)
            splits.append((range(start, split_point), range(split_point, end)))
        return splits

    def _grid_search(self, train_data: pd.DataFrame) -> Tuple[Dict, float]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        best_sharpe = -np.inf
        best_params = {}

        for combo in product(*values):
            params = dict(zip(keys, combo))
            strat = deepcopy(self.strategy)
            strat.set_params(params)
            try:
                engine = BacktestEngine(
                    train_data, strat, self.initial_cash, self.slippage_pct, self.fee_pct
                )
                results = engine.run()
                metrics = compute_metrics(results["Equity_Realistic"])
                if metrics["sharpe"] > best_sharpe:
                    best_sharpe = metrics["sharpe"]
                    best_params = params
            except Exception:
                continue

        return best_params, best_sharpe

    def _aggregate_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        train_sharpes = [f["train_sharpe"] for f in fold_results]
        test_sharpes = [f["test_sharpe"] for f in fold_results]

        avg_train = np.mean(train_sharpes) if train_sharpes else 0
        avg_test = np.mean(test_sharpes) if test_sharpes else 0

        sharpe_decay = (
            (avg_train - avg_test) / abs(avg_train) if avg_train != 0 else 0
        )
        overfit_flag = sharpe_decay > 0.4

        return {
            "folds": fold_results,
            "avg_train_sharpe": avg_train,
            "avg_test_sharpe": avg_test,
            "sharpe_decay_pct": sharpe_decay * 100,
            "overfit_status": "DANGEROUS_OVERFIT" if overfit_flag else "OK",
        }
