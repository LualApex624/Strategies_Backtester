import pandas as pd
import numpy as np

from strategies.base import BaseStrategy
from backtester.execution import ExecutionModel


class BacktestEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.001,
        fee_pct: float = 0.001,
        leverage: float = 1.0,
    ):
        self.data = data.copy()
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.leverage = leverage
        self.execution = ExecutionModel(slippage_pct, fee_pct)
        self.results: pd.DataFrame | None = None

    def run(self) -> pd.DataFrame:
        signals = self.strategy.generate_signals(self.data)
        signals = signals.reindex(self.data.index).fillna(0)
        signals = signals * self.leverage

        self.results = self._compute_equity(signals)
        return self.results

    def _compute_equity(self, signals: pd.Series) -> pd.DataFrame:
        prices = self.data["Close"]

        naive = self._naive_equity(signals, prices)
        realistic = self.execution.apply_realistic_execution(
            signals, prices, self.initial_cash
        )

        result = pd.DataFrame(index=self.data.index)
        result["Close"] = prices
        result["Signal"] = signals
        result["Equity_Naive"] = naive
        result["Equity_Realistic"] = realistic["Equity_Realistic"]
        result["Position"] = realistic["Position"]
        result["Cash"] = realistic["Cash"]
        return result

    def _naive_equity(self, signals: pd.Series, prices: pd.Series) -> pd.Series:
        returns = prices.pct_change().fillna(0)
        # shift signals by 1 to avoid look-ahead
        strategy_returns = signals.shift(1).fillna(0) * returns
        equity = self.initial_cash * (1 + strategy_returns).cumprod()
        return equity
