import numpy as np
import pandas as pd


class ExecutionModel:
    def __init__(self, slippage_pct: float = 0.001, fee_pct: float = 0.001):
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct

    def execution_price(self, market_price: float, direction: int) -> float:
        return market_price * (1 + direction * self.slippage_pct)

    def transaction_cost(self, notional: float) -> float:
        return abs(notional) * self.fee_pct

    def apply_realistic_execution(
        self, signals: pd.Series, prices: pd.Series, initial_cash: float = 100_000.0
    ) -> pd.DataFrame:
        """Signals are target allocation fractions: 1.0 = 100% long, -1.0 = 100% short."""
        n = len(prices)
        cash = np.full(n, np.nan)
        shares = np.full(n, 0.0)
        equity = np.full(n, np.nan)

        cash[0] = initial_cash
        shares[0] = 0.0
        equity[0] = initial_cash

        sig = signals.values
        px = prices.values

        for i in range(1, n):
            prev_shares = shares[i - 1]
            prev_cash = cash[i - 1]
            prev_equity = prev_cash + prev_shares * px[i]

            target_alloc = sig[i - 1]  # previous bar signal to avoid look-ahead
            target_notional = target_alloc * prev_equity
            target_shares = target_notional / px[i]
            trade_shares = target_shares - prev_shares

            if abs(trade_shares) > 1e-9:
                direction = 1 if trade_shares > 0 else -1
                exec_price = self.execution_price(px[i], direction)
                trade_notional = abs(trade_shares) * exec_price
                cost = self.transaction_cost(trade_notional)
                cash[i] = prev_cash - trade_shares * exec_price - cost
            else:
                cash[i] = prev_cash

            shares[i] = target_shares
            equity[i] = cash[i] + shares[i] * px[i]

        return pd.DataFrame(
            {"Cash": cash, "Position": shares, "Equity_Realistic": equity},
            index=prices.index,
        )
