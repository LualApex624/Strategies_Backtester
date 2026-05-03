import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class RSIStrategy(BaseStrategy):
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__(
            {"period": period, "oversold": oversold, "overbought": overbought}
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(self.params["period"]).mean()
        avg_loss = loss.rolling(self.params["period"]).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=data.index, dtype=float)
        signal[rsi < self.params["oversold"]] = 1.0
        signal[rsi > self.params["overbought"]] = -1.0
        return signal

    def get_params(self):
        return self.params.copy()
