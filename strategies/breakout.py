import pandas as pd
from strategies.base import BaseStrategy


class Breakout(BaseStrategy):
    def __init__(self, lookback: int = 20):
        super().__init__({"lookback": lookback})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        lb = self.params["lookback"]
        high_max = data["High"].rolling(lb).max()
        low_min = data["Low"].rolling(lb).min()
        close = data["Close"]

        signal = pd.Series(0, index=data.index, dtype=float)
        signal[close >= high_max] = 1.0
        signal[close <= low_min] = -1.0
        return signal

    def get_params(self):
        return self.params.copy()
