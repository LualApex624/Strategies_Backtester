import pandas as pd
from strategies.base import BaseStrategy


class SMACross(BaseStrategy):
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__({"fast_period": fast_period, "slow_period": slow_period})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast = data["Close"].rolling(self.params["fast_period"]).mean()
        slow = data["Close"].rolling(self.params["slow_period"]).mean()
        signal = pd.Series(0, index=data.index, dtype=float)
        signal[fast > slow] = 1.0
        signal[fast < slow] = -1.0
        return signal

    def get_params(self):
        return self.params.copy()
