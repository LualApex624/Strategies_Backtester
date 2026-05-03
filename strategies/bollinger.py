import pandas as pd
from strategies.base import BaseStrategy


class BollingerBands(BaseStrategy):
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__({"period": period, "num_std": num_std})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        ma = close.rolling(self.params["period"]).mean()
        std = close.rolling(self.params["period"]).std()
        upper = ma + self.params["num_std"] * std
        lower = ma - self.params["num_std"] * std

        signal = pd.Series(0, index=data.index, dtype=float)
        signal[close < lower] = 1.0
        signal[close > upper] = -1.0
        return signal

    def get_params(self):
        return self.params.copy()
