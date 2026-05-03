# Quant-Core — Backtesting Engine

Realistic backtesting engine with walk-forward optimization, overfitting detection, multi-strategy ranking and interactive Streamlit dashboard.

## Features

- **Realistic execution model** — slippage, transaction fees, and dual equity curves (naive vs. realistic)
- **Strategy library** — SMA Crossover, RSI, Bollinger Bands, Donchian Breakout, with auto-discovery of new strategies
- **Performance metrics** — Sharpe, Sortino, Calmar, Max Drawdown (amplitude + duration), Win Rate, CAGR
- **Walk-forward optimization** — sliding window train/test splits with grid search and automatic overfitting detection (40% Sharpe decay threshold)
- **Ranking engine** — weighted scoring formula to compare and rank strategies
- **Streamlit dashboard** — interactive UI with single backtest, side-by-side comparison, full ranking table, and walk-forward analysis

## Project Structure

```
project/
├── data/               # CSV cache and data loaders
├── strategies/         # One class per file, inherits BaseStrategy
├── backtester/         # Core engine and realistic execution model
├── metrics/            # Pure math: Sharpe, Sortino, Calmar, drawdowns
├── optimization/       # Walk-forward optimizer and grid search
├── ranking/            # Weighted multi-criteria scoring
├── dashboard/          # Streamlit interactive UI
├── main.py             # Single entry point (CLI + dashboard)
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline in terminal
python main.py

# Launch the interactive dashboard
python main.py --dashboard
```

## Adding a Strategy

Create a new file in `strategies/` (e.g. `strategies/macd.py`):

```python
import pandas as pd
from strategies.base import BaseStrategy

class MACD(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__({"fast": fast, "slow": slow, "signal": signal})

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["Close"]
        ema_fast = close.ewm(span=self.params["fast"]).mean()
        ema_slow = close.ewm(span=self.params["slow"]).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.params["signal"]).mean()
        signal = pd.Series(0, index=data.index, dtype=float)
        signal[macd_line > signal_line] = 1.0
        signal[macd_line < signal_line] = -1.0
        return signal

    def get_params(self):
        return self.params.copy()
```

It will automatically appear in the dashboard and CLI — no registration needed.

## Ranking Formula

```
Score = (0.3 * Sharpe) + (0.2 * Sortino) + (0.2 * Calmar) - (0.2 * MaxDD) + (0.1 * WinRate)
```

## Overfitting Detection

The walk-forward optimizer splits data into rolling train/test windows. If the average Sharpe ratio drops by more than 40% from train to test, the strategy is flagged as `DANGEROUS_OVERFIT`.

## Tech Stack

- Python 3.10+
- Pandas / NumPy (vectorized computations)
- Streamlit (dashboard)
- Plotly (interactive charts)
