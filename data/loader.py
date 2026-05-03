import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent


def load_csv(filepath: str, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.set_index(date_col)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df


def generate_sample_data(
    ticker: str = "SAMPLE",
    days: int = 2000,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp("2026-04-30"), periods=days)
    returns = rng.normal(0.0003, 0.015, size=days)
    prices = start_price * np.exp(np.cumsum(returns))

    high_noise = np.abs(rng.normal(0, 0.005, size=days))
    low_noise = np.abs(rng.normal(0, 0.005, size=days))
    volume = rng.integers(100_000, 5_000_000, size=days)

    df = pd.DataFrame(
        {
            "Open": prices * (1 + rng.normal(0, 0.002, size=days)),
            "High": prices * (1 + high_noise),
            "Low": prices * (1 - low_noise),
            "Close": prices,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    cache_path = DATA_DIR / f"{ticker}.csv"
    df.to_csv(cache_path)
    return df
