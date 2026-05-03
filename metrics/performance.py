import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_metrics(
    equity: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> Dict[str, Any]:
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return _empty_metrics()

    excess = returns - risk_free_rate / periods_per_year

    mu = returns.mean() * periods_per_year
    sigma = returns.std() * np.sqrt(periods_per_year)
    sharpe = excess.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_std = downside.std() * np.sqrt(periods_per_year)
    else:
        downside_std = 0.0
    sortino = (mu - risk_free_rate) / downside_std if downside_std > 1e-9 else 0.0

    dd_series = equity / equity.cummax() - 1
    max_dd = dd_series.min()
    calmar = (mu - risk_free_rate) / abs(max_dd) if max_dd != 0 else 0.0

    dd_duration = _max_drawdown_duration(equity)

    winning = returns[returns > 0]
    win_rate = len(winning) / len(returns) if len(returns) > 0 else 0.0

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    n_years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": sigma,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_duration_days": dd_duration,
        "win_rate": win_rate,
        "num_trades": int(_count_trades(equity)),
        "annualized_return": mu,
    }


def _max_drawdown_duration(equity: pd.Series) -> int:
    peak = equity.cummax()
    in_drawdown = equity < peak
    if not in_drawdown.any():
        return 0
    groups = (~in_drawdown).cumsum()
    dd_groups = groups[in_drawdown]
    if dd_groups.empty:
        return 0
    return int(dd_groups.value_counts().max())


def _count_trades(equity: pd.Series) -> int:
    returns = equity.pct_change().fillna(0)
    sign_changes = (np.sign(returns).diff().fillna(0) != 0).sum()
    return max(sign_changes // 2, 0)


def _empty_metrics() -> Dict[str, Any]:
    return {
        "total_return": 0.0,
        "cagr": 0.0,
        "volatility": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_duration_days": 0,
        "win_rate": 0.0,
        "num_trades": 0,
        "annualized_return": 0.0,
    }
