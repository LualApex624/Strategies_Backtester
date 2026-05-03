import pandas as pd
from typing import Dict, Any, List


def compute_score(metrics: Dict[str, Any]) -> float:
    sharpe = max(metrics.get("sharpe", 0), 0)
    sortino = max(metrics.get("sortino", 0), 0)
    calmar = max(metrics.get("calmar", 0), 0)
    max_dd = abs(metrics.get("max_drawdown", 0))
    win_rate = metrics.get("win_rate", 0)

    score = (
        0.3 * sharpe
        + 0.2 * sortino
        + 0.2 * calmar
        - 0.2 * max_dd
        + 0.1 * win_rate
    )
    return round(score, 4)


def rank_strategies(
    strategy_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for entry in strategy_results:
        name = entry["name"]
        metrics = entry["metrics"]
        score = compute_score(metrics)
        rows.append(
            {
                "Strategy": name,
                "Score": score,
                "Sharpe": round(metrics.get("sharpe", 0), 3),
                "Sortino": round(metrics.get("sortino", 0), 3),
                "Calmar": round(metrics.get("calmar", 0), 3),
                "MaxDD": round(metrics.get("max_drawdown", 0), 3),
                "WinRate": round(metrics.get("win_rate", 0), 3),
                "CAGR": round(metrics.get("cagr", 0), 4),
                "Overfit": entry.get("overfit_status", "N/A"),
            }
        )

    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df
