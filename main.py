#!/usr/bin/env python3
"""QuantCore Backtesting Engine."""

import sys
import argparse
import pandas as pd

from data.loader import generate_sample_data
from strategies.registry import discover_strategies
from backtester.engine import BacktestEngine
from metrics.performance import compute_metrics
from ranking.scorer import rank_strategies, compute_score
from optimization.walk_forward import WalkForwardOptimizer


PARAM_GRIDS = {
    "SMACross": {"fast_period": [10, 20, 30], "slow_period": [40, 50, 60, 80]},
    "RSIStrategy": {"period": [10, 14, 21], "oversold": [20, 30], "overbought": [70, 80]},
    "BollingerBands": {"period": [15, 20, 30], "num_std": [1.5, 2.0, 2.5]},
    "Breakout": {"lookback": [10, 20, 30, 50]},
}


def run_full_pipeline():
    print("=" * 60)
    print("  QUANTCORE Moteur de Backtesting")
    print("=" * 60)

    data = generate_sample_data()
    strategies = discover_strategies()
    print(f"\nStratégies détectées : {list(strategies.keys())}")

    all_results = []

    for name, cls in strategies.items():
        print(f"\n{'─' * 40}")
        print(f"  Stratégie : {name}")
        print(f"{'─' * 40}")

        strat = cls()
        engine = BacktestEngine(data, strat)
        results = engine.run()

        m_naive = compute_metrics(results["Equity_Naive"])
        m_real = compute_metrics(results["Equity_Realistic"])

        print(f"  Sharpe (naïf)     : {m_naive['sharpe']:.3f}")
        print(f"  Sharpe (réaliste) : {m_real['sharpe']:.3f}")
        print(f"  Max Drawdown      : {m_real['max_drawdown']:.2%}")
        print(f"  CAGR              : {m_real['cagr']:.2%}")
        print(f"  Win Rate          : {m_real['win_rate']:.2%}")

        grid = PARAM_GRIDS.get(name)
        overfit_status = "N/A"
        if grid:
            print(f"  Walk-Forward optimisation...")
            optimizer = WalkForwardOptimizer(data, strat, grid, n_splits=5)
            wf = optimizer.run()
            overfit_status = wf["overfit_status"]
            print(f"  Train Sharpe moy. : {wf['avg_train_sharpe']:.3f}")
            print(f"  Test Sharpe moy.  : {wf['avg_test_sharpe']:.3f}")
            print(f"  Déclin Sharpe     : {wf['sharpe_decay_pct']:.1f}%")
            print(f"  Statut            : {overfit_status}")

        all_results.append({
            "name": name,
            "metrics": m_real,
            "overfit_status": overfit_status,
        })

    print(f"\n{'=' * 60}")
    print("  RANKING FINAL")
    print(f"{'=' * 60}")
    ranking = rank_strategies(all_results)
    print(ranking.to_string())

    print(f"\n{'=' * 60}")
    print("  RAPPORT")
    print(f"{'=' * 60}")
    print("J'ai construit un moteur de backtesting réaliste avec")
    print("optimisation walk-forward, détection d'overfitting et")
    print("ranking automatique de stratégies.")
    print(f"{'=' * 60}")


def run_dashboard():
    import subprocess
    dashboard_path = str(__import__("pathlib").Path(__file__).parent / "dashboard" / "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])


def main():
    parser = argparse.ArgumentParser(description="Quant-Core Backtesting Engine")
    parser.add_argument("--dashboard", action="store_true", help="Lancer le dashboard Streamlit")
    args = parser.parse_args()

    if args.dashboard:
        run_dashboard()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
