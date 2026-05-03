import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.loader import generate_sample_data, load_csv
from strategies.registry import discover_strategies
from backtester.engine import BacktestEngine
from metrics.performance import compute_metrics
from ranking.scorer import rank_strategies, compute_score
from optimization.walk_forward import WalkForwardOptimizer

st.set_page_config(page_title="QuantCore Backtester", layout="wide")
st.title("QuantCore Moteur de Backtesting")

PARAM_GRIDS = {
    "SMACross": {"fast_period": [10, 20, 30], "slow_period": [40, 50, 60, 80]},
    "RSIStrategy": {"period": [10, 14, 21], "oversold": [20, 30], "overbought": [70, 80]},
    "BollingerBands": {"period": [15, 20, 30], "num_std": [1.5, 2.0, 2.5]},
    "Breakout": {"lookback": [10, 20, 30, 50]},
}


@st.cache_data
def get_data(source: str, ticker: str, filepath: str | None):
    if source == "Données synthétiques":
        return generate_sample_data(ticker=ticker)
    return load_csv(filepath)


@st.cache_data
def run_backtest(_strategy_cls, params, data_json, cash, slip, fee, leverage):
    data = pd.read_json(data_json)
    strat = _strategy_cls(**params)
    engine = BacktestEngine(data, strat, cash, slip, fee, leverage)
    results = engine.run()
    metrics_naive = compute_metrics(results["Equity_Naive"])
    metrics_real = compute_metrics(results["Equity_Realistic"])
    return results, metrics_naive, metrics_real


# --- Sidebar ---
st.sidebar.header("Configuration")

data_source = st.sidebar.radio("Source de données", ["Données synthétiques", "Fichier CSV"])
csv_path = None
if data_source == "Fichier CSV":
    csv_path = st.sidebar.text_input("Chemin du fichier CSV")

initial_cash = st.sidebar.number_input("Capital initial", value=100_000, step=10_000)
slippage = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.1, 0.01) / 100
fees = st.sidebar.slider("Frais (%)", 0.0, 1.0, 0.1, 0.01) / 100
leverage = st.sidebar.slider("Levier", 0.5, 5.0, 1.0, 0.1)

data = get_data(data_source, "SAMPLE", csv_path)

st.sidebar.subheader("Filtre de dates")
min_date = data.index.min().date()
max_date = data.index.max().date()
date_range = st.sidebar.date_input("Période", [min_date, max_date], min_value=min_date, max_value=max_date)
if len(date_range) == 2:
    data = data.loc[str(date_range[0]) : str(date_range[1])]

strategies = discover_strategies()

# --- Tab layout ---
tab_single, tab_compare, tab_ranking, tab_wf = st.tabs(
    ["Backtest unique", "Comparaison", "Ranking", "Walk-Forward"]
)

with tab_single:
    strat_name = st.selectbox("Stratégie", list(strategies.keys()), key="single")
    strat_cls = strategies[strat_name]
    strat_instance = strat_cls()
    default_params = strat_instance.get_params()

    st.subheader("Paramètres")
    user_params = {}
    cols = st.columns(len(default_params))
    for i, (k, v) in enumerate(default_params.items()):
        with cols[i]:
            if isinstance(v, int):
                user_params[k] = st.number_input(k, value=v, step=1, key=f"s_{k}")
            else:
                user_params[k] = st.number_input(k, value=float(v), step=0.1, key=f"s_{k}")

    if st.button("Lancer le backtest", key="run_single"):
        data_json = data.to_json()
        results, m_naive, m_real = run_backtest(
            strat_cls, user_params, data_json, initial_cash, slippage, fees, leverage
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe (réaliste)", f"{m_real['sharpe']:.3f}")
        col2.metric("Sortino", f"{m_real['sortino']:.3f}")
        col3.metric("Max Drawdown", f"{m_real['max_drawdown']:.2%}")
        col4.metric("CAGR", f"{m_real['cagr']:.2%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.index, y=results["Equity_Naive"], name="Equity Naive", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=results.index, y=results["Equity_Realistic"], name="Equity Réaliste"))
        fig.update_layout(title="Courbes d'Equity", xaxis_title="Date", yaxis_title="Valeur (€)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        dd = results["Equity_Realistic"] / results["Equity_Realistic"].cummax() - 1
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown"))
        fig_dd.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", height=300)
        st.plotly_chart(fig_dd, use_container_width=True)

        st.subheader("Métriques détaillées")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.write("**Naïf (sans coûts)**")
            st.json(m_naive)
        with mcol2:
            st.write("**Réaliste (avec slippage + frais)**")
            st.json(m_real)

with tab_compare:
    col_a, col_b = st.columns(2)
    with col_a:
        name_a = st.selectbox("Stratégie A", list(strategies.keys()), key="cmp_a")
    with col_b:
        name_b = st.selectbox("Stratégie B", list(strategies.keys()), key="cmp_b", index=1)

    if st.button("Comparer", key="run_compare"):
        data_json = data.to_json()
        res_a, _, ma = run_backtest(strategies[name_a], strategies[name_a]().get_params(), data_json, initial_cash, slippage, fees, leverage)
        res_b, _, mb = run_backtest(strategies[name_b], strategies[name_b]().get_params(), data_json, initial_cash, slippage, fees, leverage)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=res_a.index, y=res_a["Equity_Realistic"], name=name_a), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_b.index, y=res_b["Equity_Realistic"], name=name_b), row=1, col=1)

        dd_a = res_a["Equity_Realistic"] / res_a["Equity_Realistic"].cummax() - 1
        dd_b = res_b["Equity_Realistic"] / res_b["Equity_Realistic"].cummax() - 1
        fig.add_trace(go.Scatter(x=dd_a.index, y=dd_a, name=f"DD {name_a}", fill="tozeroy"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dd_b.index, y=dd_b, name=f"DD {name_b}", fill="tozeroy"), row=2, col=1)
        fig.update_layout(height=700, title="Comparaison côte à côte")
        st.plotly_chart(fig, use_container_width=True)

        cmp = pd.DataFrame({"Métrique": list(ma.keys()), name_a: list(ma.values()), name_b: list(mb.values())})
        st.dataframe(cmp, use_container_width=True)

with tab_ranking:
    if st.button("Classer toutes les stratégies", key="run_rank"):
        all_results = []
        data_json = data.to_json()
        for sname, scls in strategies.items():
            _, _, m = run_backtest(scls, scls().get_params(), data_json, initial_cash, slippage, fees, leverage)
            all_results.append({"name": sname, "metrics": m})

        ranking_df = rank_strategies(all_results)
        st.dataframe(ranking_df, use_container_width=True)

        fig = go.Figure(go.Bar(x=ranking_df["Strategy"], y=ranking_df["Score"], marker_color=["#2ecc71" if s > 0 else "#e74c3c" for s in ranking_df["Score"]]))
        fig.update_layout(title="Score par stratégie", xaxis_title="Stratégie", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

with tab_wf:
    wf_strat = st.selectbox("Stratégie à optimiser", list(strategies.keys()), key="wf_strat")
    n_folds = st.slider("Nombre de folds", 3, 10, 5)

    if st.button("Lancer Walk-Forward", key="run_wf"):
        strat_cls = strategies[wf_strat]
        grid = PARAM_GRIDS.get(wf_strat, strat_cls().get_params())

        grid_for_search = {}
        for k, v in grid.items():
            grid_for_search[k] = v if isinstance(v, list) else [v]

        with st.spinner("Optimisation en cours..."):
            optimizer = WalkForwardOptimizer(
                data, strat_cls(), grid_for_search, n_splits=n_folds,
                initial_cash=initial_cash, slippage_pct=slippage, fee_pct=fees,
            )
            wf_results = optimizer.run()

        status_color = "🔴" if wf_results["overfit_status"] == "DANGEROUS_OVERFIT" else "🟢"
        st.subheader(f"{status_color} Statut : {wf_results['overfit_status']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe moyen (Train)", f"{wf_results['avg_train_sharpe']:.3f}")
        col2.metric("Sharpe moyen (Test)", f"{wf_results['avg_test_sharpe']:.3f}")
        col3.metric("Déclin Sharpe (%)", f"{wf_results['sharpe_decay_pct']:.1f}%")

        fold_df = pd.DataFrame([
            {"Fold": f["fold"] + 1, "Params": str(f["best_params"]),
             "Sharpe Train": round(f["train_sharpe"], 3), "Sharpe Test": round(f["test_sharpe"], 3)}
            for f in wf_results["folds"]
        ])
        st.dataframe(fold_df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Train", x=fold_df["Fold"], y=fold_df["Sharpe Train"], marker_color="#3498db"))
        fig.add_trace(go.Bar(name="Test", x=fold_df["Fold"], y=fold_df["Sharpe Test"], marker_color="#e67e22"))
        fig.update_layout(barmode="group", title="Sharpe par Fold (Train vs Test)")
        st.plotly_chart(fig, use_container_width=True)
