from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.analytics.common import build_insight, build_result

try:
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
except ImportError:  # pragma: no cover
    EfficientFrontier = None
    expected_returns = None
    risk_models = None


def calculate_returns(
    df: pd.DataFrame,
    price_col: str,
    entity_col: str | None = None,
    simple: bool = True,
) -> dict[str, Any]:
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataset.")

    working_df = df.copy()
    if entity_col and entity_col in working_df.columns:
        returns = working_df.groupby(entity_col)[price_col].pct_change()
    else:
        returns = working_df[price_col].pct_change()

    if not simple:
        returns = np.log1p(returns)

    clean_returns = returns.dropna()
    if clean_returns.empty:
        return build_result(insights=["No valid return observations available."])

    mean_return = float(clean_returns.mean())
    insights = [f"Computed {len(clean_returns)} return observations with mean return {mean_return:.4f}."]
    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="calculate_returns",
                title="Return summary",
                message="Return series computed from price history.",
                evidence={"observations": len(clean_returns), "mean_return": mean_return},
            )
        ],
        charts=[{"type": "line", "title": f"Returns for {price_col}"}],
        data={"returns": clean_returns.round(6).tolist(), "mean_return": mean_return},
    )


def measure_risk(
    df: pd.DataFrame,
    price_col: str,
    benchmark_col: str | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataset.")

    returns = df[price_col].pct_change().dropna()
    if returns.empty:
        return build_result(insights=["Risk analysis skipped because no returns could be computed."])

    volatility = float(returns.std(ddof=1) * np.sqrt(252))
    excess_mean = float((returns.mean() - risk_free_rate / 252) * 252)
    sharpe = float(excess_mean / volatility) if volatility else 0.0

    beta = None
    correlation = None
    if benchmark_col and benchmark_col in df.columns:
        benchmark_returns = df[benchmark_col].pct_change().dropna()
        merged = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if not merged.empty:
            correlation = float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))
            benchmark_variance = float(merged.iloc[:, 1].var(ddof=1))
            if benchmark_variance:
                beta = float(merged.iloc[:, 0].cov(merged.iloc[:, 1]) / benchmark_variance)

    insights = [f"Annualized volatility is {volatility:.2%} with Sharpe ratio {sharpe:.2f}."]
    if beta is not None:
        insights.append(f"Beta versus '{benchmark_col}' is {beta:.2f} with correlation {correlation:.2f}.")

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="measure_risk",
                title="Risk metrics",
                message="Risk metrics computed from historical returns.",
                evidence={"volatility": volatility, "sharpe_ratio": sharpe, "beta": beta, "correlation": correlation},
            )
        ],
        data={"volatility": volatility, "sharpe_ratio": sharpe, "beta": beta, "correlation": correlation},
    )


def measure_drawdown(df: pd.DataFrame, price_col: str) -> dict[str, Any]:
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in dataset.")

    prices = df[price_col].astype(float)
    running_max = prices.cummax()
    drawdown = prices / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    trough_index = int(drawdown.idxmin()) if not drawdown.empty else None

    insights = [f"Maximum drawdown for '{price_col}' is {max_drawdown:.2%}."]
    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="measure_drawdown",
                title="Drawdown summary",
                message="Drawdown path computed from price history.",
                evidence={"max_drawdown": max_drawdown, "trough_index": trough_index},
            )
        ],
        charts=[{"type": "line", "title": f"Drawdown for {price_col}"}],
        data={"drawdown": drawdown.round(6).tolist(), "max_drawdown": max_drawdown, "trough_index": trough_index},
    )


def detect_volume_spikes(
    df: pd.DataFrame,
    volume_col: str,
    date_col: str | None = None,
    z_threshold: float = 2.0,
) -> dict[str, Any]:
    if volume_col not in df.columns:
        raise ValueError(f"Volume column '{volume_col}' not found in dataset.")

    working_df = df[[volume_col] + ([date_col] if date_col and date_col in df.columns else [])].copy()
    volumes = working_df[volume_col].astype(float)
    std = volumes.std(ddof=1)
    if not std:
        return build_result(insights=["Volume spike detection skipped because volume has no variation."])

    z_scores = (volumes - volumes.mean()) / std
    spike_df = working_df.loc[z_scores.abs() >= z_threshold].copy()
    if spike_df.empty:
        return build_result(insights=[f"No volume spikes detected using z-threshold {z_threshold}."], data={"spikes": []})

    rows = []
    insights = []
    for index, row in spike_df.iterrows():
        label = str(row[date_col]) if date_col and date_col in spike_df.columns else str(index)
        value = float(row[volume_col])
        z_score = float(z_scores.loc[index])
        rows.append({"label": label, "volume": value, "z_score": z_score})
        insights.append(f"Detected volume spike at {label} with volume {value:.2f} (z={z_score:.2f}).")

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="detect_volume_spikes",
                title="Volume spike summary",
                message="Abnormal volume observations detected.",
                evidence={"spikes": rows, "threshold": z_threshold},
            )
        ],
        data={"spikes": rows},
    )


def optimize_portfolio(
    df: pd.DataFrame,
    entity_col: str,
    price_col: str,
    date_col: str,
) -> dict[str, Any]:
    for column in (entity_col, price_col, date_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    working_df = df[[entity_col, price_col, date_col]].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    price_matrix = (
        working_df.dropna()
        .pivot_table(index=date_col, columns=entity_col, values=price_col, aggfunc="last")
        .sort_index()
        .dropna(axis=1, how="all")
        .ffill()
        .dropna()
    )

    if price_matrix.shape[1] < 2:
        return build_result(
            insights=["Portfolio optimization skipped because fewer than two assets had usable price histories."],
            diagnostics=["At least two assets are required for portfolio optimization."],
        )

    diagnostics: list[str] = []
    if EfficientFrontier is not None and expected_returns is not None and risk_models is not None:
        mu = expected_returns.mean_historical_return(price_matrix)
        sigma = risk_models.sample_cov(price_matrix)
        optimizer = EfficientFrontier(mu, sigma)
        optimizer.max_sharpe()
        cleaned = {asset: float(weight) for asset, weight in optimizer.clean_weights().items() if weight}
    else:
        diagnostics.append("PyPortfolioOpt not installed; used inverse-volatility fallback allocation.")
        returns = price_matrix.pct_change().dropna()
        vol = returns.std(ddof=1).replace(0, np.nan).dropna()
        inverse = 1 / vol
        cleaned = {asset: float(weight) for asset, weight in (inverse / inverse.sum()).to_dict().items()}

    insights = [f"Generated portfolio weights across {len(cleaned)} assets."]
    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="optimize_portfolio",
                title="Portfolio allocation",
                message="Portfolio optimization completed.",
                evidence={"weights": cleaned},
            )
        ],
        diagnostics=diagnostics,
        charts=[{"type": "pie", "title": "Portfolio weights"}],
        data={"weights": cleaned},
    )


def backtest_signal(
    df: pd.DataFrame,
    price_col: str,
    signal_col: str,
) -> dict[str, Any]:
    for column in (price_col, signal_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    working_df = df[[price_col, signal_col]].dropna().copy()
    if len(working_df) < 2:
        return build_result(insights=["Backtest skipped because fewer than two rows were available."])

    returns = working_df[price_col].pct_change().fillna(0.0)
    shifted_signal = working_df[signal_col].shift(1).fillna(0.0)
    strategy_returns = returns * shifted_signal
    equity_curve = (1 + strategy_returns).cumprod()
    total_return = float(equity_curve.iloc[-1] - 1)
    win_rate = float((strategy_returns > 0).mean())

    return build_result(
        insights=[f"Backtest total return is {total_return:.2%} with win rate {win_rate:.1%}."],
        insight_objects=[
            build_insight(
                tool="backtest_signal",
                title="Backtest summary",
                message="Naive single-signal backtest completed.",
                evidence={"total_return": total_return, "win_rate": win_rate},
            )
        ],
        charts=[{"type": "line", "title": "Strategy equity curve"}],
        data={"equity_curve": equity_curve.round(6).tolist(), "total_return": total_return, "win_rate": win_rate},
    )
