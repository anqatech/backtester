"""Backtest summary helpers."""

from __future__ import annotations

import pandas as pd

from .common import max_drawdown
from .result import BacktestResult


def summarize_backtest(result: BacktestResult) -> pd.Series:
    """Return a compact set of headline metrics for a backtest result."""

    trades = result.trades.copy()
    portfolio = result.portfolio.copy()

    if trades.empty:
        summary = {
            "trade_count": 0,
            "winning_trade_pct": 0.0,
            "average_trade_return": 0.0,
            "median_trade_return": 0.0,
            "average_trade_pnl": 0.0,
            "total_realized_pnl": 0.0,
            "average_holding_days": 0.0,
            "median_holding_days": 0.0,
            "average_entry_notional": 0.0,
            "average_entry_realized_vol_3m": 0.0,
            "max_active_positions": 0,
            "average_active_positions": 0.0,
            "max_gross_market_value": 0.0,
            "average_gross_market_value": 0.0,
            "max_drawdown_pnl": 0.0,
            "final_total_pnl": 0.0,
        }
        return pd.Series(summary, dtype="object")

    if portfolio.empty:
        portfolio = pd.DataFrame(
            {
                "active_positions": pd.Series(dtype="int64"),
                "gross_market_value": pd.Series(dtype="float64"),
                "total_pnl": pd.Series(dtype="float64"),
            }
        )

    total_pnl = portfolio["total_pnl"] if "total_pnl" in portfolio.columns else pd.Series(dtype="float64")
    summary = {
        "trade_count": int(len(trades)),
        "winning_trade_pct": float((trades["pnl"] > 0).mean()),
        "average_trade_return": float(trades["return"].mean()),
        "median_trade_return": float(trades["return"].median()),
        "average_trade_pnl": float(trades["pnl"].mean()),
        "total_realized_pnl": float(trades["pnl"].sum()),
        "average_holding_days": float(trades["holding_days"].mean()),
        "median_holding_days": float(trades["holding_days"].median()),
        "average_entry_notional": float(trades["notional"].mean()),
        "average_entry_realized_vol_3m": float(trades["entry_realized_vol_3m"].mean()),
        "max_active_positions": int(portfolio["active_positions"].max()) if not portfolio.empty else 0,
        "average_active_positions": float(portfolio["active_positions"].mean()) if not portfolio.empty else 0.0,
        "max_gross_market_value": float(portfolio["gross_market_value"].max()) if not portfolio.empty else 0.0,
        "average_gross_market_value": float(portfolio["gross_market_value"].mean()) if not portfolio.empty else 0.0,
        "max_drawdown_pnl": float(max_drawdown(total_pnl)),
        "final_total_pnl": float(total_pnl.iloc[-1]) if not total_pnl.empty else 0.0,
    }
    return pd.Series(summary)


def summarize_trades_by_ticker(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade outcomes by ticker for quick strategy diagnostics."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "trade_count",
                "winning_trade_pct",
                "average_trade_return",
                "median_trade_return",
                "average_trade_pnl",
                "total_pnl",
                "average_holding_days",
                "average_entry_notional",
                "average_entry_realized_vol_3m",
            ]
        )

    return (
        trades.groupby("ticker", as_index=False)
        .agg(
            trade_count=("ticker", "size"),
            winning_trade_pct=("pnl", lambda values: float((values > 0).mean())),
            average_trade_return=("return", "mean"),
            median_trade_return=("return", "median"),
            average_trade_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
            average_holding_days=("holding_days", "mean"),
            average_entry_notional=("notional", "mean"),
            average_entry_realized_vol_3m=("entry_realized_vol_3m", "mean"),
        )
        .sort_values(["total_pnl", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )


def summarize_trades_by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade outcomes by exit reason for quick strategy diagnostics."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "exit_reason",
                "trade_count",
                "winning_trade_pct",
                "average_trade_return",
                "median_trade_return",
                "average_trade_pnl",
                "total_pnl",
                "average_holding_days",
            ]
        )

    return (
        trades.groupby("exit_reason", as_index=False)
        .agg(
            trade_count=("exit_reason", "size"),
            winning_trade_pct=("pnl", lambda values: float((values > 0).mean())),
            average_trade_return=("return", "mean"),
            median_trade_return=("return", "median"),
            average_trade_pnl=("pnl", "mean"),
            total_pnl=("pnl", "sum"),
            average_holding_days=("holding_days", "mean"),
        )
        .sort_values(["trade_count", "exit_reason"], ascending=[False, True])
        .reset_index(drop=True)
    )
