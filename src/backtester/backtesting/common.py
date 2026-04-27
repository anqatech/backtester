"""Shared helpers for backtest execution and aggregation."""

from __future__ import annotations

from typing import Iterable, TypeAlias

import pandas as pd

from ..data import BacktesterDataLoader
from .result import BacktestResult

TradeRecord: TypeAlias = dict[str, object]
PositionRecord: TypeAlias = dict[str, object]


def prepare_history(history: pd.DataFrame, ticker: str | None) -> pd.DataFrame:
    required_columns = {"date", "close", "sma_50_to_sma_200", "realized_vol_3m", "realized_vol_1y"}
    missing_columns = required_columns.difference(history.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"History is missing required columns: {missing}")

    frame = history.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["sma_50_to_sma_200"] = pd.to_numeric(frame["sma_50_to_sma_200"], errors="coerce")
    frame["realized_vol_3m"] = pd.to_numeric(frame["realized_vol_3m"], errors="coerce")
    frame["realized_vol_1y"] = pd.to_numeric(frame["realized_vol_1y"], errors="coerce")
    frame = frame.dropna(
        subset=["date", "close", "sma_50_to_sma_200", "realized_vol_3m", "realized_vol_1y"]
    ).sort_values("date").reset_index(drop=True)
    frame = frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    elif ticker is not None:
        frame["ticker"] = BacktesterDataLoader.normalize_ticker(ticker)
    else:
        raise KeyError("History must contain a 'ticker' column or receive ticker=...")

    if ticker is not None:
        normalized_ticker = BacktesterDataLoader.normalize_ticker(ticker)
        frame = frame.loc[frame["ticker"] == normalized_ticker].reset_index(drop=True)
        frame["ticker"] = normalized_ticker

    return frame


def scaled_trade_notional(
    base_notional: float,
    entry_realized_vol_3m: float,
    reference_vol: float,
    min_notional: float,
    max_notional: float,
) -> float:
    if reference_vol <= 0:
        raise ValueError("reference_vol must be positive.")
    if min_notional <= 0 or max_notional <= 0:
        raise ValueError("min_notional and max_notional must be positive.")
    if min_notional > max_notional:
        raise ValueError("min_notional cannot be greater than max_notional.")
    if entry_realized_vol_3m <= 0:
        raise ValueError("entry_realized_vol_3m must be positive for volatility scaling.")

    raw_notional = base_notional * (reference_vol / entry_realized_vol_3m)
    return float(min(max(raw_notional, min_notional), max_notional))


def aggregate_portfolio_series(
    positions: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if positions.empty and trades.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "active_positions",
                "gross_market_value",
                "unrealized_pnl",
                "realized_pnl",
                "cumulative_realized_pnl",
                "total_pnl",
            ]
        )

    if positions.empty:
        positions_daily = pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "active_positions": pd.Series(dtype="int64"),
                "gross_market_value": pd.Series(dtype="float64"),
                "unrealized_pnl": pd.Series(dtype="float64"),
            }
        )
    else:
        positions_daily = (
            positions.groupby("date", as_index=False)
            .agg(
                active_positions=("ticker", "nunique"),
                gross_market_value=("market_value", "sum"),
                unrealized_pnl=("unrealized_pnl", "sum"),
            )
        )

    if trades.empty:
        realized_daily = pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "realized_pnl": pd.Series(dtype="float64"),
            }
        )
    else:
        realized_daily = (
            trades.groupby("exit_date", as_index=False)
            .agg(realized_pnl=("pnl", "sum"))
            .rename(columns={"exit_date": "date"})
        )

    portfolio = positions_daily.merge(realized_daily, on="date", how="outer").sort_values("date").reset_index(drop=True)
    for column in ["active_positions", "gross_market_value", "unrealized_pnl", "realized_pnl"]:
        if column not in portfolio.columns:
            portfolio[column] = 0.0
    portfolio["active_positions"] = pd.to_numeric(
        portfolio["active_positions"], errors="coerce"
    ).fillna(0).astype("int64")
    portfolio["gross_market_value"] = pd.to_numeric(
        portfolio["gross_market_value"], errors="coerce"
    ).fillna(0.0)
    portfolio["unrealized_pnl"] = pd.to_numeric(
        portfolio["unrealized_pnl"], errors="coerce"
    ).fillna(0.0)
    portfolio["realized_pnl"] = pd.to_numeric(
        portfolio["realized_pnl"], errors="coerce"
    ).fillna(0.0)
    portfolio["cumulative_realized_pnl"] = portfolio["realized_pnl"].cumsum()
    portfolio["total_pnl"] = portfolio["cumulative_realized_pnl"] + portfolio["unrealized_pnl"]

    return portfolio


def resolve_tickers(loader: BacktesterDataLoader, tickers: Iterable[str] | None) -> list[str]:
    if tickers is None:
        universe = loader.load_universe(columns=["ticker"])
        tickers = universe["ticker"].tolist()

    normalized = [
        loader.normalize_ticker(ticker)
        for ticker in tickers
        if str(ticker).strip()
    ]
    return list(dict.fromkeys(normalized))


def trade_columns() -> list[str]:
    return [
        "ticker",
        "signal_date",
        "entry_signal_value",
        "entry_realized_vol_3m",
        "entry_realized_vol_1y",
        "entry_date",
        "entry_price",
        "shares",
        "notional",
        "expiry_threshold_date",
        "exit_signal_date",
        "exit_date",
        "exit_signal_value",
        "exit_realized_vol_3m",
        "exit_realized_vol_1y",
        "exit_price",
        "exit_reason",
        "holding_days",
        "pnl",
        "return",
    ]


def position_columns() -> list[str]:
    return [
        "ticker",
        "date",
        "entry_date",
        "shares",
        "close",
        "market_value",
        "unrealized_pnl",
    ]


def max_drawdown(total_pnl: pd.Series) -> float:
    if total_pnl.empty:
        return 0.0

    running_peak = total_pnl.cummax()
    drawdowns = running_peak - total_pnl
    return float(drawdowns.max())


def empty_backtest_result() -> BacktestResult:
    empty_trades = pd.DataFrame(columns=trade_columns())
    empty_positions = pd.DataFrame(columns=position_columns())
    empty_portfolio = aggregate_portfolio_series(empty_positions, empty_trades)
    return BacktestResult(trades=empty_trades, positions=empty_positions, portfolio=empty_portfolio)
