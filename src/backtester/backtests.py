"""Simple backtest runners for early strategy validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .data import BacktesterDataLoader


@dataclass(slots=True)
class BacktestResult:
    """Container for trade logs, daily position history, and portfolio series."""

    trades: pd.DataFrame
    positions: pd.DataFrame
    portfolio: pd.DataFrame


@dataclass(slots=True, frozen=True)
class SmaCrossStrategy:
    """Parameter bundle and rule definitions for the first SMA-cross strategy."""

    trade_notional: float = 10_000.0
    reference_vol: float = 0.35
    min_notional: float = 5_000.0
    max_notional: float = 20_000.0
    entry_threshold: float = 1.0
    exit_threshold: float = 0.99
    holding_period_months: int = 1
    force_exit_at_end: bool = True

    def required_columns(self) -> list[str]:
        return [
            "date",
            "close",
            "sma_50_to_sma_200",
            "realized_vol_3m",
            "realized_vol_1y",
        ]

    def entry_signal(self, previous_row: pd.Series, row: pd.Series) -> bool:
        return (
            float(previous_row["sma_50_to_sma_200"]) < self.entry_threshold
            and float(row["sma_50_to_sma_200"]) >= self.entry_threshold
        )

    def exit_signal(self, row: pd.Series) -> bool:
        return float(row["sma_50_to_sma_200"]) < self.exit_threshold

    def position_notional(self, entry_realized_vol_3m: float) -> float:
        return _scaled_trade_notional(
            base_notional=self.trade_notional,
            entry_realized_vol_3m=entry_realized_vol_3m,
            reference_vol=self.reference_vol,
            min_notional=self.min_notional,
            max_notional=self.max_notional,
        )


class SingleTickerBacktester:
    """Stateful simulator for running a strategy against one ticker history."""

    def __init__(self, strategy: SmaCrossStrategy) -> None:
        self.strategy = strategy

    def run(self, history: pd.DataFrame, ticker: str | None = None) -> BacktestResult:
        frame = _prepare_history(history=history, ticker=ticker)
        if len(frame) < 2:
            return _empty_backtest_result()

        ticker_value = frame["ticker"].iloc[0]
        trades: list[dict] = []
        positions: list[dict] = []

        state = "flat"
        pending_entry_index: int | None = None
        pending_entry_signal: dict | None = None
        pending_exit_index: int | None = None
        pending_exit_signal_date = pd.NaT
        open_trade: dict | None = None

        for index in range(1, len(frame)):
            row = frame.iloc[index]
            previous_row = frame.iloc[index - 1]
            current_date = row["date"]

            if state == "flat" and pending_entry_index == index and pending_entry_signal is not None:
                open_trade = self._open_trade(
                    ticker=ticker_value,
                    signal_data=pending_entry_signal,
                    entry_row=row,
                    entry_index=index,
                )
                state = "long"
                pending_entry_index = None
                pending_entry_signal = None

            if state == "long" and open_trade is not None:
                if current_date >= open_trade["expiry_threshold_date"]:
                    trades.append(
                        _finalize_trade(
                            open_trade=open_trade,
                            exit_row=row,
                            exit_reason="time_exit",
                        )
                    )
                    positions.extend(
                        _position_rows_for_trade(
                            frame=frame,
                            trade=open_trade,
                            exit_index=index,
                        )
                    )
                    open_trade = None
                    state = "flat"
                    pending_exit_index = None
                    pending_exit_signal_date = pd.NaT
                elif pending_exit_index == index:
                    trades.append(
                        _finalize_trade(
                            open_trade=open_trade,
                            exit_row=row,
                            exit_reason="signal_exit",
                            exit_signal_date=pending_exit_signal_date,
                        )
                    )
                    positions.extend(
                        _position_rows_for_trade(
                            frame=frame,
                            trade=open_trade,
                            exit_index=index,
                        )
                    )
                    open_trade = None
                    state = "flat"
                    pending_exit_index = None
                    pending_exit_signal_date = pd.NaT
                elif self.strategy.exit_signal(row) and index + 1 < len(frame):
                    pending_exit_index = index + 1
                    pending_exit_signal_date = current_date

            if (
                state == "flat"
                and pending_entry_index is None
                and self.strategy.entry_signal(previous_row, row)
                and index + 1 < len(frame)
            ):
                pending_entry_index = index + 1
                pending_entry_signal = {
                    "signal_date": current_date,
                    "entry_signal_value": float(row["sma_50_to_sma_200"]),
                    "entry_realized_vol_3m": float(row["realized_vol_3m"]),
                    "entry_realized_vol_1y": float(row["realized_vol_1y"]),
                }

        if state == "long" and open_trade is not None and self.strategy.force_exit_at_end:
            final_index = len(frame) - 1
            final_row = frame.iloc[final_index]
            trades.append(
                _finalize_trade(
                    open_trade=open_trade,
                    exit_row=final_row,
                    exit_reason="end_of_data",
                )
            )
            positions.extend(
                _position_rows_for_trade(
                    frame=frame,
                    trade=open_trade,
                    exit_index=final_index,
                )
            )

        trades_df = pd.DataFrame(trades, columns=_trade_columns())
        positions_df = pd.DataFrame(positions, columns=_position_columns())
        portfolio_df = _aggregate_portfolio_series(positions_df, trades_df)

        return BacktestResult(trades=trades_df, positions=positions_df, portfolio=portfolio_df)

    def _open_trade(
        self,
        ticker: str,
        signal_data: dict,
        entry_row: pd.Series,
        entry_index: int,
    ) -> dict:
        entry_price = float(entry_row["close"])
        scaled_notional = self.strategy.position_notional(signal_data["entry_realized_vol_3m"])
        shares = scaled_notional / entry_price
        expiry_threshold_date = entry_row["date"] + pd.DateOffset(months=self.strategy.holding_period_months)
        return {
            "ticker": ticker,
            "signal_date": signal_data["signal_date"],
            "entry_signal_value": signal_data["entry_signal_value"],
            "entry_realized_vol_3m": signal_data["entry_realized_vol_3m"],
            "entry_realized_vol_1y": signal_data["entry_realized_vol_1y"],
            "entry_date": entry_row["date"],
            "entry_price": entry_price,
            "shares": shares,
            "notional": scaled_notional,
            "expiry_threshold_date": expiry_threshold_date,
            "exit_signal_date": pd.NaT,
            "exit_date": pd.NaT,
            "exit_price": pd.NA,
            "exit_reason": pd.NA,
            "_entry_index": entry_index,
        }


class UniverseBacktester:
    """Coordinator for running one strategy across a ticker universe."""

    def __init__(
        self,
        strategy: SmaCrossStrategy,
        data_loader: BacktesterDataLoader | None = None,
    ) -> None:
        self.strategy = strategy
        self.data_loader = data_loader or BacktesterDataLoader.from_env()
        self.single_ticker_backtester = SingleTickerBacktester(strategy)

    def run(self, tickers: Iterable[str] | None = None) -> BacktestResult:
        selected_tickers = _resolve_tickers(self.data_loader, tickers)

        all_trades: list[pd.DataFrame] = []
        all_positions: list[pd.DataFrame] = []
        columns = ["ticker", *self.strategy.required_columns()]
        for ticker in selected_tickers:
            history = self.data_loader.load_signal_history(ticker, columns=columns)
            result = self.single_ticker_backtester.run(history=history, ticker=ticker)
            if not result.trades.empty:
                all_trades.append(result.trades)
            if not result.positions.empty:
                all_positions.append(result.positions)

        trades_df = (
            pd.concat(all_trades, ignore_index=True)
            if all_trades
            else pd.DataFrame(columns=_trade_columns())
        )
        positions_df = (
            pd.concat(all_positions, ignore_index=True)
            if all_positions
            else pd.DataFrame(columns=_position_columns())
        )
        portfolio_df = _aggregate_portfolio_series(positions_df, trades_df)

        return BacktestResult(trades=trades_df, positions=positions_df, portfolio=portfolio_df)


def simulate_sma_cross_strategy(
    history: pd.DataFrame,
    ticker: str | None = None,
    trade_notional: float = 10_000.0,
    reference_vol: float = 0.35,
    min_notional: float = 5_000.0,
    max_notional: float = 20_000.0,
    entry_threshold: float = 1.0,
    exit_threshold: float = 0.99,
    holding_period_months: int = 1,
    force_exit_at_end: bool = True,
) -> BacktestResult:
    """Simulate the first SMA cross strategy for one ticker history."""

    strategy = SmaCrossStrategy(
        trade_notional=trade_notional,
        reference_vol=reference_vol,
        min_notional=min_notional,
        max_notional=max_notional,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        holding_period_months=holding_period_months,
        force_exit_at_end=force_exit_at_end,
    )
    return SingleTickerBacktester(strategy).run(history=history, ticker=ticker)


def run_sma_cross_universe_backtest(
    data_loader: BacktesterDataLoader | None = None,
    tickers: Iterable[str] | None = None,
    trade_notional: float = 10_000.0,
    reference_vol: float = 0.35,
    min_notional: float = 5_000.0,
    max_notional: float = 20_000.0,
    entry_threshold: float = 1.0,
    exit_threshold: float = 0.99,
    holding_period_months: int = 1,
    force_exit_at_end: bool = True,
) -> BacktestResult:
    """Run the SMA cross strategy independently across a ticker universe."""

    strategy = SmaCrossStrategy(
        trade_notional=trade_notional,
        reference_vol=reference_vol,
        min_notional=min_notional,
        max_notional=max_notional,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        holding_period_months=holding_period_months,
        force_exit_at_end=force_exit_at_end,
    )
    return UniverseBacktester(strategy=strategy, data_loader=data_loader).run(tickers=tickers)


def _empty_backtest_result() -> BacktestResult:
    empty_trades = pd.DataFrame(columns=_trade_columns())
    empty_positions = pd.DataFrame(columns=_position_columns())
    empty_portfolio = _aggregate_portfolio_series(empty_positions, empty_trades)
    return BacktestResult(trades=empty_trades, positions=empty_positions, portfolio=empty_portfolio)


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
    max_drawdown_pnl = _max_drawdown(total_pnl)

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
        "max_drawdown_pnl": float(max_drawdown_pnl),
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

    summary = (
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
    return summary


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

    summary = (
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
    return summary


def _prepare_history(history: pd.DataFrame, ticker: str | None) -> pd.DataFrame:
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


def _finalize_trade(
    open_trade: dict,
    exit_row: pd.Series,
    exit_reason: str,
    exit_signal_date: pd.Timestamp | None = None,
) -> dict:
    exit_price = float(exit_row["close"])
    pnl = open_trade["shares"] * (exit_price - open_trade["entry_price"])
    return {
        "ticker": open_trade["ticker"],
        "signal_date": open_trade["signal_date"],
        "entry_signal_value": open_trade["entry_signal_value"],
        "entry_realized_vol_3m": open_trade["entry_realized_vol_3m"],
        "entry_realized_vol_1y": open_trade["entry_realized_vol_1y"],
        "entry_date": open_trade["entry_date"],
        "entry_price": open_trade["entry_price"],
        "shares": open_trade["shares"],
        "notional": open_trade["notional"],
        "expiry_threshold_date": open_trade["expiry_threshold_date"],
        "exit_signal_date": exit_signal_date if exit_signal_date is not None else pd.NaT,
        "exit_date": exit_row["date"],
        "exit_signal_value": float(exit_row["sma_50_to_sma_200"]),
        "exit_realized_vol_3m": float(exit_row["realized_vol_3m"]),
        "exit_realized_vol_1y": float(exit_row["realized_vol_1y"]),
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_days": int((exit_row["date"] - open_trade["entry_date"]).days),
        "pnl": pnl,
        "return": (exit_price / open_trade["entry_price"]) - 1.0,
    }


def _scaled_trade_notional(
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


def _position_rows_for_trade(
    frame: pd.DataFrame,
    trade: dict,
    exit_index: int,
) -> list[dict]:
    position_rows: list[dict] = []
    for position_index in range(trade["_entry_index"], exit_index):
        row = frame.iloc[position_index]
        market_value = trade["shares"] * float(row["close"])
        position_rows.append(
            {
                "ticker": trade["ticker"],
                "date": row["date"],
                "entry_date": trade["entry_date"],
                "shares": trade["shares"],
                "close": float(row["close"]),
                "market_value": market_value,
                "unrealized_pnl": market_value - trade["notional"],
            }
        )
    return position_rows


def _aggregate_portfolio_series(
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
        positions_daily = pd.DataFrame(columns=["date", "active_positions", "gross_market_value", "unrealized_pnl"])
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
        realized_daily = pd.DataFrame(columns=["date", "realized_pnl"])
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
    portfolio["active_positions"] = portfolio["active_positions"].fillna(0).astype("int64")
    portfolio["gross_market_value"] = portfolio["gross_market_value"].fillna(0.0)
    portfolio["unrealized_pnl"] = portfolio["unrealized_pnl"].fillna(0.0)
    portfolio["realized_pnl"] = portfolio["realized_pnl"].fillna(0.0)
    portfolio["cumulative_realized_pnl"] = portfolio["realized_pnl"].cumsum()
    portfolio["total_pnl"] = portfolio["cumulative_realized_pnl"] + portfolio["unrealized_pnl"]

    return portfolio


def _max_drawdown(total_pnl: pd.Series) -> float:
    if total_pnl.empty:
        return 0.0

    running_peak = total_pnl.cummax()
    drawdowns = running_peak - total_pnl
    return float(drawdowns.max())


def _resolve_tickers(loader: BacktesterDataLoader, tickers: Iterable[str] | None) -> list[str]:
    if tickers is None:
        universe = loader.load_universe(columns=["ticker"])
        tickers = universe["ticker"].tolist()

    normalized = [
        loader.normalize_ticker(ticker)
        for ticker in tickers
        if str(ticker).strip()
    ]
    return list(dict.fromkeys(normalized))


def _trade_columns() -> list[str]:
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


def _position_columns() -> list[str]:
    return [
        "ticker",
        "date",
        "entry_date",
        "shares",
        "close",
        "market_value",
        "unrealized_pnl",
    ]
