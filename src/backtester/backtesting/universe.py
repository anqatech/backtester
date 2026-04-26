"""Universe-level backtest engine."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..data import BacktesterDataLoader
from ..strategies import SmaCrossStrategy
from ._shared import aggregate_portfolio_series, position_columns, resolve_tickers, trade_columns
from .result import BacktestResult
from .single_ticker import SingleTickerBacktester


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
        selected_tickers = resolve_tickers(self.data_loader, tickers)

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
            else pd.DataFrame(columns=trade_columns())
        )
        positions_df = (
            pd.concat(all_positions, ignore_index=True)
            if all_positions
            else pd.DataFrame(columns=position_columns())
        )
        portfolio_df = aggregate_portfolio_series(positions_df, trades_df)

        return BacktestResult(trades=trades_df, positions=positions_df, portfolio=portfolio_df)


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
