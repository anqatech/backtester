"""Single-ticker backtest engine."""

from __future__ import annotations

import pandas as pd

from ..strategies import SmaCrossStrategy
from ._shared import (
    PendingEntrySignal,
    PositionRecord,
    OpenTrade,
    TradeRecord,
    aggregate_portfolio_series,
    empty_backtest_result,
    finalize_trade,
    position_rows_for_trade,
    prepare_history,
    scaled_trade_notional,
    position_columns,
    trade_columns,
)
from .result import BacktestResult


class SingleTickerBacktester:
    """Stateful simulator for running a strategy against one ticker history."""

    def __init__(self, strategy: SmaCrossStrategy) -> None:
        self.strategy = strategy

    def run(self, history: pd.DataFrame, ticker: str | None = None) -> BacktestResult:
        frame = prepare_history(history=history, ticker=ticker)
        if len(frame) < 2:
            return empty_backtest_result()

        ticker_value = frame["ticker"].iloc[0]
        trades: list[TradeRecord] = []
        positions: list[PositionRecord] = []

        state = "flat"
        pending_entry_index: int | None = None
        pending_entry_signal: PendingEntrySignal | None = None
        pending_exit_index: int | None = None
        pending_exit_signal_date = pd.NaT
        open_trade: OpenTrade | None = None

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
                if current_date >= open_trade.expiry_threshold_date:
                    trades.append(
                        finalize_trade(
                            open_trade=open_trade,
                            exit_row=row,
                            exit_reason="time_exit",
                        )
                    )
                    positions.extend(
                        position_rows_for_trade(
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
                        finalize_trade(
                            open_trade=open_trade,
                            exit_row=row,
                            exit_reason="signal_exit",
                            exit_signal_date=pending_exit_signal_date,
                        )
                    )
                    positions.extend(
                        position_rows_for_trade(
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
                pending_entry_signal = PendingEntrySignal(
                    signal_date=current_date,
                    entry_signal_value=float(row["sma_50_to_sma_200"]),
                    entry_realized_vol_3m=float(row["realized_vol_3m"]),
                    entry_realized_vol_1y=float(row["realized_vol_1y"]),
                )

        if state == "long" and open_trade is not None and self.strategy.force_exit_at_end:
            final_index = len(frame) - 1
            final_row = frame.iloc[final_index]
            trades.append(
                finalize_trade(
                    open_trade=open_trade,
                    exit_row=final_row,
                    exit_reason="end_of_data",
                )
            )
            positions.extend(
                position_rows_for_trade(
                    frame=frame,
                    trade=open_trade,
                    exit_index=final_index,
                )
            )

        trades_df = pd.DataFrame(trades, columns=trade_columns())
        positions_df = pd.DataFrame(positions, columns=position_columns())
        portfolio_df = aggregate_portfolio_series(positions_df, trades_df)

        return BacktestResult(trades=trades_df, positions=positions_df, portfolio=portfolio_df)

    def _open_trade(
        self,
        ticker: str,
        signal_data: PendingEntrySignal,
        entry_row: pd.Series,
        entry_index: int,
    ) -> OpenTrade:
        entry_price = float(entry_row["close"])
        scaled_notional = scaled_trade_notional(
            base_notional=self.strategy.trade_notional,
            entry_realized_vol_3m=signal_data.entry_realized_vol_3m,
            reference_vol=self.strategy.reference_vol,
            min_notional=self.strategy.min_notional,
            max_notional=self.strategy.max_notional,
        )
        shares = scaled_notional / entry_price
        expiry_threshold_date = entry_row["date"] + pd.DateOffset(months=self.strategy.holding_period_months)
        return OpenTrade(
            ticker=ticker,
            signal_date=signal_data.signal_date,
            entry_signal_value=signal_data.entry_signal_value,
            entry_realized_vol_3m=signal_data.entry_realized_vol_3m,
            entry_realized_vol_1y=signal_data.entry_realized_vol_1y,
            entry_date=entry_row["date"],
            entry_price=entry_price,
            shares=shares,
            notional=scaled_notional,
            expiry_threshold_date=expiry_threshold_date,
            entry_index=entry_index,
        )


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
