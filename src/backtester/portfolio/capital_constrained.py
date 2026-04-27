"""Capital-constrained portfolio backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from ..backtesting.common import prepare_history, resolve_tickers, scaled_trade_notional
from ..data import BacktesterDataLoader
from ..strategies import SmaCrossStrategy


@dataclass(slots=True)
class PortfolioBacktestResult:
    """Container for capital-constrained portfolio backtest outputs."""

    trades: pd.DataFrame
    positions: pd.DataFrame
    portfolio: pd.DataFrame
    rejected_trades: pd.DataFrame


@dataclass(slots=True, frozen=True)
class PendingEntrySignal:
    """Signal captured on the decision day awaiting next-trading-day execution."""

    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    entry_signal_value: float
    entry_realized_vol_3m: float
    entry_realized_vol_1y: float


@dataclass(slots=True)
class OpenPosition:
    """Open portfolio position state for one ticker."""

    ticker: str
    signal_date: pd.Timestamp
    entry_signal_value: float
    entry_realized_vol_3m: float
    entry_realized_vol_1y: float
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    target_notional: float
    filled_notional: float
    expiry_threshold_date: pd.Timestamp
    last_close: float
    exit_signal_date: pd.Timestamp | None = None
    pending_exit_execution_date: pd.Timestamp | None = None


@dataclass(slots=True)
class TickerCursor:
    """Per-ticker traversal state while walking the market calendar."""

    frame: pd.DataFrame
    next_row_index: int = 0
    pending_entry: PendingEntrySignal | None = None
    position: OpenPosition | None = None


class CapitalConstrainedPortfolioBacktester:
    """Run one strategy across a universe subject to shared capital constraints."""

    def __init__(
        self,
        strategy: SmaCrossStrategy,
        data_loader: BacktesterDataLoader | None = None,
        initial_capital: float = 500_000.0,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")

        self.strategy = strategy
        self.data_loader = data_loader or BacktesterDataLoader.from_env()
        self.initial_capital = float(initial_capital)

    def run(self, tickers: Iterable[str] | None = None) -> PortfolioBacktestResult:
        selected_tickers = resolve_tickers(self.data_loader, tickers)
        histories = self._load_histories(selected_tickers)
        if not histories:
            return self._empty_result()

        cursors = {ticker: TickerCursor(frame=frame) for ticker, frame in histories.items()}
        all_dates = sorted(
            {
                pd.Timestamp(date).normalize()
                for frame in histories.values()
                for date in frame["date"].tolist()
            }
        )

        cash = self.initial_capital
        trades: list[dict[str, object]] = []
        positions: list[dict[str, object]] = []
        portfolio_rows: list[dict[str, object]] = []
        rejected_trades: list[dict[str, object]] = []
        cumulative_realized_pnl = 0.0

        for current_date in all_dates:
            rows_for_date = self._rows_for_date(cursors, current_date)
            realized_pnl_today = 0.0

            exiting_tickers = sorted(
                ticker
                for ticker in rows_for_date
                if self._exit_reason_for_today(cursors[ticker], current_date) is not None
            )
            for ticker in exiting_tickers:
                cursor = cursors[ticker]
                position = cursor.position
                row = rows_for_date[ticker]
                exit_reason = self._exit_reason_for_today(cursor, current_date)
                if position is None or row is None:
                    continue

                trade_record, proceeds = self._close_position(
                    position=position,
                    exit_row=row,
                    current_date=current_date,
                    exit_reason=exit_reason,
                )
                trades.append(trade_record)
                cash += proceeds
                realized_pnl_today += float(trade_record["pnl"])
                cursor.position = None

            entering_tickers = sorted(
                ticker
                for ticker in rows_for_date
                if self._should_enter_today(cursors[ticker], current_date)
            )
            for ticker in entering_tickers:
                cursor = cursors[ticker]
                pending_entry = cursor.pending_entry
                row = rows_for_date[ticker]
                if pending_entry is None or row is None:
                    continue

                target_notional = scaled_trade_notional(
                    base_notional=self.strategy.trade_notional,
                    entry_realized_vol_3m=pending_entry.entry_realized_vol_3m,
                    reference_vol=self.strategy.reference_vol,
                    min_notional=self.strategy.min_notional,
                    max_notional=self.strategy.max_notional,
                )
                cash_before = cash
                filled_notional = min(target_notional, cash)

                if filled_notional <= 0:
                    rejected_trades.append(
                        {
                            "ticker": ticker,
                            "signal_date": pending_entry.signal_date,
                            "execution_date": current_date,
                            "target_notional": target_notional,
                            "available_cash": cash_before,
                            "rejection_reason": "no_cash",
                        }
                    )
                    cursor.pending_entry = None
                    continue

                entry_price = float(row["close"])
                shares = filled_notional / entry_price
                cursor.position = OpenPosition(
                    ticker=ticker,
                    signal_date=pending_entry.signal_date,
                    entry_signal_value=pending_entry.entry_signal_value,
                    entry_realized_vol_3m=pending_entry.entry_realized_vol_3m,
                    entry_realized_vol_1y=pending_entry.entry_realized_vol_1y,
                    entry_date=current_date,
                    entry_price=entry_price,
                    shares=shares,
                    target_notional=target_notional,
                    filled_notional=filled_notional,
                    expiry_threshold_date=current_date + pd.DateOffset(months=self.strategy.holding_period_months),
                    last_close=entry_price,
                )
                cursor.pending_entry = None
                cash -= filled_notional

            for ticker, row in rows_for_date.items():
                cursor = cursors[ticker]
                if cursor.position is not None and row is not None:
                    cursor.position.last_close = float(row["close"])

            for ticker, row in rows_for_date.items():
                cursor = cursors[ticker]
                if row is None:
                    continue

                if cursor.position is not None and cursor.position.pending_exit_execution_date is None:
                    if self.strategy.exit_signal(row) and self._has_next_row(cursor):
                        next_row = cursor.frame.iloc[cursor.next_row_index + 1]
                        cursor.position.exit_signal_date = current_date
                        cursor.position.pending_exit_execution_date = pd.Timestamp(next_row["date"]).normalize()

                if cursor.position is None and cursor.pending_entry is None:
                    current_index = cursor.next_row_index
                    if current_index > 0 and self._has_next_row(cursor):
                        previous_row = cursor.frame.iloc[current_index - 1]
                        if self.strategy.entry_signal(previous_row, row):
                            next_row = cursor.frame.iloc[current_index + 1]
                            cursor.pending_entry = PendingEntrySignal(
                                signal_date=current_date,
                                execution_date=pd.Timestamp(next_row["date"]).normalize(),
                                entry_signal_value=float(row["sma_50_to_sma_200"]),
                                entry_realized_vol_3m=float(row["realized_vol_3m"]),
                                entry_realized_vol_1y=float(row["realized_vol_1y"]),
                            )

            gross_market_value = 0.0
            unrealized_pnl = 0.0
            active_positions = 0
            for ticker in sorted(cursors):
                position = cursors[ticker].position
                if position is None:
                    continue

                market_value = position.shares * position.last_close
                position_unrealized_pnl = market_value - position.filled_notional
                positions.append(
                    {
                        "ticker": ticker,
                        "date": current_date,
                        "entry_date": position.entry_date,
                        "shares": position.shares,
                        "close": position.last_close,
                        "market_value": market_value,
                        "unrealized_pnl": position_unrealized_pnl,
                        "target_notional": position.target_notional,
                        "filled_notional": position.filled_notional,
                    }
                )
                gross_market_value += market_value
                unrealized_pnl += position_unrealized_pnl
                active_positions += 1

            cumulative_realized_pnl += realized_pnl_today
            equity = cash + gross_market_value
            portfolio_rows.append(
                {
                    "date": current_date,
                    "cash": cash,
                    "gross_market_value": gross_market_value,
                    "equity": equity,
                    "active_positions": active_positions,
                    "realized_pnl": realized_pnl_today,
                    "cumulative_realized_pnl": cumulative_realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "total_pnl": equity - self.initial_capital,
                }
            )

            for ticker, row in rows_for_date.items():
                if row is not None:
                    cursors[ticker].next_row_index += 1

        return PortfolioBacktestResult(
            trades=pd.DataFrame(trades, columns=self._trade_columns()),
            positions=pd.DataFrame(positions, columns=self._position_columns()),
            portfolio=pd.DataFrame(portfolio_rows, columns=self._portfolio_columns()),
            rejected_trades=pd.DataFrame(rejected_trades, columns=self._rejected_columns()),
        )

    def _load_histories(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        columns = ["ticker", *self.strategy.required_columns()]
        histories: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            history = self.data_loader.load_signal_history(ticker, columns=columns)
            prepared = prepare_history(history=history, ticker=ticker)
            if not prepared.empty:
                histories[ticker] = prepared
        return histories

    @staticmethod
    def _rows_for_date(
        cursors: dict[str, TickerCursor],
        current_date: pd.Timestamp,
    ) -> dict[str, pd.Series | None]:
        rows: dict[str, pd.Series | None] = {}
        for ticker, cursor in cursors.items():
            if cursor.next_row_index < len(cursor.frame):
                candidate = cursor.frame.iloc[cursor.next_row_index]
                candidate_date = pd.Timestamp(candidate["date"]).normalize()
                rows[ticker] = candidate if candidate_date == current_date else None
            else:
                rows[ticker] = None
        return rows

    @staticmethod
    def _has_next_row(cursor: TickerCursor) -> bool:
        return cursor.next_row_index + 1 < len(cursor.frame)

    def _exit_reason_for_today(
        self,
        cursor: TickerCursor,
        current_date: pd.Timestamp,
    ) -> str | None:
        position = cursor.position
        if position is None:
            return None
        if current_date >= position.expiry_threshold_date:
            return "time_exit"
        if position.pending_exit_execution_date == current_date:
            return "signal_exit"
        if self.strategy.force_exit_at_end and cursor.next_row_index == len(cursor.frame) - 1:
            return "end_of_data"
        return None

    @staticmethod
    def _should_enter_today(cursor: TickerCursor, current_date: pd.Timestamp) -> bool:
        if cursor.position is not None or cursor.pending_entry is None:
            return False
        return cursor.pending_entry.execution_date == current_date

    @staticmethod
    def _close_position(
        position: OpenPosition,
        exit_row: pd.Series,
        current_date: pd.Timestamp,
        exit_reason: str | None,
    ) -> tuple[dict[str, object], float]:
        exit_price = float(exit_row["close"])
        proceeds = position.shares * exit_price
        pnl = proceeds - position.filled_notional
        if exit_reason == "time_exit":
            exit_signal_date = pd.NaT
        elif exit_reason == "signal_exit":
            exit_signal_date = position.exit_signal_date if position.exit_signal_date is not None else pd.NaT
        else:
            exit_reason = "end_of_data"
            exit_signal_date = pd.NaT

        trade_record = {
            "ticker": position.ticker,
            "signal_date": position.signal_date,
            "entry_signal_value": position.entry_signal_value,
            "entry_realized_vol_3m": position.entry_realized_vol_3m,
            "entry_realized_vol_1y": position.entry_realized_vol_1y,
            "entry_date": position.entry_date,
            "entry_price": position.entry_price,
            "shares": position.shares,
            "target_notional": position.target_notional,
            "filled_notional": position.filled_notional,
            "notional": position.filled_notional,
            "expiry_threshold_date": position.expiry_threshold_date,
            "exit_signal_date": exit_signal_date,
            "exit_date": current_date,
            "exit_signal_value": float(exit_row["sma_50_to_sma_200"]),
            "exit_realized_vol_3m": float(exit_row["realized_vol_3m"]),
            "exit_realized_vol_1y": float(exit_row["realized_vol_1y"]),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "holding_days": int((current_date - position.entry_date).days),
            "pnl": pnl,
            "return": (exit_price / position.entry_price) - 1.0,
            "entry_status": "partial" if position.filled_notional < position.target_notional else "filled",
        }
        return trade_record, proceeds

    @staticmethod
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
            "target_notional",
            "filled_notional",
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
            "entry_status",
        ]

    @staticmethod
    def _position_columns() -> list[str]:
        return [
            "ticker",
            "date",
            "entry_date",
            "shares",
            "close",
            "market_value",
            "unrealized_pnl",
            "target_notional",
            "filled_notional",
        ]

    @staticmethod
    def _portfolio_columns() -> list[str]:
        return [
            "date",
            "cash",
            "gross_market_value",
            "equity",
            "active_positions",
            "realized_pnl",
            "cumulative_realized_pnl",
            "unrealized_pnl",
            "total_pnl",
        ]

    @staticmethod
    def _rejected_columns() -> list[str]:
        return [
            "ticker",
            "signal_date",
            "execution_date",
            "target_notional",
            "available_cash",
            "rejection_reason",
        ]

    def _empty_result(self) -> PortfolioBacktestResult:
        return PortfolioBacktestResult(
            trades=pd.DataFrame(columns=self._trade_columns()),
            positions=pd.DataFrame(columns=self._position_columns()),
            portfolio=pd.DataFrame(columns=self._portfolio_columns()),
            rejected_trades=pd.DataFrame(columns=self._rejected_columns()),
        )
