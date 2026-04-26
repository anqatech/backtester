"""Backtesting engines, results, and analytics."""

from ..strategies import SmaCrossStrategy
from .analytics import (
    summarize_backtest,
    summarize_trades_by_exit_reason,
    summarize_trades_by_ticker,
)
from .result import BacktestResult
from .single_ticker import SingleTickerBacktester, simulate_sma_cross_strategy
from .universe import UniverseBacktester, run_sma_cross_universe_backtest

__all__ = [
    "BacktestResult",
    "SingleTickerBacktester",
    "SmaCrossStrategy",
    "UniverseBacktester",
    "run_sma_cross_universe_backtest",
    "simulate_sma_cross_strategy",
    "summarize_backtest",
    "summarize_trades_by_exit_reason",
    "summarize_trades_by_ticker",
]
