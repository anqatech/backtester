"""Public package interface for the backtester scaffold."""

from .backtests import (
    BacktestResult,
    run_sma_cross_universe_backtest,
    simulate_sma_cross_strategy,
    summarize_backtest,
    summarize_trades_by_exit_reason,
    summarize_trades_by_ticker,
)
from .data import BacktesterDataLoader, DataPaths, TickerDataBundle
from .signals import (
    SignalBuildResult,
    SignalStoreBuilder,
    TrendSignalCalculator,
    TrendSignalSettings,
)
from .universe import UniverseSnapshotLoader

__all__ = [
    "BacktestResult",
    "BacktesterDataLoader",
    "DataPaths",
    "SignalBuildResult",
    "SignalStoreBuilder",
    "TickerDataBundle",
    "TrendSignalCalculator",
    "TrendSignalSettings",
    "UniverseSnapshotLoader",
    "run_sma_cross_universe_backtest",
    "simulate_sma_cross_strategy",
    "summarize_backtest",
    "summarize_trades_by_exit_reason",
    "summarize_trades_by_ticker",
]
