"""Public package interface for the backtester scaffold."""

from .backtesting import (
    BacktestResult,
    SingleTickerBacktester,
    UniverseBacktester,
    run_sma_cross_universe_backtest,
    simulate_sma_cross_strategy,
    summarize_backtest,
    summarize_trades_by_exit_reason,
    summarize_trades_by_ticker,
)
from .data import BacktesterDataLoader, DataPaths, TickerDataBundle
from .portfolio import (
    CapitalConstrainedPortfolioBacktester,
    PortfolioBacktestResult,
)
from .signals import (
    SignalBuildResult,
    SignalStoreBuilder,
    TrendSignalCalculator,
    TrendSignalSettings,
)
from .strategies import SmaCrossStrategy
from .universe import UniverseSnapshotLoader

__all__ = [
    "BacktestResult",
    "BacktesterDataLoader",
    "CapitalConstrainedPortfolioBacktester",
    "DataPaths",
    "PortfolioBacktestResult",
    "SignalBuildResult",
    "SignalStoreBuilder",
    "SingleTickerBacktester",
    "SmaCrossStrategy",
    "TickerDataBundle",
    "TrendSignalCalculator",
    "TrendSignalSettings",
    "UniverseSnapshotLoader",
    "UniverseBacktester",
    "run_sma_cross_universe_backtest",
    "simulate_sma_cross_strategy",
    "summarize_backtest",
    "summarize_trades_by_exit_reason",
    "summarize_trades_by_ticker",
]
