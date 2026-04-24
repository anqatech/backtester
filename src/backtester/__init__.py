"""Public package interface for the backtester scaffold."""

from .data import BacktesterDataLoader, DataPaths, TickerDataBundle
from .signals import (
    SignalBuildResult,
    SignalStoreBuilder,
    TrendSignalCalculator,
    TrendSignalSettings,
)
from .universe import UniverseSnapshotLoader

__all__ = [
    "BacktesterDataLoader",
    "DataPaths",
    "SignalBuildResult",
    "SignalStoreBuilder",
    "TickerDataBundle",
    "TrendSignalCalculator",
    "TrendSignalSettings",
    "UniverseSnapshotLoader",
]
