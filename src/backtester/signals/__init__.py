"""Signal feature engineering and persistence."""

from .features import TrendSignalCalculator, TrendSignalSettings
from .store import SignalBuildResult, SignalStoreBuilder

__all__ = [
    "SignalBuildResult",
    "SignalStoreBuilder",
    "TrendSignalCalculator",
    "TrendSignalSettings",
]
