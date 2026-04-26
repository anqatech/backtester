"""Backtest result containers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    """Container for trade logs, daily position history, and portfolio series."""

    trades: pd.DataFrame
    positions: pd.DataFrame
    portfolio: pd.DataFrame
