"""SMA-cross strategy definitions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
