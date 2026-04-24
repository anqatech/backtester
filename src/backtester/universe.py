"""Universe-level loaders built from per-ticker signal histories."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .data import BacktesterDataLoader


class UniverseSnapshotLoader:
    """Load cross-sectional as-of snapshots from per-ticker signal files."""

    CROSS_SECTIONAL_DEPENDENCIES = {
        "z_tsmom_3m": ["tsmom_3m"],
        "z_tsmom_6m": ["tsmom_6m"],
        "z_tsmom_1y": ["tsmom_1y"],
        "z_relmom_12_1": ["relmom_12_1"],
        "z_price_to_sma_200": ["price_to_sma_200"],
        "z_sma_50_to_sma_200": ["sma_50_to_sma_200"],
        "ma_confirm": ["price_to_sma_200", "sma_50_to_sma_200"],
        "z_ma_confirm": ["price_to_sma_200", "sma_50_to_sma_200"],
        "trend_raw": ["tsmom_3m", "tsmom_6m", "tsmom_1y"],
        "z_trend_raw": ["tsmom_3m", "tsmom_6m", "tsmom_1y"],
        "trend_composite": [
            "tsmom_3m",
            "tsmom_6m",
            "tsmom_1y",
            "relmom_12_1",
            "price_to_sma_200",
            "sma_50_to_sma_200",
        ],
    }
    NON_PERSISTED_COLUMNS = set(CROSS_SECTIONAL_DEPENDENCIES)

    def __init__(self, data_loader: BacktesterDataLoader | None = None) -> None:
        self.data_loader = data_loader or BacktesterDataLoader.from_env()

    def load_snapshot(
        self,
        date: str | pd.Timestamp,
        tickers: Iterable[str] | None = None,
        columns: Iterable[str] | None = None,
        exact_match: bool = False,
        ready_only: bool = False,
    ) -> pd.DataFrame:
        """Load one as-of universe snapshot for the requested date."""

        snapshot_date = pd.Timestamp(date).normalize()
        selected_tickers = self._resolve_tickers(tickers)
        requested_columns = list(columns) if columns is not None else None
        history_columns = self._history_columns(requested_columns, ready_only=ready_only)

        rows: list[pd.DataFrame] = []
        for ticker in selected_tickers:
            snapshot_row = self._load_ticker_snapshot(
                ticker=ticker,
                snapshot_date=snapshot_date,
                columns=history_columns,
                exact_match=exact_match,
            )
            if snapshot_row is not None:
                rows.append(snapshot_row)

        if not rows:
            empty_columns = requested_columns or ["ticker", "date"]
            return pd.DataFrame(columns=empty_columns)

        snapshot = pd.concat(rows, ignore_index=True)
        snapshot = self._add_cross_sectional_columns(snapshot=snapshot, requested_columns=requested_columns)

        if ready_only:
            snapshot = snapshot.loc[snapshot["trend_signal_ready"].fillna(False)].reset_index(drop=True)

        if requested_columns is not None:
            snapshot = snapshot.loc[:, requested_columns]

        if "ticker" in snapshot.columns:
            snapshot = snapshot.sort_values("ticker").reset_index(drop=True)

        return snapshot

    def _load_ticker_snapshot(
        self,
        ticker: str,
        snapshot_date: pd.Timestamp,
        columns: list[str] | None,
        exact_match: bool,
    ) -> pd.DataFrame | None:
        history = self.data_loader.load_signal_history(ticker, columns=columns)

        if "date" not in history.columns:
            raise KeyError(f"Signal history for {ticker} is missing the required 'date' column.")

        history = history.copy()
        history["date"] = pd.to_datetime(history["date"]).dt.normalize()
        history = history.sort_values("date").reset_index(drop=True)

        if exact_match:
            eligible = history.loc[history["date"] == snapshot_date]
        else:
            eligible = history.loc[history["date"] <= snapshot_date]

        if eligible.empty:
            return None

        return eligible.tail(1).reset_index(drop=True)

    def _resolve_tickers(self, tickers: Iterable[str] | None) -> list[str]:
        if tickers is not None:
            return self._normalize_tickers(tickers)

        universe = self.data_loader.load_universe(columns=["ticker"])
        if "ticker" not in universe.columns:
            raise KeyError("Universe file is missing the required 'ticker' column.")

        return self._normalize_tickers(universe["ticker"].tolist())

    def _normalize_tickers(self, tickers: Iterable[str]) -> list[str]:
        normalized = [
            self.data_loader.normalize_ticker(ticker)
            for ticker in tickers
            if str(ticker).strip()
        ]
        return list(dict.fromkeys(normalized))

    @classmethod
    def _history_columns(cls, columns: list[str] | None, ready_only: bool) -> list[str] | None:
        if columns is None:
            return None

        required_columns = ["ticker", "date"]
        if ready_only:
            required_columns.append("trend_signal_ready")

        expanded_columns = list(columns)
        for column in columns:
            expanded_columns.extend(cls.CROSS_SECTIONAL_DEPENDENCIES.get(column, []))

        all_columns = [*required_columns, *expanded_columns]
        persisted_columns = [
            column for column in all_columns if column not in cls.NON_PERSISTED_COLUMNS
        ]
        return list(dict.fromkeys(persisted_columns))

    @classmethod
    def _add_cross_sectional_columns(
        cls,
        snapshot: pd.DataFrame,
        requested_columns: list[str] | None,
    ) -> pd.DataFrame:
        should_compute = requested_columns is None or any(
            column in cls.CROSS_SECTIONAL_DEPENDENCIES for column in requested_columns
        )
        if not should_compute:
            return snapshot

        snapshot = snapshot.copy()

        if "tsmom_3m" in snapshot.columns:
            snapshot["z_tsmom_3m"] = cls._zscore(snapshot["tsmom_3m"])
        if "tsmom_6m" in snapshot.columns:
            snapshot["z_tsmom_6m"] = cls._zscore(snapshot["tsmom_6m"])
        if "tsmom_1y" in snapshot.columns:
            snapshot["z_tsmom_1y"] = cls._zscore(snapshot["tsmom_1y"])
        if "relmom_12_1" in snapshot.columns:
            snapshot["z_relmom_12_1"] = cls._zscore(snapshot["relmom_12_1"])
        if "price_to_sma_200" in snapshot.columns:
            snapshot["z_price_to_sma_200"] = cls._zscore(snapshot["price_to_sma_200"])
        if "sma_50_to_sma_200" in snapshot.columns:
            snapshot["z_sma_50_to_sma_200"] = cls._zscore(snapshot["sma_50_to_sma_200"])

        if {
            "z_price_to_sma_200",
            "z_sma_50_to_sma_200",
        }.issubset(snapshot.columns):
            snapshot["ma_confirm"] = (
                0.5 * snapshot["z_price_to_sma_200"] + 0.5 * snapshot["z_sma_50_to_sma_200"]
            )
            snapshot["z_ma_confirm"] = cls._zscore(snapshot["ma_confirm"])

        if {
            "z_tsmom_3m",
            "z_tsmom_6m",
            "z_tsmom_1y",
        }.issubset(snapshot.columns):
            snapshot["trend_raw"] = (
                0.25 * snapshot["z_tsmom_3m"]
                + 0.35 * snapshot["z_tsmom_6m"]
                + 0.40 * snapshot["z_tsmom_1y"]
            )
            snapshot["z_trend_raw"] = cls._zscore(snapshot["trend_raw"])

        if {
            "z_trend_raw",
            "z_relmom_12_1",
            "z_ma_confirm",
        }.issubset(snapshot.columns):
            snapshot["trend_composite"] = (
                0.50 * snapshot["z_trend_raw"]
                + 0.30 * snapshot["z_relmom_12_1"]
                + 0.20 * snapshot["z_ma_confirm"]
            )

        return snapshot

    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        std = numeric.std(ddof=1)
        if pd.isna(std) or std == 0:
            return pd.Series(pd.NA, index=series.index, dtype="Float64")
        return ((numeric - numeric.mean()) / std).astype("Float64")
