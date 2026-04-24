"""Dataset loading helpers for the local parquet-backed backtester."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_DAILY_BARS_DIR = Path(
    "/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/daily-bars"
)
DEFAULT_FRAMES_DIR = Path(
    "/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/frames"
)
TREND_SIGNALS_FILENAME = "daily-bars-trend-signals.parquet"
REALIZED_VOLATILITY_FILENAME = "daily-bars-realized-volatility.parquet"


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Filesystem locations for the local market-data parquet files."""

    daily_bars_dir: Path
    frames_dir: Path

    @classmethod
    def from_env(cls) -> "DataPaths":
        return cls(
            daily_bars_dir=Path(
                os.environ.get("BACKTESTER_DAILY_BARS_DIR", DEFAULT_DAILY_BARS_DIR)
            ),
            frames_dir=Path(
                os.environ.get("BACKTESTER_FRAMES_DIR", DEFAULT_FRAMES_DIR)
            ),
        )

    @property
    def trend_signals_path(self) -> Path:
        return self.frames_dir / TREND_SIGNALS_FILENAME

    @property
    def realized_volatility_path(self) -> Path:
        return self.frames_dir / REALIZED_VOLATILITY_FILENAME

    def price_history_path(self, ticker: str) -> Path:
        return self.daily_bars_dir / f"{BacktesterDataLoader.normalize_ticker(ticker)}.parquet"


@dataclass(slots=True)
class TickerDataBundle:
    """Ticker-specific dataframes returned by the loader."""

    ticker: str
    price_history: pd.DataFrame
    trend_signals: pd.DataFrame
    realized_volatility: pd.DataFrame


class BacktesterDataLoader:
    """Load local parquet datasets for a specific ticker."""

    def __init__(self, paths: DataPaths | None = None) -> None:
        self.paths = paths or DataPaths.from_env()

    @classmethod
    def from_env(cls) -> "BacktesterDataLoader":
        return cls(paths=DataPaths.from_env())

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        normalized = str(ticker).strip().upper()
        if not normalized:
            raise ValueError("Ticker must be a non-empty string.")
        return normalized

    def load_price_history(
        self,
        ticker: str,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        path = self.paths.price_history_path(ticker)
        self._ensure_exists(path)
        return pd.read_parquet(path, columns=self._as_list(columns))

    def load_trend_signals(
        self,
        ticker: str | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self._load_shared_frame(
            path=self.paths.trend_signals_path,
            ticker=ticker,
            columns=columns,
        )

    def load_realized_volatility(
        self,
        ticker: str | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self._load_shared_frame(
            path=self.paths.realized_volatility_path,
            ticker=ticker,
            columns=columns,
        )

    def load_ticker_bundle(
        self,
        ticker: str,
        price_columns: Iterable[str] | None = None,
        frame_columns: Iterable[str] | None = None,
    ) -> TickerDataBundle:
        normalized_ticker = self.normalize_ticker(ticker)
        return TickerDataBundle(
            ticker=normalized_ticker,
            price_history=self.load_price_history(
                normalized_ticker,
                columns=price_columns,
            ),
            trend_signals=self.load_trend_signals(
                normalized_ticker,
                columns=frame_columns,
            ),
            realized_volatility=self.load_realized_volatility(
                normalized_ticker,
                columns=frame_columns,
            ),
        )

    def _load_shared_frame(
        self,
        path: Path,
        ticker: str | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        self._ensure_exists(path)

        requested_columns = self._as_list(columns)
        parquet_columns = self._columns_for_shared_frame(requested_columns, ticker)
        frame = pd.read_parquet(path, columns=parquet_columns)

        if ticker is None:
            return frame

        normalized_ticker = self.normalize_ticker(ticker)
        if "ticker" not in frame.columns:
            raise KeyError(f"Expected a 'ticker' column in {path}.")

        mask = frame["ticker"].astype(str).str.upper().str.strip() == normalized_ticker
        filtered = frame.loc[mask].reset_index(drop=True)

        if requested_columns is not None and "ticker" not in requested_columns:
            filtered = filtered.loc[:, requested_columns]

        return filtered

    @staticmethod
    def _ensure_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

    @staticmethod
    def _as_list(columns: Iterable[str] | None) -> list[str] | None:
        if columns is None:
            return None
        return list(columns)

    @staticmethod
    def _columns_for_shared_frame(
        requested_columns: list[str] | None,
        ticker: str | None,
    ) -> list[str] | None:
        if requested_columns is None:
            return None
        if ticker is None or "ticker" in requested_columns:
            return requested_columns
        return ["ticker", *requested_columns]
