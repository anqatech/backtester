"""Dataset loading helpers for the local parquet-backed backtester."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Filesystem locations for the local market-data parquet files."""

    daily_bars_dir: Path
    frames_dir: Path
    universe_csv_path: Path
    daily_signals_dir: Path
    database_status_filename: str = "daily-bars-database-status-with-market-cap.parquet"

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "DataPaths":
        resolved_env_file = cls._resolve_env_file(env_file)
        if resolved_env_file is not None and resolved_env_file.exists():
            from dotenv import load_dotenv

            load_dotenv(resolved_env_file, override=False)

        daily_bars_dir = os.environ.get("BACKTESTER_DAILY_BARS_DIR")
        frames_dir = os.environ.get("BACKTESTER_FRAMES_DIR")
        universe_csv_path = os.environ.get("BACKTESTER_UNIVERSE_CSV_PATH")
        daily_signals_dir = os.environ.get("BACKTESTER_DAILY_SIGNALS_DIR")
        missing = [
            variable_name
            for variable_name, value in (
                ("BACKTESTER_DAILY_BARS_DIR", daily_bars_dir),
                ("BACKTESTER_FRAMES_DIR", frames_dir),
                ("BACKTESTER_UNIVERSE_CSV_PATH", universe_csv_path),
                ("BACKTESTER_DAILY_SIGNALS_DIR", daily_signals_dir),
            )
            if not value
        ]

        if missing:
            missing_variables = ", ".join(missing)
            raise ValueError(
                f"Missing required environment variables: {missing_variables}. "
                "Set them in the project .env file or in the shell environment."
            )

        return cls(
            daily_bars_dir=Path(daily_bars_dir).expanduser(),
            frames_dir=Path(frames_dir).expanduser(),
            universe_csv_path=Path(universe_csv_path).expanduser(),
            daily_signals_dir=Path(daily_signals_dir).expanduser(),
        )

    @property
    def database_status_path(self) -> Path:
        return self.frames_dir / self.database_status_filename

    def price_history_path(self, ticker: str) -> Path:
        return self.daily_bars_dir / f"{BacktesterDataLoader.normalize_ticker(ticker)}.parquet"

    def signal_history_path(self, ticker: str) -> Path:
        return self.daily_signals_dir / f"{BacktesterDataLoader.normalize_ticker(ticker)}.parquet"

    @staticmethod
    def _resolve_env_file(env_file: str | Path | None) -> Path | None:
        if env_file is not None:
            return Path(env_file).expanduser()

        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / ".env"
        if candidate.exists():
            return candidate
        return None


@dataclass(slots=True)
class TickerDataBundle:
    """Ticker-specific dataframes returned by the loader."""

    ticker: str
    price_history: pd.DataFrame
    database_status: pd.DataFrame


class BacktesterDataLoader:
    """Load local parquet datasets for a specific ticker."""

    def __init__(self, paths: DataPaths | None = None) -> None:
        self.paths = paths or DataPaths.from_env()

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "BacktesterDataLoader":
        return cls(paths=DataPaths.from_env(env_file=env_file))

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

    def load_signal_history(
        self,
        ticker: str,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        path = self.paths.signal_history_path(ticker)
        self._ensure_exists(path)
        return pd.read_parquet(path, columns=self._as_list(columns))

    def load_universe(
        self,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        path = self.paths.universe_csv_path
        self._ensure_exists(path)
        return pd.read_csv(path, usecols=self._as_list(columns))

    def load_database_status(
        self,
        ticker: str | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        return self._load_shared_frame(
            path=self.paths.database_status_path,
            ticker=ticker,
            columns=columns,
        )

    def load_ticker_bundle(
        self,
        ticker: str,
        price_columns: Iterable[str] | None = None,
        status_columns: Iterable[str] | None = None,
    ) -> TickerDataBundle:
        normalized_ticker = self.normalize_ticker(ticker)
        return TickerDataBundle(
            ticker=normalized_ticker,
            price_history=self.load_price_history(
                normalized_ticker,
                columns=price_columns,
            ),
            database_status=self.load_database_status(
                normalized_ticker,
                columns=status_columns,
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
