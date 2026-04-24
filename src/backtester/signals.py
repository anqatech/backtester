"""Daily per-ticker trend-feature engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import BacktesterDataLoader


@dataclass(frozen=True, slots=True)
class TrendSignalSettings:
    """Configuration for per-ticker daily trend feature engineering."""

    annualization_days: int = 252
    realized_volatility_offsets: tuple[tuple[str, pd.DateOffset], ...] = field(
        default_factory=lambda: (
            ("realized_vol_1m", pd.DateOffset(months=1)),
            ("realized_vol_3m", pd.DateOffset(months=3)),
            ("realized_vol_6m", pd.DateOffset(months=6)),
            ("realized_vol_1y", pd.DateOffset(years=1)),
        )
    )
    trend_offsets: tuple[tuple[str, pd.DateOffset], ...] = field(
        default_factory=lambda: (
            ("tsmom_3m", pd.DateOffset(months=3)),
            ("tsmom_6m", pd.DateOffset(months=6)),
            ("tsmom_1y", pd.DateOffset(years=1)),
        )
    )
    relative_momentum_short_offset: pd.DateOffset = field(
        default_factory=lambda: pd.DateOffset(months=1)
    )
    relative_momentum_long_offset: pd.DateOffset = field(
        default_factory=lambda: pd.DateOffset(years=1)
    )
    realized_volatility_for_trend_offset: pd.DateOffset = field(
        default_factory=lambda: pd.DateOffset(months=3)
    )
    sma_fast_window: int = 50
    sma_slow_window: int = 200

    @property
    def annualization_factor(self) -> float:
        return float(np.sqrt(self.annualization_days))


class TrendSignalCalculator:
    """Compute daily realized volatility and trend features for one ticker."""

    def __init__(
        self,
        data_loader: BacktesterDataLoader | None = None,
        settings: TrendSignalSettings | None = None,
    ) -> None:
        self.data_loader = data_loader
        self.settings = settings or TrendSignalSettings()

    def build_enriched_price_history(self, ticker: str) -> pd.DataFrame:
        """Load one ticker's history and compute daily trend features."""

        loader = self.data_loader or BacktesterDataLoader.from_env()
        price_history = loader.load_price_history(ticker)
        return self.enrich_price_history(price_history=price_history, ticker=ticker)

    def enrich_price_history(
        self,
        price_history: pd.DataFrame,
        ticker: str | None = None,
    ) -> pd.DataFrame:
        """Return a copy of the price history with daily trend features added."""

        frame = self._prepare_price_history(price_history=price_history, ticker=ticker)
        if frame.empty:
            return frame

        frame["log_return"] = self._compute_log_returns(frame["close"])

        dates = pd.DatetimeIndex(frame["date"])
        close = frame["close"].astype("float64").to_numpy(copy=False)
        log_returns = frame["log_return"].astype("float64").to_numpy(copy=False)

        daily_sigma_3m, sigma_3m_counts, sigma_3m_ref_dates = self._compute_realized_volatility(
            dates=dates,
            log_returns=log_returns,
            offset=self.settings.realized_volatility_for_trend_offset,
            annualize=False,
        )
        frame["reference_date_sigma_3m"] = sigma_3m_ref_dates
        frame["observations_sigma_3m"] = pd.array(sigma_3m_counts, dtype="Int64")
        frame["sigma_3m"] = daily_sigma_3m

        for column_name, offset in self.settings.realized_volatility_offsets:
            realized_vol, observation_counts, reference_dates = self._compute_realized_volatility(
                dates=dates,
                log_returns=log_returns,
                offset=offset,
                annualize=True,
            )
            frame[f"reference_date_{column_name}"] = reference_dates
            frame[f"observations_{column_name}"] = pd.array(observation_counts, dtype="Int64")
            frame[column_name] = realized_vol

        for column_name, offset in self.settings.trend_offsets:
            feature_frame = self._compute_trend_momentum(
                dates=dates,
                close=close,
                daily_sigma=daily_sigma_3m,
                offset=offset,
                column_name=column_name,
            )
            for feature_name, values in feature_frame.items():
                frame[feature_name] = values

        relative_momentum = self._compute_relative_momentum(dates=dates, close=close)
        for feature_name, values in relative_momentum.items():
            frame[feature_name] = values

        frame["sma_50"] = frame["close"].rolling(
            window=self.settings.sma_fast_window,
            min_periods=self.settings.sma_fast_window,
        ).mean()
        frame["sma_200"] = frame["close"].rolling(
            window=self.settings.sma_slow_window,
            min_periods=self.settings.sma_slow_window,
        ).mean()
        frame["price_to_sma_200"] = np.where(
            frame["sma_200"] > 0,
            frame["close"] / frame["sma_200"],
            np.nan,
        )
        frame["sma_50_to_sma_200"] = np.where(
            frame["sma_200"] > 0,
            frame["sma_50"] / frame["sma_200"],
            np.nan,
        )

        frame["trend_signal_ready"] = (
            frame["tsmom_3m"].notna()
            & frame["tsmom_6m"].notna()
            & frame["tsmom_1y"].notna()
            & frame["relmom_12_1"].notna()
            & frame["price_to_sma_200"].notna()
            & frame["sma_50_to_sma_200"].notna()
        )

        return frame

    def reference_dates(
        self,
        dates: pd.Series | pd.DatetimeIndex,
        offset: pd.DateOffset,
    ) -> pd.Series:
        """Map each date to the exact-or-previous available date after applying an offset."""

        date_index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        reference_positions = self._reference_positions(date_index, offset)
        reference_dates = pd.Series(pd.NaT, index=np.arange(len(date_index)), dtype="datetime64[ns]")

        valid = reference_positions >= 0
        if valid.any():
            reference_dates.iloc[valid] = date_index.take(reference_positions[valid]).to_numpy()

        return reference_dates

    @staticmethod
    def _prepare_price_history(
        price_history: pd.DataFrame,
        ticker: str | None = None,
    ) -> pd.DataFrame:
        required_columns = {"date", "close"}
        missing_columns = required_columns.difference(price_history.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise KeyError(f"Price history is missing required columns: {missing}")

        frame = price_history.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        frame = frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

        if ticker is not None or "ticker" not in frame.columns:
            normalized_ticker = BacktesterDataLoader.normalize_ticker(
                ticker if ticker is not None else frame.get("ticker", pd.Series(["UNKNOWN"])).iloc[0]
            )
            frame["ticker"] = normalized_ticker
        else:
            frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()

        return frame

    @staticmethod
    def _compute_log_returns(close: pd.Series) -> pd.Series:
        positive_close = close.where(close > 0)
        return np.log(positive_close / positive_close.shift(1))

    @staticmethod
    def _reference_positions(dates: pd.DatetimeIndex, offset: pd.DateOffset) -> np.ndarray:
        target_dates = (dates - offset).asi8
        available_dates = dates.asi8
        positions = np.searchsorted(available_dates, target_dates, side="right") - 1
        return positions.astype(np.int64, copy=False)

    def _compute_realized_volatility(
        self,
        dates: pd.DatetimeIndex,
        log_returns: np.ndarray,
        offset: pd.DateOffset,
        annualize: bool,
    ) -> tuple[pd.Series, np.ndarray, pd.Series]:
        reference_positions = self._reference_positions(dates, offset)
        end_positions = np.arange(len(dates), dtype=np.int64)
        observation_counts, sample_std = self._window_sample_std(
            values=log_returns,
            start_positions=reference_positions + 1,
            end_positions=end_positions,
        )

        result = np.full(len(dates), np.nan, dtype=np.float64)
        valid = observation_counts >= 2
        result[valid] = sample_std[valid]
        if annualize:
            result[valid] = result[valid] * self.settings.annualization_factor

        reference_dates = pd.Series(pd.NaT, index=np.arange(len(dates)), dtype="datetime64[ns]")
        valid_reference = reference_positions >= 0
        if valid_reference.any():
            reference_dates.iloc[valid_reference] = dates.take(reference_positions[valid_reference]).to_numpy()

        return pd.Series(result, index=np.arange(len(dates)), dtype="float64"), observation_counts, reference_dates

    def _compute_trend_momentum(
        self,
        dates: pd.DatetimeIndex,
        close: np.ndarray,
        daily_sigma: pd.Series,
        offset: pd.DateOffset,
        column_name: str,
    ) -> dict[str, pd.Series]:
        reference_positions = self._reference_positions(dates, offset)
        reference_dates = pd.Series(pd.NaT, index=np.arange(len(dates)), dtype="datetime64[ns]")
        reference_close = np.full(len(dates), np.nan, dtype=np.float64)
        observation_counts = np.zeros(len(dates), dtype=np.int64)
        values = np.full(len(dates), np.nan, dtype=np.float64)

        valid_reference = reference_positions >= 0
        if valid_reference.any():
            reference_dates.iloc[valid_reference] = dates.take(reference_positions[valid_reference]).to_numpy()
            reference_close[valid_reference] = close[reference_positions[valid_reference]]
            observation_counts[valid_reference] = (
                np.arange(len(dates), dtype=np.int64)[valid_reference] - reference_positions[valid_reference]
            )

        valid_signal = (
            valid_reference
            & np.isfinite(close)
            & np.isfinite(reference_close)
            & (close > 0)
            & (reference_close > 0)
            & daily_sigma.notna().to_numpy()
            & (daily_sigma.to_numpy() > 0)
            & (observation_counts > 0)
        )
        if valid_signal.any():
            values[valid_signal] = (
                np.log(close[valid_signal] / reference_close[valid_signal])
                / (daily_sigma.to_numpy()[valid_signal] * np.sqrt(observation_counts[valid_signal]))
            )

        return {
            f"reference_date_{column_name}": reference_dates,
            f"reference_close_{column_name}": pd.Series(reference_close, dtype="float64"),
            f"observations_{column_name}": pd.Series(
                pd.array(observation_counts, dtype="Int64")
            ),
            column_name: pd.Series(values, dtype="float64"),
        }

    def _compute_relative_momentum(
        self,
        dates: pd.DatetimeIndex,
        close: np.ndarray,
    ) -> dict[str, pd.Series]:
        short_positions = self._reference_positions(dates, self.settings.relative_momentum_short_offset)
        long_positions = self._reference_positions(dates, self.settings.relative_momentum_long_offset)

        short_dates = pd.Series(pd.NaT, index=np.arange(len(dates)), dtype="datetime64[ns]")
        long_dates = pd.Series(pd.NaT, index=np.arange(len(dates)), dtype="datetime64[ns]")
        short_close = np.full(len(dates), np.nan, dtype=np.float64)
        long_close = np.full(len(dates), np.nan, dtype=np.float64)
        values = np.full(len(dates), np.nan, dtype=np.float64)

        valid_short = short_positions >= 0
        valid_long = long_positions >= 0
        if valid_short.any():
            short_dates.iloc[valid_short] = dates.take(short_positions[valid_short]).to_numpy()
            short_close[valid_short] = close[short_positions[valid_short]]
        if valid_long.any():
            long_dates.iloc[valid_long] = dates.take(long_positions[valid_long]).to_numpy()
            long_close[valid_long] = close[long_positions[valid_long]]

        valid_signal = (
            valid_short
            & valid_long
            & np.isfinite(short_close)
            & np.isfinite(long_close)
            & (short_close > 0)
            & (long_close > 0)
        )
        if valid_signal.any():
            values[valid_signal] = np.log(short_close[valid_signal] / long_close[valid_signal])

        return {
            "reference_date_relmom_1m": short_dates,
            "reference_date_relmom_1y": long_dates,
            "reference_close_relmom_1m": pd.Series(short_close, dtype="float64"),
            "reference_close_relmom_1y": pd.Series(long_close, dtype="float64"),
            "relmom_12_1": pd.Series(values, dtype="float64"),
        }

    @staticmethod
    def _window_sample_std(
        values: np.ndarray,
        start_positions: np.ndarray,
        end_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        valid_values = np.isfinite(values)
        clean_values = np.where(valid_values, values, 0.0)
        clean_values_sq = clean_values * clean_values

        prefix_count = np.concatenate(([0], valid_values.astype(np.int64).cumsum()))
        prefix_sum = np.concatenate(([0.0], clean_values.cumsum()))
        prefix_sum_sq = np.concatenate(([0.0], clean_values_sq.cumsum()))

        counts = np.zeros(len(values), dtype=np.int64)
        sample_std = np.full(len(values), np.nan, dtype=np.float64)

        valid_windows = (start_positions >= 0) & (start_positions <= end_positions)
        if not valid_windows.any():
            return counts, sample_std

        window_starts = start_positions[valid_windows]
        window_ends = end_positions[valid_windows] + 1

        counts[valid_windows] = prefix_count[window_ends] - prefix_count[window_starts]
        valid_stats = valid_windows & (counts >= 2)
        if not valid_stats.any():
            return counts, sample_std

        stats_starts = start_positions[valid_stats]
        stats_ends = end_positions[valid_stats] + 1
        stats_counts = counts[valid_stats].astype(np.float64)
        sums = prefix_sum[stats_ends] - prefix_sum[stats_starts]
        sums_sq = prefix_sum_sq[stats_ends] - prefix_sum_sq[stats_starts]

        numerator = sums_sq - (sums * sums / stats_counts)
        numerator = np.maximum(numerator, 0.0)
        sample_std[valid_stats] = np.sqrt(numerator / (stats_counts - 1.0))

        return counts, sample_std


@dataclass(frozen=True, slots=True)
class SignalBuildResult:
    """Outcome of building one enriched per-ticker signal file."""

    ticker: str
    output_path: Path
    rows_written: int
    skipped: bool = False


class SignalStoreBuilder:
    """Build and save enriched per-ticker signal histories."""

    def __init__(
        self,
        data_loader: BacktesterDataLoader | None = None,
        calculator: TrendSignalCalculator | None = None,
    ) -> None:
        self.data_loader = data_loader or BacktesterDataLoader.from_env()
        self.calculator = calculator or TrendSignalCalculator(data_loader=self.data_loader)

    def build_ticker(
        self,
        ticker: str,
        overwrite: bool = False,
    ) -> SignalBuildResult:
        """Build and save the enriched signal history for one ticker."""

        normalized_ticker = self.data_loader.normalize_ticker(ticker)
        output_path = self.data_loader.paths.signal_history_path(normalized_ticker)

        if output_path.exists() and not overwrite:
            return SignalBuildResult(
                ticker=normalized_ticker,
                output_path=output_path,
                rows_written=0,
                skipped=True,
            )

        enriched = self.calculator.build_enriched_price_history(normalized_ticker)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(output_path, index=False)

        return SignalBuildResult(
            ticker=normalized_ticker,
            output_path=output_path,
            rows_written=len(enriched),
            skipped=False,
        )

    def build_universe(
        self,
        tickers: Iterable[str] | None = None,
        overwrite: bool = False,
    ) -> list[SignalBuildResult]:
        """Build and save enriched signal histories for a set of universe tickers."""

        selected_tickers = self._resolve_tickers(tickers)
        return [
            self.build_ticker(ticker=ticker, overwrite=overwrite)
            for ticker in selected_tickers
        ]

    def _resolve_tickers(self, tickers: Iterable[str] | None) -> list[str]:
        if tickers is not None:
            return self._normalize_ticker_list(tickers)

        universe = self.data_loader.load_universe(columns=["ticker"])
        if "ticker" not in universe.columns:
            raise KeyError("Universe file is missing the required 'ticker' column.")

        return self._normalize_ticker_list(universe["ticker"].tolist())

    def _normalize_ticker_list(self, tickers: Iterable[str]) -> list[str]:
        normalized = [
            self.data_loader.normalize_ticker(ticker)
            for ticker in tickers
            if str(ticker).strip()
        ]
        return list(dict.fromkeys(normalized))
