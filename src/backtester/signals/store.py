"""Signal store building and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..data import BacktesterDataLoader
from .features import TrendSignalCalculator


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
