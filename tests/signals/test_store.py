from pathlib import Path

import pandas as pd
import pytest

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.signals import SignalStoreBuilder, TrendSignalCalculator


def test_build_ticker_saves_to_expected_parquet_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = DataPaths(
        daily_bars_dir=tmp_path / "daily-bars",
        frames_dir=tmp_path / "frames",
        universe_csv_path=tmp_path / "tickers_enriched.csv",
        daily_signals_dir=tmp_path / "daily-bars-signals",
    )
    loader = BacktesterDataLoader(paths=paths)
    calculator = TrendSignalCalculator(data_loader=loader)
    builder = SignalStoreBuilder(data_loader=loader, calculator=calculator)

    enriched = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": pd.to_datetime(["2026-01-02"]),
            "close": [100.0],
        }
    )

    build_calls: list[str] = []
    parquet_calls: list[tuple[Path, str]] = []

    def fake_build_enriched_price_history(ticker: str) -> pd.DataFrame:
        build_calls.append(ticker)
        return enriched

    def fake_to_parquet(self: pd.DataFrame, path: Path, *args, **kwargs) -> None:
        parquet_calls.append((path, self.iloc[0]["ticker"]))

    monkeypatch.setattr(
        builder.calculator,
        "build_enriched_price_history",
        fake_build_enriched_price_history,
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    result = builder.build_ticker("aapl", overwrite=True)

    assert build_calls == ["AAPL"]
    assert parquet_calls == [(paths.daily_signals_dir / "AAPL.parquet", "AAPL")]
    assert result.ticker == "AAPL"
    assert result.output_path == paths.daily_signals_dir / "AAPL.parquet"
    assert result.rows_written == 1
    assert not result.skipped


def test_build_universe_reads_tickers_from_universe_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT", "AAPL"]}).to_csv(universe_path, index=False)

    paths = DataPaths(
        daily_bars_dir=tmp_path / "daily-bars",
        frames_dir=tmp_path / "frames",
        universe_csv_path=universe_path,
        daily_signals_dir=tmp_path / "daily-bars-signals",
    )
    loader = BacktesterDataLoader(paths=paths)
    builder = SignalStoreBuilder(
        data_loader=loader,
        calculator=TrendSignalCalculator(data_loader=loader),
    )

    build_calls: list[tuple[str, bool]] = []

    def fake_build_ticker(*, ticker: str, overwrite: bool):
        build_calls.append((ticker, overwrite))
        return None

    monkeypatch.setattr(builder, "build_ticker", fake_build_ticker)

    builder.build_universe(overwrite=False)

    assert build_calls == [("AAPL", False), ("MSFT", False)]
