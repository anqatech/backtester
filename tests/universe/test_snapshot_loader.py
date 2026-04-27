from pathlib import Path

import pandas as pd
import pytest

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.universe import UniverseSnapshotLoader


def test_load_snapshot_uses_previous_available_row_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)

    paths = DataPaths(
        daily_bars_dir=tmp_path / "daily-bars",
        frames_dir=tmp_path / "frames",
        universe_csv_path=universe_path,
        daily_signals_dir=tmp_path / "daily-bars-signals",
    )
    loader = BacktesterDataLoader(paths=paths)
    snapshot_loader = UniverseSnapshotLoader(data_loader=loader)

    histories = {
        "AAPL": pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL"],
                "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                "close": [100.0, 101.0],
                "trend_signal_ready": [False, True],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "date": pd.to_datetime(["2026-01-03"]),
                "close": [200.0],
                "trend_signal_ready": [True],
            }
        ),
    }

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: histories[ticker])
    snapshot = snapshot_loader.load_snapshot(
        "2026-01-04",
        columns=["ticker", "date", "close", "trend_signal_ready"],
    )

    assert snapshot["ticker"].tolist() == ["AAPL", "MSFT"]
    assert snapshot["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-03"]
    assert snapshot["close"].tolist() == [100.0, 200.0]


def test_load_snapshot_ready_only_filters_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)

    paths = DataPaths(
        daily_bars_dir=tmp_path / "daily-bars",
        frames_dir=tmp_path / "frames",
        universe_csv_path=universe_path,
        daily_signals_dir=tmp_path / "daily-bars-signals",
    )
    loader = BacktesterDataLoader(paths=paths)
    snapshot_loader = UniverseSnapshotLoader(data_loader=loader)

    histories = {
        "AAPL": pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": pd.to_datetime(["2026-01-05"]),
                "close": [101.0],
                "trend_signal_ready": [False],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "date": pd.to_datetime(["2026-01-05"]),
                "close": [201.0],
                "trend_signal_ready": [True],
            }
        ),
    }

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: histories[ticker])
    snapshot = snapshot_loader.load_snapshot(
        "2026-01-05",
        columns=["ticker", "date", "close"],
        ready_only=True,
    )

    assert snapshot["ticker"].tolist() == ["MSFT"]
    assert snapshot.columns.tolist() == ["ticker", "date", "close"]


def test_load_snapshot_exact_match_requires_exact_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL"]}).to_csv(universe_path, index=False)

    paths = DataPaths(
        daily_bars_dir=tmp_path / "daily-bars",
        frames_dir=tmp_path / "frames",
        universe_csv_path=universe_path,
        daily_signals_dir=tmp_path / "daily-bars-signals",
    )
    loader = BacktesterDataLoader(paths=paths)
    snapshot_loader = UniverseSnapshotLoader(data_loader=loader)

    history = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "date": pd.to_datetime(["2026-01-02"]),
            "close": [100.0],
        }
    )

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: history)
    snapshot = snapshot_loader.load_snapshot(
        "2026-01-03",
        columns=["ticker", "date", "close"],
        exact_match=True,
    )

    assert snapshot.empty


def test_load_snapshot_computes_cross_sectional_trend_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA"]}).to_csv(universe_path, index=False)

    paths = DataPaths(
        daily_bars_dir=tmp_path / "daily-bars",
        frames_dir=tmp_path / "frames",
        universe_csv_path=universe_path,
        daily_signals_dir=tmp_path / "daily-bars-signals",
    )
    loader = BacktesterDataLoader(paths=paths)
    snapshot_loader = UniverseSnapshotLoader(data_loader=loader)

    histories = {
        "AAPL": pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "date": pd.to_datetime(["2026-01-05"]),
                "tsmom_3m": [0.2],
                "tsmom_6m": [0.4],
                "tsmom_1y": [0.6],
                "relmom_12_1": [0.3],
                "price_to_sma_200": [1.02],
                "sma_50_to_sma_200": [1.01],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "ticker": ["MSFT"],
                "date": pd.to_datetime(["2026-01-05"]),
                "tsmom_3m": [0.1],
                "tsmom_6m": [0.2],
                "tsmom_1y": [0.1],
                "relmom_12_1": [0.15],
                "price_to_sma_200": [0.99],
                "sma_50_to_sma_200": [1.00],
            }
        ),
        "NVDA": pd.DataFrame(
            {
                "ticker": ["NVDA"],
                "date": pd.to_datetime(["2026-01-05"]),
                "tsmom_3m": [0.5],
                "tsmom_6m": [0.7],
                "tsmom_1y": [0.9],
                "relmom_12_1": [0.45],
                "price_to_sma_200": [1.10],
                "sma_50_to_sma_200": [1.06],
            }
        ),
    }

    def load_signal_history(ticker, columns=None):
        assert "trend_raw" not in columns
        assert "ma_confirm" not in columns
        assert "trend_composite" not in columns
        return histories[ticker]

    monkeypatch.setattr(loader, "load_signal_history", load_signal_history)
    snapshot = snapshot_loader.load_snapshot(
        "2026-01-05",
        columns=["ticker", "trend_raw", "trend_composite"],
    )

    assert snapshot.columns.tolist() == ["ticker", "trend_raw", "trend_composite"]
    assert snapshot["ticker"].tolist() == ["AAPL", "MSFT", "NVDA"]
    assert snapshot["trend_raw"].notna().all()
    assert snapshot["trend_composite"].notna().all()
