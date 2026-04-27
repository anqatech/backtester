from pathlib import Path

import pandas as pd
import pytest

from backtester.backtesting import UniverseBacktester, run_sma_cross_universe_backtest
from backtester.data import BacktesterDataLoader, DataPaths
from backtester.strategies import SmaCrossStrategy


def test_universe_runner_combines_trades_across_tickers(
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

    histories = {
        "AAPL": pd.DataFrame(
            {
                "ticker": ["AAPL"] * 6,
                "date": pd.bdate_range("2026-01-05", periods=6),
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98, 0.97, 0.96],
                "realized_vol_3m": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
                "realized_vol_1y": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "ticker": ["MSFT"] * 6,
                "date": pd.bdate_range("2026-01-05", periods=6),
                "close": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
                "sma_50_to_sma_200": [0.95, 0.96, 1.01, 1.02, 1.03, 0.98],
                "realized_vol_3m": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
                "realized_vol_1y": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
            }
        ),
    }

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: histories[ticker])
    result = run_sma_cross_universe_backtest(data_loader=loader)

    assert sorted(result.trades["ticker"].tolist()) == ["AAPL", "MSFT"]
    assert "active_positions" in result.portfolio.columns
    assert "total_pnl" in result.portfolio.columns


def test_universe_backtester_class_runs_with_loader(
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
    strategy = SmaCrossStrategy()

    histories = {
        "AAPL": pd.DataFrame(
            {
                "ticker": ["AAPL"] * 6,
                "date": pd.bdate_range("2026-01-05", periods=6),
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98, 0.97, 0.96],
                "realized_vol_3m": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
                "realized_vol_1y": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "ticker": ["MSFT"] * 6,
                "date": pd.bdate_range("2026-01-05", periods=6),
                "close": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
                "sma_50_to_sma_200": [0.95, 0.96, 1.01, 1.02, 1.03, 0.98],
                "realized_vol_3m": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
                "realized_vol_1y": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
            }
        ),
    }

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: histories[ticker])
    result = UniverseBacktester(strategy=strategy, data_loader=loader).run()

    assert sorted(result.trades["ticker"].tolist()) == ["AAPL", "MSFT"]
    assert "active_positions" in result.portfolio.columns
