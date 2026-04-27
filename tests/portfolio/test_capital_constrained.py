from pathlib import Path

import pandas as pd
import pytest

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.portfolio import CapitalConstrainedPortfolioBacktester
from backtester.strategies import SmaCrossStrategy


def test_full_fill_with_sufficient_cash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    history = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 4,
            "date": pd.bdate_range("2026-01-05", periods=4),
            "close": [100.0, 101.0, 102.0, 103.0],
            "sma_50_to_sma_200": [0.98, 1.02, 1.03, 1.03],
            "realized_vol_3m": [0.35, 0.35, 0.35, 0.35],
            "realized_vol_1y": [0.40, 0.40, 0.40, 0.40],
        }
    )

    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL"]}).to_csv(universe_path, index=False)
    loader = BacktesterDataLoader(
        paths=DataPaths(
            daily_bars_dir=tmp_path / "daily-bars",
            frames_dir=tmp_path / "frames",
            universe_csv_path=universe_path,
            daily_signals_dir=tmp_path / "daily-bars-signals",
        )
    )
    strategy = SmaCrossStrategy()
    backtester = CapitalConstrainedPortfolioBacktester(
        strategy=strategy,
        data_loader=loader,
        initial_capital=500_000.0,
    )

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: history)
    result = backtester.run(["AAPL"])

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["target_notional"] == 10_000.0
    assert trade["filled_notional"] == 10_000.0
    assert trade["entry_status"] == "filled"
    assert result.rejected_trades.empty


def test_partial_fill_when_cash_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)
    loader = BacktesterDataLoader(
        paths=DataPaths(
            daily_bars_dir=tmp_path / "daily-bars",
            frames_dir=tmp_path / "frames",
            universe_csv_path=universe_path,
            daily_signals_dir=tmp_path / "daily-bars-signals",
        )
    )
    strategy = SmaCrossStrategy()
    backtester = CapitalConstrainedPortfolioBacktester(
        strategy=strategy,
        data_loader=loader,
        initial_capital=15_000.0,
    )

    histories = {
        "AAPL": pd.DataFrame(
            {
                "ticker": ["AAPL"] * 4,
                "date": pd.bdate_range("2026-01-05", periods=4),
                "close": [100.0, 101.0, 102.0, 103.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 1.03],
                "realized_vol_3m": [0.35, 0.35, 0.35, 0.35],
                "realized_vol_1y": [0.40, 0.40, 0.40, 0.40],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "ticker": ["MSFT"] * 4,
                "date": pd.bdate_range("2026-01-05", periods=4),
                "close": [200.0, 201.0, 202.0, 203.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 1.03],
                "realized_vol_3m": [0.35, 0.35, 0.35, 0.35],
                "realized_vol_1y": [0.40, 0.40, 0.40, 0.40],
            }
        ),
    }

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: histories[ticker])
    result = backtester.run(["AAPL", "MSFT"])

    assert len(result.trades) == 2
    trade_aapl = result.trades.loc[result.trades["ticker"] == "AAPL"].iloc[0]
    trade_msft = result.trades.loc[result.trades["ticker"] == "MSFT"].iloc[0]
    assert trade_aapl["filled_notional"] == 10_000.0
    assert trade_msft["filled_notional"] == 5_000.0
    assert trade_msft["entry_status"] == "partial"
    assert result.rejected_trades.empty


def test_rejects_trade_when_no_cash_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    universe_path = tmp_path / "tickers_enriched.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)
    loader = BacktesterDataLoader(
        paths=DataPaths(
            daily_bars_dir=tmp_path / "daily-bars",
            frames_dir=tmp_path / "frames",
            universe_csv_path=universe_path,
            daily_signals_dir=tmp_path / "daily-bars-signals",
        )
    )
    strategy = SmaCrossStrategy()
    backtester = CapitalConstrainedPortfolioBacktester(
        strategy=strategy,
        data_loader=loader,
        initial_capital=10_000.0,
    )

    histories = {
        "AAPL": pd.DataFrame(
            {
                "ticker": ["AAPL"] * 4,
                "date": pd.bdate_range("2026-01-05", periods=4),
                "close": [100.0, 101.0, 102.0, 103.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 1.03],
                "realized_vol_3m": [0.35, 0.35, 0.35, 0.35],
                "realized_vol_1y": [0.40, 0.40, 0.40, 0.40],
            }
        ),
        "MSFT": pd.DataFrame(
            {
                "ticker": ["MSFT"] * 4,
                "date": pd.bdate_range("2026-01-05", periods=4),
                "close": [200.0, 201.0, 202.0, 203.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 1.03],
                "realized_vol_3m": [0.35, 0.35, 0.35, 0.35],
                "realized_vol_1y": [0.40, 0.40, 0.40, 0.40],
            }
        ),
    }

    monkeypatch.setattr(loader, "load_signal_history", lambda ticker, columns=None: histories[ticker])
    result = backtester.run(["AAPL", "MSFT"])

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["ticker"] == "AAPL"
    assert len(result.rejected_trades) == 1
    rejected = result.rejected_trades.iloc[0]
    assert rejected["ticker"] == "MSFT"
    assert rejected["rejection_reason"] == "no_cash"
    assert rejected["available_cash"] == 0.0
