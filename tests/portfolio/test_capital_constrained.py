import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.portfolio import CapitalConstrainedPortfolioBacktester
from backtester.strategies import SmaCrossStrategy


class CapitalConstrainedPortfolioBacktesterTests(unittest.TestCase):
    def test_full_fill_with_sufficient_cash(self) -> None:
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

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_path = root / "tickers_enriched.csv"
            pd.DataFrame({"ticker": ["AAPL"]}).to_csv(universe_path, index=False)
            loader = BacktesterDataLoader(
                paths=DataPaths(
                    daily_bars_dir=root / "daily-bars",
                    frames_dir=root / "frames",
                    universe_csv_path=universe_path,
                    daily_signals_dir=root / "daily-bars-signals",
                )
            )
            strategy = SmaCrossStrategy()
            backtester = CapitalConstrainedPortfolioBacktester(
                strategy=strategy,
                data_loader=loader,
                initial_capital=500_000.0,
            )

            with patch.object(loader, "load_signal_history", return_value=history):
                result = backtester.run(["AAPL"])

        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertAlmostEqual(trade["target_notional"], 10_000.0)
        self.assertAlmostEqual(trade["filled_notional"], 10_000.0)
        self.assertEqual(trade["entry_status"], "filled")
        self.assertTrue(result.rejected_trades.empty)

    def test_partial_fill_when_cash_is_insufficient(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_path = root / "tickers_enriched.csv"
            pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)
            loader = BacktesterDataLoader(
                paths=DataPaths(
                    daily_bars_dir=root / "daily-bars",
                    frames_dir=root / "frames",
                    universe_csv_path=universe_path,
                    daily_signals_dir=root / "daily-bars-signals",
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

            with patch.object(loader, "load_signal_history", side_effect=lambda ticker, columns=None: histories[ticker]):
                result = backtester.run(["AAPL", "MSFT"])

        self.assertEqual(len(result.trades), 2)
        trade_aapl = result.trades.loc[result.trades["ticker"] == "AAPL"].iloc[0]
        trade_msft = result.trades.loc[result.trades["ticker"] == "MSFT"].iloc[0]
        self.assertAlmostEqual(trade_aapl["filled_notional"], 10_000.0)
        self.assertAlmostEqual(trade_msft["filled_notional"], 5_000.0)
        self.assertEqual(trade_msft["entry_status"], "partial")
        self.assertTrue(result.rejected_trades.empty)

    def test_rejects_trade_when_no_cash_is_available(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_path = root / "tickers_enriched.csv"
            pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)
            loader = BacktesterDataLoader(
                paths=DataPaths(
                    daily_bars_dir=root / "daily-bars",
                    frames_dir=root / "frames",
                    universe_csv_path=universe_path,
                    daily_signals_dir=root / "daily-bars-signals",
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

            with patch.object(loader, "load_signal_history", side_effect=lambda ticker, columns=None: histories[ticker]):
                result = backtester.run(["AAPL", "MSFT"])

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["ticker"], "AAPL")
        self.assertEqual(len(result.rejected_trades), 1)
        rejected = result.rejected_trades.iloc[0]
        self.assertEqual(rejected["ticker"], "MSFT")
        self.assertEqual(rejected["rejection_reason"], "no_cash")
        self.assertAlmostEqual(rejected["available_cash"], 0.0)


if __name__ == "__main__":
    unittest.main()
