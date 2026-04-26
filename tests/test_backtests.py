import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.backtests import run_sma_cross_universe_backtest, simulate_sma_cross_strategy
from backtester.data import BacktesterDataLoader, DataPaths


class SmaCrossBacktestTests(unittest.TestCase):
    def test_cross_entry_executes_next_day_and_signal_exit_executes_next_day(self) -> None:
        history = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 6,
                "date": pd.bdate_range("2026-01-05", periods=6),
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98, 0.97, 0.96],
            }
        )

        result = simulate_sma_cross_strategy(history)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["signal_date"], pd.Timestamp("2026-01-06"))
        self.assertAlmostEqual(trade["entry_signal_value"], 1.02)
        self.assertEqual(trade["entry_date"], pd.Timestamp("2026-01-07"))
        self.assertEqual(trade["exit_signal_date"], pd.Timestamp("2026-01-08"))
        self.assertEqual(trade["exit_date"], pd.Timestamp("2026-01-09"))
        self.assertAlmostEqual(trade["exit_signal_value"], 0.97)
        self.assertEqual(trade["exit_reason"], "signal_exit")
        self.assertAlmostEqual(trade["shares"], 10_000.0 / 102.0)
        self.assertAlmostEqual(trade["pnl"], trade["shares"] * (104.0 - 102.0))
        self.assertEqual(result.positions["date"].tolist(), [pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-08")])

    def test_time_exit_uses_first_trading_day_on_or_after_expiry(self) -> None:
        history = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 30,
                "date": pd.bdate_range("2026-01-01", periods=30),
                "close": [100.0 + index for index in range(30)],
                "sma_50_to_sma_200": [0.98, 1.02] + [1.05] * 28,
            }
        )

        result = simulate_sma_cross_strategy(history)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["entry_date"], pd.Timestamp("2026-01-05"))
        self.assertEqual(trade["expiry_threshold_date"], pd.Timestamp("2026-02-05"))
        self.assertEqual(trade["exit_date"], pd.Timestamp("2026-02-05"))
        self.assertAlmostEqual(trade["entry_signal_value"], 1.02)
        self.assertAlmostEqual(trade["exit_signal_value"], 1.05)
        self.assertEqual(trade["exit_reason"], "time_exit")
        self.assertTrue(pd.isna(trade["exit_signal_date"]))

    def test_reentry_requires_fresh_cross_after_exit(self) -> None:
        history = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 35,
                "date": pd.bdate_range("2026-01-01", periods=35),
                "close": [100.0 + index for index in range(35)],
                "sma_50_to_sma_200": [
                    0.98,
                    1.01,
                ]
                + [1.02] * 24
                + [
                    1.03,
                    1.04,
                    0.97,
                    1.01,
                    1.02,
                    1.02,
                    1.02,
                    1.02,
                    1.02,
                ],
            }
        )

        result = simulate_sma_cross_strategy(history)

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades["entry_date"].tolist(), [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-12")])

    def test_universe_runner_combines_trades_across_tickers(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_path = root / "tickers_enriched.csv"
            pd.DataFrame({"ticker": ["AAPL", "MSFT"]}).to_csv(universe_path, index=False)

            paths = DataPaths(
                daily_bars_dir=root / "daily-bars",
                frames_dir=root / "frames",
                universe_csv_path=universe_path,
                daily_signals_dir=root / "daily-bars-signals",
            )
            loader = BacktesterDataLoader(paths=paths)

            histories = {
                "AAPL": pd.DataFrame(
                    {
                        "ticker": ["AAPL"] * 6,
                        "date": pd.bdate_range("2026-01-05", periods=6),
                        "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                        "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98, 0.97, 0.96],
                    }
                ),
                "MSFT": pd.DataFrame(
                    {
                        "ticker": ["MSFT"] * 6,
                        "date": pd.bdate_range("2026-01-05", periods=6),
                        "close": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0],
                        "sma_50_to_sma_200": [0.95, 0.96, 1.01, 1.02, 1.03, 0.98],
                    }
                ),
            }

            with patch.object(loader, "load_signal_history", side_effect=lambda ticker, columns=None: histories[ticker]):
                result = run_sma_cross_universe_backtest(data_loader=loader)

            self.assertEqual(sorted(result.trades["ticker"].tolist()), ["AAPL", "MSFT"])
            self.assertIn("active_positions", result.portfolio.columns)
            self.assertIn("total_pnl", result.portfolio.columns)


if __name__ == "__main__":
    unittest.main()
