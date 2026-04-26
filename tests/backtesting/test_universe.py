import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.backtesting import UniverseBacktester, run_sma_cross_universe_backtest
from backtester.data import BacktesterDataLoader, DataPaths
from backtester.strategies import SmaCrossStrategy


class UniverseBacktesterTests(unittest.TestCase):
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

            with patch.object(loader, "load_signal_history", side_effect=lambda ticker, columns=None: histories[ticker]):
                result = run_sma_cross_universe_backtest(data_loader=loader)

            self.assertEqual(sorted(result.trades["ticker"].tolist()), ["AAPL", "MSFT"])
            self.assertIn("active_positions", result.portfolio.columns)
            self.assertIn("total_pnl", result.portfolio.columns)

    def test_universe_backtester_class_runs_with_loader(self) -> None:
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

            with patch.object(loader, "load_signal_history", side_effect=lambda ticker, columns=None: histories[ticker]):
                result = UniverseBacktester(strategy=strategy, data_loader=loader).run()

            self.assertEqual(sorted(result.trades["ticker"].tolist()), ["AAPL", "MSFT"])
            self.assertIn("active_positions", result.portfolio.columns)


if __name__ == "__main__":
    unittest.main()
