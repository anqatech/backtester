import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.universe import UniverseSnapshotLoader


class UniverseSnapshotLoaderTests(unittest.TestCase):
    def test_load_snapshot_uses_previous_available_row_by_default(self) -> None:
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

            with patch.object(loader, "load_signal_history", side_effect=lambda ticker, columns=None: histories[ticker]):
                snapshot = snapshot_loader.load_snapshot(
                    "2026-01-04",
                    columns=["ticker", "date", "close", "trend_signal_ready"],
                )

            self.assertEqual(snapshot["ticker"].tolist(), ["AAPL", "MSFT"])
            self.assertEqual(snapshot["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-02", "2026-01-03"])
            self.assertEqual(snapshot["close"].tolist(), [100.0, 200.0])

    def test_load_snapshot_ready_only_filters_rows(self) -> None:
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

            with patch.object(loader, "load_signal_history", side_effect=lambda ticker, columns=None: histories[ticker]):
                snapshot = snapshot_loader.load_snapshot(
                    "2026-01-05",
                    columns=["ticker", "date", "close"],
                    ready_only=True,
                )

            self.assertEqual(snapshot["ticker"].tolist(), ["MSFT"])
            self.assertEqual(snapshot.columns.tolist(), ["ticker", "date", "close"])

    def test_load_snapshot_exact_match_requires_exact_date(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_path = root / "tickers_enriched.csv"
            pd.DataFrame({"ticker": ["AAPL"]}).to_csv(universe_path, index=False)

            paths = DataPaths(
                daily_bars_dir=root / "daily-bars",
                frames_dir=root / "frames",
                universe_csv_path=universe_path,
                daily_signals_dir=root / "daily-bars-signals",
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

            with patch.object(loader, "load_signal_history", return_value=history):
                snapshot = snapshot_loader.load_snapshot(
                    "2026-01-03",
                    columns=["ticker", "date", "close"],
                    exact_match=True,
                )

            self.assertTrue(snapshot.empty)

    def test_load_snapshot_computes_cross_sectional_trend_columns(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_path = root / "tickers_enriched.csv"
            pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA"]}).to_csv(universe_path, index=False)

            paths = DataPaths(
                daily_bars_dir=root / "daily-bars",
                frames_dir=root / "frames",
                universe_csv_path=universe_path,
                daily_signals_dir=root / "daily-bars-signals",
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
                self.assertNotIn("trend_raw", columns)
                self.assertNotIn("ma_confirm", columns)
                self.assertNotIn("trend_composite", columns)
                return histories[ticker]

            with patch.object(loader, "load_signal_history", side_effect=load_signal_history):
                snapshot = snapshot_loader.load_snapshot(
                    "2026-01-05",
                    columns=["ticker", "trend_raw", "trend_composite"],
                )

            self.assertEqual(
                snapshot.columns.tolist(),
                ["ticker", "trend_raw", "trend_composite"],
            )
            self.assertEqual(snapshot["ticker"].tolist(), ["AAPL", "MSFT", "NVDA"])
            self.assertTrue(snapshot["trend_raw"].notna().all())
            self.assertTrue(snapshot["trend_composite"].notna().all())


if __name__ == "__main__":
    unittest.main()
