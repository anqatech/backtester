import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.signals import SignalStoreBuilder, TrendSignalCalculator
from backtester.universe import UniverseSnapshotLoader


class TrendSignalCalculatorTests(unittest.TestCase):
    def test_reference_dates_fall_back_to_previous_available_date(self) -> None:
        calculator = TrendSignalCalculator()
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])

        reference_dates = calculator.reference_dates(dates, pd.DateOffset(days=1))

        expected = pd.to_datetime(["NaT", "2026-01-02", "2026-01-05"])
        pd.testing.assert_series_equal(
            reference_dates.reset_index(drop=True),
            pd.Series(expected),
        )

    def test_enrich_price_history_adds_daily_features(self) -> None:
        calculator = TrendSignalCalculator()
        dates = pd.bdate_range("2025-01-02", periods=320)
        close = pd.Series(range(100, 100 + len(dates)), dtype="float64")
        price_history = pd.DataFrame(
            {
                "date": dates,
                "close": close,
            }
        )

        enriched = calculator.enrich_price_history(price_history, ticker="AAPL")

        self.assertIn("log_return", enriched.columns)
        self.assertIn("realized_vol_1m", enriched.columns)
        self.assertIn("realized_vol_3m", enriched.columns)
        self.assertIn("realized_vol_6m", enriched.columns)
        self.assertIn("realized_vol_1y", enriched.columns)
        self.assertIn("tsmom_3m", enriched.columns)
        self.assertIn("tsmom_6m", enriched.columns)
        self.assertIn("tsmom_1y", enriched.columns)
        self.assertIn("relmom_12_1", enriched.columns)
        self.assertIn("price_to_sma_200", enriched.columns)
        self.assertIn("sma_50_to_sma_200", enriched.columns)
        self.assertIn("trend_signal_ready", enriched.columns)

        self.assertFalse(bool(enriched["trend_signal_ready"].iloc[50]))
        self.assertTrue(bool(enriched["trend_signal_ready"].iloc[-1]))
        self.assertEqual(enriched["ticker"].iloc[-1], "AAPL")
        self.assertTrue(pd.notna(enriched["realized_vol_1y"].iloc[-1]))
        self.assertTrue(pd.notna(enriched["tsmom_1y"].iloc[-1]))
        self.assertTrue(pd.notna(enriched["relmom_12_1"].iloc[-1]))


class SignalStoreBuilderTests(unittest.TestCase):
    def test_build_ticker_saves_to_expected_parquet_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = DataPaths(
                daily_bars_dir=root / "daily-bars",
                frames_dir=root / "frames",
                universe_csv_path=root / "tickers_enriched.csv",
                daily_signals_dir=root / "daily-bars-signals",
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

            with patch.object(
                builder.calculator,
                "build_enriched_price_history",
                return_value=enriched,
            ) as build_mock, patch.object(pd.DataFrame, "to_parquet") as parquet_mock:
                result = builder.build_ticker("aapl", overwrite=True)

            build_mock.assert_called_once_with("AAPL")
            parquet_mock.assert_called_once()
            self.assertEqual(result.ticker, "AAPL")
            self.assertEqual(result.output_path, paths.daily_signals_dir / "AAPL.parquet")
            self.assertEqual(result.rows_written, 1)
            self.assertFalse(result.skipped)

    def test_build_universe_reads_tickers_from_universe_csv(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            universe_path = root / "tickers_enriched.csv"
            pd.DataFrame({"ticker": ["AAPL", "MSFT", "AAPL"]}).to_csv(universe_path, index=False)

            paths = DataPaths(
                daily_bars_dir=root / "daily-bars",
                frames_dir=root / "frames",
                universe_csv_path=universe_path,
                daily_signals_dir=root / "daily-bars-signals",
            )
            loader = BacktesterDataLoader(paths=paths)
            builder = SignalStoreBuilder(
                data_loader=loader,
                calculator=TrendSignalCalculator(data_loader=loader),
            )

            with patch.object(builder, "build_ticker") as build_ticker_mock:
                builder.build_universe(overwrite=False)

            self.assertEqual(
                [call.kwargs["ticker"] for call in build_ticker_mock.call_args_list],
                ["AAPL", "MSFT"],
            )


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
