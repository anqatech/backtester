import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.signals import SignalStoreBuilder, TrendSignalCalculator


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


if __name__ == "__main__":
    unittest.main()
