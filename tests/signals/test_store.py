import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.data import BacktesterDataLoader, DataPaths
from backtester.signals import SignalStoreBuilder, TrendSignalCalculator


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
