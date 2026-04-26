import unittest

import pandas as pd

from backtester.signals import TrendSignalCalculator


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


if __name__ == "__main__":
    unittest.main()
