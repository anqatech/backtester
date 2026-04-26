import unittest

import pandas as pd

from backtester.backtesting import SingleTickerBacktester, simulate_sma_cross_strategy
from backtester.strategies import SmaCrossStrategy


class SmaCrossSingleTickerTests(unittest.TestCase):
    def test_cross_entry_executes_next_day_and_signal_exit_executes_next_day(self) -> None:
        history = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 6,
                "date": pd.bdate_range("2026-01-05", periods=6),
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98, 0.97, 0.96],
                "realized_vol_3m": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
                "realized_vol_1y": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45],
            }
        )

        result = simulate_sma_cross_strategy(history)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["signal_date"], pd.Timestamp("2026-01-06"))
        self.assertAlmostEqual(trade["entry_signal_value"], 1.02)
        self.assertAlmostEqual(trade["entry_realized_vol_3m"], 0.31)
        self.assertAlmostEqual(trade["entry_realized_vol_1y"], 0.41)
        self.assertEqual(trade["entry_date"], pd.Timestamp("2026-01-07"))
        self.assertEqual(trade["exit_signal_date"], pd.Timestamp("2026-01-08"))
        self.assertEqual(trade["exit_date"], pd.Timestamp("2026-01-09"))
        self.assertAlmostEqual(trade["exit_signal_value"], 0.97)
        self.assertAlmostEqual(trade["exit_realized_vol_3m"], 0.34)
        self.assertAlmostEqual(trade["exit_realized_vol_1y"], 0.44)
        self.assertEqual(trade["exit_reason"], "signal_exit")
        self.assertAlmostEqual(trade["notional"], 10_000.0 * (0.35 / 0.31))
        self.assertAlmostEqual(trade["shares"], trade["notional"] / 102.0)
        self.assertAlmostEqual(trade["pnl"], trade["shares"] * (104.0 - 102.0))
        self.assertEqual(result.positions["date"].tolist(), [pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-08")])

    def test_time_exit_uses_first_trading_day_on_or_after_expiry(self) -> None:
        history = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 30,
                "date": pd.bdate_range("2026-01-01", periods=30),
                "close": [100.0 + index for index in range(30)],
                "sma_50_to_sma_200": [0.98, 1.02] + [1.05] * 28,
                "realized_vol_3m": [0.30 + 0.001 * index for index in range(30)],
                "realized_vol_1y": [0.40 + 0.001 * index for index in range(30)],
            }
        )

        result = simulate_sma_cross_strategy(history)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["entry_date"], pd.Timestamp("2026-01-05"))
        self.assertEqual(trade["expiry_threshold_date"], pd.Timestamp("2026-02-05"))
        self.assertEqual(trade["exit_date"], pd.Timestamp("2026-02-05"))
        self.assertAlmostEqual(trade["entry_signal_value"], 1.02)
        self.assertAlmostEqual(trade["entry_realized_vol_3m"], 0.301)
        self.assertAlmostEqual(trade["entry_realized_vol_1y"], 0.401)
        self.assertAlmostEqual(trade["exit_signal_value"], 1.05)
        self.assertAlmostEqual(trade["exit_realized_vol_3m"], 0.325)
        self.assertAlmostEqual(trade["exit_realized_vol_1y"], 0.425)
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
                "realized_vol_3m": [0.30 + 0.001 * index for index in range(35)],
                "realized_vol_1y": [0.40 + 0.001 * index for index in range(35)],
            }
        )

        result = simulate_sma_cross_strategy(history)

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.trades["entry_date"].tolist(), [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-12")])

    def test_volatility_scaled_notional_respects_reference_and_bounds(self) -> None:
        high_vol_history = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 4,
                "date": pd.bdate_range("2026-01-05", periods=4),
                "close": [100.0, 101.0, 102.0, 103.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98],
                "realized_vol_3m": [0.80, 0.80, 0.80, 0.80],
                "realized_vol_1y": [0.40, 0.40, 0.40, 0.40],
            }
        )
        low_vol_history = pd.DataFrame(
            {
                "ticker": ["MSFT"] * 4,
                "date": pd.bdate_range("2026-01-05", periods=4),
                "close": [200.0, 201.0, 202.0, 203.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98],
                "realized_vol_3m": [0.10, 0.10, 0.10, 0.10],
                "realized_vol_1y": [0.30, 0.30, 0.30, 0.30],
            }
        )

        high_vol_result = simulate_sma_cross_strategy(high_vol_history)
        low_vol_result = simulate_sma_cross_strategy(low_vol_history)

        self.assertAlmostEqual(high_vol_result.trades.iloc[0]["notional"], 5_000.0)
        self.assertAlmostEqual(low_vol_result.trades.iloc[0]["notional"], 20_000.0)

    def test_single_ticker_backtester_class_matches_wrapper_behavior(self) -> None:
        history = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 6,
                "date": pd.bdate_range("2026-01-05", periods=6),
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "sma_50_to_sma_200": [0.98, 1.02, 1.03, 0.98, 0.97, 0.96],
                "realized_vol_3m": [0.30, 0.31, 0.32, 0.33, 0.34, 0.35],
                "realized_vol_1y": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45],
            }
        )

        strategy = SmaCrossStrategy()
        class_result = SingleTickerBacktester(strategy).run(history)
        wrapper_result = simulate_sma_cross_strategy(history)

        pd.testing.assert_frame_equal(class_result.trades, wrapper_result.trades)
        pd.testing.assert_frame_equal(class_result.positions, wrapper_result.positions)
        pd.testing.assert_frame_equal(class_result.portfolio, wrapper_result.portfolio)


if __name__ == "__main__":
    unittest.main()
