import unittest

import pandas as pd

from backtester.backtesting import (
    BacktestResult,
    summarize_backtest,
    summarize_trades_by_exit_reason,
    summarize_trades_by_ticker,
)


class BacktestAnalyticsTests(unittest.TestCase):
    def test_summarize_backtest_returns_expected_headline_metrics(self) -> None:
        result = BacktestResult(
            trades=pd.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT"],
                    "signal_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
                    "entry_signal_value": [1.02, 1.04],
                    "entry_realized_vol_3m": [0.25, 0.35],
                    "entry_realized_vol_1y": [0.30, 0.40],
                    "entry_date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")],
                    "entry_price": [100.0, 200.0],
                    "shares": [100.0, 50.0],
                    "notional": [10_000.0, 10_000.0],
                    "expiry_threshold_date": [pd.Timestamp("2026-02-02"), pd.Timestamp("2026-02-03")],
                    "exit_signal_date": [pd.Timestamp("2026-01-04"), pd.NaT],
                    "exit_date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-10")],
                    "exit_signal_value": [0.98, 1.03],
                    "exit_realized_vol_3m": [0.27, 0.33],
                    "exit_realized_vol_1y": [0.32, 0.38],
                    "exit_price": [105.0, 196.0],
                    "exit_reason": ["signal_exit", "time_exit"],
                    "holding_days": [3, 7],
                    "pnl": [500.0, -200.0],
                    "return": [0.05, -0.02],
                }
            ),
            positions=pd.DataFrame(
                {
                    "ticker": ["AAPL", "MSFT", "MSFT"],
                    "date": [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-05")],
                    "entry_date": [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-03")],
                    "shares": [100.0, 50.0, 50.0],
                    "close": [103.0, 202.0, 198.0],
                    "market_value": [10_300.0, 10_100.0, 9_900.0],
                    "unrealized_pnl": [300.0, 100.0, -100.0],
                }
            ),
            portfolio=pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-10")],
                    "active_positions": [1, 1, 1, 0],
                    "gross_market_value": [10_300.0, 10_100.0, 9_900.0, 0.0],
                    "unrealized_pnl": [300.0, 100.0, -100.0, 0.0],
                    "realized_pnl": [0.0, 0.0, 500.0, -200.0],
                    "cumulative_realized_pnl": [0.0, 0.0, 500.0, 300.0],
                    "total_pnl": [300.0, 100.0, 400.0, 300.0],
                }
            ),
        )

        summary = summarize_backtest(result)

        self.assertEqual(summary["trade_count"], 2)
        self.assertAlmostEqual(summary["winning_trade_pct"], 0.5)
        self.assertAlmostEqual(summary["average_trade_return"], 0.015)
        self.assertAlmostEqual(summary["median_trade_return"], 0.015)
        self.assertAlmostEqual(summary["average_trade_pnl"], 150.0)
        self.assertAlmostEqual(summary["total_realized_pnl"], 300.0)
        self.assertAlmostEqual(summary["average_holding_days"], 5.0)
        self.assertAlmostEqual(summary["median_holding_days"], 5.0)
        self.assertAlmostEqual(summary["average_entry_notional"], 10_000.0)
        self.assertAlmostEqual(summary["average_entry_realized_vol_3m"], 0.30)
        self.assertEqual(summary["max_active_positions"], 1)
        self.assertAlmostEqual(summary["average_active_positions"], 0.75)
        self.assertAlmostEqual(summary["max_gross_market_value"], 10_300.0)
        self.assertAlmostEqual(summary["average_gross_market_value"], 7_575.0)
        self.assertAlmostEqual(summary["max_drawdown_pnl"], 200.0)
        self.assertAlmostEqual(summary["final_total_pnl"], 300.0)

    def test_trade_diagnostic_summaries_group_as_expected(self) -> None:
        trades = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "MSFT"],
                "pnl": [100.0, -40.0, 90.0],
                "return": [0.01, -0.004, 0.009],
                "holding_days": [5, 8, 6],
                "notional": [10_000.0, 8_000.0, 12_000.0],
                "entry_realized_vol_3m": [0.30, 0.40, 0.25],
                "exit_reason": ["signal_exit", "time_exit", "time_exit"],
            }
        )

        by_ticker = summarize_trades_by_ticker(trades)
        by_exit_reason = summarize_trades_by_exit_reason(trades)

        aapl_row = by_ticker.loc[by_ticker["ticker"] == "AAPL"].iloc[0]
        msft_row = by_ticker.loc[by_ticker["ticker"] == "MSFT"].iloc[0]
        time_exit_row = by_exit_reason.loc[by_exit_reason["exit_reason"] == "time_exit"].iloc[0]

        self.assertEqual(aapl_row["trade_count"], 2)
        self.assertAlmostEqual(aapl_row["winning_trade_pct"], 0.5)
        self.assertAlmostEqual(aapl_row["total_pnl"], 60.0)
        self.assertAlmostEqual(aapl_row["average_entry_notional"], 9_000.0)
        self.assertEqual(msft_row["trade_count"], 1)
        self.assertAlmostEqual(msft_row["total_pnl"], 90.0)
        self.assertEqual(time_exit_row["trade_count"], 2)
        self.assertAlmostEqual(time_exit_row["winning_trade_pct"], 0.5)
        self.assertAlmostEqual(time_exit_row["total_pnl"], 50.0)


if __name__ == "__main__":
    unittest.main()
