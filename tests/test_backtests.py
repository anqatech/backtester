import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from backtester.backtests import (
    BacktestResult,
    run_sma_cross_universe_backtest,
    simulate_sma_cross_strategy,
    summarize_backtest,
    summarize_trades_by_exit_reason,
    summarize_trades_by_ticker,
)
from backtester.data import BacktesterDataLoader, DataPaths


class SmaCrossBacktestTests(unittest.TestCase):
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
