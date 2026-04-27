import pandas as pd

from backtester.backtesting import (
    BacktestResult,
    summarize_backtest,
    summarize_trades_by_exit_reason,
    summarize_trades_by_ticker,
)


def test_summarize_backtest_handles_empty_result() -> None:
    result = BacktestResult(
        trades=pd.DataFrame(),
        positions=pd.DataFrame(),
        portfolio=pd.DataFrame(),
    )

    summary = summarize_backtest(result)

    assert summary["trade_count"] == 0
    assert summary["winning_trade_pct"] == 0.0
    assert summary["final_total_pnl"] == 0.0
    assert summary["max_active_positions"] == 0


def test_summarize_backtest_handles_empty_portfolio_with_nonempty_trades() -> None:
    result = BacktestResult(
        trades=pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "pnl": [100.0],
                "return": [0.01],
                "holding_days": [5],
                "notional": [10_000.0],
                "entry_realized_vol_3m": [0.25],
            }
        ),
        positions=pd.DataFrame(),
        portfolio=pd.DataFrame(),
    )

    summary = summarize_backtest(result)

    assert summary["trade_count"] == 1
    assert summary["max_active_positions"] == 0
    assert summary["average_gross_market_value"] == 0.0
    assert summary["final_total_pnl"] == 0.0


def test_trade_summary_helpers_return_empty_frames_for_empty_input() -> None:
    by_ticker = summarize_trades_by_ticker(pd.DataFrame())
    by_exit_reason = summarize_trades_by_exit_reason(pd.DataFrame())

    assert by_ticker.columns.tolist() == [
        "ticker",
        "trade_count",
        "winning_trade_pct",
        "average_trade_return",
        "median_trade_return",
        "average_trade_pnl",
        "total_pnl",
        "average_holding_days",
        "average_entry_notional",
        "average_entry_realized_vol_3m",
    ]
    assert by_exit_reason.columns.tolist() == [
        "exit_reason",
        "trade_count",
        "winning_trade_pct",
        "average_trade_return",
        "median_trade_return",
        "average_trade_pnl",
        "total_pnl",
        "average_holding_days",
    ]
    assert by_ticker.empty
    assert by_exit_reason.empty
