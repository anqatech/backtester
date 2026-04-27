import pytest
import pandas as pd

from backtester.backtesting import SingleTickerBacktester, simulate_sma_cross_strategy
from backtester.strategies import SmaCrossStrategy


def test_cross_entry_executes_next_day_and_signal_exit_executes_next_day() -> None:
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

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["signal_date"] == pd.Timestamp("2026-01-06")
    assert trade["entry_signal_value"] == pytest.approx(1.02)
    assert trade["entry_realized_vol_3m"] == pytest.approx(0.31)
    assert trade["entry_realized_vol_1y"] == pytest.approx(0.41)
    assert trade["entry_date"] == pd.Timestamp("2026-01-07")
    assert trade["exit_signal_date"] == pd.Timestamp("2026-01-08")
    assert trade["exit_date"] == pd.Timestamp("2026-01-09")
    assert trade["exit_signal_value"] == pytest.approx(0.97)
    assert trade["exit_realized_vol_3m"] == pytest.approx(0.34)
    assert trade["exit_realized_vol_1y"] == pytest.approx(0.44)
    assert trade["exit_reason"] == "signal_exit"
    assert trade["notional"] == pytest.approx(10_000.0 * (0.35 / 0.31))
    assert trade["shares"] == pytest.approx(trade["notional"] / 102.0)
    assert trade["pnl"] == pytest.approx(trade["shares"] * (104.0 - 102.0))
    assert result.positions["date"].tolist() == [pd.Timestamp("2026-01-07"), pd.Timestamp("2026-01-08")]


def test_time_exit_uses_first_trading_day_on_or_after_expiry() -> None:
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

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_date"] == pd.Timestamp("2026-01-05")
    assert trade["expiry_threshold_date"] == pd.Timestamp("2026-02-05")
    assert trade["exit_date"] == pd.Timestamp("2026-02-05")
    assert trade["entry_signal_value"] == pytest.approx(1.02)
    assert trade["entry_realized_vol_3m"] == pytest.approx(0.301)
    assert trade["entry_realized_vol_1y"] == pytest.approx(0.401)
    assert trade["exit_signal_value"] == pytest.approx(1.05)
    assert trade["exit_realized_vol_3m"] == pytest.approx(0.325)
    assert trade["exit_realized_vol_1y"] == pytest.approx(0.425)
    assert trade["exit_reason"] == "time_exit"
    assert pd.isna(trade["exit_signal_date"])


def test_reentry_requires_fresh_cross_after_exit() -> None:
    history = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 35,
            "date": pd.bdate_range("2026-01-01", periods=35),
            "close": [100.0 + index for index in range(35)],
            "sma_50_to_sma_200": [0.98, 1.01]
            + [1.02] * 24
            + [1.03, 1.04, 0.97, 1.01, 1.02, 1.02, 1.02, 1.02, 1.02],
            "realized_vol_3m": [0.30 + 0.001 * index for index in range(35)],
            "realized_vol_1y": [0.40 + 0.001 * index for index in range(35)],
        }
    )

    result = simulate_sma_cross_strategy(history)

    assert len(result.trades) == 2
    assert result.trades["entry_date"].tolist() == [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-02-12")]


def test_volatility_scaled_notional_respects_reference_and_bounds() -> None:
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

    assert high_vol_result.trades.iloc[0]["notional"] == pytest.approx(5_000.0)
    assert low_vol_result.trades.iloc[0]["notional"] == pytest.approx(20_000.0)


def test_single_ticker_backtester_class_matches_wrapper_behavior() -> None:
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
