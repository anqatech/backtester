import pandas as pd
import pytest

from backtester.backtesting.common import (
    aggregate_portfolio_series,
    empty_backtest_result,
    max_drawdown,
    position_columns,
    prepare_history,
    resolve_tickers,
    scaled_trade_notional,
    trade_columns,
)
from backtester.data import BacktesterDataLoader


def test_prepare_history_normalizes_filters_and_deduplicates() -> None:
    history = pd.DataFrame(
        {
            "ticker": [" aapl ", "AAPL", "MSFT", "AAPL"],
            "date": [
                "2026-01-02 09:30:00",
                "2026-01-02 16:00:00",
                "2026-01-03 16:00:00",
                "2026-01-04 16:00:00",
            ],
            "close": ["100", "101", "bad", "102"],
            "sma_50_to_sma_200": ["1.01", "1.02", "1.03", "1.04"],
            "realized_vol_3m": ["0.30", "0.31", "0.32", "0.33"],
            "realized_vol_1y": ["0.40", "0.41", "0.42", "0.43"],
        }
    )

    prepared = prepare_history(history, ticker="aapl")

    assert prepared["ticker"].tolist() == ["AAPL", "AAPL"]
    assert prepared["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-02", "2026-01-04"]
    assert prepared["close"].tolist() == [101, 102]


def test_prepare_history_adds_ticker_when_missing() -> None:
    history = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "close": [100.0],
            "sma_50_to_sma_200": [1.02],
            "realized_vol_3m": [0.3],
            "realized_vol_1y": [0.4],
        }
    )

    prepared = prepare_history(history, ticker="msft")

    assert prepared["ticker"].tolist() == ["MSFT"]


def test_prepare_history_raises_for_missing_columns() -> None:
    history = pd.DataFrame({"date": ["2026-01-02"], "close": [100.0]})

    with pytest.raises(KeyError, match="History is missing required columns"):
        prepare_history(history, ticker="AAPL")


def test_prepare_history_requires_ticker_column_or_argument() -> None:
    history = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "close": [100.0],
            "sma_50_to_sma_200": [1.02],
            "realized_vol_3m": [0.3],
            "realized_vol_1y": [0.4],
        }
    )

    with pytest.raises(KeyError, match="History must contain a 'ticker' column"):
        prepare_history(history, ticker=None)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reference_vol": 0.0, "min_notional": 5_000.0, "max_notional": 20_000.0, "entry_realized_vol_3m": 0.3}, "reference_vol must be positive"),
        ({"reference_vol": 0.35, "min_notional": 0.0, "max_notional": 20_000.0, "entry_realized_vol_3m": 0.3}, "min_notional and max_notional must be positive"),
        ({"reference_vol": 0.35, "min_notional": 10_000.0, "max_notional": 5_000.0, "entry_realized_vol_3m": 0.3}, "min_notional cannot be greater than max_notional"),
        ({"reference_vol": 0.35, "min_notional": 5_000.0, "max_notional": 20_000.0, "entry_realized_vol_3m": 0.0}, "entry_realized_vol_3m must be positive"),
    ],
)
def test_scaled_trade_notional_validation_errors(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        scaled_trade_notional(base_notional=10_000.0, **kwargs)


def test_scaled_trade_notional_applies_floor_and_ceiling() -> None:
    assert scaled_trade_notional(10_000.0, 0.35, 0.35, 5_000.0, 20_000.0) == 10_000.0
    assert scaled_trade_notional(10_000.0, 0.70, 0.35, 5_000.0, 20_000.0) == 5_000.0
    assert scaled_trade_notional(10_000.0, 0.10, 0.35, 5_000.0, 20_000.0) == 20_000.0


def test_aggregate_portfolio_series_handles_empty_inputs() -> None:
    portfolio = aggregate_portfolio_series(pd.DataFrame(), pd.DataFrame())

    assert portfolio.columns.tolist() == [
        "date",
        "active_positions",
        "gross_market_value",
        "unrealized_pnl",
        "realized_pnl",
        "cumulative_realized_pnl",
        "total_pnl",
    ]
    assert portfolio.empty


def test_aggregate_portfolio_series_with_only_realized_trades() -> None:
    trades = pd.DataFrame(
        {
            "exit_date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
            "pnl": [100.0, -25.0, 30.0],
        }
    )

    portfolio = aggregate_portfolio_series(pd.DataFrame(), trades)

    assert portfolio["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-05", "2026-01-06"]
    assert portfolio["realized_pnl"].tolist() == [75.0, 30.0]
    assert portfolio["cumulative_realized_pnl"].tolist() == [75.0, 105.0]
    assert portfolio["active_positions"].tolist() == [0, 0]


def test_resolve_tickers_normalizes_and_deduplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    loader = BacktesterDataLoader.__new__(BacktesterDataLoader)
    loader.normalize_ticker = staticmethod(BacktesterDataLoader.normalize_ticker)

    monkeypatch.setattr(
        loader,
        "load_universe",
        lambda columns=None: pd.DataFrame({"ticker": [" aapl ", "", "MSFT", "AAPL"]}),
    )

    assert resolve_tickers(loader, [" msft ", "AAPL", "MSFT", ""]) == ["MSFT", "AAPL"]
    assert resolve_tickers(loader, None) == ["AAPL", "MSFT"]


def test_max_drawdown_and_empty_backtest_result() -> None:
    assert max_drawdown(pd.Series(dtype="float64")) == 0.0
    assert max_drawdown(pd.Series([100.0, 80.0, 120.0, 90.0])) == 30.0

    result = empty_backtest_result()

    assert result.trades.columns.tolist() == trade_columns()
    assert result.positions.columns.tolist() == position_columns()
    assert result.portfolio.columns.tolist() == [
        "date",
        "active_positions",
        "gross_market_value",
        "unrealized_pnl",
        "realized_pnl",
        "cumulative_realized_pnl",
        "total_pnl",
    ]
