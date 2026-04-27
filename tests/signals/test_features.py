import pandas as pd

from backtester.signals import TrendSignalCalculator


def test_reference_dates_fall_back_to_previous_available_date() -> None:
    calculator = TrendSignalCalculator()
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])

    reference_dates = calculator.reference_dates(dates, pd.DateOffset(days=1))

    expected = pd.to_datetime(["NaT", "2026-01-02", "2026-01-05"])
    pd.testing.assert_series_equal(
        reference_dates.reset_index(drop=True),
        pd.Series(expected),
    )


def test_enrich_price_history_adds_daily_features() -> None:
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

    assert "log_return" in enriched.columns
    assert "realized_vol_1m" in enriched.columns
    assert "realized_vol_3m" in enriched.columns
    assert "realized_vol_6m" in enriched.columns
    assert "realized_vol_1y" in enriched.columns
    assert "tsmom_3m" in enriched.columns
    assert "tsmom_6m" in enriched.columns
    assert "tsmom_1y" in enriched.columns
    assert "relmom_12_1" in enriched.columns
    assert "price_to_sma_200" in enriched.columns
    assert "sma_50_to_sma_200" in enriched.columns
    assert "trend_signal_ready" in enriched.columns

    assert not bool(enriched["trend_signal_ready"].iloc[50])
    assert bool(enriched["trend_signal_ready"].iloc[-1])
    assert enriched["ticker"].iloc[-1] == "AAPL"
    assert pd.notna(enriched["realized_vol_1y"].iloc[-1])
    assert pd.notna(enriched["tsmom_1y"].iloc[-1])
    assert pd.notna(enriched["relmom_12_1"].iloc[-1])
