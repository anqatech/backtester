# backtester

Small initial scaffold for a Python backtesting package built around your local parquet datasets.

## Included in this first step

- installable `src/` package layout
- `BacktesterDataLoader` class for loading:
  - one ticker's daily bars from `daily-bars/<TICKER>.parquet`
  - one ticker's row from `daily-bars-database-status-with-market-cap.parquet`
- `TrendSignalCalculator` for computing daily:
  - log returns
  - realized volatility over 1m, 3m, 6m, and 1y windows
  - per-ticker trend components from the trend-signal note
- configurable dataset paths loaded from a local `.env` file

## Environment file

Create a `.env` file in the project root with:

```env
BACKTESTER_DAILY_BARS_DIR=/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/daily-bars
BACKTESTER_FRAMES_DIR=/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/frames
BACKTESTER_UNIVERSE_CSV_PATH=/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/sp500/tickers_enriched.csv
BACKTESTER_DAILY_SIGNALS_DIR=/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/daily-bars-signals
```

`BacktesterDataLoader.from_env()` will load that file automatically.

## Example

```python
from backtester import BacktesterDataLoader

loader = BacktesterDataLoader.from_env()
bundle = loader.load_ticker_bundle("AAPL")

prices = bundle.price_history
status = bundle.database_status
```

```python
from backtester import TrendSignalCalculator

calculator = TrendSignalCalculator()
enriched = calculator.build_enriched_price_history("AAPL")

enriched[[
    "date",
    "close",
    "log_return",
    "realized_vol_3m",
    "tsmom_3m",
    "tsmom_6m",
    "tsmom_1y",
    "relmom_12_1",
    "price_to_sma_200",
    "sma_50_to_sma_200",
    "trend_signal_ready",
]].tail()
```

The reference-date logic uses calendar offsets and then walks back to the previous available trading day if the exact target date is missing.

```python
from backtester import SignalStoreBuilder

builder = SignalStoreBuilder()
builder.build_ticker("AAPL", overwrite=True)
builder.build_universe(overwrite=False)
```

The builder uses `tickers_enriched.csv` as the canonical universe list and writes one enriched parquet per ticker into `daily-bars-signals`.

```python
from backtester import UniverseSnapshotLoader

snapshot_loader = UniverseSnapshotLoader()
snapshot = snapshot_loader.load_snapshot(
    "2026-04-22",
    columns=["ticker", "date", "close", "tsmom_1y", "relmom_12_1", "trend_signal_ready"],
    ready_only=True,
)
```

By default the snapshot loader returns the latest available row on or before the requested date for each ticker. Set `exact_match=True` if you want only rows from the exact date.

## Tests and coverage

Install the optional development dependency group to get `pytest` and `coverage.py`:

```bash
pip install -e '.[dev]'
```

Run the test suite with:

```bash
python -m pytest
```

The existing `unittest`-style tests are fully compatible with `pytest`, so there is no need to rewrite them first.

If you want to keep using the standard library runner, this still works too:

```bash
python -m unittest discover -s tests -p 'test*.py'
```

Then run line and branch coverage with:

```bash
python -m coverage run -m pytest
python -m coverage report -m
```

You can also generate an HTML report:

```bash
python -m coverage html
```

The coverage configuration lives in [pyproject.toml](/Users/jalalelhazzat/Documents/Codex-Projects/backtester/pyproject.toml) and is scoped to the `backtester` package.
