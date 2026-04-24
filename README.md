# backtester

Small initial scaffold for a Python backtesting package built around your local parquet datasets.

## Included in this first step

- installable `src/` package layout
- `BacktesterDataLoader` class for loading:
  - one ticker's daily bars from `daily-bars/<TICKER>.parquet`
  - one ticker's row from `daily-bars-trend-signals.parquet`
  - one ticker's row from `daily-bars-realized-volatility.parquet`
- configurable dataset paths, with sensible defaults for your current `jnbooks` directories

## Default data locations

- daily bars: `/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/daily-bars`
- frames: `/Users/jalalelhazzat/Documents/Codex-Projects/jnbooks/data/frames`

You can override them with:

- `BACKTESTER_DAILY_BARS_DIR`
- `BACKTESTER_FRAMES_DIR`

## Example

```python
from backtester import BacktesterDataLoader

loader = BacktesterDataLoader.from_env()
bundle = loader.load_ticker_bundle("AAPL")

prices = bundle.price_history
trend = bundle.trend_signals
realized_vol = bundle.realized_volatility
```
