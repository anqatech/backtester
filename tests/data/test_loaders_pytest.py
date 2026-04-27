from pathlib import Path

import pandas as pd
import pytest

from backtester.data import BacktesterDataLoader, DataPaths


def build_paths(root: Path) -> DataPaths:
    daily_bars_dir = root / "daily-bars"
    frames_dir = root / "frames"
    daily_signals_dir = root / "daily-bars-signals"
    universe_csv_path = root / "tickers_enriched.csv"

    daily_bars_dir.mkdir()
    frames_dir.mkdir()
    daily_signals_dir.mkdir()

    return DataPaths(
        daily_bars_dir=daily_bars_dir,
        frames_dir=frames_dir,
        universe_csv_path=universe_csv_path,
        daily_signals_dir=daily_signals_dir,
    )


def test_datapaths_from_env_reads_env_file_and_expands_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"BACKTESTER_DAILY_BARS_DIR={tmp_path / 'daily-bars'}",
                f"BACKTESTER_FRAMES_DIR={tmp_path / 'frames'}",
                f"BACKTESTER_UNIVERSE_CSV_PATH={tmp_path / 'tickers.csv'}",
                f"BACKTESTER_DAILY_SIGNALS_DIR={tmp_path / 'daily-bars-signals'}",
            ]
        )
    )

    for name in [
        "BACKTESTER_DAILY_BARS_DIR",
        "BACKTESTER_FRAMES_DIR",
        "BACKTESTER_UNIVERSE_CSV_PATH",
        "BACKTESTER_DAILY_SIGNALS_DIR",
    ]:
        monkeypatch.delenv(name, raising=False)

    paths = DataPaths.from_env(env_file=env_file)

    assert paths.daily_bars_dir == tmp_path / "daily-bars"
    assert paths.frames_dir == tmp_path / "frames"
    assert paths.universe_csv_path == tmp_path / "tickers.csv"
    assert paths.daily_signals_dir == tmp_path / "daily-bars-signals"


def test_datapaths_from_env_raises_for_missing_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"BACKTESTER_DAILY_BARS_DIR={tmp_path / 'daily-bars'}\n")

    for name in [
        "BACKTESTER_DAILY_BARS_DIR",
        "BACKTESTER_FRAMES_DIR",
        "BACKTESTER_UNIVERSE_CSV_PATH",
        "BACKTESTER_DAILY_SIGNALS_DIR",
    ]:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="Missing required environment variables"):
        DataPaths.from_env(env_file=env_file)


def test_datapaths_helpers_and_default_env_resolution() -> None:
    default_env = DataPaths._resolve_env_file(None)

    assert default_env is not None
    assert default_env.name == ".env"

    explicit = DataPaths._resolve_env_file("~/custom.env")

    assert explicit == Path("~/custom.env").expanduser()


def test_loader_from_env_builds_loader_with_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"BACKTESTER_DAILY_BARS_DIR={tmp_path / 'daily-bars'}",
                f"BACKTESTER_FRAMES_DIR={tmp_path / 'frames'}",
                f"BACKTESTER_UNIVERSE_CSV_PATH={tmp_path / 'tickers.csv'}",
                f"BACKTESTER_DAILY_SIGNALS_DIR={tmp_path / 'daily-bars-signals'}",
            ]
        )
    )
    for name in [
        "BACKTESTER_DAILY_BARS_DIR",
        "BACKTESTER_FRAMES_DIR",
        "BACKTESTER_UNIVERSE_CSV_PATH",
        "BACKTESTER_DAILY_SIGNALS_DIR",
    ]:
        monkeypatch.delenv(name, raising=False)

    loader = BacktesterDataLoader.from_env(env_file=env_file)

    assert loader.paths.daily_bars_dir == tmp_path / "daily-bars"


def test_normalize_ticker_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
        BacktesterDataLoader.normalize_ticker("   ")


def test_load_price_signal_and_universe_history_use_expected_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_paths(tmp_path)
    loader = BacktesterDataLoader(paths=paths)

    price_path = paths.daily_bars_dir / "AAPL.parquet"
    signal_path = paths.daily_signals_dir / "AAPL.parquet"
    price_path.touch()
    signal_path.touch()

    universe = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "sector": ["Tech", "Tech"]})
    universe.to_csv(paths.universe_csv_path, index=False)

    parquet_calls: list[tuple[Path, list[str] | None]] = []

    def fake_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        parquet_calls.append((path, columns))
        return pd.DataFrame({"ticker": ["AAPL"], "close": [123.0]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    price_history = loader.load_price_history(" aapl ", columns=("ticker", "close"))
    signal_history = loader.load_signal_history("AAPL", columns=["ticker"])
    loaded_universe = loader.load_universe(columns=["ticker"])

    assert parquet_calls == [
        (price_path, ["ticker", "close"]),
        (signal_path, ["ticker"]),
    ]
    assert price_history["close"].tolist() == [123.0]
    assert signal_history["ticker"].tolist() == ["AAPL"]
    assert loaded_universe.columns.tolist() == ["ticker"]


def test_load_database_status_filters_ticker_and_preserves_requested_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = build_paths(tmp_path)
    loader = BacktesterDataLoader(paths=paths)
    paths.database_status_path.touch()

    read_calls: list[list[str] | None] = []

    def fake_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        read_calls.append(columns)
        frame = pd.DataFrame(
            {
                "ticker": ["aapl", "MSFT"],
                "market_cap": [10.0, 20.0],
                "status": ["ok", "ok"],
            }
        )
        if columns is None:
            return frame
        return frame.loc[:, columns]

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    filtered = loader.load_database_status("AAPL", columns=["market_cap"])
    unfiltered = loader.load_database_status(columns=["ticker", "status"])

    assert read_calls == [["ticker", "market_cap"], ["ticker", "status"]]
    assert filtered.columns.tolist() == ["market_cap"]
    assert filtered["market_cap"].tolist() == [10.0]
    assert unfiltered.columns.tolist() == ["ticker", "status"]


def test_load_database_status_raises_when_ticker_column_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = build_paths(tmp_path)
    loader = BacktesterDataLoader(paths=paths)
    paths.database_status_path.touch()

    monkeypatch.setattr(pd, "read_parquet", lambda path, columns=None: pd.DataFrame({"market_cap": [10.0]}))

    with pytest.raises(KeyError, match="Expected a 'ticker' column"):
        loader.load_database_status("AAPL", columns=["market_cap"])


def test_load_ticker_bundle_combines_price_and_status_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = build_paths(tmp_path)
    loader = BacktesterDataLoader(paths=paths)
    (paths.daily_bars_dir / "AAPL.parquet").touch()
    paths.database_status_path.touch()

    def fake_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if path == paths.daily_bars_dir / "AAPL.parquet":
            return pd.DataFrame({"ticker": ["AAPL"], "close": [111.0]})
        return pd.DataFrame({"ticker": ["AAPL", "MSFT"], "market_cap": [10.0, 20.0]})

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    bundle = loader.load_ticker_bundle("aapl", price_columns=["close"], status_columns=["market_cap"])

    assert bundle.ticker == "AAPL"
    assert bundle.price_history["close"].tolist() == [111.0]
    assert bundle.database_status.columns.tolist() == ["market_cap"]
    assert bundle.database_status["market_cap"].tolist() == [10.0]


def test_load_price_history_raises_for_missing_file(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    loader = BacktesterDataLoader(paths=paths)

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        loader.load_price_history("AAPL")


def test_columns_for_shared_frame_behaves_as_expected() -> None:
    assert BacktesterDataLoader._as_list(None) is None
    assert BacktesterDataLoader._as_list(("a", "b")) == ["a", "b"]
    assert BacktesterDataLoader._columns_for_shared_frame(None, "AAPL") is None
    assert BacktesterDataLoader._columns_for_shared_frame(["ticker", "market_cap"], "AAPL") == [
        "ticker",
        "market_cap",
    ]
    assert BacktesterDataLoader._columns_for_shared_frame(["market_cap"], None) == ["market_cap"]
    assert BacktesterDataLoader._columns_for_shared_frame(["market_cap"], "AAPL") == ["ticker", "market_cap"]
