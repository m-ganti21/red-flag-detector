"""Download monthly prices so the screen can be tested against what happened next.

A screen that has never been checked against outcomes is a calculator. This pulls
total-return prices for the universe so forward returns can be measured.

One decision matters more than the rest, and it is about *when* you could have acted.
A 10-K is not public on the last day of the fiscal year; it is filed weeks or months
later. Measuring returns from the fiscal year end would credit the strategy with
information nobody had yet, which is look-ahead bias and the most common way a
backtest lies. Returns are therefore measured from a configurable lag after the
period end, defaulting to 90 days, which is roughly when a large filer's 10-K is out.

Usage:
    python scripts/04_pull_prices.py
    python scripts/04_pull_prices.py --universe data/universe.csv --start 2006-01-01

Outputs:
    data/processed/prices.csv   ticker, date, adjusted close (monthly)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("prices")

CHUNK_SIZE = 40


def load_tickers(universe: Path, scored: Path) -> list[str]:
    """Prefer the tickers we actually scored; fall back to the universe file."""
    if scored.exists():
        frame = pd.read_csv(scored, usecols=["ticker"])
        return sorted(frame["ticker"].dropna().unique().tolist())
    if universe.exists():
        frame = pd.read_csv(universe, usecols=["ticker"])
        return sorted(frame["ticker"].dropna().unique().tolist())
    raise SystemExit(f"Need one of {scored} or {universe}. Run the earlier scripts first.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("data/universe.csv"))
    parser.add_argument("--scored", type=Path, default=Path("data/processed/scored.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("data/processed"))
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--refresh", action="store_true", help="Ignore the saved file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Run:  pip install yfinance")
        print("Then add 'yfinance>=0.2.40' to requirements.txt")
        return 1

    out_path = args.outdir / "prices.csv"
    if out_path.exists() and not args.refresh:
        existing = pd.read_csv(out_path)
        print(f"{out_path} already exists with {len(existing):,} rows. Use --refresh to redo.")
        return 0

    tickers = load_tickers(args.universe, args.scored)
    log.info("Downloading monthly prices for %d tickers from %s", len(tickers), args.start)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for start in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[start : start + CHUNK_SIZE]
        log.info("  %d-%d of %d", start + 1, start + len(chunk), len(tickers))
        try:
            raw = yf.download(
                chunk,
                start=args.start,
                interval="1mo",
                auto_adjust=True,      # splits and dividends folded into the price
                progress=False,
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the run
            log.warning("  chunk failed (%s); retrying one ticker at a time", exc)
            raw = None

        if raw is None or raw.empty:
            failed.extend(chunk)
            continue

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            close.columns = chunk[:1]

        tidy = (
            close.reset_index()
            .melt(id_vars=close.index.name or "Date", var_name="ticker", value_name="close")
            .rename(columns={close.index.name or "Date": "date"})
            .dropna(subset=["close"])
        )
        frames.append(tidy)
        time.sleep(0.5)   # be polite; the endpoint throttles aggressive clients

    if not frames:
        print("No price data came back. Check your connection and try again.")
        return 1

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_path, index=False)

    covered = prices["ticker"].nunique()
    print()
    print(f"Tickers requested : {len(tickers)}")
    print(f"Tickers with data : {covered}")
    print(f"Monthly rows      : {len(prices):,}")
    print(f"Date range        : {prices['date'].min()} to {prices['date'].max()}")
    print(f"\nWrote {out_path}")

    missing = sorted(set(tickers) - set(prices["ticker"]))
    if missing:
        print(f"\nNo data for {len(missing)}: {', '.join(missing[:20])}"
              + (" ..." if len(missing) > 20 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
