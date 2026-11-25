"""Download annual financials from SEC EDGAR and save them as a tidy table.

Usage:
    python scripts/01_pull_fundamentals.py                      # the 8-company smoke test
    python scripts/01_pull_fundamentals.py --tickers AAPL MSFT  # specific companies
    python scripts/01_pull_fundamentals.py --universe data/universe.txt
    python scripts/01_pull_fundamentals.py --tickers AAPL --refresh

Outputs:
    data/processed/fundamentals.csv  one row per company-year
    data/processed/restatements.csv  where later filings revised earlier numbers
    data/raw/                        cached JSON, so reruns are free
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Make "src" importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.edgar import EdgarClient, EdgarError  # noqa: E402
from src.parse import annual_frame, restatement_report  # noqa: E402

# A deliberately mixed smoke-test set: a hardware giant, a software firm, two retailers
# with January year-ends, a bank with an unclassified balance sheet, an industrial, a
# pharma, and a company that has restated. If the pipeline survives these eight, the
# tag-fallback logic is doing its job.
SMOKE_TEST_TICKERS = ["AAPL", "MSFT", "WMT", "TGT", "JPM", "CAT", "PFE", "GE"]

log = logging.getLogger("pull")


def load_universe(path: Path) -> list[str]:
    """Read tickers from a text file (one per line) or a CSV with a 'ticker' column."""
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        column = next(
            (c for c in frame.columns if c.strip().lower() in {"ticker", "symbol"}), None
        )
        if column is None:
            raise SystemExit(f"{path} needs a 'ticker' or 'symbol' column")
        values = frame[column].dropna().tolist()
    else:
        values = [line.strip() for line in path.read_text().splitlines()]
    return [str(v).upper().strip() for v in values if str(v).strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", help="Tickers to pull, e.g. AAPL MSFT")
    parser.add_argument("--universe", type=Path, help="File of tickers, one per line")
    parser.add_argument(
        "--min-required",
        type=int,
        default=None,
        help="Keep a year if at least this many required concepts are present "
        "(default: require all of them)",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore the disk cache")
    parser.add_argument("--outdir", type=Path, default=Path("data/processed"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    load_dotenv()

    if args.universe:
        tickers = load_universe(args.universe)
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = SMOKE_TEST_TICKERS
        log.info("No tickers given, running the %d-company smoke test", len(tickers))

    try:
        client = EdgarClient()
    except EdgarError as exc:
        log.error("%s", exc)
        return 1

    log.info("Fetching the ticker directory from SEC.gov")
    cik_map = client.ticker_to_cik(refresh=args.refresh)

    unknown = [t for t in tickers if t not in cik_map]
    if unknown:
        log.warning("Not in the SEC directory, skipping: %s", ", ".join(unknown))
    tickers = [t for t in tickers if t in cik_map]
    if not tickers:
        log.error("No usable tickers left")
        return 1

    financial_frames: list[pd.DataFrame] = []
    restatement_frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    for ticker in tqdm(tickers, desc="Companies", unit="co"):
        cik = cik_map[ticker]
        try:
            facts = client.company_facts(cik, refresh=args.refresh)
        except EdgarError as exc:
            failures.append((ticker, str(exc)))
            continue

        annual = annual_frame(facts, min_required=args.min_required)
        if annual.empty:
            failures.append((ticker, "no fiscal year had all required concepts"))
        else:
            annual.insert(0, "ticker", ticker)
            financial_frames.append(annual)

        restated = restatement_report(facts)
        if not restated.empty:
            restated.insert(0, "ticker", ticker)
            restatement_frames.append(restated)

    if not financial_frames:
        log.error("Nothing usable came back. Try --min-required 10 to loosen the filter.")
        return 1

    fundamentals = pd.concat(financial_frames, ignore_index=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    fundamentals_path = args.outdir / "fundamentals.csv"
    fundamentals.to_csv(fundamentals_path, index=False)

    restatements = (
        pd.concat(restatement_frames, ignore_index=True)
        if restatement_frames
        else pd.DataFrame()
    )
    restatements_path = args.outdir / "restatements.csv"
    restatements.to_csv(restatements_path, index=False)

    # -- summary ---------------------------------------------------------
    print()
    print(f"Companies requested : {len(tickers)}")
    print(f"Companies with data : {fundamentals['ticker'].nunique()}")
    print(f"Company-years       : {len(fundamentals)}")
    print(
        f"Years covered       : {int(fundamentals['fiscal_year'].min())}"
        f"-{int(fundamentals['fiscal_year'].max())}"
    )
    print(f"Restatements found  : {len(restatements)}")
    print()
    print(f"Wrote {fundamentals_path}")
    print(f"Wrote {restatements_path}")

    if failures:
        print(f"\n{len(failures)} company(ies) produced nothing:")
        for ticker, reason in failures:
            print(f"  {ticker}: {reason}")

    print("\nMost recent year per company:")
    latest = fundamentals.sort_values("fiscal_year").groupby("ticker").tail(1)
    preview = latest[["ticker", "fiscal_year", "revenue", "net_income", "total_assets"]]
    with pd.option_context("display.float_format", lambda v: f"{v:,.0f}"):
        print(preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
