"""Test whether the red-flag signals predicted forward returns.

Usage:
    python scripts/05_backtest.py
    python scripts/05_backtest.py --lag 60 --hold 730
    python scripts/05_backtest.py --no-sector-neutral

Outputs:
    data/processed/backtest.csv       every company-year with its forward return
    data/processed/backtest_summary.txt  the numbers for the README
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import (  # noqa: E402
    DEFAULT_FILING_LAG_DAYS,
    DEFAULT_HOLDING_DAYS,
    attach_prices,
    decile_table,
    spread_test,
    yearly_spread,
)

SIGNALS = ["accruals_ratio", "m_score", "receivables_vs_sales", "dso_change"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, default=Path("data/processed/scored.csv"))
    parser.add_argument("--prices", type=Path, default=Path("data/processed/prices.csv"))
    parser.add_argument("--universe", type=Path, default=Path("data/universe.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("data/processed"))
    parser.add_argument("--lag", type=int, default=DEFAULT_FILING_LAG_DAYS,
                        help="Days after period end before the filing is public")
    parser.add_argument("--hold", type=int, default=DEFAULT_HOLDING_DAYS)
    parser.add_argument("--no-sector-neutral", action="store_true",
                        help="Neutralise by year only, leaving any sector tilt in")
    args = parser.parse_args()

    for path in (args.scored, args.prices):
        if not path.exists():
            print(f"Missing {path}. Run the earlier scripts first.")
            return 1

    scored = pd.read_csv(args.scored)
    prices = pd.read_csv(args.prices)

    if args.universe.exists():
        sectors = pd.read_csv(args.universe)[["ticker", "sector"]]
        scored = scored.merge(sectors, on="ticker", how="left")

    scored = scored[scored["m_score"].notna()]
    panel = attach_prices(scored, prices, filing_lag_days=args.lag, holding_days=args.hold)

    usable = panel["forward_return"].notna()
    neutral = ("entry_year",) if args.no_sector_neutral else ("entry_year", "sector")
    if "sector" not in panel.columns:
        neutral = ("entry_year",)

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("BACKTEST")
    say(f"  entry   : period end + {args.lag} days (filing lag, no look-ahead)")
    say(f"  holding : {args.hold} days")
    say(f"  peers   : {' x '.join(neutral)}")
    say(f"  usable  : {int(usable.sum())} of {len(panel)} scoreable company-years")
    say()

    say("TOP-MINUS-BOTTOM QUINTILE SPREAD (negative = high values underperformed)")
    results = []
    for signal in SIGNALS:
        try:
            result = spread_test(panel, signal, neutral_by=neutral)
        except (ValueError, KeyError) as exc:
            say(f"  {signal:<18} skipped ({exc})")
            continue
        results.append(result)
        say(f"  {result}")
    say()
    say("  *** p<0.01   ** p<0.05")
    say()

    best = min(results, key=lambda r: r.p_value) if results else None
    if best is not None:
        say(f"DECILES FOR THE STRONGEST SIGNAL ({best.signal})")
        table = decile_table(panel, best.signal)
        for decile, row in table.iterrows():
            say(f"  {decile:2d}  n={int(row['count']):4d}  "
                f"mean {row['mean'] * 100:+6.1f}%  median {row['median'] * 100:+6.1f}%")
        say()

        yearly = yearly_spread(panel, best.signal)
        if not yearly.empty:
            wins = int((yearly["spread"] < 0).sum())
            say(f"YEAR BY YEAR ({best.signal})")
            say(f"  high quintile underperformed in {wins} of {len(yearly)} years")
            say()

    # The M-Score's sales-growth term makes it part growth proxy. Holding growth
    # roughly fixed is the cleanest way to see whether the flag means anything.
    if "sgi" in panel.columns and "m_score_flag" in panel.columns:
        from scipy import stats

        # Peer-relative, like every other test here: a raw-return comparison would
        # conflate the signal with whichever years happened to hold more flags.
        from src.backtest import add_excess_returns

        measured = add_excess_returns(panel[usable], neutral_by=neutral)
        slow = measured[measured["sgi"] <= measured["sgi"].median()]
        flagged = slow[slow["m_score_flag"] == True]  # noqa: E712
        rest = slow[slow["m_score_flag"] != True]  # noqa: E712
        if len(flagged) >= 5:
            t, p = stats.ttest_ind(
                flagged["excess_return"], rest["excess_return"], equal_var=False
            )
            say("M-SCORE FLAG AMONG SLOW-GROWTH COMPANIES ONLY")
            say("  (the flag loads on growth; holding growth roughly fixed isolates it)")
            say(f"  flagged   n={len(flagged):4d}  mean excess {flagged['excess_return'].mean() * 100:+6.1f}%")
            say(f"  others    n={len(rest):4d}  mean excess {rest['excess_return'].mean() * 100:+6.1f}%")
            say(f"  difference {(flagged['excess_return'].mean() - rest['excess_return'].mean()) * 100:+.1f} pp"
                f"  p={p:.3f}  (n is small; indicative only)")
            say()

    say("CAVEATS")
    say("  Universe is the CURRENT S&P 500, so companies that collapsed and were")
    say("  delisted are absent - exactly the ones a fraud screen should catch.")
    say("  This survivorship bias understates any real signal.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.outdir / "backtest.csv", index=False)
    (args.outdir / "backtest_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {args.outdir / 'backtest.csv'}")
    print(f"Wrote {args.outdir / 'backtest_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
