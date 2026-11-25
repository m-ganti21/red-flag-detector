"""Score every company-year for accounting red flags.

Reads the tidy financials produced by 01_pull_fundamentals.py, computes the Beneish
M-Score and the accrual and divergence signals, and writes the result plus a ranked
watchlist.

Usage:
    python scripts/03_score.py
    python scripts/03_score.py --lenient       # score partial data with substitutions
    python scripts/03_score.py --top 25

Outputs:
    data/processed/scored.csv      every company-year with every signal
    data/processed/watchlist.csv   the flagged years, worst first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ratios import M_SCORE_THRESHOLD, score_all  # noqa: E402

DISPLAY_COLUMNS = [
    "ticker",
    "fiscal_year",
    "m_score",
    "accruals_ratio",
    "receivables_vs_sales",
    "revenue_growth",
    "asset_growth",
    "review_required",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infile", type=Path, default=Path("data/processed/fundamentals.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Substitute neutral values for missing Beneish components so partially "
        "reported companies still receive a score",
    )
    parser.add_argument("--top", type=int, default=15, help="Rows to print")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip winsorisation and show the uncapped scores (for comparison)",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Rank only company-years with no structural distortion "
        "(no acquisition, no extreme revenue swing, no negative margin)",
    )
    args = parser.parse_args()

    if not args.infile.exists():
        print(f"No input at {args.infile}. Run 01_pull_fundamentals.py first.")
        return 1

    fundamentals = pd.read_csv(args.infile)
    scored = score_all(
        fundamentals, neutral_substitution=args.lenient, winsorize=not args.raw
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    scored_path = args.outdir / "scored.csv"
    scored.to_csv(scored_path, index=False)

    scoreable = scored["m_score"].notna()
    flagged = scored[scored["m_score_flag"].fillna(False)].copy()
    flagged = flagged.sort_values("m_score", ascending=False)
    watchlist_path = args.outdir / "watchlist.csv"
    flagged.to_csv(watchlist_path, index=False)

    print()
    print(f"Company-years scored   : {int(scoreable.sum())} of {len(scored)}")
    print(f"Flagged (M > {M_SCORE_THRESHOLD})   : {len(flagged)}")
    if scoreable.any():
        print(f"Median M-Score         : {scored.loc[scoreable, 'm_score'].median():.2f}")
    print()
    print(f"Wrote {scored_path}")
    print(f"Wrote {watchlist_path}")

    if not scoreable.any():
        print("\nNothing scoreable. Most likely a required column is missing;")
        print("try --lenient to see which companies are close.")
        return 0

    unscored = scored.loc[~scoreable, "ticker"].value_counts()
    if not unscored.empty:
        print("\nCompany-years that could not be scored (missing components):")
        for ticker, count in unscored.items():
            print(f"  {ticker}: {count}")

    # How much of the tail is explained by something structural rather than by the
    # accounts? This is the number worth quoting.
    distorted = scored["review_required"].fillna(False)
    top_hundred = scored.loc[scoreable].nlargest(min(100, int(scoreable.sum())), "m_score")
    print()
    print("Quality review:")
    print(
        f"  Scoreable years needing review : {int((distorted & scoreable).sum())} "
        f"({(distorted & scoreable).sum() / max(1, scoreable.sum()) * 100:.1f}%)"
    )
    print(
        f"  ...among the {len(top_hundred)} highest scores  : "
        f"{int(top_hundred['review_required'].sum())} "
        f"({top_hundred['review_required'].mean() * 100:.0f}%)"
    )
    print(f"  Acquisition years (assets +25%) : {int((scored['asset_jump'] & scoreable).sum())}")
    print(f"  Indices capped by winsorising   : {int((scored['m_components_winsorized'] > 0).sum())}")

    pool = scored.loc[scoreable & ~distorted] if args.clean_only else scored.loc[scoreable]
    label = "excluding structurally distorted years" if args.clean_only else "all scoreable years"

    print(f"\nTop {args.top} by M-Score ({label}):")
    preview = pool[DISPLAY_COLUMNS].sort_values("m_score", ascending=False).head(args.top)
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(preview.to_string(index=False))

    print("\nA high score means the numbers are unusual, not that anyone cheated.")
    print("Rows marked review_required are distorted by an acquisition or a revenue")
    print("swing; the model is reading that, not the accounting. Use --clean-only to")
    print("rank without them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
