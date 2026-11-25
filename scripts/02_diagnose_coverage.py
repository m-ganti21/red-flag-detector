"""Explain why a company produced no usable rows, using only the cached JSON.

The pull script reports that a company "had no fiscal year with all required concepts"
but not *which* concept was missing or what the company reports instead. This script
answers both, offline, by reading ``data/raw/`` -- no further SEC requests.

For each company it prints:

* the fiscal years where every required concept was found,
* per concept, the years covered and which XBRL tag supplied them,
* candidate tags the company *does* report that look like a missing concept.

That last section is the useful one. It is how you grow ``src/concepts.py`` from a
dictionary that works for Apple into one that works for banks, industrials with captive
finance arms, and companies that re-tagged their statements mid-decade.

Usage:
    python scripts/02_diagnose_coverage.py                  # every cached company
    python scripts/02_diagnose_coverage.py --tickers GE CAT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.concepts import CONCEPTS, REQUIRED_NAMES, Concept  # noqa: E402
from src.edgar import cached_company_files, read_cached_json  # noqa: E402
from src.parse import ANNUAL_FORMS, MAX_PERIOD_DAYS, MIN_PERIOD_DAYS, fiscal_year_of  # noqa: E402
import datetime as dt  # noqa: E402

# Words that hint a tag might carry a given concept. Deliberately loose: the point is to
# surface candidates for a human to judge, not to guess automatically.
SEARCH_HINTS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "sales"),
    "cogs": ("cost",),
    "sga": ("sellinggeneral", "generaladministrative", "administrative"),
    "depreciation": ("depreciation", "amortization"),
    "net_income": ("netincome", "profitloss"),
    "income_continuing": ("continuingoperations",),
    "cfo": ("operatingactivities",),
    "receivables": ("receivable",),
    "inventory": ("inventor",),
    "current_assets": ("assetscurrent",),
    "total_assets": ("assets",),
    "ppe_net": ("propertyplant",),
    "securities": ("securities", "investments"),
    "current_liabilities": ("liabilitiescurrent",),
    "long_term_debt": ("longtermdebt",),
    "total_liabilities": ("liabilities",),
    "equity": ("stockholdersequity",),
}


def annual_years_for_tag(facts: dict, taxonomy: str, tag: str, unit: str) -> set[int]:
    """Fiscal years in which this tag has a usable annual 10-K observation."""
    block = facts.get("facts", {}).get(taxonomy, {}).get(tag)
    if not block:
        return set()
    years: set[int] = set()
    for entry in block.get("units", {}).get(unit, []):
        if entry.get("form") not in ANNUAL_FORMS or not entry.get("end"):
            continue
        try:
            end = dt.date.fromisoformat(entry["end"])
        except ValueError:
            continue
        if entry.get("start"):
            try:
                start = dt.date.fromisoformat(entry["start"])
            except ValueError:
                continue
            if not MIN_PERIOD_DAYS <= (end - start).days <= MAX_PERIOD_DAYS:
                continue
        years.add(fiscal_year_of(end))
    return years


def concept_coverage(facts: dict, concept: Concept) -> tuple[set[int], dict[str, set[int]]]:
    """Years covered by the concept once every candidate tag is merged.

    Mirrors the parser: each tag contributes the years it alone covers, with
    higher-priority tags winning any year two of them share.
    """
    claimed: set[int] = set()
    contributions: dict[str, set[int]] = {}
    for tag in concept.tags:
        years = annual_years_for_tag(facts, concept.taxonomy, tag, concept.unit)
        fresh = years - claimed
        if fresh:
            contributions[tag] = fresh
            claimed |= fresh
    return claimed, contributions


def candidate_tags(facts: dict, concept: Concept, known: set[str]) -> list[tuple[str, set[int]]]:
    """Tags the company reports that look like this concept but are not in our map."""
    hints = SEARCH_HINTS.get(concept.name, ())
    if not hints:
        return []
    found: list[tuple[str, set[int]]] = []
    for tag in facts.get("facts", {}).get(concept.taxonomy, {}):
        if tag in known:
            continue
        lowered = tag.lower()
        if not any(h in lowered for h in hints):
            continue
        years = annual_years_for_tag(facts, concept.taxonomy, tag, concept.unit)
        if years:
            found.append((tag, years))
    found.sort(key=lambda pair: (-len(pair[1]), pair[0]))
    return found


def span(years: set[int]) -> str:
    if not years:
        return "none"
    lo, hi = min(years), max(years)
    return f"{lo}-{hi} ({len(years)} yrs)" if lo != hi else f"{lo} (1 yr)"


def diagnose(ticker: str, facts: dict, max_candidates: int = 4) -> None:
    name = facts.get("entityName", "?")
    print("=" * 78)
    print(f"{ticker}  --  {name}")
    print("=" * 78)

    coverage: dict[str, set[int]] = {}
    tags_used: dict[str, dict[str, set[int]]] = {}
    for concept in CONCEPTS:
        years, contributions = concept_coverage(facts, concept)
        coverage[concept.name] = years
        tags_used[concept.name] = contributions

    all_years = set().union(*coverage.values()) if coverage else set()
    complete = {y for y in all_years if all(y in coverage[n] for n in REQUIRED_NAMES)}

    print(f"\n  Years seen in filings     : {span(all_years)}")
    print(f"  Years with ALL required   : {span(complete)}")

    print("\n  Required concepts:")
    gaps: list[Concept] = []
    for concept in CONCEPTS:
        if concept.name not in REQUIRED_NAMES:
            continue
        years = coverage[concept.name]
        missing = all_years - years
        marker = "  <-- GAP" if missing else ""
        contributions = tags_used[concept.name]
        print(f"    {concept.name:<18} {span(years):<20}{marker}")
        if not contributions:
            print(f"      {'(no candidate tag reported)':<60}")
        for tag, tag_years in contributions.items():
            print(f"      {tag:<58} {span(tag_years)}")
        if missing:
            gaps.append(concept)

    if not gaps:
        print("\n  Nothing missing. If this company still produced no rows, the gap is")
        print("  in a different required concept or a year-alignment issue.")
        return

    print("\n  Tags this company reports that we are NOT using:")
    for concept in gaps:
        known = set(concept.tags)
        candidates = candidate_tags(facts, concept, known)
        missing_years = all_years - coverage[concept.name]
        print(f"\n    {concept.name}  (missing {span(missing_years)})")
        if not candidates:
            print("      no similar tags found; this company may genuinely not report it")
            continue
        for tag, years in candidates[:max_candidates]:
            covers = len(years & missing_years)
            flag = "  <== fills the gap" if covers == len(missing_years) and covers else ""
            print(f"      {tag:<62} {span(years)}{flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", help="Limit to these tickers")
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-candidates", type=int, default=4)
    args = parser.parse_args()

    directory = next(
        (p for p in (args.cache / "company_tickers.json.gz", args.cache / "company_tickers.json")
         if p.exists()),
        None,
    )
    if directory is None:
        print(f"No cached ticker directory in {args.cache}. Run 01_pull_fundamentals.py first.")
        return 1
    raw = read_cached_json(directory)
    cik_to_ticker = {f"{int(e['cik_str']):010d}": e["ticker"].upper() for e in raw.values()}

    wanted = {t.upper() for t in args.tickers} if args.tickers else None
    cached = cached_company_files(args.cache)
    if not cached:
        print(f"No cached company facts in {args.cache}. Run 01_pull_fundamentals.py first.")
        return 1

    def cik_of(path: Path) -> str:
        return path.name.removesuffix(".gz").removesuffix(".json").removeprefix("companyfacts_")

    shown = 0
    for path in cached:
        ticker = cik_to_ticker.get(cik_of(path), cik_of(path))
        if wanted and ticker not in wanted:
            continue
        diagnose(ticker, read_cached_json(path), args.max_candidates)
        shown += 1
        print()

    if shown == 0:
        print("Nothing matched. Cached companies:")
        for path in cached:
            print(f"  {cik_to_ticker.get(cik_of(path), cik_of(path))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
