"""A small, polite client for the SEC EDGAR structured-data APIs.

The SEC publishes every number that companies tag in their filings through two free
JSON endpoints, with no API key and no registration:

* ``https://www.sec.gov/files/company_tickers.json`` -- ticker to CIK directory.
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` -- every XBRL fact
  a company has ever reported, across every filing.

The SEC asks automated clients to identify themselves with a contact email in the
User-Agent header and to stay under 10 requests per second. Ignoring either gets your
IP blocked, so both are enforced here rather than left to the caller.

See https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class EdgarError(RuntimeError):
    """Raised when EDGAR cannot be reached or returns something unusable."""


class EdgarClient:
    """Fetches EDGAR JSON with rate limiting, retries, and an on-disk cache.

    Args:
        user_agent: Contact string the SEC requires, e.g. ``"Jane Doe jane@uni.edu"``.
            Read from the ``SEC_USER_AGENT`` environment variable when not passed.
        cache_dir: Directory for cached raw JSON. Re-running the pipeline then costs
            no network calls, which matters because a 500-company pull takes minutes.
        min_interval: Seconds to wait between requests. The SEC limit is 10 per second;
            the default of 0.12 leaves headroom.
        max_retries: Attempts per URL before giving up.

    Example:
        >>> client = EdgarClient(user_agent="Jane Doe jane@uni.edu")
        >>> cik = client.ticker_to_cik()["AAPL"]
        >>> facts = client.company_facts(cik)
        >>> facts["entityName"]
        'Apple Inc.'
    """

    def __init__(
        self,
        user_agent: str | None = None,
        cache_dir: str | Path = "data/raw",
        min_interval: float = 0.12,
        max_retries: int = 4,
    ) -> None:
        resolved = (user_agent or os.environ.get("SEC_USER_AGENT", "")).strip()
        if not resolved or "@" not in resolved:
            raise EdgarError(
                "The SEC requires a User-Agent containing a contact email.\n"
                'Set SEC_USER_AGENT="Your Name your.email@example.com" in your .env file, '
                "or pass user_agent= directly."
            )
        if "example.com" in resolved.lower() or resolved.lower().startswith("your name"):
            raise EdgarError(
                f"SEC_USER_AGENT is still the placeholder ({resolved!r}).\n"
                "Edit .env and put your own name and email there before contacting the SEC."
            )
        self.user_agent = resolved
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            }
        )
        self._last_request_at = 0.0

    # -- internals ----------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep just long enough to stay under the SEC's rate limit."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _get_json(self, url: str) -> dict[str, Any]:
        """GET a URL and parse JSON, retrying on transient failures."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self._session.get(url, timeout=30)
                if response.status_code == 404:
                    raise EdgarError(f"EDGAR has no document at {url}")
                if response.status_code == 429:
                    # Rate limited: back off hard before trying again.
                    wait = 2.0 * attempt
                    log.warning("Rate limited by EDGAR, sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except EdgarError:
                raise
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                wait = 1.5 * attempt
                log.warning(
                    "Attempt %d/%d failed for %s (%s); retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise EdgarError(f"Gave up on {url} after {self.max_retries} attempts") from last_error

    def _cached_json(self, cache_name: str, url: str, refresh: bool = False) -> dict[str, Any]:
        """Return parsed JSON for ``url``, reading from and writing to the disk cache.

        Payloads are stored gzipped. A company's facts run to several megabytes of
        plain JSON, so a 400-company universe would need gigabytes uncompressed;
        gzip cuts that by roughly 90% for a few milliseconds of CPU per file.
        Uncompressed files written by earlier versions are still read.
        """
        gz_path = self.cache_dir / f"{cache_name}.gz"
        legacy_path = self.cache_dir / cache_name

        if not refresh:
            for path, opener in ((gz_path, gzip.open), (legacy_path, open)):
                if not path.exists():
                    continue
                try:
                    with opener(path, "rt", encoding="utf-8") as handle:
                        return json.load(handle)
                except (json.JSONDecodeError, OSError, EOFError):
                    log.warning("Cache file %s is unreadable; refetching", path)

        payload = self._get_json(url)
        with gzip.open(gz_path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        # Drop any stale uncompressed copy so the two cannot drift apart.
        legacy_path.unlink(missing_ok=True)
        return payload

    # -- public API ---------------------------------------------------------

    def ticker_to_cik(self, refresh: bool = False) -> dict[str, str]:
        """Map every exchange-listed ticker to its zero-padded 10-digit CIK.

        CIKs are the SEC's company identifier. The API needs them padded to 10 digits
        ("0000320193"), while the directory file stores them as plain integers.

        Returns:
            ``{"AAPL": "0000320193", "MSFT": "0000789019", ...}``
        """
        raw = self._cached_json("company_tickers.json", TICKER_URL, refresh=refresh)
        mapping: dict[str, str] = {}
        for entry in raw.values():
            ticker = str(entry["ticker"]).upper().strip()
            mapping[ticker] = f"{int(entry['cik_str']):010d}"
        if not mapping:
            raise EdgarError("Ticker directory came back empty")
        return mapping

    def company_facts(self, cik: str, refresh: bool = False) -> dict[str, Any]:
        """Fetch every XBRL fact a company has reported.

        Args:
            cik: Zero-padded 10-digit CIK, as returned by :meth:`ticker_to_cik`.
            refresh: Bypass the disk cache and refetch.

        Returns:
            The raw companyfacts payload: ``{"cik", "entityName", "facts": {...}}``.
        """
        cik = str(cik).zfill(10)
        return self._cached_json(
            f"companyfacts_{cik}.json",
            COMPANY_FACTS_URL.format(cik=cik),
            refresh=refresh,
        )

    def submissions(self, cik: str, refresh: bool = False) -> dict[str, Any]:
        """Fetch a company's filing history (used later for 10-K text and SIC code)."""
        cik = str(cik).zfill(10)
        return self._cached_json(
            f"submissions_{cik}.json",
            SUBMISSIONS_URL.format(cik=cik),
            refresh=refresh,
        )

    def is_cached(self, cik: str) -> bool:
        """Whether this company's facts are already on disk, compressed or not."""
        stem = f"companyfacts_{str(cik).zfill(10)}.json"
        return (self.cache_dir / f"{stem}.gz").exists() or (self.cache_dir / stem).exists()


def read_cached_json(path: Path) -> dict[str, Any]:
    """Read a cache file whether or not it is gzipped."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def cached_company_files(cache_dir: str | Path) -> list[Path]:
    """Every cached companyfacts file, newest format preferred when both exist."""
    cache_dir = Path(cache_dir)
    by_stem: dict[str, Path] = {}
    for path in sorted(cache_dir.glob("companyfacts_*.json*")):
        stem = path.name.removesuffix(".gz")
        # A .gz written later supersedes a legacy plain file of the same name.
        if stem not in by_stem or path.suffix == ".gz":
            by_stem[stem] = path
    return [by_stem[k] for k in sorted(by_stem)]
