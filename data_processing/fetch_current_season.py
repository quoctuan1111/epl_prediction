"""
fetch_current_season.py
=======================
Downloads the current EPL season CSV from football-data.co.uk and saves it
to data/raw/, overwriting the previous copy only if new results are available.

Supported seasons (URL map at bottom of file — extend as needed each August).

Usage:
    python data_processing/fetch_current_season.py
    # returns exit code 0 whether updated or skipped

Importable:
    from data_processing.fetch_current_season import fetch
    updated = fetch()   # True = new data written, False = skipped
"""

import os
import io
import logging

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Map season label → football-data.co.uk URL
# Add new seasons here each August (e.g. 2025_2026 → "2526/E0.csv")
SEASON_URLS: dict[str, str] = {
    "2023_2024": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024_2025": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
}

# Current active season — the one that gets refreshed on each call
CURRENT_SEASON = "2024_2025"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _row_count(path: str) -> int:
    """Return the number of data rows in a CSV (0 if file doesn't exist)."""
    if not os.path.exists(path):
        return 0
    try:
        return len(pd.read_csv(path, encoding="latin-1"))
    except Exception:
        return 0


def fetch(season: str = CURRENT_SEASON) -> bool:
    """
    Download `season`'s CSV from football-data.co.uk.

    Returns
    -------
    True  – file was updated (new results found)
    False – skipped (no new data or download failed)
    """
    url = SEASON_URLS.get(season)
    if url is None:
        log.error("Unknown season '%s'. Add it to SEASON_URLS.", season)
        return False

    dest = os.path.join(RAW_DIR, f"{season}.csv")
    old_rows = _row_count(dest)

    log.info("Downloading %s …", url)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Download failed: %s", exc)
        return False

    # Parse the downloaded content
    try:
        new_df = pd.read_csv(io.StringIO(resp.text), encoding="latin-1")
        # Drop completely blank trailing rows
        new_df = new_df.dropna(how="all")
        new_rows = len(new_df)
    except Exception as exc:
        log.error("Could not parse downloaded CSV: %s", exc)
        return False

    if new_rows <= old_rows:
        log.info(
            "⏭  Skipped — no new results (remote: %d rows, local: %d rows).",
            new_rows, old_rows,
        )
        return False

    # Write updated file
    os.makedirs(RAW_DIR, exist_ok=True)
    new_df.to_csv(dest, index=False)
    log.info(
        "✅  Updated %s.csv  (%d → %d rows, +%d new results).",
        season, old_rows, new_rows, new_rows - old_rows,
    )
    return True


def fetch_all() -> dict[str, bool]:
    """Fetch every season in SEASON_URLS. Returns {season: updated}."""
    return {season: fetch(season) for season in SEASON_URLS}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    updated = fetch(CURRENT_SEASON)
    if not updated:
        print("No new data available. Local file is already up-to-date.")
