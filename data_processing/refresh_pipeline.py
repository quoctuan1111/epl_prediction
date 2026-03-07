"""
refresh_pipeline.py
===================
Orchestrates the full data refresh pipeline:

    1. fetch_current_season  — download latest EPL season CSV
    2. data_merge            — re-merge all season CSVs → merged_data.csv
    3. feature_engineering   — recompute all features → featured_data.csv
    4. Write data/last_refresh.json

Usage:
    python data_processing/refresh_pipeline.py

Importable:
    from data_processing.refresh_pipeline import run_refresh
    result = run_refresh()   # {"updated": bool, "matches": int, "timestamp": str}
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR        = os.path.join(os.path.dirname(__file__), "..")
MERGED_PATH     = os.path.join(BASE_DIR, "data", "processed_data", "merged_data.csv")
FEATURED_PATH   = os.path.join(BASE_DIR, "data", "processed_data", "featured_data.csv")
REFRESH_LOG     = os.path.join(BASE_DIR, "data", "last_refresh.json")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_fetch() -> bool:
    """Returns True if fresh data was downloaded."""
    log.info("── Step 1/3: Fetching current season data …")
    # Add parent directory to path so imports work when called from Render cron
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data_processing.fetch_current_season import fetch
    updated = fetch()
    if updated:
        log.info("   New data downloaded.")
    else:
        log.info("   No new data — continuing to rebuild features from existing data.")
    return updated


def _step_merge():
    """Re-merge all raw season CSVs into merged_data.csv."""
    log.info("── Step 2/3: Merging all seasons …")
    import pandas as pd
    from data_processing.data_merge import merge_all_seasons, validate, RAW_DIR

    merged = merge_all_seasons(RAW_DIR)
    merged = merged.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    validate(merged)

    os.makedirs(os.path.dirname(MERGED_PATH), exist_ok=True)
    merged.to_csv(MERGED_PATH, index=False)
    log.info("   Saved merged_data.csv  (%d rows).", len(merged))
    return merged


def _step_feature_engineering(merged) -> int:
    """Run feature engineering on the merged dataframe. Returns row count."""
    import pandas as pd
    log.info("── Step 3/3: Engineering features …")
    from data_processing.feature_engineering import (
        add_basic_cols, add_rolling_features, add_strength_features,
        add_h2h_features, add_elo, add_days_rest,
        ROLLING_WINDOW, ELO_K, ELO_START,
    )

    df = merged.copy()
    df = add_basic_cols(df)
    df = add_rolling_features(df, window=ROLLING_WINDOW)
    df = add_strength_features(df)
    df = add_h2h_features(df)
    df = add_elo(df, k=ELO_K, start=ELO_START)
    df = add_days_rest(df)

    os.makedirs(os.path.dirname(FEATURED_PATH), exist_ok=True)
    df.to_csv(FEATURED_PATH, index=False)
    log.info("   Saved featured_data.csv  (%d rows, %d cols).", len(df), df.shape[1])
    return len(df)


def _write_refresh_log(matches: int, updated: bool) -> str:
    """Write last_refresh.json and return ISO timestamp."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "timestamp": ts,
        "matches":   matches,
        "new_data":  updated,
    }
    os.makedirs(os.path.dirname(REFRESH_LOG), exist_ok=True)
    with open(REFRESH_LOG, "w") as f:
        json.dump(payload, f)
    return ts


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def run_refresh() -> dict:
    """
    Run the full refresh pipeline.

    Returns
    -------
    {
        "updated":   bool,   # True if new match data was downloaded
        "matches":   int,    # total row count after feature engineering
        "timestamp": str,    # UTC ISO-8601 completion time
    }
    """
    log.info("=" * 55)
    log.info("EPL REFRESH PIPELINE")
    log.info("=" * 55)

    try:
        updated = _step_fetch()
        merged  = _step_merge()
        count   = _step_feature_engineering(merged)
        ts      = _write_refresh_log(count, updated)

        log.info("=" * 55)
        log.info("✅  Pipeline complete.  matches=%d  updated=%s  ts=%s",
                 count, updated, ts)
        return {"updated": updated, "matches": count, "timestamp": ts}

    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_refresh()
    print(f"\n✅ Refresh complete — {result['matches']:,} matches | "
          f"new data: {result['updated']} | {result['timestamp']}")
