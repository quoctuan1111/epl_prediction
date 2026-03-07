"""
upcoming_fixtures.py
====================
Fetches upcoming EPL fixtures from the football-data.org REST API (free tier).
Falls back to scraping football-data.co.uk's current season CSV for matches
that have no result yet (future-dated rows), if the API key is not available.

Environment variable:
    FOOTBALL_DATA_API_KEY  — free key from https://www.football-data.org/client/register
                             If absent, the fallback (CSV-based) method is used.

Usage:
    python data_processing/upcoming_fixtures.py

Importable:
    from data_processing.upcoming_fixtures import get_fixtures
    fixtures = get_fixtures()  # list of dicts
"""

import os
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR        = os.path.join(os.path.dirname(__file__), "..")
FEATURED_CSV    = os.path.join(BASE_DIR, "data", "processed_data", "featured_data.csv")
RAW_CURRENT_CSV = os.path.join(BASE_DIR, "data", "raw", "2024_2025.csv")

API_URL = "https://api.football-data.org/v4/competitions/PL/matches"

TIMEOUT = 15

# Team name map — API names → your model's internal names
# Extend if you see mismatches in logs
API_TEAM_MAP: dict[str, str] = {
    "Arsenal FC":                  "Arsenal",
    "Aston Villa FC":              "Aston Villa",
    "AFC Bournemouth":             "Bournemouth",
    "Brentford FC":                "Brentford",
    "Brighton & Hove Albion FC":   "Brighton",
    "Chelsea FC":                  "Chelsea",
    "Crystal Palace FC":           "Crystal Palace",
    "Everton FC":                  "Everton",
    "Fulham FC":                   "Fulham",
    "Ipswich Town FC":             "Ipswich Town",
    "Leicester City FC":           "Leicester City",
    "Liverpool FC":                "Liverpool",
    "Manchester City FC":          "Manchester City",
    "Manchester United FC":        "Manchester United",
    "Newcastle United FC":         "Newcastle United",
    "Nottingham Forest FC":        "Nottingham Forest",
    "Southampton FC":              "Southampton",
    "Tottenham Hotspur FC":        "Tottenham Hotspur",
    "West Ham United FC":          "West Ham United",
    "Wolverhampton Wanderers FC":  "Wolverhampton Wanderers",
    # Short forms that might appear
    "Man City":    "Manchester City",
    "Man United":  "Manchester United",
    "Spurs":       "Tottenham Hotspur",
    "Wolves":      "Wolverhampton Wanderers",
    "Nott'm Forest": "Nottingham Forest",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    return API_TEAM_MAP.get(name, name)


def _last_match_dates() -> dict[str, date]:
    """Read featured_data.csv and return {team: last_match_date}."""
    if not os.path.exists(FEATURED_CSV):
        return {}
    try:
        df = pd.read_csv(FEATURED_CSV, usecols=["Date", "HomeTeam", "AwayTeam"],
                         parse_dates=["Date"])
        last: dict[str, date] = {}
        for _, row in df.iterrows():
            d = row["Date"].date() if pd.notna(row["Date"]) else None
            if d is None:
                continue
            for team in (row["HomeTeam"], row["AwayTeam"]):
                if team not in last or d > last[team]:
                    last[team] = d
        return last
    except Exception as exc:
        log.warning("Could not read featured_data.csv for rest days: %s", exc)
        return {}


def _compute_rest(team: str, fixture_date: date,
                  last_dates: dict[str, date]) -> int:
    """Days since last match. Returns 7 as a sensible default."""
    last = last_dates.get(team)
    if last is None:
        return 7
    return max(1, (fixture_date - last).days)


# ---------------------------------------------------------------------------
# Method A: football-data.org API
# ---------------------------------------------------------------------------

def _fetch_via_api(last_dates: dict[str, date]) -> list[dict]:
    """Use football-data.org v4 API to get upcoming PL fixtures."""
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        return []
    today = date.today()
    date_from = today.strftime("%Y-%m-%d")
    date_to   = (today + timedelta(days=14)).strftime("%Y-%m-%d")

    params  = {"status": "SCHEDULED", "dateFrom": date_from, "dateTo": date_to}
    headers = {"X-Auth-Token": api_key}

    try:
        resp = requests.get(API_URL, params=params, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("football-data.org API request failed: %s", exc)
        return []

    data = resp.json()
    matches = data.get("matches", [])

    fixtures = []
    for m in matches:
        raw_home = m.get("homeTeam", {}).get("name", "")
        raw_away = m.get("awayTeam", {}).get("name", "")
        utc_date = m.get("utcDate", "")

        try:
            fixture_date = datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            continue

        home = _normalise(raw_home)
        away = _normalise(raw_away)

        fixtures.append({
            "home_team":  home,
            "away_team":  away,
            "date":       fixture_date.isoformat(),
            "home_rest":  _compute_rest(home, fixture_date, last_dates),
            "away_rest":  _compute_rest(away, fixture_date, last_dates),
        })

    log.info("API: found %d upcoming fixtures.", len(fixtures))
    return fixtures


# ---------------------------------------------------------------------------
# Method B: CSV fallback — unplayed future rows in current season file
# ---------------------------------------------------------------------------

def _fetch_via_csv(last_dates: dict[str, date]) -> list[dict]:
    """
    Parse the current season CSV (2024_2025.csv) for rows that have a future
    date but no FTR result (i.e. they are published fixture placeholders).
    football-data.co.uk includes upcoming fixtures with blank FTR/FTHG/FTAG.
    """
    if not os.path.exists(RAW_CURRENT_CSV):
        log.warning("Current season CSV not found at %s. Run fetch_current_season.py first.",
                    RAW_CURRENT_CSV)
        return []

    try:
        df = pd.read_csv(RAW_CURRENT_CSV, encoding="latin-1")
        df = df.dropna(how="all")
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed",
                                     errors="coerce")
    except Exception as exc:
        log.warning("Could not parse current season CSV: %s", exc)
        return []

    today = pd.Timestamp(date.today())
    # Upcoming = future date AND no final result
    upcoming = df[
        (df["Date"] >= today) &
        (df["FTR"].isna() | (df["FTR"].astype(str).str.strip() == ""))
    ].sort_values("Date")

    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_processing.data_merge import TEAM_NAME_MAP  # reuse normalisation

    fixtures = []
    for _, row in upcoming.iterrows():
        raw_home = str(row.get("HomeTeam", "")).strip()
        raw_away = str(row.get("AwayTeam", "")).strip()
        home = TEAM_NAME_MAP.get(raw_home, raw_home)
        away = TEAM_NAME_MAP.get(raw_away, raw_away)
        fixture_date = row["Date"].date()

        fixtures.append({
            "home_team": home,
            "away_team": away,
            "date":      fixture_date.isoformat(),
            "home_rest": _compute_rest(home, fixture_date, last_dates),
            "away_rest": _compute_rest(away, fixture_date, last_dates),
        })

    log.info("CSV fallback: found %d upcoming fixtures.", len(fixtures))
    return fixtures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fixtures(limit: int = 20) -> list[dict]:
    """
    Return upcoming EPL fixtures (up to `limit` matches).

    Tries football-data.org API first (requires FOOTBALL_DATA_API_KEY env var).
    Falls back to parsing the current-season CSV for future-dated rows.

    Each item: {"home_team", "away_team", "date", "home_rest", "away_rest"}
    """
    last_dates = _last_match_dates()

    fixtures = []
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if api_key:
        fixtures = _fetch_via_api(last_dates)

    if not fixtures:
        log.info("Using CSV fallback for fixtures.")
        fixtures = _fetch_via_csv(last_dates)

    return fixtures[:limit]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    games = get_fixtures()
    if not games:
        print("No upcoming fixtures found.")
    else:
        print(f"\n{'DATE':<12} {'HOME TEAM':<28} {'AWAY TEAM':<28} REST(H/A)")
        print("-" * 80)
        for g in games:
            print(f"{g['date']:<12} {g['home_team']:<28} {g['away_team']:<28} "
                  f"{g['home_rest']}d / {g['away_rest']}d")
