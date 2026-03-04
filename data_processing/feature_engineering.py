"""
feature_engineering.py
=======================
Reads data/processed_data/merged_data.csv and engineers all features
needed for model training.

Features created
----------------
Basic
  TotalGoals      : FTHG + FTAG
  GoalDiff        : FTHG - FTAG
  Over_2_5        : 1 if TotalGoals > 2.5 else 0
  Result_encoded  : H→2, D→1, A→0  (for model target)

Rolling (last N matches, computed per team, time-ordered, NO data leakage)
  home_scored_roll5     : avg goals scored by home team (home games only, last 5)
  home_conceded_roll5   : avg goals conceded by home team (home games only, last 5)
  away_scored_roll5     : avg goals scored by away team (away games only, last 5)
  away_conceded_roll5   : avg goals conceded by away team (away games only, last 5)
  home_pts_roll5        : avg points earned by home team (all games, last 5)
  away_pts_roll5        : avg points earned by away team (all games, last 5)
  home_gd_roll5         : avg goal diff for home team (all games, last 5)
  away_gd_roll5         : avg goal diff for away team (all games, last 5)

Elo
  home_elo_before       : home team Elo rating before this match
  away_elo_before       : away team Elo rating before this match
  elo_diff              : home_elo_before - away_elo_before

Rest
  home_days_rest        : days since home team's previous match
  away_days_rest        : days since away team's previous match

Output → data/processed_data/featured_data.csv

Usage:
    python data_processing/feature_engineering.py
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data", "merged_data.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data", "featured_data.csv")

ROLLING_WINDOW = 5
ELO_K          = 32
ELO_START      = 1500


# ---------------------------------------------------------------------------
# 1. Basic derived columns
# ---------------------------------------------------------------------------

def add_basic_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalGoals"]     = df["FTHG"] + df["FTAG"]
    df["GoalDiff"]       = df["FTHG"] - df["FTAG"]
    df["Over_2_5"]       = (df["TotalGoals"] > 2.5).astype(int)
    df["Result_encoded"] = df["FTR"].map({"H": 2, "D": 1, "A": 0})
    return df


def _points_from_result(ftr: str, side: str) -> int:
    """Return points (3/1/0) earned by 'side' (H or A) given full-time result."""
    if ftr == side:
        return 3
    elif ftr == "D":
        return 1
    return 0


# ---------------------------------------------------------------------------
# 2. Rolling features (strict look-back — no leakage)
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    For each match row, compute rolling stats from the team's OWN past matches
    using a strictly time-ordered, expanding/rolling window.

    Approach:
      - Build two team-game history series: one for home games, one for all games.
      - For each match, look up the team's history *before* this date and take
        the last `window` games.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Dictionaries: team → list of past game records (appended as we iterate)
    # Each record: {"date": ..., "scored": ..., "conceded": ..., "pts": ..., "gd": ..., "venue": "H"/"A"}
    history: dict[str, list] = defaultdict(list)

    # Output columns — pre-fill with NaN
    cols = [
        "home_scored_roll5", "home_conceded_roll5",
        "away_scored_roll5", "away_conceded_roll5",
        "home_pts_roll5",    "away_pts_roll5",
        "home_gd_roll5",     "away_gd_roll5",
    ]
    for col in cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        ht = row["HomeTeam"]
        at = row["AwayTeam"]

        def _rolling_stats(team: str, venue_filter: str | None):
            """
            Compute rolling mean of scored/conceded/pts/gd over the last `window`
            past games for `team`. If venue_filter is 'H' or 'A', restrict to
            home-only or away-only games.
            """
            past = history[team]
            if venue_filter:
                past = [g for g in past if g["venue"] == venue_filter]
            last_n = past[-window:]
            if not last_n:
                return np.nan, np.nan, np.nan, np.nan
            scored    = np.mean([g["scored"]   for g in last_n])
            conceded  = np.mean([g["conceded"] for g in last_n])
            pts       = np.mean([g["pts"]      for g in last_n])
            gd        = np.mean([g["gd"]       for g in last_n])
            return scored, conceded, pts, gd

        # --- Home team stats (home venue filter) ---
        hs, hc, hp, hg = _rolling_stats(ht, "H")
        df.at[idx, "home_scored_roll5"]   = hs
        df.at[idx, "home_conceded_roll5"] = hc
        df.at[idx, "home_pts_roll5"]      = hp
        df.at[idx, "home_gd_roll5"]       = hg

        # --- Away team stats (away venue filter) ---
        as_, ac, ap, ag = _rolling_stats(at, "A")
        df.at[idx, "away_scored_roll5"]   = as_
        df.at[idx, "away_conceded_roll5"] = ac
        df.at[idx, "away_pts_roll5"]      = ap
        df.at[idx, "away_gd_roll5"]       = ag

        # --- Update history AFTER computing features (avoid leakage) ---
        h_pts = _points_from_result(row["FTR"], "H")
        a_pts = _points_from_result(row["FTR"], "A")
        h_gd  = int(row["FTHG"]) - int(row["FTAG"])
        a_gd  = -h_gd

        history[ht].append({
            "date": row["Date"], "scored": row["FTHG"], "conceded": row["FTAG"],
            "pts": h_pts, "gd": h_gd, "venue": "H"
        })
        history[at].append({
            "date": row["Date"], "scored": row["FTAG"], "conceded": row["FTHG"],
            "pts": a_pts, "gd": a_gd, "venue": "A"
        })

    return df


# ---------------------------------------------------------------------------
# 3. Elo ratings
# ---------------------------------------------------------------------------

def _expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


def _elo_outcome(ftr: str) -> tuple[float, float]:
    """Return (home_score, away_score) in Elo terms (1/0/0.5)."""
    if ftr == "H":
        return 1.0, 0.0
    elif ftr == "A":
        return 0.0, 1.0
    return 0.5, 0.5


def add_elo(df: pd.DataFrame, k: int = ELO_K, start: int = ELO_START) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    elo: dict[str, float] = defaultdict(lambda: float(start))

    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        he, ae = elo[ht], elo[at]

        home_elos.append(he)
        away_elos.append(ae)

        exp_h = _expected_score(he, ae)
        exp_a = _expected_score(ae, he)
        s_h, s_a = _elo_outcome(row["FTR"])

        elo[ht] = he + k * (s_h - exp_h)
        elo[at] = ae + k * (s_a - exp_a)

    df["home_elo_before"] = home_elos
    df["away_elo_before"] = away_elos
    df["elo_diff"]        = df["home_elo_before"] - df["away_elo_before"]
    return df


# ---------------------------------------------------------------------------
# 4. Days rest
# ---------------------------------------------------------------------------

def add_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("Date").reset_index(drop=True)
    last_match: dict[str, pd.Timestamp] = {}

    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        date   = row["Date"]

        home_rest.append((date - last_match[ht]).days if ht in last_match else np.nan)
        away_rest.append((date - last_match[at]).days if at in last_match else np.nan)

        last_match[ht] = date
        last_match[at] = date

    df["home_days_rest"] = home_rest
    df["away_days_rest"] = away_rest
    return df


# ---------------------------------------------------------------------------
# 5. Validate & summarise
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    feature_cols = [
        "home_scored_roll5", "home_conceded_roll5",
        "away_scored_roll5", "away_conceded_roll5",
        "home_pts_roll5",    "away_pts_roll5",
        "home_gd_roll5",     "away_gd_roll5",
        "home_elo_before",   "away_elo_before",   "elo_diff",
        "home_days_rest",    "away_days_rest",
        "TotalGoals",        "GoalDiff",
        "Over_2_5",          "Result_encoded",
    ]
    print("\n" + "=" * 55)
    print("FEATURED DATASET SUMMARY")
    print("=" * 55)
    print(f"  Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Feature columns  : {len(feature_cols)}")
    print(f"\n  NaN counts in feature columns:")
    for col in feature_cols:
        n = df[col].isna().sum()
        pct = 100 * n / len(df)
        flag = "  (expected — first few games per team)" if n > 0 else ""
        print(f"    {col:30s}: {n:4d}  ({pct:.1f}%){flag}")
    print("\n  Sample stats:")
    print(df[feature_cols].describe().round(2).to_string())
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("EPL FEATURE ENGINEERING")
    print("=" * 55)
    print(f"Input  : {os.path.abspath(INPUT_PATH)}")
    print(f"Output : {os.path.abspath(OUTPUT_PATH)}\n")

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    print(f"Loaded {len(df):,} matches.")

    print("\n[1/4] Adding basic columns …")
    df = add_basic_cols(df)

    print("[2/4] Computing rolling features (this may take ~30s) …")
    df = add_rolling_features(df, window=ROLLING_WINDOW)

    print("[3/4] Computing Elo ratings …")
    df = add_elo(df, k=ELO_K, start=ELO_START)

    print("[4/4] Computing days rest …")
    df = add_days_rest(df)

    validate(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved → {os.path.abspath(OUTPUT_PATH)}")
    print(f"   Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
