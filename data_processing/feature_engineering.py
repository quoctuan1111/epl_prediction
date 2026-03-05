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

Rolling — last 5 matches per team, NO data leakage
  home_scored_roll5           : avg goals scored  (home games, last 5)
  home_conceded_roll5         : avg goals conceded (home games, last 5)
  away_scored_roll5           : avg goals scored  (away games, last 5)
  away_conceded_roll5         : avg goals conceded (away games, last 5)
  home_pts_roll5              : avg pts (all games, last 5)
  away_pts_roll5              : avg pts (all games, last 5)
  home_gd_roll5               : avg GD (all games, last 5)
  away_gd_roll5               : avg GD (all games, last 5)

Shot-based (proxy xG) — rolling 5, venue-specific
  home_shots_roll5            : avg shots by home team (home games, last 5)
  home_shots_on_target_roll5  : avg SOT by home team (home games, last 5)
  away_shots_roll5            : avg shots by away team (away games, last 5)
  away_shots_on_target_roll5  : avg SOT by away team (away games, last 5)

Form
  home_form3                  : pts per game — home team, last 3 all-venue games
  away_form3                  : pts per game — away team, last 3 all-venue games
  home_win_streak             : consecutive wins ending before this match (all venues)
  away_win_streak             : consecutive wins ending before this match (all venues)

Attack / Defence strength (relative to league average rolling over entire history)
  home_attack_str             : home team avg scored / global avg scored
  away_attack_str             : away team avg scored / global avg scored
  home_defence_str            : home team avg conceded / global avg conceded
  away_defence_str            : away team avg conceded / global avg conceded

Head-to-Head
  h2h_home_win_rate           : home team win rate vs this away team (last 6 H2H)

Elo
  home_elo_before             : home team Elo before this match
  away_elo_before             : away team Elo before this match
  elo_diff                    : home_elo_before - away_elo_before

Rest
  home_days_rest              : days since home team's previous match
  away_days_rest              : days since away team's previous match

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
    Compute rolling stats from each team's past matches — venue split + shots.
    History is appended AFTER features are computed, preventing leakage.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Each record: scored, conceded, pts, gd, shots, shots_on_target, venue
    history: dict[str, list] = defaultdict(list)

    cols = [
        "home_scored_roll5", "home_conceded_roll5",
        "away_scored_roll5", "away_conceded_roll5",
        "home_pts_roll5",    "away_pts_roll5",
        "home_gd_roll5",     "away_gd_roll5",
        # shot-based
        "home_shots_roll5", "home_shots_on_target_roll5",
        "away_shots_roll5", "away_shots_on_target_roll5",
        # form
        "home_form3", "away_form3",
        "home_win_streak", "away_win_streak",
    ]
    for col in cols:
        df[col] = np.nan

    for idx, row in df.iterrows():
        ht = row["HomeTeam"]
        at = row["AwayTeam"]

        # Safely read optional shot columns (not in all seasons)
        h_shots = row.get("HS", np.nan) if "HS" in df.columns else np.nan
        h_sot   = row.get("HST", np.nan) if "HST" in df.columns else np.nan
        a_shots = row.get("AS", np.nan) if "AS" in df.columns else np.nan
        a_sot   = row.get("AST", np.nan) if "AST" in df.columns else np.nan

        def _rolling_stats(team: str, venue_filter: str | None, w: int = window):
            past = history[team]
            if venue_filter:
                past = [g for g in past if g["venue"] == venue_filter]
            last_n = past[-w:]
            if not last_n:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            scored   = np.mean([g["scored"]   for g in last_n])
            conceded = np.mean([g["conceded"] for g in last_n])
            pts      = np.mean([g["pts"]      for g in last_n])
            gd       = np.mean([g["gd"]       for g in last_n])
            shots    = np.nanmean([g.get("shots", np.nan) for g in last_n])
            sot      = np.nanmean([g.get("sot", np.nan)   for g in last_n])
            return scored, conceded, pts, gd, shots, sot

        # Home rolling (home venue)
        hs, hc, hp, hg, hsh, hsot = _rolling_stats(ht, "H")
        df.at[idx, "home_scored_roll5"]           = hs
        df.at[idx, "home_conceded_roll5"]         = hc
        df.at[idx, "home_pts_roll5"]              = hp
        df.at[idx, "home_gd_roll5"]               = hg
        df.at[idx, "home_shots_roll5"]            = hsh
        df.at[idx, "home_shots_on_target_roll5"]  = hsot

        # Away rolling (away venue)
        as_, ac, ap, ag, ash, asot = _rolling_stats(at, "A")
        df.at[idx, "away_scored_roll5"]           = as_
        df.at[idx, "away_conceded_roll5"]         = ac
        df.at[idx, "away_pts_roll5"]              = ap
        df.at[idx, "away_gd_roll5"]               = ag
        df.at[idx, "away_shots_roll5"]            = ash
        df.at[idx, "away_shots_on_target_roll5"]  = asot

        # Form (last 3, all venues)
        def _form3(team: str):
            past = history[team][-3:]
            if not past:
                return np.nan
            return np.mean([g["pts"] for g in past])

        df.at[idx, "home_form3"] = _form3(ht)
        df.at[idx, "away_form3"] = _form3(at)

        # Win streak (all venues)
        def _win_streak(team: str):
            past = list(reversed(history[team]))
            streak = 0
            for g in past:
                if g["pts"] == 3:
                    streak += 1
                else:
                    break
            return streak

        df.at[idx, "home_win_streak"] = _win_streak(ht)
        df.at[idx, "away_win_streak"] = _win_streak(at)

        # ── Update history AFTER computing features (avoid leakage) ──
        h_pts = _points_from_result(row["FTR"], "H")
        a_pts = _points_from_result(row["FTR"], "A")
        h_gd  = int(row["FTHG"]) - int(row["FTAG"])
        a_gd  = -h_gd

        history[ht].append({
            "date": row["Date"], "scored": row["FTHG"], "conceded": row["FTAG"],
            "pts": h_pts, "gd": h_gd, "venue": "H",
            "shots": h_shots, "sot": h_sot,
        })
        history[at].append({
            "date": row["Date"], "scored": row["FTAG"], "conceded": row["FTHG"],
            "pts": a_pts, "gd": a_gd, "venue": "A",
            "shots": a_shots, "sot": a_sot,
        })

    return df


# ---------------------------------------------------------------------------
# 3. Attack / Defence strength (relative to global rolling average)
# ---------------------------------------------------------------------------

def add_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attack strength  = team's rolling avg goals scored / global avg goals scored
    Defence strength = team's rolling avg goals conceded / global avg goals conceded
    (uses home_scored_roll5 / away_scored_roll5 already computed)
    """
    df = df.copy()

    global_avg_scored   = df[["home_scored_roll5",   "away_scored_roll5"]].stack().mean()
    global_avg_conceded = df[["home_conceded_roll5",  "away_conceded_roll5"]].stack().mean()

    df["home_attack_str"]  = df["home_scored_roll5"]   / global_avg_scored   if global_avg_scored   > 0 else np.nan
    df["away_attack_str"]  = df["away_scored_roll5"]   / global_avg_scored   if global_avg_scored   > 0 else np.nan
    df["home_defence_str"] = df["home_conceded_roll5"] / global_avg_conceded if global_avg_conceded > 0 else np.nan
    df["away_defence_str"] = df["away_conceded_roll5"] / global_avg_conceded if global_avg_conceded > 0 else np.nan

    return df


# ---------------------------------------------------------------------------
# 4. Head-to-Head feature
# ---------------------------------------------------------------------------

def add_h2h_features(df: pd.DataFrame, h2h_window: int = 6) -> pd.DataFrame:
    """
    h2h_home_win_rate : fraction of last `h2h_window` meetings between (HomeTeam, AwayTeam)
                        where HomeTeam won.  NaN if < 2 meetings exist.
    Uses strict look-back — only past meetings are used.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # h2h_history[frozenset({home, away})] = list of {"home": team, "result": ftr}
    h2h_history: dict[frozenset, list] = defaultdict(list)
    win_rates = []

    for _, row in df.iterrows():
        ht, at = row["HomeTeam"], row["AwayTeam"]
        key    = frozenset({ht, at})
        past   = h2h_history[key][-h2h_window:]

        if len(past) < 2:
            win_rates.append(np.nan)
        else:
            # Count home-team wins in past meetings (home side matches current home team)
            wins = sum(
                1 for g in past
                if g["home"] == ht and g["result"] == "H"
                or g["home"] == at and g["result"] == "A"   # away team of past match = current ht
            )
            win_rates.append(wins / len(past))

        # Update history AFTER computing
        h2h_history[key].append({"home": ht, "result": row["FTR"]})

    df["h2h_home_win_rate"] = win_rates
    return df


# ---------------------------------------------------------------------------
# 5. Elo ratings
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
# 6. Days rest
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
# 7. Validate & summarise
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "home_scored_roll5", "home_conceded_roll5",
    "away_scored_roll5", "away_conceded_roll5",
    "home_pts_roll5",    "away_pts_roll5",
    "home_gd_roll5",     "away_gd_roll5",
    # shot-based
    "home_shots_roll5", "home_shots_on_target_roll5",
    "away_shots_roll5", "away_shots_on_target_roll5",
    # form
    "home_form3", "away_form3",
    "home_win_streak", "away_win_streak",
    # attack/defence strength
    "home_attack_str", "away_attack_str",
    "home_defence_str", "away_defence_str",
    # H2H
    "h2h_home_win_rate",
    # Elo
    "home_elo_before", "away_elo_before", "elo_diff",
    # rest
    "home_days_rest", "away_days_rest",
]


def validate(df: pd.DataFrame) -> None:
    all_cols = FEATURE_COLS + ["TotalGoals", "GoalDiff", "Over_2_5", "Result_encoded"]
    print("\n" + "=" * 58)
    print("FEATURED DATASET SUMMARY")
    print("=" * 58)
    print(f"  Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Feature columns  : {len(FEATURE_COLS)}")
    print(f"\n  NaN counts in feature columns:")
    for col in all_cols:
        if col not in df.columns:
            print(f"    {col:38s}: MISSING")
            continue
        n   = df[col].isna().sum()
        pct = 100 * n / len(df)
        flag = "  ← expected (first games / missing shots)" if n > 0 else ""
        print(f"    {col:38s}: {n:4d}  ({pct:.1f}%){flag}")
    print("\n  Sample stats:")
    avail = [c for c in FEATURE_COLS if c in df.columns]
    print(df[avail].describe().round(2).to_string())
    print("=" * 58)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 58)
    print("EPL FEATURE ENGINEERING  (enhanced)")
    print("=" * 58)
    print(f"Input  : {os.path.abspath(INPUT_PATH)}")
    print(f"Output : {os.path.abspath(OUTPUT_PATH)}\n")

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    print(f"Loaded {len(df):,} matches.\n")

    print("[1/6] Adding basic columns …")
    df = add_basic_cols(df)

    print("[2/6] Computing rolling features (may take ~60s) …")
    df = add_rolling_features(df, window=ROLLING_WINDOW)

    print("[3/6] Computing attack/defence strength …")
    df = add_strength_features(df)

    print("[4/6] Computing head-to-head features …")
    df = add_h2h_features(df)

    print("[5/6] Computing Elo ratings …")
    df = add_elo(df, k=ELO_K, start=ELO_START)

    print("[6/6] Computing days rest …")
    df = add_days_rest(df)

    validate(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {os.path.abspath(OUTPUT_PATH)}")
    print(f"   Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
