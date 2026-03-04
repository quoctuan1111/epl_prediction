"""
data_merge.py
=============
Loads all EPL season CSV files from data/raw/, standardizes columns,
and merges them into a single clean dataset saved to data/merged_data.csv.

Usage:
    python data_processing/data_merge.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data", "merged_data.csv")

# Core columns to keep from each season file
CORE_COLS = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",   # Full-time home goals
    "FTAG",   # Full-time away goals
    "FTR",    # Full-time result: H / D / A
    "HTHG",   # Half-time home goals
    "HTAG",   # Half-time away goals
    "HTR",    # Half-time result
    "HS",     # Home shots
    "AS",     # Away shots
    "HST",    # Home shots on target
    "AST",    # Away shots on target
    "HC",     # Home corners
    "AC",     # Away corners
    "HY",     # Home yellow cards
    "AY",     # Away yellow cards
    "HR",     # Home red cards
    "AR",     # Away red cards
    "B365H",  # Bet365 home win odds
    "B365D",  # Bet365 draw odds
    "B365A",  # Bet365 away win odds
]

# Team name normalisation map (handles historical renames)
TEAM_NAME_MAP = {
    "Birmingham":        "Birmingham City",
    "Blackburn":         "Blackburn Rovers",
    "Bolton":            "Bolton Wanderers",
    "Bradford":          "Bradford City",
    "Brighton":          "Brighton",
    "Cardiff":           "Cardiff City",
    "Charlton":          "Charlton Athletic",
    "Coventry":          "Coventry City",
    "Derby":             "Derby County",
    "Hull":              "Hull City",
    "Ipswich":           "Ipswich Town",
    "Leeds":             "Leeds United",
    "Leicester":         "Leicester City",
    "Man City":          "Manchester City",
    "Man United":        "Manchester United",
    "Middlesbrough":     "Middlesbrough",
    "Newcastle":         "Newcastle United",
    "Norwich":           "Norwich City",
    "Nott'm Forest":     "Nottingham Forest",
    "Nottm Forest":      "Nottingham Forest",
    "QPR":               "Queens Park Rangers",
    "Sheffield United":  "Sheffield United",
    "Sheffield Weds":    "Sheffield Wednesday",
    "Stoke":             "Stoke City",
    "Swansea":           "Swansea City",
    "Tottenham":         "Tottenham Hotspur",
    "West Brom":         "West Bromwich Albion",
    "West Ham":          "West Ham United",
    "Wigan":             "Wigan Athletic",
    "Wolves":            "Wolverhampton Wanderers",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_season(filepath: str, season: str) -> pd.DataFrame:
    """Load a single season CSV and return a standardised DataFrame."""
    df = pd.read_csv(filepath, encoding="latin-1")

    # Drop completely empty rows (some CSVs have trailing blank rows)
    df = df.dropna(how="all")

    # Add season label
    df["Season"] = season

    # Keep only columns that exist in this file
    cols_present = [c for c in CORE_COLS if c in df.columns]
    df = df[cols_present].copy()

    return df


def normalise_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Apply team name standardisation."""
    df["HomeTeam"] = df["HomeTeam"].str.strip().replace(TEAM_NAME_MAP)
    df["AwayTeam"] = df["AwayTeam"].str.strip().replace(TEAM_NAME_MAP)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the Date column — handles both dd/mm/yy and dd/mm/yyyy formats."""
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    return df


def merge_all_seasons(raw_dir: str) -> pd.DataFrame:
    """Load every CSV in raw_dir and concatenate into one DataFrame."""
    csv_files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {raw_dir}")

    print(f"Found {len(csv_files)} season files.\n")

    frames = []
    for fname in csv_files:
        season = fname.replace(".csv", "")          # e.g. "2010_2011"
        fpath  = os.path.join(raw_dir, fname)
        df     = load_season(fpath, season)
        df     = normalise_teams(df)
        df     = parse_dates(df)
        frames.append(df)
        print(f"  ✓ {season:12s}  →  {len(df)} matches  |  cols: {list(df.columns)}")

    merged = pd.concat(frames, ignore_index=True)
    return merged


def validate(df: pd.DataFrame) -> None:
    """Print a quick diagnostic summary of the merged dataset."""
    print("\n" + "=" * 55)
    print("MERGED DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total matches : {len(df):,}")
    print(f"  Seasons       : {df['Season'].nunique()}  ({df['Season'].min()} → {df['Season'].max()})")
    print(f"  Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"  Unique teams  : {pd.unique(df[['HomeTeam','AwayTeam']].values.ravel()).shape[0]}")
    print(f"  Columns       : {list(df.columns)}")

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        print(f"\n  ⚠ Columns with NaN values:")
        for col, n in null_counts.items():
            print(f"      {col:15s}: {n} NaN ({100*n/len(df):.1f}%)")
    else:
        print("\n  ✓ No NaN values in core columns.")

    print("\n  FTR distribution:")
    print(df["FTR"].value_counts().to_string())
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("EPL DATA MERGE")
    print("=" * 55)
    print(f"Source : {os.path.abspath(RAW_DIR)}")
    print(f"Output : {os.path.abspath(OUTPUT_PATH)}\n")

    merged = merge_all_seasons(RAW_DIR)
    merged = merged.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

    validate(merged)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved merged dataset → {os.path.abspath(OUTPUT_PATH)}")
    print(f"   Shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")
