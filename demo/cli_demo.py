"""
cli_demo.py
===========
Interactive CLI demo — predict EPL match outcome using the trained WDL model.

How it works:
  1. Loads featured_data.csv to extract each team's latest rolling stats & Elo
  2. You pick Home team and Away team from a numbered list
  3. The model outputs Win / Draw / Lose probabilities + a prediction

Usage:
    python demo/cli_demo.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURED_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "processed_data", "featured_data.csv")
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "output", "wdl_best_model.pkl")

FEATURE_COLS  = [
    "home_scored_roll5",   "home_conceded_roll5",
    "away_scored_roll5",   "away_conceded_roll5",
    "home_pts_roll5",      "away_pts_roll5",
    "home_gd_roll5",       "away_gd_roll5",
    "home_elo_before",     "away_elo_before",  "elo_diff",
    "home_days_rest",      "away_days_rest",
]

BAR_WIDTH = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def color(text: str, code: str) -> str:
    """ANSI colour codes (works in Windows Terminal / PowerShell 7)."""
    return f"\033[{code}m{text}\033[0m"

def bold(text):   return color(text, "1")
def green(text):  return color(text, "92")
def yellow(text): return color(text, "93")
def blue(text):   return color(text, "94")
def red(text):    return color(text, "91")
def cyan(text):   return color(text, "96")

def prob_bar(prob: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(prob * width))
    bar    = "█" * filled + "░" * (width - filled)
    return bar

def pick_team(teams: list[str], prompt: str) -> str:
    """Interactive numbered team picker."""
    print()
    for i, t in enumerate(teams, 1):
        print(f"  {i:>2}. {t}")
    print()
    while True:
        raw = input(f"  {prompt} → ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(teams):
            return teams[int(raw) - 1]
        # Also allow typing team name directly
        matches = [t for t in teams if raw.lower() in t.lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"  ⚠  Multiple matches: {matches}. Be more specific.")
        else:
            print(f"  ⚠  Not found. Enter a number (1–{len(teams)}) or part of the team name.")


def get_team_features(df: pd.DataFrame, team: str, venue: str) -> dict:
    """
    Get a team's most recent rolling features.
    venue = 'home' → use rows where team was HomeTeam
    venue = 'away' → use rows where team was AwayTeam
    """
    if venue == "home":
        rows = df[df["HomeTeam"] == team].sort_values("Date")
        if rows.empty:
            rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date")
        last = rows.iloc[-1]
        return {
            "scored_roll5":   last["home_scored_roll5"],
            "conceded_roll5": last["home_conceded_roll5"],
            "pts_roll5":      last["home_pts_roll5"],
            "gd_roll5":       last["home_gd_roll5"],
            "elo":            last["home_elo_before"],
        }
    else:
        rows = df[df["AwayTeam"] == team].sort_values("Date")
        if rows.empty:
            rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date")
        last = rows.iloc[-1]
        return {
            "scored_roll5":   last["away_scored_roll5"],
            "conceded_roll5": last["away_conceded_roll5"],
            "pts_roll5":      last["away_pts_roll5"],
            "gd_roll5":       last["away_gd_roll5"],
            "elo":            last["away_elo_before"],
        }


def build_feature_vector(h: dict, a: dict, home_rest: int, away_rest: int) -> np.ndarray:
    return np.array([[
        h["scored_roll5"],   h["conceded_roll5"],
        a["scored_roll5"],   a["conceded_roll5"],
        h["pts_roll5"],      a["pts_roll5"],
        h["gd_roll5"],       a["gd_roll5"],
        h["elo"],            a["elo"],
        h["elo"] - a["elo"],
        home_rest,           away_rest,
    ]])


def print_result(home: str, away: str, probs: np.ndarray):
    labels   = ["Away Win", "Draw", "Home Win"]
    p_away   = probs[0]
    p_draw   = probs[1]
    p_home   = probs[2]
    winner   = labels[np.argmax(probs)]
    conf     = np.max(probs)

    print()
    print("  " + "═" * 52)
    print(f"  {bold('PREDICTION RESULT')}")
    print(f"  {cyan(home):>30}  vs  {cyan(away)}")
    print("  " + "═" * 52)

    # Probability bars
    for label, prob, colour_fn in [
        (f"🏠 {home} Win", p_home, green),
        ("🤝 Draw",        p_draw, yellow),
        (f"✈  {away} Win", p_away, red),
    ]:
        bar   = prob_bar(prob)
        pct   = f"{prob*100:5.1f}%"
        print(f"\n  {label:<28}")
        print(f"  {colour_fn(bar)}  {bold(pct)}")

    print()
    print("  " + "─" * 52)

    # Verdict
    verdict_icon = "🟢" if conf > 0.50 else "🟡" if conf > 0.40 else "🔴"
    conf_label   = "High" if conf > 0.50 else "Moderate" if conf > 0.40 else "Low"
    result_text  = f"🏆  Predicted: {bold(winner)}"
    conf_text    = f"{verdict_icon} Confidence: {conf_label} ({conf*100:.1f}%)"

    print(f"\n  {result_text}")
    print(f"  {conf_text}")
    print("  " + "═" * 52)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("  " + "═" * 52)
    print(f"  {bold('⚽  EPL MATCH PREDICTOR  —  CLI DEMO')}")
    print(f"  Powered by {cyan('Logistic Regression')} | Season stats up to 2024–25")
    print("  " + "═" * 52)

    # ── Load data & model ────────────────────────────────
    print(f"\n  Loading model and data …", end=" ", flush=True)
    try:
        df    = pd.read_csv(FEATURED_DATA, parse_dates=["Date"])
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError as e:
        print(red(f"\n  ❌ File not found: {e}"))
        sys.exit(1)
    print(green("✓"))

    # All teams that appear in the dataset
    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

    while True:
        # ── Pick teams ───────────────────────────────────
        print(f"\n  {bold('SELECT HOME TEAM')}")
        home = pick_team(teams, "Enter number or name")

        print(f"\n  {bold('SELECT AWAY TEAM')}")
        away_teams = [t for t in teams if t != home]
        away = pick_team(away_teams, "Enter number or name")

        # ── Days rest ────────────────────────────────────
        print(f"\n  {bold('DAYS REST (press Enter for default = 7)')}")
        try:
            raw = input("  Home team days rest → ").strip()
            home_rest = int(raw) if raw else 7
            raw = input("  Away team days rest → ").strip()
            away_rest = int(raw) if raw else 7
        except ValueError:
            home_rest = away_rest = 7

        # ── Build features ───────────────────────────────
        h_feats = get_team_features(df, home, "home")
        a_feats = get_team_features(df, away, "away")
        X = build_feature_vector(h_feats, a_feats, home_rest, away_rest)

        # ── Predict ──────────────────────────────────────
        probs = model.predict_proba(X)[0]   # shape (3,) → [Away, Draw, Home]

        # ── Show result ──────────────────────────────────
        print_result(home, away, probs)

        # ── Team stats sidebar ───────────────────────────
        print(f"\n  {bold('TEAM STATS (last 5 home/away games)')} ")
        print(f"  {'Stat':<28} {home:>15}  {away:>15}")
        print("  " + "─" * 60)
        stats = [
            ("Avg goals scored",  h_feats["scored_roll5"],   a_feats["scored_roll5"]),
            ("Avg goals conceded",h_feats["conceded_roll5"], a_feats["conceded_roll5"]),
            ("Avg points",        h_feats["pts_roll5"],      a_feats["pts_roll5"]),
            ("Avg goal diff",     h_feats["gd_roll5"],       a_feats["gd_roll5"]),
            ("Elo rating",        h_feats["elo"],            a_feats["elo"]),
        ]
        for label, hv, av in stats:
            winner_h = hv > av
            hv_str = f"{hv:.2f}" if not np.isnan(hv) else "N/A"
            av_str = f"{av:.2f}" if not np.isnan(av) else "N/A"
            hv_fmt = green(f"{hv_str:>15}") if winner_h else f"{hv_str:>15}"
            av_fmt = green(f"{av_str:>15}") if not winner_h else f"{av_str:>15}"
            print(f"  {label:<28} {hv_fmt}  {av_fmt}")

        # ── Replay? ──────────────────────────────────────
        print()
        again = input("  Predict another match? (y/n) → ").strip().lower()
        if again != "y":
            print(f"\n  {green('Thanks for using EPL Predictor!')} See you next match day ⚽\n")
            break


if __name__ == "__main__":
    main()
