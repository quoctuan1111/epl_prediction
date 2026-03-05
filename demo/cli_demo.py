"""
cli_demo.py  (enhanced)
=======================
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

# Ordered list — the model was trained on these columns in this order.
# The demo will silently use NaN/default for any column that is absent in the data.
FEATURE_COLS = [
    # original 13
    "home_scored_roll5",   "home_conceded_roll5",
    "away_scored_roll5",   "away_conceded_roll5",
    "home_pts_roll5",      "away_pts_roll5",
    "home_gd_roll5",       "away_gd_roll5",
    "home_elo_before",     "away_elo_before",  "elo_diff",
    "home_days_rest",      "away_days_rest",
    # enhanced features (v2)
    "home_shots_roll5", "home_shots_on_target_roll5",
    "away_shots_roll5", "away_shots_on_target_roll5",
    "home_form3", "away_form3",
    "home_win_streak", "away_win_streak",
    "home_attack_str", "away_attack_str",
    "home_defence_str", "away_defence_str",
    "h2h_home_win_rate",
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
def magenta(text): return color(text, "95")

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
        matches = [t for t in teams if raw.lower() in t.lower()]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"  ⚠  Multiple matches: {matches}. Be more specific.")
        else:
            print(f"  ⚠  Not found. Enter a number (1–{len(teams)}) or part of the team name.")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _last_row(df: pd.DataFrame, team: str, home_col: str, away_col: str) -> pd.Series:
    """Return the last row where team appeared as home (home_col) or away (away_col)."""
    mask = (df[home_col] == team) | (df[away_col] == team)
    rows = df[mask].sort_values("Date")
    return rows.iloc[-1] if not rows.empty else None


def get_team_features(df: pd.DataFrame, team: str, venue: str) -> dict:
    """
    Extract a team's most recent rolling features.
    venue = 'home' → prefer rows where team was HomeTeam
    venue = 'away' → prefer rows where team was AwayTeam
    """
    if venue == "home":
        rows = df[df["HomeTeam"] == team].sort_values("Date")
        col_prefix = "home"
    else:
        rows = df[df["AwayTeam"] == team].sort_values("Date")
        col_prefix = "away"

    if rows.empty:
        rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date")
        col_prefix = "home" if not rows.empty and rows.iloc[-1]["HomeTeam"] == team else "away"

    if rows.empty:
        return {}

    last = rows.iloc[-1]

    def _get(col):
        return last[col] if col in last.index else np.nan

    return {
        "scored_roll5":           _get(f"{col_prefix}_scored_roll5"),
        "conceded_roll5":         _get(f"{col_prefix}_conceded_roll5"),
        "pts_roll5":              _get(f"{col_prefix}_pts_roll5"),
        "gd_roll5":               _get(f"{col_prefix}_gd_roll5"),
        "elo":                    _get(f"{col_prefix}_elo_before"),
        "shots_roll5":            _get(f"{col_prefix}_shots_roll5"),
        "shots_on_target_roll5":  _get(f"{col_prefix}_shots_on_target_roll5"),
        "form3":                  _get(f"{col_prefix}_form3"),
        "win_streak":             _get(f"{col_prefix}_win_streak"),
        "attack_str":             _get(f"{col_prefix}_attack_str"),
        "defence_str":            _get(f"{col_prefix}_defence_str"),
    }


def get_h2h_win_rate(df: pd.DataFrame, home_team: str, away_team: str) -> float:
    """Return latest h2h_home_win_rate between these two teams, or NaN."""
    mask = ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) | \
           ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    rows = df[mask].sort_values("Date")
    if rows.empty:
        return np.nan
    last = rows.iloc[-1]
    if last["HomeTeam"] == home_team:
        return last.get("h2h_home_win_rate", np.nan)
    return np.nan


def build_feature_vector(
    df: pd.DataFrame,
    h: dict, a: dict,
    home_team: str, away_team: str,
    home_rest: int, away_rest: int,
    feature_cols: list,
) -> np.ndarray:
    """Build a single-row feature vector matching the training column order."""

    h2h = get_h2h_win_rate(df, home_team, away_team)
    elo_diff = h.get("elo", np.nan) - a.get("elo", np.nan) if h and a else np.nan

    lookup = {
        "home_scored_roll5":            h.get("scored_roll5", np.nan),
        "home_conceded_roll5":          h.get("conceded_roll5", np.nan),
        "away_scored_roll5":            a.get("scored_roll5", np.nan),
        "away_conceded_roll5":          a.get("conceded_roll5", np.nan),
        "home_pts_roll5":               h.get("pts_roll5", np.nan),
        "away_pts_roll5":               a.get("pts_roll5", np.nan),
        "home_gd_roll5":                h.get("gd_roll5", np.nan),
        "away_gd_roll5":                a.get("gd_roll5", np.nan),
        "home_elo_before":              h.get("elo", np.nan),
        "away_elo_before":              a.get("elo", np.nan),
        "elo_diff":                     elo_diff,
        "home_days_rest":               home_rest,
        "away_days_rest":               away_rest,
        "home_shots_roll5":             h.get("shots_roll5", np.nan),
        "home_shots_on_target_roll5":   h.get("shots_on_target_roll5", np.nan),
        "away_shots_roll5":             a.get("shots_roll5", np.nan),
        "away_shots_on_target_roll5":   a.get("shots_on_target_roll5", np.nan),
        "home_form3":                   h.get("form3", np.nan),
        "away_form3":                   a.get("form3", np.nan),
        "home_win_streak":              h.get("win_streak", np.nan),
        "away_win_streak":              a.get("win_streak", np.nan),
        "home_attack_str":              h.get("attack_str", np.nan),
        "away_attack_str":              a.get("attack_str", np.nan),
        "home_defence_str":             h.get("defence_str", np.nan),
        "away_defence_str":             a.get("defence_str", np.nan),
        "h2h_home_win_rate":            h2h,
    }

    row = [lookup.get(col, np.nan) for col in feature_cols]
    return np.array([row])


def _fmt(v, fmt=".2f"):
    return format(v, fmt) if not (isinstance(v, float) and np.isnan(v)) else "N/A"


def print_result(home: str, away: str, probs: np.ndarray):
    p_away, p_draw, p_home = probs[0], probs[1], probs[2]
    labels  = ["Away Win", "Draw", "Home Win"]
    winner  = labels[np.argmax(probs)]
    conf    = np.max(probs)

    print()
    print("  " + "═" * 55)
    print(f"  {bold('PREDICTION RESULT')}")
    print(f"  {cyan(home):>32}  vs  {cyan(away)}")
    print("  " + "═" * 55)

    for label, prob, colour_fn in [
        (f"🏠 {home} Win", p_home, green),
        ("🤝 Draw",         p_draw, yellow),
        (f"✈  {away} Win",  p_away, red),
    ]:
        bar = prob_bar(prob)
        pct = f"{prob*100:5.1f}%"
        print(f"\n  {label:<30}")
        print(f"  {colour_fn(bar)}  {bold(pct)}")

    print()
    print("  " + "─" * 55)
    verdict_icon = "🟢" if conf > 0.50 else "🟡" if conf > 0.40 else "🔴"
    conf_label   = "High" if conf > 0.50 else "Moderate" if conf > 0.40 else "Low"
    print(f"\n  🏆  Predicted: {bold(winner)}")
    print(f"  {verdict_icon} Confidence: {conf_label} ({conf*100:.1f}%)")
    print("  " + "═" * 55)


def print_team_stats(h: dict, a: dict, home: str, away: str):
    print(f"\n  {bold('TEAM STATS (last 5 home/away games)')}")
    print(f"  {'Stat':<32} {home:>15}  {away:>15}")
    print("  " + "─" * 65)

    stats = [
        ("Avg goals scored",         h.get("scored_roll5"),          a.get("scored_roll5")),
        ("Avg goals conceded",        h.get("conceded_roll5"),        a.get("conceded_roll5")),
        ("Avg points",               h.get("pts_roll5"),             a.get("pts_roll5")),
        ("Avg goal diff",            h.get("gd_roll5"),              a.get("gd_roll5")),
        ("Avg shots",                h.get("shots_roll5"),           a.get("shots_roll5")),
        ("Avg shots on target",      h.get("shots_on_target_roll5"), a.get("shots_on_target_roll5")),
        ("Form (last 3, pts/game)",  h.get("form3"),                 a.get("form3")),
        ("Win streak",               h.get("win_streak"),            a.get("win_streak")),
        ("Elo rating",               h.get("elo"),                   a.get("elo")),
        ("Attack strength",          h.get("attack_str"),            a.get("attack_str")),
        ("Defence strength ↓ better",h.get("defence_str"),          a.get("defence_str")),
    ]

    for label, hv, av in stats:
        if hv is None or av is None:
            continue
        hnan = isinstance(hv, float) and np.isnan(hv)
        avan = isinstance(av, float) and np.isnan(av)
        if hnan and avan:
            continue

        # For "defence strength" lower is actually better (fewer conceded)
        is_def = "Defence" in label
        winner_h = (hv < av) if is_def else (hv > av)

        hv_str = _fmt(hv) if not hnan else "N/A"
        av_str = _fmt(av) if not avan else "N/A"
        hv_fmt = green(f"{hv_str:>15}") if winner_h   else f"{hv_str:>15}"
        av_fmt = green(f"{av_str:>15}") if not winner_h else f"{av_str:>15}"
        print(f"  {label:<32} {hv_fmt}  {av_fmt}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("  " + "═" * 55)
    print(f"  {bold('⚽  EPL MATCH PREDICTOR  —  CLI DEMO  (v2)')}")
    print(f"  Powered by {cyan('Enhanced WDL Model')} | Season stats up to 2024–25")
    print("  " + "═" * 55)

    # ── Load data & model ──────────────────────────────────────
    print(f"\n  Loading model and data …", end=" ", flush=True)
    try:
        df    = pd.read_csv(FEATURED_DATA, parse_dates=["Date"])
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError as e:
        print(red(f"\n  ❌ File not found: {e}"))
        sys.exit(1)
    print(green("✓"))

    # Identify which feature columns are actually in the data
    available_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(available_feat_cols) < len(FEATURE_COLS):
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        print(yellow(f"  ⚠  {len(missing)} features not in data (will use NaN): {missing}"))

    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

    while True:
        # ── Pick teams ──────────────────────────────────────────
        print(f"\n  {bold('SELECT HOME TEAM')}")
        home = pick_team(teams, "Enter number or name")

        print(f"\n  {bold('SELECT AWAY TEAM')}")
        away_teams = [t for t in teams if t != home]
        away = pick_team(away_teams, "Enter number or name")

        # ── Days rest ───────────────────────────────────────────
        print(f"\n  {bold('DAYS REST')} (press Enter for default = 7)")
        try:
            raw = input("  Home team days rest → ").strip()
            home_rest = int(raw) if raw else 7
            raw = input("  Away team days rest → ").strip()
            away_rest = int(raw) if raw else 7
        except ValueError:
            home_rest = away_rest = 7

        # ── Build features ──────────────────────────────────────
        h_feats = get_team_features(df, home, "home")
        a_feats = get_team_features(df, away, "away")
        X = build_feature_vector(df, h_feats, a_feats, home, away, home_rest, away_rest, available_feat_cols)

        # ── Predict ─────────────────────────────────────────────
        try:
            probs = model.predict_proba(X)[0]   # [Away, Draw, Home]
        except Exception as ex:
            print(red(f"\n  ❌ Prediction error: {ex}"))
            break

        # ── Display ─────────────────────────────────────────────
        print_result(home, away, probs)
        print_team_stats(h_feats, a_feats, home, away)

        # ── H2H note ────────────────────────────────────────────
        h2h = get_h2h_win_rate(df, home, away)
        if not np.isnan(h2h):
            print(f"\n  {magenta('Head-to-Head')}: {home} wins {h2h*100:.0f}% of meetings vs {away}")

        # ── Replay? ─────────────────────────────────────────────
        print()
        again = input("  Predict another match? (y/n) → ").strip().lower()
        if again != "y":
            print(f"\n  {green('Thanks for using EPL Predictor!')} See you next match day ⚽\n")
            break


if __name__ == "__main__":
    main()
