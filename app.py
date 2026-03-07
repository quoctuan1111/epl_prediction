"""
app.py — Flask web app for EPL Match Predictor
================================================
Wraps the CLI demo logic into a web API + serves the frontend.

Routes:
  GET  /              → index.html (team picker UI)
  GET  /teams         → JSON list of available teams
  POST /predict       → JSON prediction result
  POST /refresh       → trigger full data refresh pipeline
  GET  /upcoming      → JSON list of upcoming fixtures with predictions
  GET  /last_refresh  → JSON metadata from last pipeline run
"""

import json
import os
import sys
import threading
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # loads .env from project root — safe no-op if file absent

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(__file__)
FEATURED_DATA  = os.path.join(BASE_DIR, "data", "processed_data", "featured_data.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "output", "wdl_best_model.pkl")

FEATURE_COLS = [
    "home_scored_roll5",   "home_conceded_roll5",
    "away_scored_roll5",   "away_conceded_roll5",
    "home_pts_roll5",      "away_pts_roll5",
    "home_gd_roll5",       "away_gd_roll5",
    "home_elo_before",     "away_elo_before",  "elo_diff",
    "home_days_rest",      "away_days_rest",
    "home_shots_roll5", "home_shots_on_target_roll5",
    "away_shots_roll5", "away_shots_on_target_roll5",
    "home_form3", "away_form3",
    "home_win_streak", "away_win_streak",
    "home_attack_str", "away_attack_str",
    "home_defence_str", "away_defence_str",
    "h2h_home_win_rate",
]

REFRESH_LOG  = os.path.join(BASE_DIR, "data", "last_refresh.json")
REFRESH_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Load data & model once at startup
# ---------------------------------------------------------------------------

df    = pd.read_csv(FEATURED_DATA, parse_dates=["Date"])
model = joblib.load(MODEL_PATH)

available_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
TEAMS = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))


def _reload_data():
    """Reload df and TEAMS from disk after a pipeline refresh."""
    global df, available_feat_cols, TEAMS
    df = pd.read_csv(FEATURED_DATA, parse_dates=["Date"])
    available_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    TEAMS = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    app.logger.info("Data reloaded: %d matches, %d teams.", len(df), len(TEAMS))


# ---------------------------------------------------------------------------
# Feature helpers (same logic as cli_demo.py)
# ---------------------------------------------------------------------------

def get_team_features(team: str, venue: str) -> dict:
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
        return float(last[col]) if col in last.index and not pd.isna(last[col]) else None

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


def get_h2h_win_rate(home_team: str, away_team: str):
    mask = ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team)) | \
           ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
    rows = df[mask].sort_values("Date")
    if rows.empty:
        return None
    last = rows.iloc[-1]
    if last["HomeTeam"] == home_team:
        val = last.get("h2h_home_win_rate", None)
        return None if val is None or pd.isna(val) else float(val)
    return None


def build_feature_vector(h: dict, a: dict, home_team: str, away_team: str,
                         home_rest: int, away_rest: int) -> np.ndarray:
    h2h = get_h2h_win_rate(home_team, away_team)
    h_elo = h.get("elo") or np.nan
    a_elo = a.get("elo") or np.nan
    elo_diff = (h_elo - a_elo) if (h_elo and a_elo) else np.nan

    def v(val):
        return val if val is not None else np.nan

    lookup = {
        "home_scored_roll5":            v(h.get("scored_roll5")),
        "home_conceded_roll5":          v(h.get("conceded_roll5")),
        "away_scored_roll5":            v(a.get("scored_roll5")),
        "away_conceded_roll5":          v(a.get("conceded_roll5")),
        "home_pts_roll5":               v(h.get("pts_roll5")),
        "away_pts_roll5":               v(a.get("pts_roll5")),
        "home_gd_roll5":                v(h.get("gd_roll5")),
        "away_gd_roll5":                v(a.get("gd_roll5")),
        "home_elo_before":              h_elo,
        "away_elo_before":              a_elo,
        "elo_diff":                     elo_diff,
        "home_days_rest":               home_rest,
        "away_days_rest":               away_rest,
        "home_shots_roll5":             v(h.get("shots_roll5")),
        "home_shots_on_target_roll5":   v(h.get("shots_on_target_roll5")),
        "away_shots_roll5":             v(a.get("shots_roll5")),
        "away_shots_on_target_roll5":   v(a.get("shots_on_target_roll5")),
        "home_form3":                   v(h.get("form3")),
        "away_form3":                   v(a.get("form3")),
        "home_win_streak":              v(h.get("win_streak")),
        "away_win_streak":              v(a.get("win_streak")),
        "home_attack_str":              v(h.get("attack_str")),
        "away_attack_str":              v(a.get("attack_str")),
        "home_defence_str":             v(h.get("defence_str")),
        "away_defence_str":             v(a.get("defence_str")),
        "h2h_home_win_rate":            h2h if h2h is not None else np.nan,
    }

    row = [lookup.get(col, np.nan) for col in available_feat_cols]
    return np.array([row])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", teams=TEAMS)


@app.route("/teams")
def teams():
    return jsonify(TEAMS)

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict():
    data      = request.get_json()
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    home_rest = int(data.get("home_rest", 7))
    away_rest = int(data.get("away_rest", 7))

    if home_team not in TEAMS or away_team not in TEAMS:
        return jsonify({"error": "Invalid team name"}), 400
    if home_team == away_team:
        return jsonify({"error": "Home and away teams must be different"}), 400

    h_feats = get_team_features(home_team, "home")
    a_feats = get_team_features(away_team, "away")
    X       = build_feature_vector(h_feats, a_feats, home_team, away_team, home_rest, away_rest)

    probs = model.predict_proba(X)[0]   # [Away, Draw, Home]
    p_away, p_draw, p_home = float(probs[0]), float(probs[1]), float(probs[2])

    labels  = ["Away Win", "Draw", "Home Win"]
    winner  = labels[int(np.argmax(probs))]
    conf    = float(np.max(probs))

    h2h = get_h2h_win_rate(home_team, away_team)

    def _fmt(v):
        return round(v, 2) if v is not None else None

    stats = {
        "avg_goals_scored":       (_fmt(h_feats.get("scored_roll5")),        _fmt(a_feats.get("scored_roll5"))),
        "avg_goals_conceded":     (_fmt(h_feats.get("conceded_roll5")),       _fmt(a_feats.get("conceded_roll5"))),
        "avg_points":             (_fmt(h_feats.get("pts_roll5")),            _fmt(a_feats.get("pts_roll5"))),
        "avg_goal_diff":          (_fmt(h_feats.get("gd_roll5")),             _fmt(a_feats.get("gd_roll5"))),
        "avg_shots":              (_fmt(h_feats.get("shots_roll5")),          _fmt(a_feats.get("shots_roll5"))),
        "avg_shots_on_target":    (_fmt(h_feats.get("shots_on_target_roll5")),_fmt(a_feats.get("shots_on_target_roll5"))),
        "form_last3":             (_fmt(h_feats.get("form3")),                _fmt(a_feats.get("form3"))),
        "win_streak":             (_fmt(h_feats.get("win_streak")),           _fmt(a_feats.get("win_streak"))),
        "elo_rating":             (_fmt(h_feats.get("elo")),                  _fmt(a_feats.get("elo"))),
        "attack_strength":        (_fmt(h_feats.get("attack_str")),           _fmt(a_feats.get("attack_str"))),
        "defence_strength":       (_fmt(h_feats.get("defence_str")),          _fmt(a_feats.get("defence_str"))),
    }

    return jsonify({
        "home_team":   home_team,
        "away_team":   away_team,
        "p_home":      round(p_home, 4),
        "p_draw":      round(p_draw, 4),
        "p_away":      round(p_away, 4),
        "winner":      winner,
        "confidence":  round(conf, 4),
        "h2h_home_win_rate": round(h2h, 4) if h2h is not None else None,
        "stats":       stats,
    })


# ---------------------------------------------------------------------------
# Refresh route
# ---------------------------------------------------------------------------

@app.route("/refresh", methods=["POST"])
def refresh():
    """
    Trigger the full data refresh pipeline.
    Optional query param: ?secret=YOUR_KEY (compare against env REFRESH_SECRET).
    """
    secret_env = os.environ.get("REFRESH_SECRET", "")
    if secret_env and request.args.get("secret", "") != secret_env:
        return jsonify({"error": "Unauthorized"}), 401

    if not REFRESH_LOCK.acquire(blocking=False):
        return jsonify({"error": "Refresh already in progress"}), 429

    try:
        sys.path.insert(0, BASE_DIR)
        from data_processing.refresh_pipeline import run_refresh
        result = run_refresh()
        _reload_data()
        return jsonify({"status": "ok", **result})
    except Exception as exc:
        app.logger.exception("Refresh failed: %s", exc)
        return jsonify({"error": str(exc)}), 500
    finally:
        REFRESH_LOCK.release()


# ---------------------------------------------------------------------------
# Last refresh metadata route
# ---------------------------------------------------------------------------

@app.route("/last_refresh")
def last_refresh():
    if not os.path.exists(REFRESH_LOG):
        return jsonify({"timestamp": None, "matches": None, "new_data": None})
    try:
        with open(REFRESH_LOG) as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({"timestamp": None, "matches": None, "new_data": None})


# ---------------------------------------------------------------------------
# Upcoming fixtures route
# ---------------------------------------------------------------------------

@app.route("/upcoming")
def upcoming():
    """Return upcoming EPL fixtures with model predictions."""
    try:
        sys.path.insert(0, BASE_DIR)
        from data_processing.upcoming_fixtures import get_fixtures
        fixtures = get_fixtures()
    except Exception as exc:
        app.logger.warning("Could not fetch fixtures: %s", exc)
        return jsonify({"error": str(exc), "fixtures": []}), 500

    results = []
    for fix in fixtures:
        home_team = fix["home_team"]
        away_team = fix["away_team"]
        home_rest = fix.get("home_rest", 7)
        away_rest = fix.get("away_rest", 7)

        if home_team not in TEAMS or away_team not in TEAMS:
            app.logger.debug("Skipping fixture — unknown team(s): %s vs %s",
                             home_team, away_team)
            continue

        try:
            h_feats = get_team_features(home_team, "home")
            a_feats = get_team_features(away_team, "away")
            X       = build_feature_vector(h_feats, a_feats, home_team, away_team,
                                           home_rest, away_rest)
            probs           = model.predict_proba(X)[0]
            p_away, p_draw, p_home = float(probs[0]), float(probs[1]), float(probs[2])
            labels          = ["Away Win", "Draw", "Home Win"]
            winner          = labels[int(np.argmax(probs))]

            results.append({
                "home_team": home_team,
                "away_team": away_team,
                "date":      fix["date"],
                "p_home":    round(p_home, 4),
                "p_draw":    round(p_draw, 4),
                "p_away":    round(p_away, 4),
                "winner":    winner,
                "confidence": round(float(np.max(probs)), 4),
            })
        except Exception as exc:
            app.logger.warning("Prediction failed for %s vs %s: %s",
                               home_team, away_team, exc)

    return jsonify({"fixtures": results, "count": len(results)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
