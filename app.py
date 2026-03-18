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
import requests
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from flask_session import Session
from functools import wraps
from tracking_store.prediction_store import (
    get_accuracy_dashboard,
    get_admin_predictions,
    get_prediction_accuracy,
    init_tracking_db,
    latest_prediction_for_fixture,
    normalize_client_id,
    resolve_prediction_if_needed,
    result_label_from_score,
    write_prediction,
)
from tracking_store.user_store import (
    init_user_db,
    register_user,
    login_user,
    get_user,
    update_user_profile,
    change_password,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Session configuration
# ---------------------------------------------------------------------------
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
Session(app)

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

FOOTBALL_DATA_API_URL = "https://api.football-data.org/v4/competitions/PL/matches"
TIMEOUT_SECONDS = 12

API_TEAM_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich Town",
    "Leicester City FC": "Leicester City",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle United",
    "Nottingham Forest FC": "Nottingham Forest",
    "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "West Ham United FC": "West Ham United",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
}

# ---------------------------------------------------------------------------
# Load data & model once at startup
# ---------------------------------------------------------------------------

df    = pd.read_csv(FEATURED_DATA, parse_dates=["Date"])
model = joblib.load(MODEL_PATH)

available_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
TEAMS = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
_feature_means = df[available_feat_cols].mean().to_dict()


def _normalise_team_name(name: str) -> str:
    return API_TEAM_MAP.get(name, name)


def _request_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return (request.remote_addr or "").strip()




def _get_today_matches_from_api():
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "")
    if not api_key:
        return []

    today = datetime.now(timezone.utc).date().isoformat()
    params = {"dateFrom": today, "dateTo": today}
    headers = {"X-Auth-Token": api_key}

    try:
        resp = requests.get(FOOTBALL_DATA_API_URL, params=params, headers=headers, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        app.logger.warning("Could not fetch today's matches from API: %s", exc)
        return []

    fixtures = []
    for item in payload.get("matches", []):
        raw_home = item.get("homeTeam", {}).get("name", "")
        raw_away = item.get("awayTeam", {}).get("name", "")
        home = _normalise_team_name(raw_home)
        away = _normalise_team_name(raw_away)
        if home not in TEAMS or away not in TEAMS:
            continue

        utc_date = item.get("utcDate", "")
        fixture_date = None
        try:
            fixture_date = datetime.fromisoformat(utc_date.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            fixture_date = datetime.now(timezone.utc).date().isoformat()

        full_time = item.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")

        fixtures.append(
            {
                "date": fixture_date,
                "kickoff_utc": utc_date,
                "status": item.get("status", "SCHEDULED"),
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    return fixtures


def _compute_team_form_payload(team: str, window: int = 10):
    team_rows = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].sort_values("Date")
    if team_rows.empty:
        return None

    recent = team_rows.tail(window).copy()
    results = []
    elo_points = []
    shots_for = []
    shots_against = []
    shots_on_target_for = []
    shots_on_target_against = []

    rolling_points = []
    cumulative = 0

    for _, row in recent.iterrows():
        is_home = row["HomeTeam"] == team
        opponent = row["AwayTeam"] if is_home else row["HomeTeam"]
        ftr = row.get("FTR", "")
        if is_home:
            if ftr == "H":
                result = "W"
                points = 3
            elif ftr == "D":
                result = "D"
                points = 1
            else:
                result = "L"
                points = 0
        else:
            if ftr == "A":
                result = "W"
                points = 3
            elif ftr == "D":
                result = "D"
                points = 1
            else:
                result = "L"
                points = 0

        goals_for = row.get("FTHG") if is_home else row.get("FTAG")
        goals_against = row.get("FTAG") if is_home else row.get("FTHG")
        shots_f = row.get("HS") if is_home else row.get("AS")
        shots_a = row.get("AS") if is_home else row.get("HS")
        sot_f = row.get("HST") if is_home else row.get("AST")
        sot_a = row.get("AST") if is_home else row.get("HST")
        elo_before = row.get("home_elo_before") if is_home else row.get("away_elo_before")

        cumulative += points
        rolling_points.append(cumulative)

        results.append(
            {
                "date": str(row["Date"].date()),
                "venue": "H" if is_home else "A",
                "opponent": opponent,
                "result": result,
                "points": points,
                "goals_for": int(goals_for) if pd.notna(goals_for) else None,
                "goals_against": int(goals_against) if pd.notna(goals_against) else None,
            }
        )

        elo_points.append(float(elo_before) if pd.notna(elo_before) else None)
        shots_for.append(float(shots_f) if pd.notna(shots_f) else None)
        shots_against.append(float(shots_a) if pd.notna(shots_a) else None)
        shots_on_target_for.append(float(sot_f) if pd.notna(sot_f) else None)
        shots_on_target_against.append(float(sot_a) if pd.notna(sot_a) else None)

    def _avg(values):
        nums = [v for v in values if v is not None and not pd.isna(v)]
        return round(float(np.mean(nums)), 2) if nums else None

    return {
        "team": team,
        "window": window,
        "last_results": results,
        "rolling_form_points": rolling_points,
        "elo_trajectory": elo_points,
        "shot_stats": {
            "shots_for_avg": _avg(shots_for),
            "shots_against_avg": _avg(shots_against),
            "shots_on_target_for_avg": _avg(shots_on_target_for),
            "shots_on_target_against_avg": _avg(shots_on_target_against),
        },
    }


def _reload_data():
    """Reload df and TEAMS from disk after a pipeline refresh."""
    global df, available_feat_cols, TEAMS, _feature_means
    df = pd.read_csv(FEATURED_DATA, parse_dates=["Date"])
    available_feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    TEAMS = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    _feature_means = df[available_feat_cols].mean().to_dict()
    app.logger.info("Data reloaded: %d matches, %d teams.", len(df), len(TEAMS))


init_tracking_db()
init_user_db()


# ---------------------------------------------------------------------------
# Authentication decorator
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow JSON API calls without login (they'll use client_id)
        if request.method == 'POST' and request.is_json:
            return f(*args, **kwargs)
        # For page views, require login
        if 'user_id' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function


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

    row = np.array([lookup.get(col, np.nan) for col in available_feat_cols], dtype=float)
    # Impute NaN with training-data column means so the model never sees NaN
    nan_mask = np.isnan(row)
    if nan_mask.any():
        means = np.array([_feature_means.get(col, 0.0) for col in available_feat_cols])
        row[nan_mask] = means[nan_mask]
    return row.reshape(1, -1)


# ---------------------------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "GET":
        return render_template("register.html")
    
    data = request.get_json(silent=True) or {}
    nickname = data.get("nickname", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    
    result = register_user(nickname, email, password)
    
    if result["success"]:
        # Auto-login after registration
        session['user_id'] = result['user_id']
        session['nickname'] = result['nickname']
        return jsonify({"success": True, "redirect": url_for('index')})
    
    return jsonify({"success": False, "error": result.get("error", "Registration failed")}), 400


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        # If already logged in, redirect to predictor
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template("login.html")
    
    data = request.get_json(silent=True) or {}
    nickname = data.get("nickname", "").strip()
    password = data.get("password", "").strip()
    
    result = login_user(nickname, password)
    
    if result["success"]:
        session['user_id'] = result['user_id']
        session['nickname'] = result['nickname']
        session['email'] = result['email']
        return jsonify({"success": True, "redirect": url_for('index')})
    
    return jsonify({"success": False, "error": result.get("error", "Login failed")}), 401


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))


@app.route("/api/user")
def api_user():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    user = get_user(session['user_id'])
    if not user:
        session.clear()
        return jsonify({"error": "User not found"}), 404
    
    return jsonify(user)


@app.route("/api/user/update", methods=["POST"])
def api_user_update():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json(silent=True) or {}
    nickname = data.get("nickname")
    email = data.get("email")
    
    result = update_user_profile(session['user_id'], nickname=nickname, email=email)
    
    if result["success"]:
        # Update session if nickname changed
        if nickname:
            session['nickname'] = nickname
        if email:
            session['email'] = email
        return jsonify(result)
    
    return jsonify(result), 400


@app.route("/api/user/change-password", methods=["POST"])
def api_change_password():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json(silent=True) or {}
    old_password = data.get("old_password", "")
    new_password = data.get("new_password", "")
    
    result = change_password(session['user_id'], old_password, new_password)
    return jsonify(result) if result["success"] else (jsonify(result), 400)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/env-test")
def env_test():
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    return {"key_loaded": bool(key)}

@app.route("/")
@login_required
def index():
    return render_template("index.html", teams=TEAMS)


@app.route("/team_form")
@login_required
def team_form_page():
    return render_template("team_form.html", teams=TEAMS)


@app.route("/teams")
def teams():
    return jsonify(TEAMS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    home_rest = int(data.get("home_rest", 7))
    away_rest = int(data.get("away_rest", 7))
    
    # Use logged-in user_id if available, otherwise use client_id from request
    user_id = session.get('user_id')
    source_client_id = data.get("client_id", "")
    client_id = user_id if user_id else normalize_client_id(source_client_id)
    
    fixture_date = str(data.get("fixture_date", "")).strip() or None

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

    request_ip = _request_ip()
    normalized_source_client_id = normalize_client_id(source_client_id)

    prediction_id = write_prediction(
        client_id=client_id,
        home_team=home_team,
        away_team=away_team,
        fixture_date=fixture_date,
        predicted_label=winner,
        p_home=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away=round(p_away, 4),
        confidence=round(conf, 4),
        source_client_id=source_client_id,
        request_ip=request_ip,
        user_agent=request.headers.get("User-Agent", ""),
    )

    app.logger.info(
        "Prediction saved id=%s client_id=%s source_client_id=%s ip=%s fixture=%s vs %s",
        prediction_id,
        client_id,
        normalized_source_client_id,
        request_ip,
        home_team,
        away_team,
    )

    return jsonify({
        "home_team":   home_team,
        "away_team":   away_team,
        "p_home":      round(p_home, 4),
        "p_draw":      round(p_draw, 4),
        "p_away":      round(p_away, 4),
        "winner":      winner,
        "confidence":  round(conf, 4),
        "prediction_id": prediction_id,
        "client_id": client_id,
        "h2h_home_win_rate": round(h2h, 4) if h2h is not None else None,
        "stats":       stats,
    })


# ---------------------------------------------------------------------------
# Refresh route
# ---------------------------------------------------------------------------

def _bg_refresh():
    try:
        sys.path.insert(0, BASE_DIR)
        from data_processing.refresh_pipeline import run_refresh
        run_refresh()
        _reload_data()
    except Exception as exc:
        app.logger.exception("Background refresh failed: %s", exc)
    finally:
        REFRESH_LOCK.release()

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

    thread = threading.Thread(target=_bg_refresh)
    thread.start()
    return jsonify({"status": "started", "message": "Refresh started in background"})


# ---------------------------------------------------------------------------
# Last refresh metadata route
# ---------------------------------------------------------------------------

@app.route("/last_refresh")
def last_refresh():
    payload = {
        "timestamp": None,
        "matches": None,
        "new_data": None,
        "is_refreshing": REFRESH_LOCK.locked()
    }
    if not os.path.exists(REFRESH_LOG):
        return jsonify(payload)
    try:
        with open(REFRESH_LOG) as f:
            data = json.load(f)
            payload.update(data)
            return jsonify(payload)
    except Exception:
        return jsonify(payload)


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
                "home_rest": home_rest,
                "away_rest": away_rest,
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


@app.route("/live_timeline")
def live_timeline():
    client_id = normalize_client_id(request.args.get("client_id", ""))
    today_fixtures = _get_today_matches_from_api()

    payload = []
    for fixture in today_fixtures:
        latest = latest_prediction_for_fixture(
            client_id=client_id,
            home_team=fixture["home_team"],
            away_team=fixture["away_team"],
            fixture_date=fixture["date"],
        )

        actual_label = result_label_from_score(fixture.get("home_goals"), fixture.get("away_goals"))

        if latest and actual_label is not None:
            resolve_prediction_if_needed(
                prediction_id=latest["id"],
                home_goals=fixture.get("home_goals"),
                away_goals=fixture.get("away_goals"),
            )
            latest["actual_label"] = actual_label
            latest["actual_home_goals"] = fixture.get("home_goals")
            latest["actual_away_goals"] = fixture.get("away_goals")

        payload.append(
            {
                **fixture,
                "prediction": {
                    "id": latest["id"],
                    "winner": latest["predicted_label"],
                    "confidence": latest["confidence"],
                    "p_home": latest["p_home"],
                    "p_draw": latest["p_draw"],
                    "p_away": latest["p_away"],
                    "created_at": latest["created_at"],
                    "is_correct": (latest["predicted_label"] == latest["actual_label"])
                    if latest.get("actual_label")
                    else None,
                }
                if latest
                else None,
                "actual_label": actual_label,
            }
        )

    return jsonify({"date": datetime.now(timezone.utc).date().isoformat(), "fixtures": payload})


@app.route("/prediction_accuracy")
def prediction_accuracy():
    client_id = normalize_client_id(request.args.get("client_id", ""))
    return jsonify(get_prediction_accuracy(client_id))


@app.route("/accuracy_dashboard")
def accuracy_dashboard():
    client_id = normalize_client_id(request.args.get("client_id", ""))
    return jsonify(get_accuracy_dashboard(client_id))


@app.route("/admin/predictions")
def admin_predictions():
    """
    Admin endpoint to inspect prediction rows with full attribution.

    Query params:
      client_id  — filter by client ID
      ip         — filter by request IP
      limit      — rows per page (max 500, default 100)
      offset     — pagination offset (default 0)
      secret     — must match env ADMIN_SECRET if set
    """
    secret_env = os.environ.get("ADMIN_SECRET", "")
    if secret_env and request.args.get("secret", "") != secret_env:
        return jsonify({"error": "Unauthorized"}), 401

    client_id  = normalize_client_id(request.args.get("client_id", ""), fallback="")
    request_ip = request.args.get("ip", "").strip()
    limit      = request.args.get("limit", 100)
    offset     = request.args.get("offset", 0)

    result = get_admin_predictions(
        client_id=client_id or None,
        request_ip=request_ip or None,
        limit=int(limit),
        offset=int(offset),
    )
    return jsonify(result)


@app.route("/team_form_data")
def team_form_data():
    team = request.args.get("team", "")
    window = int(request.args.get("window", 10))
    window = 5 if window <= 5 else 10

    if team not in TEAMS:
        return jsonify({"error": "Invalid team"}), 400

    payload = _compute_team_form_payload(team=team, window=window)
    if payload is None:
        return jsonify({"error": "No data available"}), 404
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
