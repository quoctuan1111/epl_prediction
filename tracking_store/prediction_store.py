import os
import sqlite3
import uuid
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRACKING_DB = os.path.join(BASE_DIR, "data", "tracking", "prediction_tracking.db")


def _db_connect():
    os.makedirs(os.path.dirname(TRACKING_DB), exist_ok=True)
    conn = sqlite3.connect(TRACKING_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_tracking_db():
    conn = _db_connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                fixture_date TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                p_home REAL NOT NULL,
                p_draw REAL NOT NULL,
                p_away REAL NOT NULL,
                confidence REAL NOT NULL,
                actual_label TEXT,
                actual_home_goals INTEGER,
                actual_away_goals INTEGER,
                resolved_at TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_client_created "
            "ON predictions(client_id, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_fixture "
            "ON predictions(home_team, away_team, fixture_date)"
        )
        conn.commit()
    finally:
        conn.close()


def result_label_from_score(home_goals, away_goals):
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return "Home Win"
    if home_goals < away_goals:
        return "Away Win"
    return "Draw"


def write_prediction(
    client_id: str,
    home_team: str,
    away_team: str,
    fixture_date,
    predicted_label: str,
    p_home: float,
    p_draw: float,
    p_away: float,
    confidence: float,
):
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    fixture_date_value = fixture_date if fixture_date else None

    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT INTO predictions (
                id, client_id, created_at, fixture_date, home_team, away_team,
                predicted_label, p_home, p_draw, p_away, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                client_id,
                created_at,
                fixture_date_value,
                home_team,
                away_team,
                predicted_label,
                p_home,
                p_draw,
                p_away,
                confidence,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return prediction_id


def latest_prediction_for_fixture(client_id: str, home_team: str, away_team: str, fixture_date: str):
    conn = _db_connect()
    try:
        row = conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE client_id = ?
              AND home_team = ?
              AND away_team = ?
              AND fixture_date = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (client_id, home_team, away_team, fixture_date),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def resolve_prediction_if_needed(prediction_id: str, home_goals, away_goals):
    actual_label = result_label_from_score(home_goals, away_goals)
    if actual_label is None:
        return

    conn = _db_connect()
    try:
        conn.execute(
            """
            UPDATE predictions
            SET actual_label = ?,
                actual_home_goals = ?,
                actual_away_goals = ?,
                resolved_at = ?
            WHERE id = ?
              AND actual_label IS NULL
            """,
            (
                actual_label,
                int(home_goals),
                int(away_goals),
                datetime.now(timezone.utc).isoformat(),
                prediction_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_prediction_accuracy(client_id: str):
    conn = _db_connect()
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS completed,
                SUM(CASE WHEN actual_label = predicted_label THEN 1 ELSE 0 END) AS correct
            FROM predictions
            WHERE client_id = ?
              AND actual_label IS NOT NULL
            """,
            (client_id,),
        ).fetchone()
    finally:
        conn.close()

    completed = int(row["completed"] or 0)
    correct = int(row["correct"] or 0)
    accuracy = round((correct / completed) * 100, 2) if completed else None
    return {"client_id": client_id, "completed": completed, "correct": correct, "accuracy_pct": accuracy}


def get_accuracy_dashboard(client_id: str):
    conn = _db_connect()
    try:
        summary_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN actual_label IS NOT NULL THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN actual_label = predicted_label THEN 1 ELSE 0 END) AS correct
            FROM predictions
            WHERE client_id = ?
            """,
            (client_id,),
        ).fetchone()

        history_rows = conn.execute(
            """
            SELECT
                substr(created_at, 1, 10) AS day,
                COUNT(*) AS predictions,
                SUM(CASE WHEN actual_label IS NOT NULL THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN actual_label = predicted_label THEN 1 ELSE 0 END) AS correct
            FROM predictions
            WHERE client_id = ?
            GROUP BY substr(created_at, 1, 10)
            ORDER BY day ASC
            """,
            (client_id,),
        ).fetchall()

        recent_rows = conn.execute(
            """
            SELECT
                id, created_at, fixture_date, home_team, away_team, predicted_label,
                p_home, p_draw, p_away, confidence,
                actual_label, actual_home_goals, actual_away_goals
            FROM predictions
            WHERE client_id = ?
            ORDER BY created_at DESC
            LIMIT 50
            """,
            (client_id,),
        ).fetchall()
    finally:
        conn.close()

    total_predictions = int(summary_row["total_predictions"] or 0)
    completed = int(summary_row["completed"] or 0)
    correct = int(summary_row["correct"] or 0)
    accuracy_pct = round((correct / completed) * 100, 2) if completed else None

    history = []
    for row in history_rows:
        comp = int(row["completed"] or 0)
        corr = int(row["correct"] or 0)
        history.append(
            {
                "day": row["day"],
                "predictions": int(row["predictions"] or 0),
                "completed": comp,
                "correct": corr,
                "accuracy_pct": round((corr / comp) * 100, 2) if comp else None,
            }
        )

    recent = [dict(r) for r in recent_rows]

    return {
        "client_id": client_id,
        "summary": {
            "total_predictions": total_predictions,
            "completed": completed,
            "correct": correct,
            "accuracy_pct": accuracy_pct,
        },
        "history": history,
        "recent": recent,
    }
