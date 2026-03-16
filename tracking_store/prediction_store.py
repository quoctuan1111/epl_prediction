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

        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "source_client_id" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN source_client_id TEXT")
        if "request_ip" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN request_ip TEXT")
        if "user_agent" not in cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN user_agent TEXT")

        conn.execute(
            """
            UPDATE predictions
            SET client_id = 'anonymous'
            WHERE client_id IS NULL OR TRIM(client_id) = ''
            """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_request_ip "
            "ON predictions(request_ip, created_at)"
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_predictions_client_id_required
            BEFORE INSERT ON predictions
            FOR EACH ROW
            WHEN NEW.client_id IS NULL OR TRIM(NEW.client_id) = ''
            BEGIN
                SELECT RAISE(ABORT, 'client_id is required');
            END;
            """
        )
        conn.commit()
    finally:
        conn.close()


def normalize_client_id(client_id, fallback: str = "anonymous") -> str:
    if client_id is None:
        return fallback
    value = str(client_id).strip()
    if not value:
        return fallback
    if value.lower() in {"none", "null", "undefined", "nan"}:
        return fallback
    return value[:128]


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
    source_client_id: str = None,
    request_ip: str = None,
    user_agent: str = None,
):
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    fixture_date_value = fixture_date if fixture_date else None
    normalized_client_id = normalize_client_id(client_id)
    normalized_source_client_id = normalize_client_id(source_client_id, fallback="") or None
    normalized_request_ip = (str(request_ip).strip() if request_ip is not None else "")[:64] or None
    normalized_user_agent = (str(user_agent).strip() if user_agent is not None else "")[:512] or None

    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT INTO predictions (
                id, client_id, created_at, fixture_date, home_team, away_team,
                predicted_label, p_home, p_draw, p_away, confidence,
                source_client_id, request_ip, user_agent
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                normalized_client_id,
                created_at,
                fixture_date_value,
                home_team,
                away_team,
                predicted_label,
                p_home,
                p_draw,
                p_away,
                confidence,
                normalized_source_client_id,
                normalized_request_ip,
                normalized_user_agent,
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


def get_admin_predictions(
    client_id: str = None,
    request_ip: str = None,
    limit: int = 100,
    offset: int = 0,
):
    """Return raw prediction rows with full attribution data for admin use."""
    conditions = []
    params = []

    if client_id:
        conditions.append("client_id = ?")
        params.append(client_id)
    if request_ip:
        conditions.append("request_ip = ?")
        params.append(request_ip)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    conn = _db_connect()
    try:
        total_row = conn.execute(
            f"SELECT COUNT(*) FROM predictions {where}", params
        ).fetchone()
        rows = conn.execute(
            f"""
            SELECT
                id, client_id, source_client_id, request_ip, user_agent,
                created_at, fixture_date, home_team, away_team,
                predicted_label, p_home, p_draw, p_away, confidence,
                actual_label, actual_home_goals, actual_away_goals, resolved_at
            FROM predictions
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()
    finally:
        conn.close()

    return {
        "total": int(total_row[0]),
        "limit": limit,
        "offset": offset,
        "rows": [dict(r) for r in rows],
    }


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
