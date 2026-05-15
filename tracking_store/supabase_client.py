"""
supabase_client.py — lightweight wrapper to optionally sync prediction rows to Supabase
"""
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

_client = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception:
        _client = None


def is_configured() -> bool:
    return _client is not None


def insert_prediction(row: dict):
    """Insert a prediction row into Supabase `predictions` table.

    This is optional — if Supabase is not configured the call is a no-op.
    Returns the Supabase response object on success or a dict with an error key on failure.
    """
    if not is_configured():
        return None

    try:
        resp = _client.table("predictions").insert(row).execute()
        return resp
    except Exception as exc:
        return {"error": str(exc)}


def insert_profile(row: dict):
    """Insert a user profile row into Supabase `profiles` table.

    Best-effort: returns the Supabase response or an error dict.
    """
    if not is_configured():
        return None

    try:
        resp = _client.table("profiles").insert(row).execute()
        return resp
    except Exception as exc:
        return {"error": str(exc)}
