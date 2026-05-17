"""
supabase_client.py — lightweight wrapper to optionally sync prediction rows to Supabase
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# Require explicit opt-in to use the service role key to avoid accidental exposure
SUPABASE_USE_SERVICE_ROLE = str(os.environ.get("SUPABASE_USE_SERVICE_ROLE", "false")).lower() in ("1", "true", "yes")

_client = None
if SUPABASE_URL:
    key_to_use = None
    if SUPABASE_USE_SERVICE_ROLE and SUPABASE_SERVICE_ROLE_KEY:
        key_to_use = SUPABASE_SERVICE_ROLE_KEY
    elif SUPABASE_ANON_KEY:
        key_to_use = SUPABASE_ANON_KEY

    if key_to_use:
        try:
            from supabase import create_client
            _client = create_client(SUPABASE_URL, key_to_use)
            if key_to_use == SUPABASE_SERVICE_ROLE_KEY:
                logging.warning("Supabase initialized with SERVICE_ROLE key. Ensure this only runs on trusted servers.")
        except Exception as exc:
            _client = None
            logging.exception("Failed to initialize Supabase client: %s", exc)


def is_configured() -> bool:
    return _client is not None


def insert_prediction(row: dict):
    """Insert a prediction row into Supabase `predictions` table.

    This is optional — if Supabase is not configured the call is a no-op.
    Returns the Supabase response object on success or a dict with an error key on failure.
    """
    if not is_configured():
        logging.debug("Supabase not configured; skipping insert_prediction")
        return None

    try:
        resp = _client.table("predictions").insert(row).execute()
        return resp
    except Exception as exc:
        logging.exception("Supabase insert_prediction failed: %s", exc)
        return {"error": str(exc)}


def insert_profile(row: dict):
    """Insert a user profile row into Supabase `profiles` table.

    Best-effort: returns the Supabase response or an error dict.
    """
    if not is_configured():
        logging.debug("Supabase not configured; skipping insert_profile")
        return None

    try:
        # Backfill and profile sync should be idempotent. Use id as the conflict
        # target so reruns update the existing profile row instead of failing.
        resp = _client.table("profiles").upsert(row, on_conflict="id").execute()
        return resp
    except Exception as exc:
        logging.exception("Supabase insert_profile failed: %s", exc)
        return {"error": str(exc)}
