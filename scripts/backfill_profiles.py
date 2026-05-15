"""Backfill local SQLite users to Supabase profiles table.

Run:
    python scripts/backfill_profiles.py

This is best-effort and idempotent: it will attempt to insert each user by id.
"""
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
import sys

# Ensure project root is on sys.path so imports like `tracking_store` work
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tracking_store.supabase_client import is_configured, insert_profile

 
TRACKING_DB = os.path.join(BASE_DIR, "data", "tracking", "prediction_tracking.db")


def main():
    if not is_configured():
        print("Supabase not configured. Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        return

    if not os.path.exists(TRACKING_DB):
        print("Local tracking DB not found:", TRACKING_DB)
        return

    conn = sqlite3.connect(TRACKING_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, nickname, email, created_at FROM users")
    rows = cur.fetchall()
    print(f"Found {len(rows)} users to backfill.")

    success = 0
    failed = 0
    for r in rows:
        row = {
            "id": r["id"],
            "nickname": r["nickname"],
            "email": r["email"],
            "created_at": r["created_at"],
        }
        res = insert_profile(row)
        if res is None:
            print("Skipped (no supabase client):", row["id"])
            failed += 1
        elif isinstance(res, dict) and res.get("error"):
            print("Error inserting", row["id"], res.get("error"))
            failed += 1
        else:
            print("Inserted", row["id"])
            success += 1

    print(f"Done. success={success}, failed={failed}")


if __name__ == '__main__':
    main()
