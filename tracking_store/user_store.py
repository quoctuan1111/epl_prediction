"""
user_store.py — User registration, authentication, and profile management
=========================================================================
Provides functions for:
  - User registration with password hashing
  - Login verification
  - Password validation
  - User profile retrieval and updates
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRACKING_DB = os.path.join(BASE_DIR, "data", "tracking", "prediction_tracking.db")


def _db_connect():
    os.makedirs(os.path.dirname(TRACKING_DB), exist_ok=True)
    conn = sqlite3.connect(TRACKING_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db():
    """Initialize the users table in the database."""
    conn = _db_connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                nickname TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_nickname ON users(nickname)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)"
        )
        conn.commit()
    finally:
        conn.close()


def user_exists(nickname: str = None, email: str = None) -> bool:
    """Check if a user exists by nickname or email."""
    if not nickname and not email:
        return False
    
    conn = _db_connect()
    try:
        if nickname:
            result = conn.execute(
                "SELECT id FROM users WHERE LOWER(nickname) = LOWER(?)",
                (nickname,)
            ).fetchone()
            if result:
                return True
        
        if email:
            result = conn.execute(
                "SELECT id FROM users WHERE LOWER(email) = LOWER(?)",
                (email,)
            ).fetchone()
            if result:
                return True
        
        return False
    finally:
        conn.close()


def register_user(nickname: str, email: str, password: str) -> dict:
    """
    Register a new user with nickname, email, and password.
    
    Returns:
        dict with keys: success (bool), user_id (str), message (str), error (str)
    """
    # Validation
    nickname = (nickname or "").strip()
    email = (email or "").strip()
    password = (password or "").strip()
    
    if not nickname or len(nickname) < 3:
        return {"success": False, "error": "Nickname must be at least 3 characters"}
    
    if not email or "@" not in email:
        return {"success": False, "error": "Valid email is required"}
    
    if not password or len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters"}
    
    if len(nickname) > 50:
        return {"success": False, "error": "Nickname must be 50 characters or less"}
    
    if len(email) > 120:
        return {"success": False, "error": "Email must be 120 characters or less"}
    
    # Check if user already exists
    if user_exists(nickname=nickname, email=email):
        return {"success": False, "error": "Username or email already taken"}
    
    # Create user
    user_id = str(uuid.uuid4())
    password_hash = generate_password_hash(password, method="pbkdf2:sha256")
    created_at = datetime.now(timezone.utc).isoformat()
    
    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT INTO users (id, nickname, email, password_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, nickname, email, password_hash, created_at, created_at)
        )
        conn.commit()
        return {
            "success": True,
            "user_id": user_id,
            "nickname": nickname,
            "message": "User registered successfully"
        }
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Username or email already taken"}
    finally:
        conn.close()


def login_user(email: str, password: str) -> dict:
    """
    Authenticate a user by email and password.
    
    Returns:
        dict with keys: success (bool), user_id (str), nickname (str), email (str), error (str)
    """
    email = (email or "").strip()
    password = (password or "").strip()
    
    if not email or not password:
        return {"success": False, "error": "Email and password are required"}
    
    conn = _db_connect()
    try:
        user = conn.execute(
            "SELECT id, nickname, email, password_hash FROM users WHERE LOWER(email) = LOWER(?)",
            (email,)
        ).fetchone()
        
        if not user:
            return {"success": False, "error": "Invalid email or password"}
        
        if not check_password_hash(user["password_hash"], password):
            return {"success": False, "error": "Invalid email or password"}
        
        return {
            "success": True,
            "user_id": user["id"],
            "nickname": user["nickname"],
            "email": user["email"]
        }
    finally:
        conn.close()


def get_user(user_id: str) -> dict:
    """Get user profile by user_id."""
    conn = _db_connect()
    try:
        user = conn.execute(
            "SELECT id, nickname, email, created_at FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        
        if not user:
            return None
        
        return {
            "id": user["id"],
            "nickname": user["nickname"],
            "email": user["email"],
            "created_at": user["created_at"]
        }
    finally:
        conn.close()


def update_user_profile(user_id: str, nickname: str = None, email: str = None) -> dict:
    """Update user profile (nickname or email)."""
    updates = {}
    params = []
    
    if nickname:
        nickname = nickname.strip()
        if len(nickname) < 3:
            return {"success": False, "error": "Nickname must be at least 3 characters"}
        if len(nickname) > 50:
            return {"success": False, "error": "Nickname must be 50 characters or less"}
        updates["nickname"] = nickname
    
    if email:
        email = email.strip()
        if "@" not in email:
            return {"success": False, "error": "Valid email is required"}
        if len(email) > 120:
            return {"success": False, "error": "Email must be 120 characters or less"}
        updates["email"] = email
    
    if not updates:
        return {"success": False, "error": "No updates provided"}
    
    conn = _db_connect()
    try:
        # Check if nickname/email already taken by another user
        if nickname:
            existing = conn.execute(
                "SELECT id FROM users WHERE LOWER(nickname) = LOWER(?) AND id != ?",
                (nickname, user_id)
            ).fetchone()
            if existing:
                return {"success": False, "error": "Nickname already taken"}
        
        if email:
            existing = conn.execute(
                "SELECT id FROM users WHERE LOWER(email) = LOWER(?) AND id != ?",
                (email, user_id)
            ).fetchone()
            if existing:
                return {"success": False, "error": "Email already taken"}
        
        # Update user
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        set_clause += ", updated_at = ?"
        values = list(updates.values()) + [datetime.now(timezone.utc).isoformat(), user_id]
        
        conn.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?",
            values + [user_id]
        )
        conn.commit()
        
        return {"success": True, "message": "Profile updated successfully"}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Nickname or email already taken"}
    finally:
        conn.close()


def change_password(user_id: str, old_password: str, new_password: str) -> dict:
    """Change user password."""
    if not new_password or len(new_password) < 6:
        return {"success": False, "error": "New password must be at least 6 characters"}
    
    conn = _db_connect()
    try:
        user = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        
        if not user:
            return {"success": False, "error": "User not found"}
        
        if not check_password_hash(user["password_hash"], old_password):
            return {"success": False, "error": "Current password is incorrect"}
        
        new_hash = generate_password_hash(new_password, method="pbkdf2:sha256")
        conn.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (new_hash, datetime.now(timezone.utc).isoformat(), user_id)
        )
        conn.commit()
        
        return {"success": True, "message": "Password changed successfully"}
    finally:
        conn.close()
