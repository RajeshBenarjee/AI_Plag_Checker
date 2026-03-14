"""
utils/auth_manager.py
----------------------
Handles user registration, login, and session management.
Stores users in database/users.json

Roles:
  - teacher : can see dashboard, all student submissions, batch upload
  - student : can only submit and see their own history
"""

import os
import json
import hashlib
import hmac
from datetime import datetime

USERS_FILE = "database/users.json"


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: dict):
    os.makedirs("database", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

def register(username: str, password: str, role: str = "student") -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success, message)
    """
    username = username.strip().lower()

    if not username or not password:
        return False, "Username and password cannot be empty."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    if role not in ("teacher", "student"):
        return False, "Invalid role."

    users = _load_users()

    if username in users:
        return False, "Username already exists."

    users[username] = {
        "password_hash": _hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "submission_count": 0
    }

    _save_users(users)
    return True, "Registration successful!"


def login(username: str, password: str) -> tuple[bool, str, dict]:
    """
    Authenticate a user.
    Returns (success, message, user_info_dict)
    """
    username = username.strip().lower()
    users = _load_users()

    if username not in users:
        return False, "Username not found.", {}

    user = users[username]

    if user["password_hash"] != _hash_password(password):
        return False, "Incorrect password.", {}

    user_info = {
        "username": username,
        "role": user["role"],
        "submission_count": user.get("submission_count", 0)
    }

    return True, "Login successful!", user_info


def increment_submission(username: str):
    """Increment submission count for a user."""
    users = _load_users()
    if username in users:
        users[username]["submission_count"] = users[username].get("submission_count", 0) + 1
        _save_users(users)


def get_all_users() -> list:
    """Return list of all users (for teacher dashboard)."""
    users = _load_users()
    return [
        {
            "username": u,
            "role": d["role"],
            "submission_count": d.get("submission_count", 0),
            "created_at": d.get("created_at", "")
        }
        for u, d in users.items()
    ]