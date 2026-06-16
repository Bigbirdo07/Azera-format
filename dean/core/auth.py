from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from database.db import execute_insert, fetch_all


ROLES = {"Viewer", "Editor", "Admin"}
ROLE_PERMISSIONS = {
    "Viewer": {
        "can_preview": True,
        "can_parse": True,
        "can_execute": False,
        "can_export": False,
        "can_admin": False,
    },
    "Editor": {
        "can_preview": True,
        "can_parse": True,
        "can_execute": True,
        "can_export": True,
        "can_admin": False,
    },
    "Admin": {
        "can_preview": True,
        "can_parse": True,
        "can_execute": True,
        "can_export": True,
        "can_admin": True,
    },
}


@dataclass(frozen=True)
class CurrentUser:
    username: str
    role: str


def has_users() -> bool:
    return bool(fetch_all("SELECT id FROM users LIMIT 1"))


def create_user(username: str, password: str, role: str) -> int:
    if role not in ROLES:
        raise ValueError(f"Unsupported role: {role}")
    if not username.strip():
        raise ValueError("Username cannot be blank.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    return execute_insert(
        """
        INSERT INTO users (username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            username.strip(),
            hash_password(password),
            role,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def authenticate_user(username: str, password: str) -> CurrentUser | None:
    rows = fetch_all(
        "SELECT username, password_hash, role FROM users WHERE username = ?",
        (username.strip(),),
    )
    if not rows:
        return None
    user = rows[0]
    if not verify_password(password, user["password_hash"]):
        return None
    return CurrentUser(username=user["username"], role=user["role"])


def get_permissions(role: str) -> dict[str, bool]:
    return ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS["Viewer"])


def list_users() -> list[dict[str, Any]]:
    return fetch_all("SELECT username, role, created_at FROM users ORDER BY username")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations),
    )
    return hmac.compare_digest(digest.hex(), digest_hex)
