"""
Analyst authentication + RBAC (Role-Based Access Control).

WHY THIS EXISTS: alert triage, case management, and webhook configuration
endpoints (see main.py / cases.py / webhooks.py) were previously
deliberately left open -- their docstrings said so explicitly, reasoning
"no login flow exists for the dashboard." This module adds that login
flow: JWT-based session auth for analysts, with two roles:

  - "analyst": day-to-day SOC work -- triage alerts, work cases, add
               notes, configure webhooks.
  - "admin":   everything an analyst can do, PLUS manage other analyst
               accounts (create/deactivate) and delete webhooks.

DESIGN: stateless JWT (HS256) rather than server-side sessions -- no
extra infrastructure (Redis, session middleware) needed for a project at
this scale, and it fits the existing lightweight FastAPI + MongoDB stack.
Tokens are short-lived (default 8h, no refresh-token flow) -- an analyst
re-logging in once a shift is an acceptable tradeoff at this scope.

SCOPE NOTE (state this in the report, same honesty pattern as the rest
of this codebase): this is analyst-facing session auth for the dashboard
UI. It is DELIBERATELY SEPARATE from the existing X-API-Key scheme on
/detect, /logs (POST), and /ingest-cert in main.py -- those protect
machine-to-machine ingestion traffic (an e-commerce site posting
events), a different trust boundary than a human analyst logging into
the console. The two schemes are not meant to be unified.

SECURITY NOTE: set SIEM_JWT_SECRET before deploying. If it's left unset,
this falls back to an insecure hardcoded value -- fine for local dev,
NOT fine for a public deployment (same pattern as SIEM_API_KEY in
main.py).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from bson import ObjectId

SECRET_KEY = os.environ.get("SIEM_JWT_SECRET", "dev-only-insecure-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 8

ROLES = {"analyst", "admin"}

# auto_error=False so a missing header raises OUR 401 (with a clear
# message) instead of FastAPI's generic one -- and so this dependency
# can be reused in places that want to check "is anyone logged in?"
# without crashing before we get a chance to explain why.
_security = HTTPBearer(auto_error=False)


# =========================
# Password hashing
# =========================
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# =========================
# JWT
# =========================
def create_access_token(user_id: str, username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": user_id, "username": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_security)):
    """
    FastAPI dependency: extracts + validates the Bearer token from the
    Authorization header, returns {"user_id", "username", "role"}.
    Raises 401 if the header is missing or the token is invalid/expired.
    """
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated -- missing Authorization header")
    payload = decode_token(credentials.credentials)
    return {"user_id": payload["sub"], "username": payload["username"], "role": payload["role"]}


def require_role(*allowed_roles):
    """
    FastAPI dependency factory. require_role("admin") -- admins only.
    require_role("analyst", "admin") -- any authenticated user, since
    those are currently the only two roles (kept explicit rather than
    just using get_current_user directly, so it's obvious at each
    endpoint which roles were intended to reach it).
    """
    def _dependency(user: dict = Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"This action requires one of roles {sorted(allowed_roles)}; "
                       f"your role is '{user['role']}'.",
            )
        return user
    return _dependency


# =========================
# User management (DB-backed) -- db.users collection
#   { _id, username, hashed_password, role, display_name, active,
#     created_at, last_login }
# =========================
def _oid(value):
    return value if isinstance(value, ObjectId) else ObjectId(value)


def get_user_by_username(db, username: str):
    return db.users.find_one({"username": username})


def create_user(db, username: str, password: str, role: str = "analyst", display_name: str = None):
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    if not username or not password:
        raise ValueError("username and password are required")
    if get_user_by_username(db, username):
        raise ValueError(f"Username '{username}' already exists")

    now = datetime.utcnow().isoformat()
    doc = {
        "username": username,
        "hashed_password": hash_password(password),
        "role": role,
        "display_name": display_name or username,
        "active": True,
        "created_at": now,
        "last_login": None,
    }
    result = db.users.insert_one(doc)
    return str(result.inserted_id)


def authenticate_user(db, username: str, password: str):
    """Returns the user doc if the credentials are valid and the account
    is active, else None. Deliberately returns the same None for 'no
    such user' and 'wrong password' -- doesn't leak which one it was."""
    user = get_user_by_username(db, username)
    if not user or not user.get("active", True):
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def list_users(db):
    results = []
    for u in db.users.find({}, {"hashed_password": 0}):
        u["_id"] = str(u["_id"])
        results.append(u)
    return results


def set_user_active(db, user_id: str, active: bool):
    result = db.users.update_one({"_id": _oid(user_id)}, {"$set": {"active": active}})
    return result.matched_count > 0


def touch_last_login(db, user_id: str):
    db.users.update_one({"_id": _oid(user_id)}, {"$set": {"last_login": datetime.utcnow().isoformat()}})