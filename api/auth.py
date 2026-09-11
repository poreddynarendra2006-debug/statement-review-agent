"""Reviewer accounts: create an account, sign in, and the token review endpoints need.

    POST /auth/register     create an account
    POST /auth/login        sign in and receive a token
    GET  /auth/me           the account a token belongs to
    POST /auth/logout       end the session; the token stops working

Passwords are kept only as salted PBKDF2-SHA256 hashes. Signing in issues a
random token and only a SHA-256 hash of it is stored, so a copy of the
database can't be used to sign in. Tokens expire after AUTH_TOKEN_HOURS.

Accounts live in the same SQLite file as saved reviews (SQLITE_DB_PATH), in
tables of their own, so there is still one file to back up.

Review endpoints need a token unless AUTH_REQUIRED is false, which is meant
for local testing only.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Iterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import ErrorResponse, LoginRequest, LoginResponse, RegisterRequest, UserResponse

logger = logging.getLogger("api.auth")

DB_ENV = "SQLITE_DB_PATH"
DEFAULT_DB = "finsight_review.db"

#: OWASP's recommended work factor for PBKDF2-SHA256.
PBKDF2_ITERATIONS = 600_000

ROLES = ("Senior Financial Auditor", "Chief Financial Officer (CFO)", "Risk & Compliance Analyst")
MIN_PASSWORD, MAX_PASSWORD = 8, 128
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SIGN_IN_FAILED = "Invalid email or password."
SIGN_IN_REQUIRED = "Sign in to continue."

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""

router = APIRouter(prefix="/auth", tags=["accounts"])
bearer = HTTPBearer(auto_error=False,
                    description="Sign in with POST /auth/login, then paste the token here.")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def auth_required() -> bool:
    """Whether review endpoints need a token. Read per request, so it can change without a restart."""
    return os.environ.get("AUTH_REQUIRED", "true").strip().lower() not in {"0", "false", "no", "off"}


def token_lifetime() -> timedelta:
    return timedelta(hours=float(os.environ.get("AUTH_TOKEN_HOURS", "12")))


# ---------------------------------------------------------------------------
# Storage and hashing
# ---------------------------------------------------------------------------


@contextmanager
def _database() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(os.environ.get(DB_ENV, DEFAULT_DB))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        with conn:
            yield conn
    finally:
        conn.close()


def prepare_accounts() -> None:
    """Create the account tables at startup. Failure is logged, never fatal."""
    _decoy_hash()  # built now, so the first unknown-email sign-in isn't slower than the rest
    try:
        with _database():
            pass
    except Exception:  # noqa: BLE001 - the service must start even if storage is unavailable
        logger.exception("could not prepare account storage")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    # One fixed format, so stored times compare correctly as text.
    return moment.isoformat(timespec="microseconds")


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "$".join(("pbkdf2_sha256", str(PBKDF2_ITERATIONS),
                     base64.b64encode(salt).decode(), base64.b64encode(derived).decode()))


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, expected = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                      base64.b64decode(salt), int(iterations))
        return hmac.compare_digest(derived, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=1)
def _decoy_hash() -> str:
    """Checked when no account matches, so an unknown email takes as long as a wrong password."""
    return hash_password(secrets.token_urlsafe(16))


def _account(row: sqlite3.Row) -> UserResponse:
    return UserResponse(id=row["id"], name=row["name"], email=row["email"], role=row["role"])


def _account_for(token: str) -> Optional[UserResponse]:
    with _database() as conn:
        row = conn.execute(
            "SELECT users.* FROM auth_tokens JOIN users ON users.id = auth_tokens.user_id "
            "WHERE auth_tokens.token_hash = ? AND auth_tokens.expires_at > ?",
            (_digest(token), _stamp(_now())),
        ).fetchone()
    return _account(row) if row else None


# ---------------------------------------------------------------------------
# Guards for other endpoints
# ---------------------------------------------------------------------------


def _sign_in_required() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=SIGN_IN_REQUIRED,
                         headers={"WWW-Authenticate": "Bearer"})


def signed_in_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> UserResponse:
    """The account behind the request's token. Always needs a valid token."""
    account = _account_for(credentials.credentials) if credentials else None
    if account is None:
        raise _sign_in_required()
    return account


def require_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> Optional[UserResponse]:
    """Guard for review endpoints: the signed-in account, or None when sign-in is turned off."""
    account = _account_for(credentials.credentials) if credentials else None
    if account is None and auth_required():
        raise _sign_in_required()
    return account


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _invalid(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=message)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED,
             responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def register(account: RegisterRequest) -> UserResponse:
    """Create a reviewer account."""
    name = account.name.strip()
    email = account.email.strip().lower()
    password = account.password

    if not name or len(name) > 120:
        raise _invalid("Enter your full name (up to 120 characters).")
    if len(email) > 254 or not EMAIL.match(email):
        raise _invalid("Enter a valid email address.")
    if account.role not in ROLES:
        raise _invalid(f"Choose one of these roles: {', '.join(ROLES)}.")
    if not (MIN_PASSWORD <= len(password) <= MAX_PASSWORD
            and re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise _invalid(f"Use a password of {MIN_PASSWORD} to {MAX_PASSWORD} characters "
                       "with at least one letter and one number.")

    password_hash = hash_password(password)
    try:
        with _database() as conn:
            cursor = conn.execute(
                "INSERT INTO users (name, email, role, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (name, email, account.role, password_hash, _stamp(_now())),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="An account with this email already exists.") from exc
    return UserResponse(id=cursor.lastrowid, name=name, email=email, role=account.role)


@router.post("/login", response_model=LoginResponse, responses={401: {"model": ErrorResponse}})
def login(credentials: LoginRequest) -> LoginResponse:
    """Sign in. Send the token back as 'Authorization: Bearer <token>'."""
    email = credentials.email.strip().lower()
    with _database() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    # Hash either way, so the response time doesn't reveal whether the email exists.
    valid = verify_password(credentials.password, row["password_hash"] if row else _decoy_hash())
    if row is None or not valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=SIGN_IN_FAILED)

    token = secrets.token_urlsafe(32)
    now = _now()
    expires = now + token_lifetime()
    with _database() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE expires_at <= ?", (_stamp(now),))
        conn.execute(
            "INSERT INTO auth_tokens (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (_digest(token), row["id"], _stamp(now), _stamp(expires)),
        )
    return LoginResponse(token=token, expires_at=expires.isoformat(timespec="seconds"), user=_account(row))


@router.get("/me", response_model=UserResponse, responses={401: {"model": ErrorResponse}})
def me(account: UserResponse = Depends(signed_in_user)) -> UserResponse:
    """The signed-in account."""
    return account


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, response_class=Response,
             dependencies=[Depends(signed_in_user)], responses={401: {"model": ErrorResponse}})
def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> Response:
    """Sign out. The token stops working immediately."""
    with _database() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token_hash = ?", (_digest(credentials.credentials),))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
