"""Reviewer accounts, and the sign-in every review endpoint needs."""

import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import api.auth as auth
from agents.orchestrator import ReviewOrchestrator
from api.dependencies import get_ingestion, get_orchestrator, get_reporting
from api.main import app

ACCOUNT = {"name": "Test Reviewer", "email": "Reviewer@Example.com",
           "password": "ledger2026", "role": "Senior Financial Auditor"}


class Storage:
    """Stands in for Risk & Reporting's database module."""

    def __init__(self):
        self.reviewers = []

    def list_reviews(self, limit=20):
        return [{"id": 1, "company": "Acme"}]

    def get_review(self, review_id):
        return {"id": review_id, "company": "Acme"} if review_id == 1 else None

    def record_action(self, review_id, finding_ref, status, note, reviewer):
        self.reviewers.append(reviewer)
        return len(self.reviewers)

    def get_actions(self, review_id):
        return []


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "accounts.sqlite"))
    monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", 1_000)  # the real work factor is deliberately slow
    for cached in (get_orchestrator, get_ingestion, get_reporting):
        cached.cache_clear()
    app.dependency_overrides.clear()
    app.dependency_overrides[get_orchestrator] = ReviewOrchestrator
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def storage():
    database = Storage()
    app.dependency_overrides[get_reporting] = lambda: SimpleNamespace(
        database=database, monitoring=None, report=None)
    return database


def register(client, **changes):
    return client.post("/auth/register", json={**ACCOUNT, **changes})


def sign_in(client, email=ACCOUNT["email"], password=ACCOUNT["password"]):
    return client.post("/auth/login", json={"email": email, "password": password})


def token_for(client):
    register(client)
    return sign_in(client).json()["token"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def stored(query):
    with sqlite3.connect(auth.os.environ["SQLITE_DB_PATH"]) as conn:
        return conn.execute(query).fetchall()


# --- creating an account --------------------------------------------------


def test_register_creates_an_account_and_never_returns_the_password(client):
    response = register(client)

    assert response.status_code == 201
    body = response.json()
    assert body == {"id": body["id"], "name": "Test Reviewer",
                    "email": "reviewer@example.com", "role": "Senior Financial Auditor"}


def test_passwords_are_stored_only_as_salted_hashes(client):
    register(client)
    register(client, email="second@example.com")

    hashes = [row[0] for row in stored("SELECT password_hash FROM users")]
    assert len(hashes) == 2
    assert all(h.startswith("pbkdf2_sha256$") and ACCOUNT["password"] not in h for h in hashes)
    assert hashes[0] != hashes[1], "the same password must hash differently for each account"


def test_an_email_can_only_be_registered_once_whatever_its_case(client):
    register(client)
    response = register(client, email="reviewer@EXAMPLE.COM")

    assert response.status_code == 409
    assert response.json()["detail"] == "An account with this email already exists."


@pytest.mark.parametrize("changes", [
    {"password": "short1"},
    {"password": "lettersonly"},
    {"password": "12345678"},
    {"email": "not-an-email"},
    {"role": "Intern"},
    {"name": "   "},
])
def test_invalid_registrations_are_refused_with_a_readable_message(client, changes):
    response = register(client, **changes)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


# --- signing in and out -----------------------------------------------------


def test_sign_in_returns_a_token_and_the_account(client):
    register(client)
    response = sign_in(client, email="REVIEWER@example.com")

    assert response.status_code == 200
    body = response.json()
    assert body["token"] and body["token_type"] == "bearer"
    assert body["user"]["email"] == "reviewer@example.com"
    assert datetime.fromisoformat(body["expires_at"]) > datetime.now(timezone.utc)


def test_wrong_password_and_unknown_email_get_the_same_answer(client):
    register(client)
    wrong_password = sign_in(client, password="not-my-password1")
    unknown_email = sign_in(client, email="nobody@example.com")

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json() == {"detail": "Invalid email or password."}


def test_tokens_are_stored_only_as_hashes(client):
    token = token_for(client)

    rows = stored("SELECT token_hash FROM auth_tokens")
    assert len(rows) == 1
    assert token not in rows[0][0]


def test_me_needs_a_valid_token(client):
    token = token_for(client)

    missing = client.get("/auth/me")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert client.get("/auth/me", headers=bearer("made-up-token")).status_code == 401
    assert client.get("/auth/me", headers=bearer(token)).json()["name"] == "Test Reviewer"


def test_signing_out_ends_the_session(client):
    token = token_for(client)

    assert client.post("/auth/logout", headers=bearer(token)).status_code == 204
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


def test_an_expired_token_is_refused(client, monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN_HOURS", "-1")
    token = token_for(client)

    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


# --- protecting review endpoints ---------------------------------------------


@pytest.mark.parametrize("method, path, kwargs", [
    ("post", "/review", {"json": {"records": [{"company": "Acme", "year": 2023}]}}),
    ("post", "/review/upload", {"files": {"file": ("s.csv", b"Year,Company\n2023,Acme\n", "text/csv")}}),
    ("get", "/reviews", {}),
    ("get", "/reviews/1", {}),
    ("get", "/reviews/1/report.pdf", {}),
    ("get", "/reviews/1/actions", {}),
    ("post", "/reviews/1/actions", {"json": {"finding_ref": "VAL_NI_04:Acme:2023", "status": "VERIFIED"}}),
    ("get", "/monitoring/summary", {}),
    ("get", "/monitoring/stages", {}),
    ("get", "/monitoring/coverage", {}),
    ("get", "/monitoring/recent", {}),
])
def test_review_endpoints_need_sign_in(client, storage, method, path, kwargs):
    response = getattr(client, method)(path, **kwargs)

    assert response.status_code == 401
    assert response.json() == {"detail": "Sign in to continue."}


def test_review_endpoints_work_once_signed_in(client, storage):
    response = client.get("/reviews", headers=bearer(token_for(client)))

    assert response.status_code == 200
    assert response.json() == [{"id": 1, "company": "Acme"}]


def test_actions_are_recorded_under_the_signed_in_account(client, storage):
    token = token_for(client)
    response = client.post("/reviews/1/actions", headers=bearer(token), json={
        "finding_ref": "VAL_NI_04:Acme:2023", "status": "VERIFIED", "reviewer": "Somebody Else"})

    assert response.status_code == 201
    assert storage.reviewers == ["Test Reviewer"]


def test_sign_in_can_be_turned_off_for_local_testing(client, storage, monkeypatch):
    monkeypatch.setenv("AUTH_REQUIRED", "false")

    assert client.get("/reviews").status_code == 200


def test_health_docs_and_account_endpoints_stay_public(client):
    assert client.get("/health").status_code == 200
    schema = client.get("/openapi.json").json()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    for path in ("/auth/register", "/auth/login", "/auth/me", "/auth/logout"):
        assert path in schema["paths"]
