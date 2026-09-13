"""A review belongs to the account that ran it.

A new account opened the history and found every upload the team had made,
because reviews carried no owner and no endpoint asked who was reading. These
run the real API against a real database with two real accounts, and check
every way a review can be reached: the history, opening it, its PDF, and its
reviewer actions.

The two demo reviews created at startup have no owner and stay visible to
everyone - they are examples, not anyone's statements.
"""

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("database.database")

from api import auth  # noqa: E402
from api.dependencies import get_ingestion, get_orchestrator, get_reporting  # noqa: E402

RECORDS = [{"company": "Acme Corporation", "year": year, "revenue": 1000.0 + year,
            "net_income": 100.0, "total_assets": 2000.0}
           for year in (2021, 2022, 2023)]


def _clear_caches():
    for cached in (get_orchestrator, get_ingestion, get_reporting):
        cached.cache_clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "private.sqlite"))
    monkeypatch.setattr(auth, "PBKDF2_ITERATIONS", 1_000)  # the real work factor is deliberately slow
    _clear_caches()
    from api.main import app
    with TestClient(app) as test_client:
        yield test_client
    _clear_caches()


def sign_in(client, who):
    account = {"name": who.title(), "email": f"{who}@example.com",
               "password": "ledger2026", "role": "Senior Financial Auditor"}
    assert client.post("/auth/register", json=account).status_code == 201
    token = client.post("/auth/login", json={"email": account["email"],
                                             "password": account["password"]}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def run_review(client, headers):
    response = client.post("/review", json={"records": RECORDS}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()["review_id"]


def history_ids(client, headers):
    response = client.get("/reviews?limit=200", headers=headers)
    assert response.status_code == 200
    return {row["review_id"] for row in response.json()}


@pytest.fixture
def two_reviewers(client):
    alice, bob = sign_in(client, "alice"), sign_in(client, "bob")
    return alice, bob, run_review(client, alice), run_review(client, bob)


# --- each sees their own ------------------------------------------------------


def test_the_history_shows_only_your_own_uploads(client, two_reviewers):
    alice, bob, alices, bobs = two_reviewers

    assert alices in history_ids(client, alice)
    assert bobs not in history_ids(client, alice)
    assert bobs in history_ids(client, bob)
    assert alices not in history_ids(client, bob)


def test_a_brand_new_account_sees_no_one_elses_uploads(client, two_reviewers):
    _, _, alices, bobs = two_reviewers
    newcomer = sign_in(client, "carol")

    assert not {alices, bobs} & history_ids(client, newcomer)


def test_you_can_open_your_own_review(client, two_reviewers):
    alice, _, alices, _ = two_reviewers

    assert client.get(f"/reviews/{alices}", headers=alice).status_code == 200


# --- and nothing of anyone else's --------------------------------------------


@pytest.mark.parametrize("method, path", [
    ("get", "/reviews/{id}"),
    ("get", "/reviews/{id}/report.pdf"),
    ("get", "/reviews/{id}/actions"),
])
def test_someone_elses_review_is_not_found(client, two_reviewers, method, path):
    alice, _, _, bobs = two_reviewers

    response = getattr(client, method)(path.format(id=bobs), headers=alice)

    assert response.status_code == 404


def test_you_cannot_record_a_decision_on_someone_elses_review(client, two_reviewers):
    alice, bob, _, bobs = two_reviewers

    response = client.post(f"/reviews/{bobs}/actions", headers=alice,
                           json={"finding_ref": "VAL_BS_01:Acme:2023", "status": "VERIFIED"})

    assert response.status_code == 404
    assert client.get(f"/reviews/{bobs}/actions", headers=bob).json() == []


def test_not_found_does_not_reveal_that_the_review_exists(client, two_reviewers):
    # Someone else's review and one that was never created must answer alike,
    # or the difference tells an outsider which ids are in use.
    alice, _, _, bobs = two_reviewers

    theirs = client.get(f"/reviews/{bobs}", headers=alice)
    missing = client.get("/reviews/99999", headers=alice)

    assert theirs.status_code == missing.status_code == 404
    assert theirs.json()["detail"].replace(str(bobs), "N") == missing.json()["detail"].replace("99999", "N")


def test_the_owner_is_not_leaked_in_the_review_itself(client, two_reviewers):
    alice, _, alices, _ = two_reviewers

    assert "owner_id" not in client.get(f"/reviews/{alices}", headers=alice).json()


# --- demo reviews stay shared --------------------------------------------------


def test_the_demo_reviews_are_visible_to_everyone(client, two_reviewers):
    alice, bob, alices, bobs = two_reviewers

    shared = history_ids(client, alice) - {alices}
    assert shared, "the startup demo reviews should appear in every history"
    assert shared == history_ids(client, bob) - {bobs}
    for review_id in shared:
        assert client.get(f"/reviews/{review_id}", headers=bob).status_code == 200


# --- a database from before this change still works ---------------------------


def test_an_older_database_gains_the_owner_column(tmp_path, monkeypatch):
    import sqlite3

    from database import database

    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as conn:  # the table as it was before reviews had owners
        conn.execute("""CREATE TABLE reviews (id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, company TEXT NOT NULL, period TEXT, record_count INTEGER,
            risk_score INTEGER, risk_level TEXT, review_mode TEXT, elapsed_seconds REAL,
            findings_count INTEGER NOT NULL, failed_checks_count INTEGER NOT NULL,
            filename TEXT, result_json TEXT NOT NULL)""")
    monkeypatch.setenv("SQLITE_DB_PATH", str(path))

    review_id = database.save_review({"company": "Acme", "owner_id": 7})

    assert database.get_review(review_id)["owner_id"] == 7
    assert [r["id"] for r in database.list_reviews(owner_id=7)] == [review_id]
    assert database.list_reviews(owner_id=8) == []
