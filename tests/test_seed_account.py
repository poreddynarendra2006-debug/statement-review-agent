"""The account the container recreates when it starts.

A deployment replaces the container, and the accounts database is a file inside
it, so every account disappears with the old container. The team found this the
hard way: an account created in the morning stopped working after an afternoon
deployment nobody on the demo side had made.

Seeding one known account on start means a deployment no longer locks anyone
out. These tests fix the behaviour that matters: it appears when it's missing,
it never touches an account that already exists, and it stays out of the way
when it isn't configured - which is the case locally and in every other test.
"""

import pytest

from api import auth


@pytest.fixture
def database(tmp_path, monkeypatch):
    """An empty database of its own, so nothing here touches the real file."""
    monkeypatch.setenv(auth.DB_ENV, str(tmp_path / "seed.db"))
    return tmp_path / "seed.db"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("SEED_ACCOUNT_EMAIL", "reviewer@example.com")
    monkeypatch.setenv("SEED_ACCOUNT_PASSWORD", "demo-password-1")
    monkeypatch.setenv("SEED_ACCOUNT_NAME", "Demo Reviewer")
    monkeypatch.setenv("SEED_ACCOUNT_ROLE", "Senior Financial Auditor")


def accounts():
    with auth._database() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM users")]


# --- what it is for ---------------------------------------------------------


def test_the_account_is_created_on_a_fresh_database(database, configured):
    auth.seed_account()

    rows = accounts()
    assert len(rows) == 1
    assert rows[0]["email"] == "reviewer@example.com"
    assert rows[0]["name"] == "Demo Reviewer"
    assert rows[0]["role"] == "Senior Financial Auditor"


def test_the_seeded_account_can_sign_in(database, configured):
    auth.seed_account()

    row = accounts()[0]
    assert auth.verify_password("demo-password-1", row["password_hash"])


def test_the_password_is_stored_as_a_hash(database, configured):
    auth.seed_account()

    assert "demo-password-1" not in accounts()[0]["password_hash"]


def test_starting_twice_leaves_one_account(database, configured):
    # Every restart calls this. It must not pile up duplicates, and the email
    # column is unique, so a second insert would raise.
    auth.seed_account()
    auth.seed_account()
    auth.seed_account()

    assert len(accounts()) == 1


def test_an_existing_account_keeps_its_own_password(database, configured, monkeypatch):
    # Somebody registered this address themselves. Their password wins - the
    # seed fills a gap, it does not reset anyone.
    auth.seed_account()
    theirs = accounts()[0]["password_hash"]

    monkeypatch.setenv("SEED_ACCOUNT_PASSWORD", "different-password-9")
    auth.seed_account()

    assert accounts()[0]["password_hash"] == theirs


def test_the_email_is_matched_however_it_is_capitalised(database, configured, monkeypatch):
    auth.seed_account()

    monkeypatch.setenv("SEED_ACCOUNT_EMAIL", "Reviewer@Example.COM")
    auth.seed_account()

    assert len(accounts()) == 1


# --- and when it should do nothing ------------------------------------------


def test_nothing_is_seeded_when_it_is_not_configured(database, monkeypatch):
    for variable in auth.SEED_VARS.values():
        monkeypatch.delenv(variable, raising=False)

    auth.seed_account()

    assert accounts() == []


@pytest.mark.parametrize("missing", ["SEED_ACCOUNT_EMAIL", "SEED_ACCOUNT_PASSWORD"])
def test_half_a_configuration_seeds_nothing(database, configured, monkeypatch, missing):
    monkeypatch.delenv(missing)

    auth.seed_account()

    assert accounts() == []


@pytest.mark.parametrize("password", ["short1", "no-digits-at-all", "12345678"])
def test_a_password_the_sign_up_form_would_reject_is_refused(database, configured,
                                                             monkeypatch, password):
    monkeypatch.setenv("SEED_ACCOUNT_PASSWORD", password)

    auth.seed_account()

    assert accounts() == [], "a seeded account must not be weaker than a registered one"


def test_an_invalid_email_is_refused(database, configured, monkeypatch):
    monkeypatch.setenv("SEED_ACCOUNT_EMAIL", "not-an-email")

    auth.seed_account()

    assert accounts() == []


def test_an_unknown_role_is_refused(database, configured, monkeypatch):
    monkeypatch.setenv("SEED_ACCOUNT_ROLE", "Head of Everything")

    auth.seed_account()

    assert accounts() == []


def test_the_name_and_role_have_sensible_defaults(database, configured, monkeypatch):
    monkeypatch.delenv("SEED_ACCOUNT_NAME")
    monkeypatch.delenv("SEED_ACCOUNT_ROLE")

    auth.seed_account()

    row = accounts()[0]
    assert row["name"]
    assert row["role"] in auth.ROLES


# --- it runs as part of starting up -----------------------------------------


def test_startup_seeds_the_account(database, configured):
    auth.prepare_accounts()

    assert [r["email"] for r in accounts()] == ["reviewer@example.com"]


def test_startup_still_works_with_nothing_configured(database, monkeypatch):
    for variable in auth.SEED_VARS.values():
        monkeypatch.delenv(variable, raising=False)

    auth.prepare_accounts()  # must not raise

    assert accounts() == []
