"""The guard that keeps tests off the user's real database.

A fingerprint of the real file's mtime before and after each test detects *any* writer,
not this process, so another process touching the file would fail a test that never used
it. Guarding the action instead removes the race: a connection to the real database is
refused when it is opened, by the process that opened it.
"""
from __future__ import annotations

import os
import pwd
import sqlite3
from pathlib import Path

import pytest


def _real_db() -> Path:
    # expanduser honours the patched HOME, so the true home comes from the password
    # database — that is what an absolute-path write would reach.
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".akasha" / "akasha.db"


@pytest.fixture
def stands_inside_the_guarded_dir(tmp_path, monkeypatch):
    """A cwd the guard treats as inside the real installation, without creating it.

    Redirecting the resolver tests the same logic against a directory that is ours to
    make, instead of creating one inside the operator's home.
    """
    import conftest

    fake_real = tmp_path / "fake-home" / ".akasha"
    inside = fake_real / "nested" / "pretend-checkout"
    inside.mkdir(parents=True)          # built first: once redirected, the guard refuses it
    monkeypatch.setattr(conftest, "real_akasha_dir", lambda: fake_real.resolve())
    monkeypatch.chdir(inside)
    return fake_real


def test_connecting_to_the_real_database_is_refused():
    """The guard must still fail loudly for the thing it exists to catch. A guard that
    stopped false-firing by never firing would be worse than the flake it replaced."""
    with pytest.raises(AssertionError, match="real database"):
        sqlite3.connect(str(_real_db()))


def test_connecting_to_anything_under_the_real_akasha_dir_is_refused():
    """The knowledge directory lives beside the database and is equally not ours."""
    with pytest.raises(AssertionError, match="real"):
        sqlite3.connect(str(_real_db().parent / "some-other.db"))


def test_a_temporary_database_still_connects(tmp_path):
    """The guard must be invisible to every legitimate test, or it becomes the thing
    people work around."""
    conn = sqlite3.connect(str(tmp_path / "s.db"))
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.close()


def test_in_memory_databases_still_connect():
    """':memory:' is not a path; resolving it must not be mistaken for a real one."""
    sqlite3.connect(":memory:").close()


def test_a_file_uri_to_a_temporary_database_still_connects(tmp_path):
    """Read-only access is a file: URI. Resolving the whole string as a relative filename
    would either miss the real directory or refuse a legitimate database."""
    db = tmp_path / "s.db"
    sqlite3.connect(str(db)).close()
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.execute("SELECT 1")
    conn.close()


def test_a_file_uri_into_the_real_directory_is_refused():
    with pytest.raises(AssertionError, match="real"):
        sqlite3.connect(f"file:{_real_db()}?mode=ro", uri=True)


def test_an_external_writer_does_not_fail_a_test():
    """Something outside this process writing the real database during a test is none of
    the test's business, and must not fail it.

    Simulated rather than performed: writing to the real database to prove a test about
    not writing to it would be its own bug. The guard does not read that file's mtime at
    all, which is what makes the race impossible.
    """
    source = (Path(__file__).parent / "conftest.py").read_text()
    assert "st_mtime_ns" not in source, (
        "the guard is fingerprinting mtime again, which reintroduces the external-writer "
        "race this file documents"
    )


def test_a_test_cannot_create_a_directory_inside_the_real_akasha_dir():
    """The connect guard covers sqlite only, so a plain mkdir would walk straight past it
    and leave stray directories in the real installation. Same shape as the connect
    guard: refuse the act in the process that performs it."""
    with pytest.raises(AssertionError, match="real"):
        (_real_db().parent / "nested" / "leaked-by-a-test").mkdir(parents=True, exist_ok=True)


def test_a_relative_path_from_inside_the_akasha_dir_still_connects(
        stands_inside_the_guarded_dir):
    """A process whose cwd is inside the guarded directory must still be able to open
    ':memory:' or a relative file. Resolving a relative path first would make it look
    like a file in there and refuse it."""
    sqlite3.connect(":memory:").close()
    conn = sqlite3.connect("relative.db")      # resolves under the guarded dir, but is not it
    conn.close()


def test_an_absolute_path_to_the_real_dir_is_still_refused_from_inside_it(
        stands_inside_the_guarded_dir):
    """The loosening above must not open a hole: naming the guarded database outright is
    still refused, wherever the test happens to be running from."""
    with pytest.raises(AssertionError, match="real database"):
        sqlite3.connect(str(stands_inside_the_guarded_dir / "akasha.db"))
