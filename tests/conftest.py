"""Suite-wide isolation from the user's real installation.

`load_config()` falls back to `~/.akasha/config.toml`, so a fixture that builds a Config
without overriding `db_path` gets one pointing at the user's database.

Every test therefore gets its own HOME. A test that wants the real one must say so
explicitly, and none currently do.
"""
from __future__ import annotations

import pytest


def real_akasha_dir():
    """The installation this suite must never touch.

    Resolved from the password database rather than `expanduser`, which honours the
    patched HOME. Called on every check rather than captured once, so a test that needs
    to stand *inside* the guarded directory can point it at a temporary one instead of
    creating the real thing.
    """
    import os
    import pwd
    from pathlib import Path

    return (Path(pwd.getpwuid(os.getuid()).pw_dir) / ".akasha").resolve()


def _is_inside_real_dir(target) -> bool:
    from pathlib import Path

    real_dir = real_akasha_dir()
    resolved = Path(target).resolve()
    return resolved == real_dir or real_dir in resolved.parents


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Point HOME at a per-test directory before anything can read the real one."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _guard_real_akasha_dir(isolated_home, monkeypatch):
    """Refuse a connection to the real ~/.akasha, rather than noticing one afterwards.

    Refusing the connect catches a read or write at the moment it happens, with its own
    stack trace, rather than a later test failing for no visible reason.
    """
    import sqlite3
    from pathlib import Path

    real_connect = sqlite3.connect

    def guarded_connect(database, *args, **kwargs):
        text = str(database)
        # A file: URI is a path in disguise; parse it, or it would be resolved as a
        # relative filename. Other URIs, such as :memory:, are not paths on this machine
        # and pass through untouched.
        if text.startswith("file:"):
            from urllib.parse import unquote, urlparse

            target = Path(unquote(urlparse(text).path))
            if _is_inside_real_dir(target):
                raise AssertionError(
                    f"test opened the real database or a file beside it: {target}. "
                    "Fixtures must set cfg.db_path (and knowledge_dir) to tmp_path."
                )
            return real_connect(database, *args, **kwargs)
        try:
            given = Path(text).expanduser()
        except (TypeError, ValueError, OSError):
            return real_connect(database, *args, **kwargs)
        # Only an absolute path can be the real database. A relative one resolves against
        # the cwd, and resolving first would make ':memory:' look like a file inside
        # the guarded directory whenever the cwd is inside it.
        if not given.is_absolute():
            return real_connect(database, *args, **kwargs)
        if _is_inside_real_dir(given):
            raise AssertionError(
                f"test opened the real database or a file beside it: {given}. "
                "Fixtures must set cfg.db_path (and knowledge_dir) to tmp_path."
            )
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)

    # A directory is not a database, so mkdir would walk past the guard above.
    real_mkdir = Path.mkdir

    def guarded_mkdir(self, *args, **kwargs):
        if _is_inside_real_dir(self):
            raise AssertionError(
                f"test created a directory inside the real installation: {self}. "
                "Point the fixture at tmp_path, or patch conftest.real_akasha_dir to a "
                "temporary directory if the test must stand inside one."
            )
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    yield


@pytest.fixture(autouse=True)
def _init_stays_keyword_only(monkeypatch):
    """`akasha init` turns dense search on when the extra is importable, which would make
    every test that runs init load an encoder. Tests that want it patch it back on."""
    from akasha import vectors

    monkeypatch.setattr(vectors, "extra_installed", lambda: False, raising=False)
