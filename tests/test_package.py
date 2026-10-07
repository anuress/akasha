"""Package import check."""

import sys


def test_import_akasha():
    """akasha imports without error."""
    import akasha  # noqa: F401


def test_akasha_main_module():
    """akasha.__main__ module exists and is callable."""
    import importlib.util

    # Check __main__ module can be located
    spec = importlib.util.find_spec("akasha.__main__")
    assert spec is not None
