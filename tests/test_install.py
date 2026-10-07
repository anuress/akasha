import json
import os
import subprocess

import pytest

from akasha import install
from akasha.install import VENDORS, detect_vendors, mcp_registered, register, register_all


class Result:
    returncode = 0
    stderr = ""


@pytest.fixture
def run_calls(monkeypatch):
    """Vendor CLIs are never run for real: every subprocess call is recorded."""
    calls = []

    def fake(argv, **kwargs):
        calls.append((argv, kwargs))
        return Result()

    monkeypatch.setattr(install.subprocess, "run", fake)
    return calls


def test_supported_vendors_are_the_three_with_a_registration_path():
    assert list(VENDORS) == ["claude", "gemini", "copilot"]
    assert "othercli" not in VENDORS


def test_detect_vendors_returns_only_installed_ones(monkeypatch):
    installed = {"gemini", "claude", "othercli"}
    monkeypatch.setattr("shutil.which",
                        lambda n: f"/usr/bin/{n}" if n in installed else None)
    assert detect_vendors() == ["claude", "gemini"]


# --- vendors with their own `mcp add` ------------------------------------------------

def test_claude_is_registered_through_its_own_cli_not_its_config_file(tmp_path, run_calls):
    """claude owns ~/.claude.json; akasha must never write it."""
    assert "registered" in register("claude", tmp_path)
    assert run_calls[0][0] == ["claude", "mcp", "add", "--scope", "user", "akasha", "--",
                               "akasha", "serve"]
    assert not (tmp_path / ".claude.json").exists()


def test_gemini_is_registered_through_its_own_cli(tmp_path, run_calls):
    register("gemini", tmp_path)
    assert run_calls[0][0][:6] == ["gemini", "mcp", "add", "--scope", "user", "--transport"]
    assert run_calls[0][0][-3:] == ["akasha", "akasha", "serve"]


def test_gemini_trust_is_scoped_to_the_akasha_server_only(tmp_path):
    """--trust skips confirmation for the server it is attached to; it must not widen."""
    cmd = register("gemini", tmp_path, dry_run=True)
    assert "--trust" in cmd
    for wider in ("--yolo", "--approval-mode", "--dangerously"):
        assert wider not in cmd


def test_vendor_calls_never_wait_on_a_terminal_or_forever(tmp_path, run_calls):
    """A vendor CLI that prompts or hangs must not stall `akasha init`."""
    register("claude", tmp_path)
    kwargs = run_calls[0][1]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize("error, reason", [
    (FileNotFoundError("no such binary"), "no such binary"),
    (subprocess.TimeoutExpired("claude", 30), "timed out"),
])
def test_a_vendor_cli_that_cannot_run_is_a_failed_result_not_a_crash(
        tmp_path, monkeypatch, error, reason):
    def boom(*a, **k):
        raise error

    monkeypatch.setattr(install.subprocess, "run", boom)
    result = register("claude", tmp_path)
    assert "failed:" in result and reason in result


def test_a_non_zero_exit_names_the_failure(tmp_path, monkeypatch):
    class Bad:
        returncode = 1
        stderr = "boom"

    monkeypatch.setattr(install.subprocess, "run", lambda *a, **k: Bad())
    assert "failed: boom" in register("gemini", tmp_path)


def test_dry_run_runs_no_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(install.subprocess, "run",
                        lambda *a, **k: pytest.fail("subprocess must not run in dry-run"))
    assert "would run" in register("claude", tmp_path, dry_run=True)
    assert "would run" in register("gemini", tmp_path, dry_run=True)


def test_an_already_registered_server_is_not_added_again(tmp_path, run_calls):
    (tmp_path / ".claude.json").write_text(json.dumps({"mcpServers": {"akasha": {}}}))
    assert "already" in register("claude", tmp_path)
    assert run_calls == []


# --- vendors with no add command: atomic file edit -----------------------------------

def _file_for(home, vendor):
    return {"copilot": home / ".copilot" / "mcp-config.json"}[vendor]


@pytest.mark.parametrize("vendor, key", [("copilot", "mcpServers")])
def test_a_vendor_without_an_add_command_gets_its_config_file_edited(tmp_path, run_calls,
                                                                     vendor, key):
    register(vendor, tmp_path)
    assert "akasha" in json.loads(_file_for(tmp_path, vendor).read_text())[key]
    assert run_calls == []


def test_file_registration_preserves_everything_else(tmp_path):
    path = _file_for(tmp_path, "copilot")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}, "k": 1}))
    register("copilot", tmp_path)
    data = json.loads(path.read_text())
    assert set(data["mcpServers"]) == {"other", "akasha"} and data["k"] == 1


def test_file_registration_is_idempotent(tmp_path):
    register("copilot", tmp_path)
    assert "already" in register("copilot", tmp_path)


def test_file_registration_dry_run_writes_nothing(tmp_path):
    assert "would" in register("copilot", tmp_path, dry_run=True)
    assert not _file_for(tmp_path, "copilot").exists()


@pytest.mark.parametrize("text", ["{not json", "[]", '{"mcpServers": []}', "\xff\xfe"])
def test_unusable_config_is_left_untouched_with_a_reason(tmp_path, text):
    path = _file_for(tmp_path, "copilot")
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode("latin-1"))
    result = register("copilot", tmp_path)
    assert path.read_bytes() == text.encode("latin-1")
    assert "left untouched" in result


def test_an_unreadable_config_is_left_untouched_not_a_crash(tmp_path):
    path = _file_for(tmp_path, "copilot")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("running as a user that ignores file modes")
        assert "left untouched" in register("copilot", tmp_path)
    finally:
        path.chmod(0o600)


@pytest.mark.parametrize("target", ["os.fsync", "os.replace"])
def test_a_failure_mid_write_leaves_the_original_intact(tmp_path, monkeypatch, target):
    """A half-written config would lose the user's other servers."""
    path = _file_for(tmp_path, "copilot")
    path.parent.mkdir(parents=True)
    original = json.dumps({"mcpServers": {"other": {"command": "x"}}})
    path.write_text(original)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(target, boom)
    result = register("copilot", tmp_path)
    assert path.read_text() == original
    assert "failed" in result
    assert [p.name for p in path.parent.iterdir()] == [path.name], "temp file left behind"


def test_a_rewrite_keeps_the_original_file_mode(tmp_path):
    path = _file_for(tmp_path, "copilot")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    path.chmod(0o640)
    register("copilot", tmp_path)
    assert path.stat().st_mode & 0o777 == 0o640


# --- checking registration ------------------------------------------------------------

def test_mcp_registered_reads_the_vendors_own_file(tmp_path):
    assert mcp_registered("claude", tmp_path) is False
    (tmp_path / ".claude.json").write_text(json.dumps({"mcpServers": {"akasha": {}}}))
    assert mcp_registered("claude", tmp_path) is True


def test_mcp_registered_is_unchecked_when_the_file_cannot_be_read(tmp_path):
    """False would tell the user to re-register over a config that merely failed to parse."""
    (tmp_path / ".claude.json").write_text("{not json")
    assert mcp_registered("claude", tmp_path) is None
    assert mcp_registered("othercli", tmp_path) is None


def test_register_all_covers_only_requested_vendors(tmp_path, run_calls):
    out = register_all(["claude", "othercli"], dry_run=True, home=tmp_path)
    assert list(out) == ["claude"]
