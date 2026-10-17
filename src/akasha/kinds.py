"""Document kind vocabulary and custom-kind validation shared by config and writes."""
from __future__ import annotations

import re

KINDS = ("reference", "plan", "spec", "finding", "review", "progress",
         "decision", "convention", "gotcha", "log")
ALIASES = {"findings": "finding"}
_CUSTOM_KIND = re.compile(r"[a-z0-9][a-z0-9-]{0,23}\Z")


def custom_kind_error(kind: object, existing: list[str] | tuple[str, ...] = ()) -> str | None:
    """Explain why a value cannot be added to the custom vocabulary."""
    if not isinstance(kind, str):
        return "must be a string"
    if not _CUSTOM_KIND.fullmatch(kind):
        return "must be 1-24 lowercase letters, digits, or hyphens"
    if kind in KINDS:
        return "is a built-in kind"
    if kind in ALIASES:
        return "is a built-in kind alias"
    if kind in existing:
        return "duplicates an earlier custom kind"
    return None


def validate_custom_kinds(value: object) -> tuple[list[str], list[str]]:
    """Return valid unique names and a reason for each dropped config value."""
    if not isinstance(value, list):
        return [], [f"{value!r}: custom_kinds must be a TOML list"]

    valid: list[str] = []
    issues: list[str] = []
    for item in value:
        reason = custom_kind_error(item, valid)
        if reason is None:
            valid.append(item)
        else:
            issues.append(f"{item!r}: {reason}")
    return valid, issues


def canonical_kind(kind: str, custom_kinds: list[str] | tuple[str, ...] = ()) -> str:
    """Resolve aliases and reject kinds outside the configured vocabulary."""
    kind = ALIASES.get(kind, kind)
    if kind not in KINDS and kind not in custom_kinds:
        allowed = (*KINDS, *custom_kinds)
        raise ValueError(
            f"unknown kind: {kind!r}. Allowed kinds: {', '.join(allowed)}. "
            "If none fits, call again with new_kind: true to create it."
        )
    return kind
