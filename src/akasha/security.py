"""Trust-boundary filtering. Runs before anything is indexed, registered, or assembled."""
from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from akasha.config import Config

# (rule name, pattern). Ordered — first match wins per span.
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("bearer", re.compile(r"(?i)\b(?:authorization:\s*)?bearer\s+[A-Za-z0-9._\-]{16,}")),
    ("generic_token", re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|token)\b\s*[=:]\s*['\"]?[A-Za-z0-9._\-]{12,}['\"]?")),
]


def is_denied(path: Path, cfg: Config) -> bool:
    """True when a path must never be indexed, registered, or read into context."""
    name = path.name
    if any(name.lower().endswith(ext.lower()) for ext in cfg.deny_extensions):
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in cfg.deny_files)


def redact(text: str) -> tuple[str, list[str]]:
    """Replace secret-shaped spans. Returns (clean_text, rule_names_that_fired)."""
    hits: list[str] = []
    for name, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            hits.append(name)
            text = pattern.sub(f"[REDACTED:{name}]", text)
    return text, hits


# (rule name, pattern). Ordered — first match wins per span, like SECRET_PATTERNS.
# Two classes by handling: control characters are stripped outright, because this corpus
# has no legitimate use for them; instruction-shaped spans are wrapped in markers, not
# deleted, so a document that quotes an attack keeps its words and a model reading the
# text sees a flag instead of an instruction. The marker relies on the reader treating
# retrieved content as data, not rules.
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("invisible", re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]")),
    ("bidi_control", re.compile(r"[\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069]")),
    ("instruction_override", re.compile(
        r"(?i)\b(?:ignore|disregard|forget|overlook)\s+"
        r"(?:all\s+|any\s+|the\s+|these\s+|those\s+|my\s+|your\s+|previous\s+|prior\s+"
        r"|above\s+|earlier\s+|given\s+|system\s+|user\s+)*"
        r"(?:instructions?|prompts?|directives?|guidelines?|guidance)\b")),
    ("reverse_guards", re.compile(
        r"(?i)\b(?:do\s+not|don'?t)\s+(?:follow|obey|heed)\s+"
        r"(?:any|the|these|those)\s+(?:above|previous|prior|system|given)?\s*"
        r"(?:instructions?|prompts?|directives?|guidelines?)\b")),
    ("reader_takeover", re.compile(
        r"(?i)\byou\s+are\s+now\s+(?:an?\s+|the\s+)?(?:ai|assistant|system|model|agent)\b")),
]

# Rules whose match is removed outright rather than wrapped around.
INJECTION_STRIP_RULES = frozenset({"invisible", "bidi_control"})

_INJECTION_OPEN = "[INJECTION:"


def _is_self_wrapped(match: re.Match, text: str, name: str) -> bool:
    """True when the match is the body of a marker pair this function itself wrote.

    Only a pair the scrub produced is trusted as proof of an earlier pass: the opener
    carries exactly this rule's name and both markers sit flush against the span. Marker-
    shaped text in input — an unclosed literal '[INJECTION:done]' ahead of a span — has
    the wrong name or a gap, so it is wrapped like any other span instead of trusted as
    an outer pair.
    """
    opener = f"{_INJECTION_OPEN}{name}]"
    closer = "[/INJECTION]"
    start, end = match.start(), match.end()
    if start < len(opener):
        return False
    return text[start - len(opener):start] == opener and text[end:end + len(closer)] == closer


def scrub_injection(text: str) -> tuple[str, list[str]]:
    """Neutralise prompt-injection shapes. Returns (clean_text, rule_names_that_fired).

    Strip-class rules remove their match outright and are harmless to re-run (they strip
    nothing the second time). Wrap-class rules flag their span in place; a match already
    inside a marker pair from an earlier boundary is left alone, but every distinct span
    in the text is wrapped — one known live span must not smuggle a second one through.
    """
    hits: list[str] = []
    for name, pattern in INJECTION_PATTERNS:
        if not pattern.search(text):
            continue
        hits.append(name)
        if name in INJECTION_STRIP_RULES:
            text = pattern.sub("", text)
        else:
            def repl(match: re.Match, _name=name, _text=text) -> str:
                if _is_self_wrapped(match, _text, _name):
                    return match.group(0)
                return f"{_INJECTION_OPEN}{_name}]" + match.group(0) + "[/INJECTION]"
            text = pattern.sub(repl, text)
    return text, hits
