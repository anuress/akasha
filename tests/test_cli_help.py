import argparse

import pytest

from akasha.cli import build_parser

KEPT_COMMANDS = {"init", "doctor", "serve", "index", "knowledge", "feature", "hook",
                 "config", "housekeeping"}
KEPT_KNOWLEDGE = {"search", "get", "write", "append", "update", "archive", "rm", "restore",
                  "purge", "stale", "stats", "fsck", "related", "timeline"}


def _subparser_actions(parser):
    return [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]


def _choices(parser):
    return {name: sub for action in _subparser_actions(parser)
            for name, sub in action.choices.items()}


def _walk(parser):
    yield parser
    for action in _subparser_actions(parser):
        for sub in action.choices.values():
            yield from _walk(sub)


def _helps(parser):
    return {c.dest: (c.help or "") for a in _subparser_actions(parser)
            for c in a._choices_actions}


def test_the_top_level_commands_are_exactly_the_kept_ones():
    assert set(_choices(build_parser())) == KEPT_COMMANDS


def test_help_lists_exactly_the_kept_commands(capsys):
    from akasha.cli import main

    with pytest.raises(SystemExit):
        main(["--help"])
    listed = capsys.readouterr().out
    section = listed.split("command", 1)[1].split("examples:")[0]
    names = {line.split()[0] for line in section.splitlines()
             if line.startswith("    ") and line.split()}
    assert names >= KEPT_COMMANDS


def test_the_knowledge_subcommands_are_exactly_the_kept_ones():
    assert set(_choices(_choices(build_parser())["knowledge"])) == KEPT_KNOWLEDGE


def test_every_command_and_subcommand_has_help_text():
    missing = []
    for parser in _walk(build_parser()):
        for action in _subparser_actions(parser):
            documented = _helps(parser)
            missing += [f"{parser.prog} {n}" for n in action.choices if not documented.get(n)]
    assert missing == []


def test_every_option_and_argument_has_help_text():
    missing = []
    for parser in _walk(build_parser()):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                continue
            if set(action.option_strings) & {"-h", "--help"}:
                continue
            if not (action.help or "").strip():
                missing.append(f"{parser.prog}: {action.option_strings or [action.dest]}")
    assert missing == []


def test_root_help_describes_the_tool_and_shows_generic_examples():
    text = build_parser().format_help()
    assert "knowledge base" in text.lower()
    assert "akasha knowledge search" in text


def test_destructive_commands_are_labelled_as_such():
    labels = _helps(_choices(build_parser())["knowledge"])
    assert "permanent" in labels["purge"].lower()
    assert "trash" in labels["rm"].lower()


def test_serve_is_marked_as_agent_facing():
    assert "mcp" in _helps(build_parser())["serve"].lower()


def test_help_renders_for_every_subparser():
    for parser in _walk(build_parser()):
        parser.format_help()


def test_subparser_actions_use_a_metavar_not_the_choice_list():
    for parser in _walk(build_parser()):
        for action in _subparser_actions(parser):
            assert action.metavar == "command", parser.prog
