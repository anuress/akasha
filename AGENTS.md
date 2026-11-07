# Working on akasha

akasha is a local knowledge base for coding agents, served over MCP and a small CLI.
Its callers are agents, not people, so output size matters: context is the scarce
resource, and every capability should be judged from the caller's side.

## How work is done here

- Measure before building. If something is being added only because it seems generally
  useful, measure first.
- Write the test first and see it fail. A test written after the fix tends to assert what
  the code does rather than what it should do.
- When a change depends on how an external tool behaves, run the tool; a passing suite
  does not show that.
- Enforce invariants in code, not in prompts.
- When something is unavailable or refused, say so in the output and surface it in
  `doctor` instead of degrading quietly.

## Conventions

- `uv run pytest -q` runs the whole suite; pytest is in the dev dependency group, so a
  bare run works in a fresh checkout.
- `uv run --python 3.11 pytest -q` checks the minimum supported Python version.
- Tests carry the reason in the name and docstring.
- Comments explain why, not what, and carry no history.
- Version is `0.0.x`; bump the patch only.
- Keep commits passing the pre-commit hooks and CI.

## Adding a file format

A format is one function in `splitters.SPLITTERS`, keyed by the lowercased extension, that
returns a `SplitResult`: the `Unit(locator, key, label, fields)` rows, the headers, the
headers it had to rename, and any warnings. The indexer needs no change, because `feed`
splits, diffs and chunks whatever the table maps to. Add tests to `tests/test_splitters.py`
and use synthetic data only: the repository is public. A new suffix also goes into
`feed.ALLOWED_SUFFIXES`, which `feed add` checks before it splits anything, and into both help
strings that name the formats: the `feed` line of `akasha --help` and `feed add`'s. Add one feed-level test
in `tests/test_feed.py`: splitter tests never reach that check, so a suffix the splitter
knows and the command does not stays green there and fails in use.

A dependency a format needs is imported inside its splitter and declared as an extra, so the
package installs without it. When the import fails, the message names the extra, the way
`xlsx` asks for `uv tool install 'akasha-mcp[xlsx]'`.
