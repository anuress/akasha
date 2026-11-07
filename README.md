# akasha

A local knowledge base that gives coding agents a memory, served over MCP.

## The problem

Every agent session starts from zero. Yesterday an agent spent an hour working out why a
test fails one run in twenty: two tests share a fixture directory and the cleanup of one
races the setup of the other. It fixed the test and the session ended. The reasoning
lived in a context window, and the window is gone. Today another agent meets the same
flake, reads the same files and derives the same answer, spending the same time and
context to get there.

What you already know is not missing, it is scattered. Some of it sits in Claude Code
project memory, some in Serena memories, some in a notes folder, some in markdown files
inside repos. Each agent writes to its own corner, and none of them searches the others.
Pasting notes into prompts does not scale: context is the scarce resource, and every
pasted page crowds out the work. Conventions ("migrations are reviewed before merge",
"never mock the database in integration tests") get retyped at the start of session
after session.

## How akasha fixes it

akasha builds one local index over the markdown you already have. It is read-only by
default and copies nothing: your files stay the source of truth, and the SQLite index
derived from them can be deleted and rebuilt at any time.

Agents search it through MCP before they start investigating, and write what they find
back as ordinary markdown with `knowledge_write` and `knowledge_append`. The next
session starts where the last one ended. Search is keyword (BM25) and, if enabled, dense
retrieval, fused by reciprocal rank. Results are sized for a context window: a handful of
capped hits, with `knowledge_get` to page into a document only when one is worth reading.

Conventions are documents of `kind=convention`. They are delivered automatically at
session start and are read-only to agents; only the CLI can create or change one. Hooks
keep the index fresh without anyone thinking about it (in Claude Code and Gemini CLI): a
session-start hook refreshes it in the background, and a post-tool hook reindexes a note
the moment an agent edits it.

Safety is enforced where the data enters. At index time, secret-shaped strings are
redacted and known prompt-injection patterns are neutralised before any agent sees the
text. Both are pattern-based: they catch the common shapes, not every secret. Destructive operations are
CLI-only; an agent can archive a document, never delete one. There is no account and no
server, and nothing leaves your machine (the one exception is a small embedding model,
downloaded once if dense search is on).

A day with akasha: an agent running in `claude` records the flaky-test cause with
`knowledge_write`. The next morning an agent in `gemini` is asked about the same flake,
calls `knowledge_search` first, and finds the finding in one hit instead of an hour of
investigation.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A supported agent CLI: `claude`, `gemini` or `copilot`. `akasha init --all` registers
  with whichever are on your PATH. Any other MCP client can run `akasha serve` (stdio)
  by hand.

## Install

The PyPI package is `akasha-mcp`; the command it installs is `akasha`.

```sh
uv tool install --python 3.12 'akasha-mcp[vectors]'
```

`--python` matters when your default Python is older than 3.11 (macOS ships 3.9): uv does
not pick an interpreter from the package's requirement on its own, and downloads 3.12 if
you don't have it. Without dense search: `uv tool install --python 3.12 akasha-mcp`. From git:

```sh
uv tool install --python 3.12 'akasha-mcp[vectors] @ git+https://github.com/anuress/akasha'
```

With `[vectors]` installed, `akasha init` writes `provider = "model2vec"` and search is
dense plus keyword. Without it, search is keyword-only: `init` says so and `akasha doctor`
keeps saying so. `akasha config set embeddings.provider none` turns dense search off. An
existing config is never rewritten; to enable it there, set `embeddings.provider model2vec`
and run `akasha index`.

### Optional: pi

pi users can add the [pi-akasha](https://github.com/anuress/pi-akasha) extension
(`pi install git:github.com/anuress/pi-akasha`). It registers the MCP server and runs
both hooks, including after compaction. `akasha init --all` does not configure pi.

## Quickstart

```sh
akasha init
```

Creates `~/.akasha/` (mode 0700) with `config.toml`, the SQLite index and a `knowledge/`
directory for documents agents write. On first run it looks for existing notes to index:
Claude Code project memories (`~/.claude/projects/*/memory`), and `.serena/memories` and
`graphify-out` directories in the immediate subdirectories of `~` (change with
`--code-root DIR`, repeatable).

```sh
akasha init --all
```

Registers the MCP server with every detected agent CLI through that CLI's own `mcp add`
command (copilot: its `mcp-config.json`), and installs the two hooks for claude and
gemini. For claude it also allows the `mcp__akasha` tools in `settings.json` so they do
not prompt; skip that with `--no-allow`. `--dry-run` prints what would be written and
changes nothing.

```sh
akasha index                         # build or refresh the index
akasha knowledge search "cache expiry" --limit 3
akasha doctor                        # what is degraded; exit 0 means healthy
```

Search defaults to the repo of the current checkout; `--all` searches every repo.

## Tools

Fourteen MCP tools:

| Tool | What it does |
|---|---|
| `knowledge_search` | Ranked chunks; current repo first, OR retry when strict finds nothing, `as_of` date |
| `knowledge_get` | One document by id, paged |
| `knowledge_write` | New document; lists fold candidates and can supersede older ones |
| `knowledge_update` | Correct a document in place, whole body or one exact span |
| `knowledge_append` | Append a dated section to a document |
| `knowledge_archive` | Hide from default search; reversible |
| `knowledge_fsck` | Integrity check, read-only |
| `knowledge_timeline` | Documents in the order written |
| `knowledge_related` | Linked documents: citations and backlinks |
| `feature_show` | How many documents carry a feature tag |
| `doctor` | What is degraded about the installation |
| `knowledge_feed` | Feed a .csv/.xlsx export as a searchable dataset; reports what changed |
| `knowledge_feed_get` | One dataset row by `row:N` or `key:V` |
| `knowledge_feed_query` | Filter rows: `F=v` exact, `F~v` contains |

## Feeding files

An exported sheet becomes a dataset: one document plus the stored copy its cells were split
from. Every non-empty cell of a kept column is its own chunk, so search finds the cell, and
the hit carries `unit` (dataset, locator, key, field) with the exact `next` call that reads
the whole row.

```sh
akasha feed add fruits.csv --as fruits --key id
```

The first feed prints `first feed: N units`. Feeding the same name again replaces its rows
and reports what changed: `+A -R ~C`, then one line per row, with a changed cell shown as
`color: -red` and `color: +green`. An eight-data-row `fruits.csv` prints `first feed: 8 units`
the first time and, once one row was added and two rows changed, `+1 -0 ~2` the next: the
line counts rows added, removed and changed, not cells. The listing stops at 50 rows with
`… N more (use --full)`; `--full` prints every row and every change line of a cell. A feed
that changes nothing prints `unchanged`.

A row is addressed two ways: `row:N` is its place in the export (`row:2` is the first data
row under a header on row 1), and `key:V` is the value of that row's `--key`. A key value
is usable as a locator only when it is non-empty and unique among the rows the feed kept, and
`--require` decides which those are: a blank or repeated key leaves the row addressable as
`row:N` alone, and the feed warns once, naming the column.

```sh
akasha feed list
akasha feed get fruits key:3
akasha feed query fruits --where season=summer --fields name,color
```

- `feed list`: name, repo, units, feed date and copy path of every dataset.
- `feed get`: one row; `--json` prints it as an object.
- `feed query`: `F=v` matches a whole cell, `F~v` a cell containing the value, both
  case-insensitive; every `--where` must match. `--limit` defaults to 20 (max 100).

Settings live on `feed add`:

| Flag | Effect |
|---|---|
| `--key COL` | Column whose value addresses a row across feeds; a first feed needs one |
| `--sheet NAME` | xlsx sheet; the first sheet unless one is named |
| `--header-row N` | Row holding the column headers (default 1) |
| `--require COL` | Keep only rows with a value in this column (repeatable) |
| `--exclude COL` | Drop this column's cells (repeatable); `--key` and `--label` still show them |
| `--label TEMPLATE` | Row label with `{column}` placeholders, e.g. `'{id} {name}'` |

Any setting flag replaces the whole stored settings object, so a re-feed that leaves `--key`
out is a feed whose rows are addressed by row number again; `feed add` prints
`settings replaced: ...` when it happens. With no setting flag at all, the last feed's
settings are reused.

`--repo NAME` feeds under a repo other than the one the current checkout resolves to;
`--repo '*'` is refused, because a dataset belongs to one repo. The name is global to the
installation, so feeding one name under a second repo is refused rather than quietly
divergent: use another name, or `feed rm` the first one.

The copy is `~/.akasha/files/<repo>/<name>.<ext>`, the document is
`~/.akasha/knowledge/<repo>/datasets/<name>.md`. The cells are split from that copy, so
re-feeding replaces it.

`xlsx` needs an extra:

```sh
uv tool install --python 3.12 'akasha-mcp[xlsx]'
```

Without it the splitter names the extra, and `doctor` reports it for as long as an xlsx
dataset exists.

The document is generated: `knowledge update`, `knowledge append`, `knowledge archive`
and `knowledge rm` refuse it and point at `feed add`; `akasha feed rm fruits` moves
document and copy to the trash and prints the id `knowledge restore` takes. A restore is
refused while a dataset of that name is live, and `knowledge purge` deletes trashed copies
too.

In detail: `fsck` reports a copy that is missing or edited (error), units that no longer
match the copy (warning; `akasha index --force` repairs them), damaged metadata, a copy no
readable document names, and a staging or backup file an interrupted feed left behind.
`doctor` reports the last failed attach of a dataset whose units are still not its own, and
the `xlsx` extra while an xlsx dataset exists. Search keeps answering from the units already
stored while either is unresolved, which is why the finding names the re-feed that fixes it.

## Hooks

Two hooks, both fail-open (they never block a session or a tool):

- `akasha hook session-start` prints the repo's conventions, a short brief and an
  integrity note into the new session. The installed command passes `--defer-refresh`
  so it does not wait for an index walk.
- `akasha hook post-tool` reads the tool payload on stdin and reindexes the one file an
  agent just wrote, if it is inside an index root.
  After 15 edits in a session with nothing recorded through `knowledge_write`,
  `knowledge_append` or `knowledge_update`, it prints a one-time reminder to record what
  was learned; the count restarts at the next write.
  It also prints the `kind=gotcha` documents scoped to the file, as context after the
  edit (advisory; never blocks). A gotcha opts in with frontmatter `paths: [globs]`,
  matched against the path relative to the file's checkout (`*` stays in one directory,
  `**` crosses them; braces are not supported), and may add `pattern: <regex>`, which
  must match the new text (keep it simple; only the first 200,000 characters are scanned).
  Repo-scoped and `default` gotchas apply; each is shown once per session per file.

Documents of `kind=convention` are standing rules: they are injected at session start and
are read-only over MCP. Only the CLI can create or change one.

## Sources and config

The native knowledge directory (`~/.akasha/knowledge`) holds documents written through
akasha. Everything else is an indexed root, one `[[index]]` block in `config.toml`:

```toml
[[index]]
path     = "~/notes"
source   = "notes"
include  = ["**/*.md"]      # optional globs
exclude  = ["**/cache/**"]
repo     = "my-repo"        # repo the whole root belongs to
writable = false            # default; true lets knowledge_append write into it
```

Roots are read-only by default. Add one without editing the file:

```sh
akasha index add ~/notes --source notes --repo my-repo
akasha index
```

## Deleting things

Agents can only archive. Everything else is CLI-only.

| Command | Effect |
|---|---|
| `akasha knowledge archive ID` | Hidden from default search; reversible with `--include-archived` |
| `akasha knowledge rm ID` | Moved to trash; recoverable |
| `akasha knowledge restore ID` | Brings a trashed document back |
| `akasha knowledge purge --yes` | Permanently deletes trashed documents (`--older-than DAYS` to limit) |

## Rebuilding and ids

The database is derived from the files:

```sh
rm ~/.akasha/akasha.db && akasha index
```

Ids of documents in indexed roots are derived from root and relative path, so links
between documents survive a rebuild. Moving or renaming a file changes its id.

## Privacy

At index time, secrets are redacted (`scan_secrets`, on by default) and prompt-injection
shapes are neutralised before text reaches an agent; common secret files (`.env*`, `*.pem`,
`*.key`, `*.pfx`, `*.kdbx`, SSH keys, `.npmrc`, `.pypirc`, `credentials.json` and similar)
are denied outright; `deny_files` and `deny_extensions` in `[security]` replace the lists.
A local audit log of events, including redacted query heads, is kept for 90 days (`akasha housekeeping` prunes it).

## Development

```sh
uv run pytest -q
uv run --python 3.11 pytest -q
```

## License

MIT
