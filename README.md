# akasha

A local knowledge base for coding agents, served over MCP.

Agents search it before investigating and write findings back, so the next session
(or the next agent) does not start from zero. Markdown files on disk are the truth; a
SQLite index derived from them is disposable. Search is keyword (BM25), plus optional
dense retrieval.

It has no account and no server: nothing leaves your machine. If you enable vector
search, a small embedding model is downloaded once on first use.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- A supported agent CLI: `claude`, `gemini` or `copilot`. `akasha init --all` registers
  with whichever are on your PATH. Any other MCP client can run `akasha serve` (stdio)
  by hand.

## Install

The PyPI package is `akasha-mcp`; the command it installs is `akasha`.

```sh
uv tool install 'akasha-mcp[vectors]'
```

Without dense search: `uv tool install akasha-mcp`. From git:

```sh
uv tool install 'akasha-mcp[vectors] @ git+https://github.com/anuress/akasha'
```

With `[vectors]` installed, `akasha init` writes `provider = "model2vec"` and search is
dense plus keyword. Without it, search is keyword-only: `init` says so and `akasha doctor`
keeps saying so. `akasha config set embeddings.provider none` turns dense search off. An
existing config is never rewritten; to enable it there, set `embeddings.provider model2vec`
and run `akasha index`.

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

## What agents get

Eleven MCP tools:

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

Two hooks, both fail-open (they never block a session or a tool):

- `akasha hook session-start` prints the repo's conventions, a short brief and an
  integrity note into the new session. The installed command passes `--defer-refresh`
  so it does not wait for an index walk.
- `akasha hook post-tool` reads the tool payload on stdin and reindexes the one file an
  agent just wrote, if it is inside an index root.

Documents of `kind=convention` are standing rules: they are injected at session start and
are read-only over MCP. Only the CLI can create or change one.

## Sources

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

## Rebuilding

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
