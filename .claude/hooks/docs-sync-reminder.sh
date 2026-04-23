#!/usr/bin/env bash
# Stop hook: if any tracked/untracked file other than CLAUDE.md and README.md
# has changed this session, block once with a reminder to update both docs.
# Sentinel file keyed by session_id prevents infinite block loops.

set -u

input=$(cat)
session_id=$(printf '%s' "$input" | jq -r '.session_id // "unknown"')
sentinel="/tmp/claude-docs-reminder-${session_id}"

[ -f "$sentinel" ] && exit 0

cd "$(git rev-parse --show-toplevel 2>/dev/null)" 2>/dev/null || exit 0

changes=$(git status --porcelain 2>/dev/null | grep -v -E '^.. (CLAUDE\.md|README\.md)$' || true)

[ -z "$changes" ] && exit 0

touch "$sentinel"

jq -n --arg c "$changes" '{
  decision: "block",
  reason: ("Before stopping: verify CLAUDE.md and README.md reflect the following repo changes — update both files now if any file was added, moved, renamed, or deleted:\n\n" + $c + "\n\n(This reminder fires once per session. Re-read both docs, apply any needed edits, then stop.)")
}'
