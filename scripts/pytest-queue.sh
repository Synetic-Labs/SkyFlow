#!/usr/bin/env bash
# Run pytest behind a MACHINE-WIDE lock, so N concurrent runs QUEUE instead of collide.
# Useful when several checkouts or worktrees share one machine: full suites started in
# parallel contend for the same CPU and thermal budget, and every suite ends up slower
# than if they had run one at a time. Sequential is not a compromise here, it is strictly
# faster in wall clock AND it keeps the machine responsive.
#
#     scripts/pytest-queue.sh tests/                    # the full suite, queued
#     scripts/pytest-queue.sh tests/test_env_contract.py -q
#     PYTEST_QUEUE_TIMEOUT=60 scripts/pytest-queue.sh tests/   # give up waiting after 60 s
#
# The lock is held for the whole run and released on exit, crash or kill (flock is tied
# to the fd, so a killed run never leaves it stuck). The lock lives in /tmp, NOT in the
# repo: any checkout or worktree calling this script joins the same machine-wide queue.
set -euo pipefail

cd "$(dirname "$0")/.."

LOCK=/tmp/pytest-queue.lock
WAIT="${PYTEST_QUEUE_TIMEOUT:-}"        # empty = wait forever

# APPEND, not truncate. `exec 9>"$LOCK"` would wipe the holder's record at open — before this
# process has even tried the lock — so the "who has it?" message below could never say. flock
# is on the inode, so append vs truncate makes no difference to the locking itself.
exec 9>>"$LOCK"

if ! flock -n 9; then
  holder="$(cat "$LOCK" 2>/dev/null || true)"
  echo "pytest-queue: another suite is running${holder:+ ($holder)} — waiting for the lock." >&2
  echo "pytest-queue: ^C to give up, or PYTEST_QUEUE_TIMEOUT=<seconds> to bound the wait." >&2
  if [[ -n "$WAIT" ]]; then
    flock -w "$WAIT" 9 || { echo "pytest-queue: still locked after ${WAIT}s — not running." >&2; exit 75; }
  else
    flock 9
  fi
fi

# Record who holds it, so the next run's wait message is informative rather than mysterious.
# Truncate-and-write by PATH (safe: we hold the lock, and truncating the inode keeps it valid).
printf 'pid %s in %s since %s\n' "$$" "$PWD" "$(date +%H:%M:%S)" > "$LOCK"

# nice, so interactive work (or a training run) stays ahead of a background suite.
echo "pytest-queue: lock acquired — running in $PWD" >&2
nice -n 10 uv run python -m pytest "$@"
