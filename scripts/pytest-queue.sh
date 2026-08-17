#!/usr/bin/env bash
# Run pytest behind a MACHINE-WIDE lock, so N concurrent sessions QUEUE instead of collide.
# Mirror of nav-train's scripts/pytest-queue.sh with the SAME /tmp lock file, so SkyFlow
# and nav-train suites share one queue on this thermally-limited box.
#
#     scripts/pytest-queue.sh tests/                    # the full suite, queued
#     scripts/pytest-queue.sh tests/test_obs_contract.py -q
#     NAV_PYTEST_TIMEOUT=60 scripts/pytest-queue.sh tests/   # give up waiting after 60 s
#
# WHY: this box is TEMPERATURE-limited, not core-limited — 32 threads, but five sessions each
# starting a 1100-test suite drove load to 40 and every suite got slower than if they had run
# one at a time. Sequential is not a compromise here, it is strictly faster in wall clock AND
# it keeps the machine responsive. The lock is held for the whole run and released on exit,
# crash or kill (flock is tied to the fd, so a killed run never leaves it stuck).
#
# The lock lives in /tmp, NOT in the repo: nav-train and nav-train-state-est are separate
# checkouts of the same project competing for the same CPU and thermal budget, so they must
# share one queue. Any worktree calling this script joins the same line.
set -euo pipefail

cd "$(dirname "$0")/.."

LOCK=/tmp/nav-pytest.lock
WAIT="${NAV_PYTEST_TIMEOUT:-}"          # empty = wait forever

# APPEND, not truncate. `exec 9>"$LOCK"` would wipe the holder's record at open — before this
# process has even tried the lock — so the "who has it?" message below could never say. flock
# is on the inode, so append vs truncate makes no difference to the locking itself.
exec 9>>"$LOCK"

if ! flock -n 9; then
  holder="$(cat "$LOCK" 2>/dev/null || true)"
  echo "pytest-queue: another suite is running${holder:+ ($holder)} — waiting for the lock." >&2
  echo "pytest-queue: ^C to give up, or NAV_PYTEST_TIMEOUT=<seconds> to bound the wait." >&2
  if [[ -n "$WAIT" ]]; then
    flock -w "$WAIT" 9 || { echo "pytest-queue: still locked after ${WAIT}s — not running." >&2; exit 75; }
  else
    flock 9
  fi
fi

# Record who holds it, so the next session's wait message is informative rather than mysterious.
# Truncate-and-write by PATH (safe: we hold the lock, and truncating the inode keeps it valid).
printf 'pid %s in %s since %s\n' "$$" "$PWD" "$(date +%H:%M:%S)" > "$LOCK"

# nice, so an interactive session (or a training run) stays ahead of a background suite.
echo "pytest-queue: lock acquired — running in $PWD" >&2
nice -n 10 uv run python -m pytest "$@"
