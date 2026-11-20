#!/usr/bin/env bash
# Full ScienceWorld slice generation in the background (resumable: rerun the same command).
# Raw rows stream to data/scienceworld.jsonl and results/scienceworld/rows_raw.jsonl; at the end
# the generator marks minimal pairs, caps any class at 50% and rewrites data/scienceworld.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/scienceworld
LIMIT="${LIMIT:-1100}"            # raw labelled rows before the class cap
VARIATIONS="${VARIATIONS:-5}"
MAX_SECONDS="${MAX_SECONDS:-14400}"
nohup .venv/bin/python -m scienceworld_slice.build_slice \
  --limit "$LIMIT" --variations "$VARIATIONS" --workers 4 --max-seconds "$MAX_SECONDS" \
  --work-dir results/scienceworld --output data/scienceworld.jsonl \
  >> results/scienceworld/gen.log 2>&1 &
echo "started pid $! ; log: results/scienceworld/gen.log"
