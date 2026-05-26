#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MIN_EVENTS="4"

TARGET_DIR="$1"
PATTERN="${2:-*.ndjson}"

find "$TARGET_DIR" -type f -name "$PATTERN" \
  | sort \
  | grep -vE '/truthTraj_[^/]*\.ndjson$' \
  | while IFS= read -r in_file; do
      echo "$in_file"
      "$PYTHON_BIN" "$SCRIPT_DIR/saveSize.py" \
        -i "$in_file" \
        --min-events "$MIN_EVENTS" \
        --occ-min \
        --exclude-entering-tracks
    done
