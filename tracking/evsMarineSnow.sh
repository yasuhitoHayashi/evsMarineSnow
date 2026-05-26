#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACKER="${SCRIPT_DIR}/cpp/build/tracker_cli"
TARGET_DIR="$1"

for csv in "$TARGET_DIR"/*.csv; do
  case "$(basename "$csv")" in
    truth_*|summary_*|*_size.csv) continue ;;
  esac

  base="${csv%.csv}"
  "$TRACKER" "$csv" "$base.ndjson" \
    --sigma_x=2.0 \
    --m_threshold=8 \
    --recent_us=1000 \
    --iso_enable=1 \
    --iso_k=1 \
    --iso_r_px=2 \
    --iso_w_us=1000 \
    --seconds=10
done
