#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from dataclasses import dataclass
import json
import glob
import os

import numpy as np

from trajectory_model import fit_track_trajectory_poly, predict_xy

NDJSON_PATTERN = "events_*.ndjson"
STREAK_BIN_MS = 10.0
CENTROID_BIN_MS = STREAK_BIN_MS
CENTROID_MIN_EVENTS = 5
MIN_EVENTS_PER_BIN = 10
WIDTH_PERCENTILES = (5.0, 95.0)
MIN_STREAK_WIDTH_PX = 0.0

OUTPUT_COLUMNS = [
    "pid",
    "particle_t_start_us",
    "particle_t_end_us",
    "particle_duration_us",
    "n_events_track",
    "len_px_median",
]


@dataclass
class ProcessingStats:
    files: int = 0
    tracks: int = 0
    rows_written: int = 0
    skipped_unsorted: int = 0
    skipped_no_centroid: int = 0


def parse_args():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--input", help="single tracks ndjson")
    src.add_argument("-d", "--dir", help="root dir to search NDJSON recursively")
    return ap.parse_args()


def iter_tracks_from_ndjson(path: str):
    with open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                raise ValueError(f"{path}:{line_no}: empty NDJSON line")
            obj = json.loads(s)

            pid = obj.get("particle_id", None)
            if pid is None:
                raise ValueError(f"{path}:{line_no}: missing particle_id")
            pid = int(pid)

            events = np.asarray(obj.get("events", []), dtype=np.float32)
            if events.ndim != 2 or events.shape[1] < 3 or events.shape[0] == 0:
                raise ValueError(f"{path}:{line_no}: events must be a non-empty Nx3+ array")
            events = events[:, :3]

            ch = obj.get("centroid_history", None)
            cents = None if ch is None else np.asarray(ch, dtype=np.float32)
            if cents is not None and (cents.ndim != 2 or cents.shape[1] < 3):
                raise ValueError(f"{path}:{line_no}: centroid_history must be an Nx3+ array")

            yield pid, events, cents


def compute_centroid_history_from_events(events, bin_ms=2.0, min_events=5):
    """Reconstruct centroid history from events."""
    e = np.asarray(events, dtype=np.float32)
    if e.ndim != 2 or e.shape[1] < 3:
        return None

    # keep finite rows
    m = np.isfinite(e[:, 0]) & np.isfinite(e[:, 1]) & np.isfinite(e[:, 2])
    e = e[m]
    if e.shape[0] < min_events:
        return None

    t = e[:, 2].astype(np.float64)
    if t.size < min_events:
        return None

    # IMPORTANT: no sort; enforce monotonic time
    if np.any(np.diff(t) < 0):
        return None

    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0:
        return None

    bin_us = float(bin_ms) * 1000.0
    n_bins = int((t1 - t0) // bin_us) + 1

    model = fit_track_trajectory_poly(e, deg=2)
    if model is None:
        return None

    cents = []
    for b in range(n_bins):
        tb0 = t0 + b * bin_us
        tb1 = tb0 + bin_us
        mb = (t >= tb0) & (t < tb1)
        if int(np.sum(mb)) < int(min_events):
            continue

        tc = float(np.median(t[mb]))
        x_pred, y_pred = predict_xy(model, np.asarray([tc], dtype=np.float64))
        cx = float(x_pred[0])
        cy = float(y_pred[0])
        cents.append((tc, cx, cy))

    if len(cents) < 2:
        return None

    return np.asarray(cents, dtype=np.float32)


def estimate_len_px_median(events, cents, bin_ms=10.0, min_events_per_bin=10, qlo=5, qhi=95):
    """Return time-median streak width [px]."""
    e = np.asarray(events, dtype=np.float32)
    c = np.asarray(cents, dtype=np.float32)
    if e.ndim != 2 or e.shape[1] < 3:
        return None
    if c.ndim != 2 or c.shape[0] < 2 or c.shape[1] < 3:
        return None

    te = e[:, 2] * 1e-3
    tc = c[:, 0] * 1e-3

    me = np.isfinite(te) & np.isfinite(e[:, 0]) & np.isfinite(e[:, 1])
    e = e[me]
    te = te[me]
    if e.shape[0] < min_events_per_bin:
        return None

    order = np.argsort(te)
    e = e[order]
    te = te[order]

    dc = np.diff(c[:, 1:3], axis=0)
    dt = np.diff(tc)

    ok_dt = dt > 1e-9
    if not np.any(ok_dt):
        return None

    v = np.zeros((c.shape[0], 2), dtype=np.float32)
    v[1:][ok_dt] = (dc[ok_dt] / dt[ok_dt, None]).astype(np.float32)
    v[0] = v[1]  # pad

    vnorm = np.hypot(v[:, 0], v[:, 1])
    ok = vnorm > 1e-6
    t_hat = np.zeros_like(v)
    t_hat[ok] = v[ok] / vnorm[ok, None]

    n_hat = np.column_stack([-t_hat[:, 1], t_hat[:, 0]]).astype(np.float32)

    tmin, tmax = float(te[0]), float(te[-1])
    if tmax <= tmin:
        return None

    n_bins = int(np.ceil((tmax - tmin) / float(bin_ms)))
    if n_bins < 1:
        return None

    L = []
    for b in range(n_bins):
        tb0 = tmin + b * float(bin_ms)
        tb1 = tb0 + float(bin_ms)
        m = (te >= tb0) & (te < tb1)
        if int(np.sum(m)) < int(min_events_per_bin):
            continue

        tmid = 0.5 * (tb0 + tb1)
        i = int(np.argmin(np.abs(tc - tmid)))

        d = np.sum((e[m, :2] - c[i, 1:3]) * n_hat[i], axis=1)
        if d.size < min_events_per_bin:
            continue

        q05, q95 = np.percentile(d, [qlo, qhi])
        if np.isfinite(q05) and np.isfinite(q95) and (q95 > q05):
            L.append(float(q95 - q05))

    if len(L) == 0:
        return None
    return float(np.median(L))


def default_output_path(ndjson_path: str):
    base = os.path.basename(ndjson_path)
    if base.endswith(".ndjson"):
        stem = base[:-7]
    else:
        raise ValueError("input must be a .ndjson file")
    return os.path.join(os.path.dirname(os.path.abspath(ndjson_path)), stem + "_sizestreak.csv")


def collect_input_files(input_path: str | None, root_dir: str | None):
    if input_path:
        files = [os.path.abspath(input_path)]
    else:
        patt = os.path.join(os.path.abspath(root_dir), "**", NDJSON_PATTERN)
        files = sorted(glob.glob(patt, recursive=True))

    files = [fp for fp in files if fp.endswith(".ndjson")]
    files = [fp for fp in files if "_sensitivity.ndjson" not in os.path.basename(fp)]
    return sorted(set(files))


def get_centroids_for_track(events: np.ndarray, cents: np.ndarray | None):
    if cents is not None and cents.ndim == 2 and cents.shape[0] >= 2:
        return cents
    return compute_centroid_history_from_events(
        events,
        bin_ms=CENTROID_BIN_MS,
        min_events=CENTROID_MIN_EVENTS,
    )


def make_output_row(
    pid: int,
    t_us: np.ndarray,
    len_px_median: float,
):
    pt0 = int(float(t_us[0]))
    pt1 = int(float(t_us[-1]))

    return [
        int(pid),
        pt0,
        pt1,
        int(max(0, pt1 - pt0)),
        int(t_us.size),
        f"{float(len_px_median):.6g}",
    ]


def process_track(pid: int, events: np.ndarray, cents: np.ndarray | None):
    t_us = events[:, 2]
    if t_us.size < 2:
        return None, "too_short"
    if np.any(np.diff(t_us) < 0):
        return None, "unsorted"

    cents = get_centroids_for_track(events, cents)
    if cents is None:
        return None, "no_centroid"

    qlo, qhi = WIDTH_PERCENTILES
    len_px_median = estimate_len_px_median(
        events,
        cents,
        bin_ms=STREAK_BIN_MS,
        min_events_per_bin=MIN_EVENTS_PER_BIN,
        qlo=qlo,
        qhi=qhi,
    )
    if (
        len_px_median is None
        or (not np.isfinite(len_px_median))
        or (len_px_median <= MIN_STREAK_WIDTH_PX)
    ):
        return None, "no_width"

    return make_output_row(pid, t_us, len_px_median), "ok"


def process_file(ndjson_path: str, stats: ProcessingStats):
    out_csv = default_output_path(ndjson_path)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    stats.files += 1
    with open(out_csv, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow(OUTPUT_COLUMNS)
        for pid, events, cents in iter_tracks_from_ndjson(ndjson_path):
            stats.tracks += 1
            row, status = process_track(pid, events, cents)
            if status == "unsorted":
                stats.skipped_unsorted += 1
            elif status == "no_centroid":
                stats.skipped_no_centroid += 1

            if row is None:
                continue
            writer.writerow(row)
            stats.rows_written += 1

    print(f"[file] wrote: {out_csv}")


def main():
    args = parse_args()
    files = collect_input_files(args.input, args.dir)
    if not files:
        raise RuntimeError(f"No files matched in dir={args.dir} with pattern={NDJSON_PATTERN}")

    stats = ProcessingStats()
    for fp in files:
        process_file(fp, stats)

    print(
        f"[done] files={stats.files} tracks_seen={stats.tracks} "
        f"rows_written={stats.rows_written} skipped_unsorted={stats.skipped_unsorted} "
        f"skipped_no_cent={stats.skipped_no_centroid}"
    )


if __name__ == "__main__":
    main()
