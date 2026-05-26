#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract track-level projected areas from tracking NDJSON files."""

import argparse
import csv
from dataclasses import dataclass
import os

import numpy as np

import saveSize_core as sc

WIN_US_FIXED = 1000
DEG_FIXED = 2
BIN_PX_FIXED = 1.0
EDGE_MARGIN_PX_FIXED = 0.0
IMG_WIDTH_PX_FIXED = 1280.0
IMG_HEIGHT_PX_FIXED = 720.0
SIZE_EVENT_SCOPE = "first_half"

OUTPUT_COLUMNS = [
    "pid",
    "particle_t_start_us",
    "particle_t_end_us",
    "particle_duration_us",
    "n_events_track",
    f"area_px2_win{WIN_US_FIXED}",
    f"n_events_window_win{WIN_US_FIXED}",
    f"rho_ev_per_px2_win{WIN_US_FIXED}",
    f"best_ts_us_win{WIN_US_FIXED}",
]


@dataclass
class ProcessingStats:
    tracks_read: int = 0
    rows_written: int = 0
    skipped_entering: int = 0


def parse_args():
    ap = argparse.ArgumentParser(description="Extract EVS projected particle area")
    ap.add_argument("-i", "--input", required=True, help="tracks ndjson")
    ap.add_argument("-o", "--output", default=None, help="output CSV path")

    ap.add_argument("--step-us", type=int, default=900)
    ap.add_argument("--min-events", type=int, default=4)
    ap.add_argument(
        "--occ-min",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use minimum occupancy across phase-shifted grids (default: enabled)",
    )
    ap.add_argument(
        "--exclude-entering-tracks",
        action="store_true",
        help="exclude tracks that touch the image boundary",
    )
    return ap.parse_args()


def _default_output_path(in_path: str) -> str:
    base = os.path.basename(in_path)
    if base.endswith(".ndjson"):
        base = base[:-len(".ndjson")]
    out_dir = os.path.dirname(os.path.abspath(in_path))
    return os.path.join(out_dir, base + "_size.csv")


def _first_half_by_time(ev_full: np.ndarray) -> np.ndarray:
    t = ev_full[:, 2]
    t_mid = float(t[0] + 0.5 * (t[-1] - t[0]))
    return ev_full[t <= t_mid]


def validate_args(args):
    if not str(args.input).endswith(".ndjson"):
        raise ValueError("input must be a .ndjson file")
    if args.step_us < 1:
        raise ValueError("step-us must be >=1")
    if args.min_events < 1:
        raise ValueError("min-events must be >=1")


def track_metadata_row(pid: int, ev_full: np.ndarray):
    t_all = ev_full[:, 2]
    pt0 = float(np.min(t_all))
    pt1 = float(np.max(t_all))
    return [
        pid,
        f"{pt0:.0f}",
        f"{pt1:.0f}",
        f"{float(pt1 - pt0):.0f}",
        int(t_all.size),
    ]


def should_skip_entering_track(ev_size: np.ndarray, exclude_entering_tracks: bool) -> bool:
    if not exclude_entering_tracks:
        return False
    return sc.is_track_touching_edge(
        ev_full=ev_size,
        img_width_px=IMG_WIDTH_PX_FIXED,
        img_height_px=IMG_HEIGHT_PX_FIXED,
        edge_margin_px=EDGE_MARGIN_PX_FIXED,
    )


def estimate_track_area(ev_size: np.ndarray, step_us: int, min_events: int, occ_min: bool):
    model = sc.fit_track_trajectory_poly(ev_size, deg=DEG_FIXED)
    if model is None:
        return None

    return sc.max_area_for_window(
        ev=ev_size,
        model=model,
        win_us=WIN_US_FIXED,
        step_us=int(step_us),
        bin_px=BIN_PX_FIXED,
        min_events=int(min_events),
        morph="fill",
        kernel_px=1,
        occ_min=bool(occ_min),
    )


def process_track(pid: int, ev: np.ndarray, args):
    ev_full = np.asarray(ev, dtype=np.float64)
    if ev_full.shape[0] < 2:
        return None, "too_short"

    ev_full = ev_full[np.argsort(ev_full[:, 2])]
    ev_size = _first_half_by_time(ev_full)
    if ev_size.shape[0] < 2:
        return None, "too_short_size_scope"

    if should_skip_entering_track(ev_size, args.exclude_entering_tracks):
        return None, "entering"

    row = track_metadata_row(pid, ev_full)
    res = estimate_track_area(
        ev_size,
        step_us=args.step_us,
        min_events=args.min_events,
        occ_min=args.occ_min,
    )
    if res is None:
        row += ["", "", "", ""]
    else:
        area, n_ev, rho, best_ts = res
        row += [f"{area:.6g}", int(n_ev), f"{rho:.6g}", int(best_ts)]
    return row, "ok"


def write_size_csv(input_path: str, output_path: str, args) -> ProcessingStats:
    stats = ProcessingStats()
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)

        for pid, ev in sc.iter_tracks_from_ndjson(input_path):
            stats.tracks_read += 1
            row, status = process_track(pid, ev, args)
            if status == "entering":
                stats.skipped_entering += 1
                continue
            if row is None:
                continue

            writer.writerow(row)
            stats.rows_written += 1

            if stats.tracks_read % 2000 == 0:
                print(
                    f"[progress] tracks read={stats.tracks_read}, "
                    f"ok={stats.rows_written}, skipped_entering={stats.skipped_entering}"
                )

    return stats


def print_summary(output_path: str, args, stats: ProcessingStats):
    print(f"[done] wrote: {output_path}")
    if args.exclude_entering_tracks:
        print(
            f"[done] entering-filter: enabled "
            f"(scope={SIZE_EVENT_SCOPE}, edge_margin_px={EDGE_MARGIN_PX_FIXED:.6g}, "
            f"img_width_px={IMG_WIDTH_PX_FIXED:.6g}, img_height_px={IMG_HEIGHT_PX_FIXED:.6g})"
        )
    print(
        f"[done] tracks read={stats.tracks_read}, ok={stats.rows_written}, "
        f"skipped_entering={stats.skipped_entering}"
    )


def main():
    args = parse_args()
    validate_args(args)

    out_csv = args.output if args.output else _default_output_path(args.input)
    stats = write_size_csv(args.input, out_csv, args)
    print_summary(out_csv, args, stats)


if __name__ == "__main__":
    main()
