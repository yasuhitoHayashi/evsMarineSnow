#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate virtual-frame detections by fps, phase, and size bin."""

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd


FPS_VALUES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
PHASE_STEP_US = 1000
MIN_D_MM = 0.5
CHUNKSIZE = 1_000_000
REFERENCE_NUMERIC_COLUMNS = (
    "d_bin_left_mm",
    "d_bin_right_mm",
    "d_mid_mm",
    "N_i",
    "ref_volume_mm3",
    "sum_tau_volume_mm3_s",
)


def collect_frame_sampling_paths(path_text: str, recursive: bool) -> list[Path]:
    path = Path(path_text)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        pattern = "**/*_frameSampling.csv" if recursive else "*_frameSampling.csv"
        paths = sorted(path.glob(pattern))
    else:
        paths = [Path(p) for p in sorted(glob.glob(path_text))]

    return sorted({p for p in paths if p.is_file() and p.name.endswith("_frameSampling.csv")})


def phase_values(frame_period_us: int):
    phase = 0
    while phase < frame_period_us:
        yield phase
        phase += PHASE_STEP_US


def reference_path_for_frame_sampling(path: Path) -> Path:
    name = path.name
    if name.endswith("_frameSampling.csv"):
        name = name[: -len("_frameSampling.csv")]
    else:
        name = path.stem
    return path.with_name(f"{name}_reference_psd.csv")


def load_reference(ref_path: Path) -> pd.DataFrame:
    ref_path = Path(ref_path)
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    ref = pd.read_csv(ref_path)
    required = {"d_bin_left_mm", "d_bin_right_mm", "d_mid_mm", "N_i"}
    missing = sorted(required - set(ref.columns))
    if missing:
        raise RuntimeError(f"{ref_path}: missing columns: {', '.join(missing)}")

    for col in REFERENCE_NUMERIC_COLUMNS:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")

    ref = ref[
        np.isfinite(ref["d_bin_left_mm"])
        & np.isfinite(ref["d_bin_right_mm"])
        & np.isfinite(ref["d_mid_mm"])
        & np.isfinite(ref["N_i"])
        & (ref["N_i"] > 0)
        & (ref["d_mid_mm"] >= MIN_D_MM)
    ].copy()
    if ref.empty:
        raise RuntimeError(f"{ref_path}: no valid reference bins")
    return ref.reset_index(drop=True)


def aggregate_frame_sampling(path: Path) -> list[dict]:
    time_key = path.name.removesuffix("_frameSampling.csv")
    ref = load_reference(reference_path_for_frame_sampling(path))

    edges = np.concatenate(
        [
            ref["d_bin_left_mm"].to_numpy(dtype=float),
            np.asarray([float(ref["d_bin_right_mm"].iloc[-1])], dtype=float),
        ]
    )
    n_bins = len(ref)

    sums = {}
    usecols = ["D_mm", "fps", "phase_index", "phase_us", "n_det"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE):
        for col in usecols:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk[
            np.isfinite(chunk["D_mm"])
            & np.isfinite(chunk["fps"])
            & np.isfinite(chunk["phase_index"])
            & np.isfinite(chunk["phase_us"])
            & np.isfinite(chunk["n_det"])
            & (chunk["D_mm"] >= MIN_D_MM)
            & (chunk["n_det"] > 0)
        ].copy()
        if chunk.empty:
            continue

        bin_idx = np.digitize(chunk["D_mm"].to_numpy(dtype=float), edges) - 1
        keep = (bin_idx >= 0) & (bin_idx < n_bins)
        if not np.any(keep):
            continue
        chunk = chunk.loc[keep].copy()
        chunk["bin_index"] = bin_idx[keep].astype(int)
        volume = (math.pi / 6.0) * np.power(chunk["D_mm"].to_numpy(dtype=float), 3.0)
        chunk["hit_volume_total_mm3"] = chunk["n_det"].to_numpy(dtype=float) * volume
        chunk["det_track_volume_mm3"] = volume
        chunk["n_det_track"] = 1

        grouped = chunk.groupby(["fps", "phase_index", "phase_us", "bin_index"], sort=False).agg(
            n_hit_total=("n_det", "sum"),
            n_det_track=("n_det_track", "sum"),
            hit_volume_total_mm3=("hit_volume_total_mm3", "sum"),
            det_track_volume_mm3=("det_track_volume_mm3", "sum"),
        )
        for key, row in grouped.iterrows():
            if key not in sums:
                sums[key] = [0.0, 0, 0.0, 0.0]
            sums[key][0] += float(row["n_hit_total"])
            sums[key][1] += int(row["n_det_track"])
            sums[key][2] += float(row["hit_volume_total_mm3"])
            sums[key][3] += float(row["det_track_volume_mm3"])

    rows = []
    for fps in FPS_VALUES:
        frame_period_us = int(round(1_000_000.0 / float(fps)))
        for phase_index, phase_us in enumerate(phase_values(frame_period_us)):
            for bin_index, ref_row in ref.iterrows():
                key = (float(fps), int(phase_index), float(phase_us), int(bin_index))
                n_hit, n_track, hit_volume, det_volume = sums.get(key, [0.0, 0, 0.0, 0.0])
                n_ref = int(round(float(ref_row["N_i"])))
                n_miss = max(0, n_ref - int(n_track))
                d_mid = float(ref_row["d_mid_mm"])
                rows.append(
                    {
                        "time": time_key,
                        "fps": float(fps),
                        "phase_index": int(phase_index),
                        "phase_us": float(phase_us),
                        "bin_index": int(bin_index),
                        "d_bin_left_mm": float(ref_row["d_bin_left_mm"]),
                        "d_bin_right_mm": float(ref_row["d_bin_right_mm"]),
                        "d_mid_mm": d_mid,
                        "n_ref": n_ref,
                        "n_det": float(n_hit),
                        "n_det_track": int(n_track),
                        "n_hit_total": float(n_hit),
                        "ref_volume_mm3": float(ref_row.get("ref_volume_mm3", np.nan)),
                        "det_track_volume_mm3": float(det_volume),
                        "hit_volume_total_mm3": float(hit_volume),
                        "sum_tau_volume_mm3_s": float(ref_row.get("sum_tau_volume_mm3_s", np.nan)),
                        "count_mode": "multicount",
                        "min_d_mm": MIN_D_MM,
                        "n_miss": int(n_miss),
                        "miss_rate": float(n_miss / n_ref) if n_ref > 0 else np.nan,
                        "obs_rate": float(n_hit / n_ref) if n_ref > 0 else np.nan,
                        "log_d": float(np.log(d_mid)),
                        "log_n_ref": float(np.log(n_ref)) if n_ref > 0 else np.nan,
                    }
                )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a phase/bin count table from *_frameSampling.csv files."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="*_frameSampling.csv, directory, or glob",
    )
    parser.add_argument("-o", "--out-csv", required=True)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = collect_frame_sampling_paths(args.input, args.recursive)
    if not paths:
        raise RuntimeError("no frameSampling CSV files")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for path in paths:
            print(f"[aggregate] {path}", flush=True)
            rows = aggregate_frame_sampling(path)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
            writer.writerows(rows)
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
