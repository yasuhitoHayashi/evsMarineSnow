#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build reference PSD tables from EVS particle-size CSV files."""

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np

DIAMETER_COL = "D_mm"
OUT_SUFFIX = "_reference_psd.csv"
MIN_D_MM = 0.5
DMIN_MM = 0.5
DMAX_MM = 13.0
BIN_RATIO = 1.1447
ASSUME_FILE_SEC = 10.0
VGEOM_L = 0.253


def collect_size_csvs(path_text: str, recursive: bool) -> list[Path]:
    path = Path(path_text)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        pattern = "**/*_size.csv" if recursive else "*_size.csv"
        paths = sorted(path.glob(pattern))
    else:
        paths = [Path(p) for p in sorted(glob.glob(path_text))]

    return sorted({p for p in paths if p.is_file() and p.name.endswith("_size.csv")})


def to_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def make_bins_by_ratio(dmin_mm: float, dmax_mm: float, ratio: float):
    if dmin_mm <= 0.0 or dmax_mm <= dmin_mm:
        raise ValueError("invalid dmin/dmax")
    if ratio <= 1.0:
        raise ValueError("bin ratio must be > 1")
    n = int(math.ceil(math.log(dmax_mm / dmin_mm) / math.log(ratio)))
    edges = dmin_mm * np.power(ratio, np.arange(n + 1, dtype=float))
    mid = np.sqrt(edges[:-1] * edges[1:])
    width = edges[1:] - edges[:-1]
    return edges, mid, width


def output_path_for_size_csv(size_path: Path, out_suffix: str) -> Path:
    name = size_path.name
    if name.endswith("_size.csv"):
        name = name[: -len("_size.csv")]
    else:
        name = size_path.stem
    return size_path.with_name(f"{name}{out_suffix}")


def read_size_tracks(size_path: Path, min_d_mm: float) -> list[tuple[float, float]]:
    tracks = []
    with size_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"particle_t_start_us", "particle_t_end_us", DIAMETER_COL}
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"{size_path}: missing columns: {', '.join(missing)}")

        for row in reader:
            t0 = to_float(row.get("particle_t_start_us"))
            t1 = to_float(row.get("particle_t_end_us"))
            d_mm = to_float(row.get(DIAMETER_COL))
            if t0 is None or t1 is None or d_mm is None:
                continue
            if t1 < t0 or d_mm <= 0.0:
                continue
            if not math.isfinite(d_mm) or d_mm < min_d_mm:
                continue
            tau_s = (t1 - t0) * 1.0e-6
            if tau_s < 0.0:
                continue
            tracks.append((d_mm, tau_s))
    return tracks


def build_reference_rows(size_path: Path):
    edges, mid, width = make_bins_by_ratio(DMIN_MM, DMAX_MM, BIN_RATIO)
    tracks = read_size_tracks(size_path, min_d_mm=MIN_D_MM)

    n_bins = len(width)
    n_i = np.zeros(n_bins, dtype=float)
    sum_tau = np.zeros(n_bins, dtype=float)
    ref_volume = np.zeros(n_bins, dtype=float)
    sum_tau_volume = np.zeros(n_bins, dtype=float)

    if tracks:
        d = np.asarray([x[0] for x in tracks], dtype=float)
        tau = np.asarray([x[1] for x in tracks], dtype=float)
        volume = (math.pi / 6.0) * np.power(d, 3.0)
        bin_idx = np.digitize(d, edges) - 1
        keep = (bin_idx >= 0) & (bin_idx < n_bins)
        if np.any(keep):
            bin_idx = bin_idx[keep]
            tau = tau[keep]
            volume = volume[keep]
            n_i = np.bincount(bin_idx, minlength=n_bins).astype(float)
            sum_tau = np.bincount(bin_idx, weights=tau, minlength=n_bins).astype(float)
            ref_volume = np.bincount(bin_idx, weights=volume, minlength=n_bins).astype(float)
            sum_tau_volume = np.bincount(
                bin_idx,
                weights=tau * volume,
                minlength=n_bins,
            ).astype(float)

    denom = ASSUME_FILE_SEC * VGEOM_L
    psd_ref = sum_tau / ((denom + 1.0e-300) * (width + 1.0e-300))
    volume_density_ref = sum_tau_volume / ((denom + 1.0e-300) * (width + 1.0e-300))

    rows = []
    for i in range(n_bins):
        rows.append(
            {
                "d_bin_left_mm": float(edges[i]),
                "d_bin_right_mm": float(edges[i + 1]),
                "d_mid_mm": float(mid[i]),
                "N_i": int(n_i[i]),
                "sum_tau_i_s": float(sum_tau[i]),
                "PSD_ref": float(psd_ref[i]),
                "ref_volume_mm3": float(ref_volume[i]),
                "sum_tau_volume_mm3_s": float(sum_tau_volume[i]),
                "volume_density_ref": float(volume_density_ref[i]),
            }
        )
    return rows


def write_reference_csv(size_path: Path):
    out_path = output_path_for_size_csv(size_path, OUT_SUFFIX)
    rows = build_reference_rows(size_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {out_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build *_reference_psd.csv from *_size.csv files.")
    parser.add_argument("-i", "--input", required=True, help="*_size.csv, directory, or glob")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    size_paths = collect_size_csvs(args.input, args.recursive)
    if not size_paths:
        raise RuntimeError("no input files")

    for size_path in size_paths:
        write_reference_csv(size_path)


if __name__ == "__main__":
    main()
