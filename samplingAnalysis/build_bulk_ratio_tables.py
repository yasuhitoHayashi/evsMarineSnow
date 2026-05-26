#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build frame-to-EVS ratios for number and volume density."""

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd


FPS_VALUES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
PHASE_STEP_US = 1000
MIN_D_MM = 0.5
DURATION_S = 10.0
VGEOM_L = 0.253
CHUNKSIZE = 1_000_000
HIT_VOLUME_COL = "hit_volume_total_mm3"
VOLUME_DENSITY_REF_COL = "volume_density_ref"
REFERENCE_COLUMNS = {
    "d_bin_left_mm",
    "d_bin_right_mm",
    "d_mid_mm",
    "N_i",
    "PSD_ref",
    VOLUME_DENSITY_REF_COL,
}
OUTPUT_COLUMNS = [
    "label",
    "time",
    "fps",
    "phase_index",
    "metric",
    "min_count",
    "min_d_mm",
    "value_frame",
    "value_evs",
    "ratio_frame_to_evs",
]


def phase_values(frame_period_us: int):
    phase = 0
    while phase < frame_period_us:
        yield phase
        phase += PHASE_STEP_US


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


def camera_label(path: Path) -> str:
    for part in reversed(path.parts):
        if part in {"cam45M", "cam85M"}:
            return part
    return path.parent.name


def time_key_from_path(path: Path) -> str:
    return path.name.removesuffix("_frameSampling.csv")


def reference_path_for_frame_sampling(path: Path) -> Path:
    return path.with_name(f"{time_key_from_path(path)}_reference_psd.csv")


def load_reference(path: Path) -> pd.DataFrame:
    ref = pd.read_csv(path)
    missing = sorted(REFERENCE_COLUMNS - set(ref.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns: {', '.join(missing)}")
    for col in REFERENCE_COLUMNS:
        ref[col] = pd.to_numeric(ref[col], errors="coerce")
    ref = ref[
        np.isfinite(ref["d_bin_left_mm"])
        & np.isfinite(ref["d_bin_right_mm"])
        & np.isfinite(ref["d_mid_mm"])
        & np.isfinite(ref["N_i"])
        & (ref["d_mid_mm"] >= MIN_D_MM)
    ].copy()
    if ref.empty:
        raise RuntimeError(f"{path}: no valid reference bins")
    return ref.reset_index(drop=True)


def reference_value(ref: pd.DataFrame, metric: str, min_count: float) -> float:
    d = ref["d_mid_mm"].to_numpy(dtype=float)
    n_ref = ref["N_i"].to_numpy(dtype=float)
    width = (ref["d_bin_right_mm"] - ref["d_bin_left_mm"]).to_numpy(dtype=float)
    mask = (
        np.isfinite(d)
        & (d >= MIN_D_MM)
        & np.isfinite(n_ref)
        & (n_ref > float(min_count))
        & np.isfinite(width)
        & (width > 0)
    )
    if metric == "number_density":
        y = ref["PSD_ref"].to_numpy(dtype=float)
    elif metric == "volume":
        y = ref[VOLUME_DENSITY_REF_COL].to_numpy(dtype=float)
    else:
        raise RuntimeError(f"unsupported metric: {metric}")
    mask &= np.isfinite(y) & (y > 0)
    if not np.any(mask):
        return np.nan
    return float(np.sum(y[mask] * width[mask]))


def aggregate_hits(frame_path: Path, ref: pd.DataFrame) -> dict:
    edges = np.concatenate(
        [
            ref["d_bin_left_mm"].to_numpy(dtype=float),
            np.asarray([float(ref["d_bin_right_mm"].iloc[-1])], dtype=float),
        ]
    )
    n_bins = len(ref)
    sums = {}
    usecols = ["D_mm", "fps", "phase_index", "phase_us", "n_det"]
    for chunk in pd.read_csv(frame_path, usecols=usecols, chunksize=CHUNKSIZE):
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
        chunk[HIT_VOLUME_COL] = chunk["n_det"].to_numpy(dtype=float) * volume

        grouped = chunk.groupby(["fps", "phase_index", "bin_index"], sort=False).agg(
            n_hit_total=("n_det", "sum"),
            hit_volume_total_mm3=(HIT_VOLUME_COL, "sum"),
        )
        for key, row in grouped.iterrows():
            if key not in sums:
                sums[key] = [0.0, 0.0]
            sums[key][0] += float(row["n_hit_total"])
            sums[key][1] += float(row["hit_volume_total_mm3"])
    return sums


def frame_value(
    hit_sums: dict,
    ref: pd.DataFrame,
    fps: float,
    phase_index: int,
    metric: str,
    min_count: float,
) -> float:
    nframes = float(np.rint(DURATION_S * float(fps)))
    denom = nframes * VGEOM_L
    if (not np.isfinite(denom)) or denom <= 0:
        return np.nan

    ref_n = ref["N_i"].to_numpy(dtype=float)
    ref_d = ref["d_mid_mm"].to_numpy(dtype=float)
    total = 0.0
    any_valid = False
    for bin_index in range(len(ref)):
        if not (np.isfinite(ref_n[bin_index]) and ref_n[bin_index] > float(min_count)):
            continue
        if not (np.isfinite(ref_d[bin_index]) and ref_d[bin_index] >= MIN_D_MM):
            continue
        n_hit, hit_volume = hit_sums.get((float(fps), int(phase_index), int(bin_index)), [0.0, 0.0])
        if not (np.isfinite(n_hit) and n_hit > float(min_count)):
            continue
        any_valid = True
        if metric == "number_density":
            total += float(n_hit)
        elif metric == "volume":
            total += float(hit_volume)
        else:
            raise RuntimeError(f"unsupported metric: {metric}")
    if not any_valid:
        return np.nan
    return float(total / denom)


def build_rows_for_file(frame_path: Path) -> list[dict]:
    label = camera_label(frame_path)
    time_key = time_key_from_path(frame_path)
    ref = load_reference(reference_path_for_frame_sampling(frame_path))
    hit_sums = aggregate_hits(frame_path, ref)

    rows = []
    for metric in ("number_density", "volume"):
        for min_count, suffix in ((0.0, "all"), (9.0, "ge10")):
            evs_val = reference_value(ref, metric=metric, min_count=min_count)
            for fps in FPS_VALUES:
                frame_period_us = int(round(1_000_000.0 / float(fps)))
                for phase_index, _phase_us in enumerate(phase_values(frame_period_us)):
                    frame_val = frame_value(
                        hit_sums,
                        ref,
                        fps=fps,
                        phase_index=phase_index,
                        metric=metric,
                        min_count=min_count,
                    )
                    ratio = np.nan
                    if np.isfinite(frame_val) and np.isfinite(evs_val) and evs_val > 0:
                        ratio = float(frame_val / evs_val)
                    rows.append(
                        {
                            "label": label,
                            "time": time_key,
                            "fps": float(fps),
                            "phase_index": int(phase_index),
                            "metric": metric,
                            "min_count": float(min_count),
                            "min_d_mm": MIN_D_MM,
                            "value_frame": float(frame_val) if np.isfinite(frame_val) else np.nan,
                            "value_evs": float(evs_val) if np.isfinite(evs_val) else np.nan,
                            "ratio_frame_to_evs": ratio,
                            "_suffix": suffix,
                        }
                    )
    return rows


def write_metric_tables(rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    outputs = {
        ("number_density", "all"): "bulk_number_density_ratio_total_all.csv",
        ("number_density", "ge10"): "bulk_number_density_ratio_total_ge10.csv",
        ("volume", "all"): "bulk_volume_ratio_total_all.csv",
        ("volume", "ge10"): "bulk_volume_ratio_total_ge10.csv",
    }
    for (metric, suffix), filename in outputs.items():
        out = df[(df["metric"] == metric) & (df["_suffix"] == suffix)].copy()
        out = out.drop(columns=["_suffix"])
        out.to_csv(out_dir / filename, index=False, columns=OUTPUT_COLUMNS)
        print(f"[done] {out_dir / filename} rows={len(out)}", flush=True)


def parse_args():
    here = Path(__file__).resolve()
    default_input = here.parent.parent / "inSituData"
    default_out = here.parent / "data_output" / "bulk_ratio_alltimes"
    parser = argparse.ArgumentParser(
        description="Build density and volume ratio tables from *_frameSampling.csv files."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(default_input),
        help="*_frameSampling.csv, directory, or glob",
    )
    parser.add_argument("-o", "--out-dir", default=str(default_out))
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    recursive = True if input_path.is_dir() else bool(args.recursive)
    paths = collect_frame_sampling_paths(args.input, recursive=recursive)
    if not paths:
        raise RuntimeError("no *_frameSampling.csv files")
    rows = []
    for path in paths:
        print(f"[bulk] {path}", flush=True)
        rows.extend(build_rows_for_file(path))
    write_metric_tables(rows, args.out_dir)


if __name__ == "__main__":
    main()
