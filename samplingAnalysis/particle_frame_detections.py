#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build virtual-frame detections from EVS particle-size tables."""

import argparse
import csv
from dataclasses import dataclass
import glob
import math
from pathlib import Path


FPS_VALUES = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
PHASE_STEP_US = 1000
EXPOSURE_US = 100
AREA_COL = "area_px2_win1000"
DIAMETER_COL = "D_mm"
OUT_SUFFIX = "_frameSampling.csv"

OUTPUT_COLUMNS = [
    "source_file",
    "time",
    "pid",
    "particle_t_start_us",
    "particle_t_end_us",
    "particle_duration_us",
    "area_px2",
    "D_mm",
    "fps",
    "frame_period_us",
    "phase_index",
    "phase_us",
    "exposure_us",
    "n_det",
]


@dataclass(frozen=True)
class ParticleRecord:
    pid: str
    t0_us: float
    t1_us: float
    duration_us: float
    area_px2: float | None
    diameter_mm: float | None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build *_frameSampling.csv files from *_size.csv files."
    )
    parser.add_argument("-i", "--input", required=True, help="*_size.csv, directory, or glob")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


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


def read_particles(path: Path) -> list[ParticleRecord]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"pid", "particle_t_start_us", "particle_t_end_us", AREA_COL, DIAMETER_COL}
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"{path}: missing columns: {', '.join(missing)}")

        particles = []
        for line_no, row in enumerate(reader, start=2):
            pid = row.get("pid", "")
            t0 = to_float(row.get("particle_t_start_us"))
            t1 = to_float(row.get("particle_t_end_us"))
            if t0 is None or t1 is None or t1 < t0:
                raise ValueError(f"{path}:{line_no}: invalid particle time range")

            area = to_float(row.get(AREA_COL))
            diameter = to_float(row.get(DIAMETER_COL))

            particles.append(
                ParticleRecord(
                    pid=pid,
                    t0_us=t0,
                    t1_us=t1,
                    duration_us=t1 - t0,
                    area_px2=area,
                    diameter_mm=diameter,
                )
            )

    return particles


def detected_phase_counts(
    t0_us: float,
    t1_us: float,
    frame_period_us: int,
    phase_step_us: int,
    exposure_us: int,
) -> dict[int, int]:
    half = 0.5 * float(exposure_us)
    start = t0_us - half
    end = t1_us + half
    max_phase = ((int(frame_period_us) - 1) // int(phase_step_us)) * int(phase_step_us)
    k_min = math.floor((start - max_phase) / frame_period_us)
    k_max = math.floor(end / frame_period_us)

    counts = {}
    for k in range(k_min, k_max + 1):
        phase_min = max(0.0, start - k * frame_period_us)
        phase_max = min(float(max_phase), end - k * frame_period_us)
        if phase_max < phase_min:
            continue

        first_phase = int(math.ceil(phase_min / phase_step_us) * phase_step_us)
        last_phase = int(math.floor(phase_max / phase_step_us) * phase_step_us)
        for phase_us in range(first_phase, last_phase + 1, int(phase_step_us)):
            counts[phase_us] = counts.get(phase_us, 0) + 1

    return counts


def output_path_for_size_csv(size_path: Path, out_suffix: str) -> Path:
    name = size_path.name
    if name.endswith("_size.csv"):
        name = name[: -len("_size.csv")]
    else:
        name = size_path.stem
    return size_path.with_name(f"{name}{out_suffix}")


def frame_sampling_rows(size_path: Path):
    particles = read_particles(size_path)
    time_label = size_path.name.removesuffix("_size.csv")

    for fps in FPS_VALUES:
        frame_period_us = int(round(1_000_000.0 / float(fps)))
        for particle in particles:
            phase_counts = detected_phase_counts(
                t0_us=particle.t0_us,
                t1_us=particle.t1_us,
                frame_period_us=frame_period_us,
                phase_step_us=PHASE_STEP_US,
                exposure_us=EXPOSURE_US,
            )
            for phase_us, n_det in sorted(phase_counts.items()):
                yield {
                    "source_file": str(size_path),
                    "time": time_label,
                    "pid": particle.pid,
                    "particle_t_start_us": particle.t0_us,
                    "particle_t_end_us": particle.t1_us,
                    "particle_duration_us": particle.duration_us,
                    "area_px2": particle.area_px2,
                    "D_mm": particle.diameter_mm,
                    "fps": float(fps),
                    "frame_period_us": frame_period_us,
                    "phase_index": int(phase_us // PHASE_STEP_US),
                    "phase_us": phase_us,
                    "exposure_us": int(EXPOSURE_US),
                    "n_det": int(n_det),
                }


def write_rows_for_size_csv(writer: csv.DictWriter, size_path: Path):
    for row in frame_sampling_rows(size_path):
        writer.writerow(row)


def write_one_output(output_path: Path, size_paths: list[Path]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {output_path}", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for size_path in size_paths:
            write_rows_for_size_csv(writer, size_path)


def write_detection_table(size_paths: list[Path]):
    for size_path in size_paths:
        write_one_output(output_path_for_size_csv(size_path, OUT_SUFFIX), [size_path])


def main():
    args = parse_args()
    size_paths = collect_size_csvs(args.input, args.recursive)
    if not size_paths:
        raise RuntimeError("no input files")

    write_detection_table(size_paths)


if __name__ == "__main__":
    main()
