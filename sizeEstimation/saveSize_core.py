#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Core routines for EVS particle projected-area extraction.

The public workflow is `saveSize.py`.  This module intentionally contains only
the reusable mechanics: streaming tracks from NDJSON, fitting/warping particle
motion, and measuring the largest filled occupancy area in 1 ms windows.
"""

from __future__ import annotations

import json
from typing import Iterator, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi

from trajectory_model import TrajectoryModel, fit_track_trajectory_poly, warp_events_with_model


def iter_tracks_from_ndjson(path: str) -> Iterator[Tuple[int, np.ndarray]]:
    with open(path, "rt") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                raise ValueError(f"{path}:{line_no}: empty NDJSON line")
            obj = json.loads(s)

            pid = obj.get("particle_id")
            if pid is None:
                raise ValueError(f"{path}:{line_no}: missing particle_id")
            pid_i = int(pid)

            events = np.asarray(obj.get("events", []), dtype=np.float32)
            if events.ndim != 2 or events.shape[1] < 3 or events.shape[0] == 0:
                raise ValueError(f"{path}:{line_no}: events must be a non-empty Nx3+ array")
            yield pid_i, events[:, :3]


def is_track_touching_edge(
    ev_full: np.ndarray,
    img_width_px: float,
    img_height_px: float,
    edge_margin_px: float = 0.0,
) -> bool:
    """Return True when a track touches the image boundary."""
    ev = np.asarray(ev_full, dtype=np.float64)
    if ev.ndim != 2 or ev.shape[0] == 0 or ev.shape[1] < 2:
        raise ValueError("ev_full must be a non-empty Nx2+ array")

    x = ev[:, 0]
    y = ev[:, 1]
    m = float(edge_margin_px)
    x_max = float(img_width_px) - 1.0 - m
    y_max = float(img_height_px) - 1.0 - m
    return bool(np.any(x <= m) or np.any(x >= x_max) or np.any(y <= m) or np.any(y >= y_max))


def occupancy_area_mask(
    xy: np.ndarray,
    bin_px: float = 1.0,
    morph: str = "fill",
    kernel_px: int = 1,
) -> np.ndarray:
    """Rasterize warped events and return the filled occupancy mask."""
    if morph != "fill":
        raise ValueError("only morph='fill' is supported")
    if int(kernel_px) != 1:
        raise ValueError("only kernel_px=1 is supported")

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        raise ValueError("xy must be a non-empty Nx2+ array")
    if not (np.isfinite(bin_px) and bin_px > 0):
        raise ValueError("bin_px must be > 0")

    gx = np.floor(xy[:, 0] / bin_px).astype(np.int64)
    gy = np.floor(xy[:, 1] / bin_px).astype(np.int64)
    cells = np.unique(np.column_stack([gx, gy]), axis=0)

    x_min = int(cells[:, 0].min())
    x_max = int(cells[:, 0].max())
    y_min = int(cells[:, 1].min())
    y_max = int(cells[:, 1].max())
    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
    mask[(cells[:, 1] - y_min).astype(np.int64), (cells[:, 0] - x_min).astype(np.int64)] = True
    return ndi.binary_fill_holes(mask)


def occupancy_area(
    xy: np.ndarray,
    bin_px: float = 1.0,
    morph: str = "fill",
    kernel_px: int = 1,
) -> float:
    """Filled occupancy area in pixel-squared units."""
    mask = occupancy_area_mask(xy, bin_px=bin_px, morph=morph, kernel_px=kernel_px)
    return float(mask.sum()) * float(bin_px) * float(bin_px)


def occupancy_area_occ_min(
    xy: np.ndarray,
    bin_px: float = 1.0,
    morph: str = "fill",
    kernel_px: int = 1,
    shifts=(0.0, 0.25, 0.5, 0.75),
) -> float:
    """Filled occupancy area after taking the minimum over sub-pixel grid shifts."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return 0.0

    best_n = None
    for dx in shifts:
        for dy in shifts:
            xy_shift = xy.copy()
            xy_shift[:, 0] += float(dx) * float(bin_px)
            xy_shift[:, 1] += float(dy) * float(bin_px)
            mask = occupancy_area_mask(
                xy_shift,
                bin_px=bin_px,
                morph=morph,
                kernel_px=kernel_px,
            )
            n_on = int(mask.sum())
            if best_n is None or n_on < best_n:
                best_n = n_on

    return 0.0 if best_n is None else float(best_n) * float(bin_px) * float(bin_px)


def max_area_for_window(
    ev: np.ndarray,
    model: TrajectoryModel,
    win_us: int,
    step_us: int,
    bin_px: float,
    min_events: int,
    morph: str,
    kernel_px: int,
    occ_min: bool = False,
) -> Optional[Tuple[float, int, float, int]]:
    """Return the maximum projected area measured over sliding time windows."""
    t = np.asarray(ev[:, 2], dtype=np.int64)
    t0, t1 = int(t[0]), int(t[-1])
    if (t1 - t0) < int(win_us):
        return None

    best_area = -np.inf
    best_n_ev = 0
    best_rho = np.nan
    best_ts = -1

    for ts in range(t0, t1 - int(win_us) + 1, int(step_us)):
        te = ts + int(win_us)
        j0 = int(np.searchsorted(t, ts, side="left"))
        j1 = int(np.searchsorted(t, te, side="right"))
        n_ev = int(j1 - j0)
        if n_ev < int(min_events):
            continue

        xy_warp = warp_events_with_model(ev[j0:j1], model)
        if occ_min:
            area = occupancy_area_occ_min(
                xy_warp,
                bin_px=bin_px,
                morph=morph,
                kernel_px=kernel_px,
            )
        else:
            area = occupancy_area(
                xy_warp,
                bin_px=bin_px,
                morph=morph,
                kernel_px=kernel_px,
            )
        if area <= 0:
            continue

        if area > best_area:
            best_area = float(area)
            best_n_ev = n_ev
            best_rho = float(n_ev) / float(area)
            best_ts = ts

    if best_area <= 0:
        return None
    return float(best_area), int(best_n_ev), float(best_rho), int(best_ts)
