#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrajectoryModel:
    deg: int
    t_ref_sec: float
    tc_mean: float
    tc_std: float
    cx: np.ndarray
    cy: np.ndarray


def fit_track_trajectory_poly(ev_sorted: np.ndarray, deg: int = 2) -> Optional[TrajectoryModel]:
    # Fit a smooth trajectory model x(t), y(t) using the track start as time reference.
    ev = np.asarray(ev_sorted, dtype=np.float64)
    if ev.ndim != 2 or ev.shape[1] < 3 or ev.shape[0] < max(6, deg + 2):
        return None

    x = ev[:, 0]
    y = ev[:, 1]
    t_sec = ev[:, 2] * 1e-6

    t_ref_sec = float(t_sec[0])
    tc = t_sec - t_ref_sec
    tc_mean = float(np.mean(tc))
    tc_std = float(np.std(tc))
    if not np.isfinite(tc_std) or tc_std < 1e-12:
        return None

    # Standardize time before polyfit for numerical stability.
    z = (tc - tc_mean) / tc_std
    deg = int(deg)
    n_unique = int(np.unique(z).size)
    if n_unique < deg + 1:
        return None

    try:
        cx = np.polyfit(z, x, deg=deg)
        cy = np.polyfit(z, y, deg=deg)
    except Exception:
        return None

    return TrajectoryModel(
        deg=deg,
        t_ref_sec=t_ref_sec,
        tc_mean=tc_mean,
        tc_std=tc_std,
        cx=cx,
        cy=cy,
    )


def warp_events_with_model(ev_window: np.ndarray, model: TrajectoryModel) -> np.ndarray:
    # Move events to the window start using the fitted trajectory.
    evw = np.asarray(ev_window, dtype=np.float64)
    if evw.ndim != 2 or evw.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    x = evw[:, 0]
    y = evw[:, 1]
    t_sec = evw[:, 2] * 1e-6

    t_ref_sec = float(t_sec[0])
    tc = t_sec - model.t_ref_sec
    z = (tc - model.tc_mean) / model.tc_std

    x_pred = np.polyval(model.cx, z)
    y_pred = np.polyval(model.cy, z)

    # Predicted position at the reference time of this window.
    tc0 = t_ref_sec - model.t_ref_sec
    z0 = (tc0 - model.tc_mean) / model.tc_std
    x0 = float(np.polyval(model.cx, z0))
    y0 = float(np.polyval(model.cy, z0))

    # Subtract the predicted displacement, keeping particle shape compact.
    xw = x - (x_pred - x0)
    yw = y - (y_pred - y0)
    return np.column_stack([xw, yw])


def predict_xy(model: TrajectoryModel, t_us: np.ndarray):
    t_sec = np.asarray(t_us, dtype=np.float64) * 1e-6
    tc = t_sec - model.t_ref_sec
    z = (tc - model.tc_mean) / model.tc_std
    x_pred = np.polyval(model.cx, z)
    y_pred = np.polyval(model.cy, z)
    return x_pred, y_pred
