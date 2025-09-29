"""Utility helpers for constructing smooth facial masks."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def _select_skin_points(
    kps: np.ndarray,
    spec: str = "auto",
    available: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Select a subset of facial landmarks that mostly cover skin."""
    n = len(kps)
    if spec == "auto":
        spec = "468" if n >= 200 else "68"

    if spec == "68":
        idx = list(range(2, 15))
        idx += [31, 35]
        idx += [1, 15]
    elif spec == "468":
        idx = [
            93,
            132,
            58,
            172,
            136,
            150,
            149,
            176,
            148,
            152,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            383,
            127,
            234,
            454,
            227,
            447,
            205,
            50,
            280,
            101,
            330,
        ]
        idx = sorted(set(i for i in idx if 0 <= i < n))
    else:
        idx = list(range(n))

    if available is not None:
        avail = set(available)
        idx = [i for i in idx if i in avail]
        if not idx:
            idx = list(range(n))
    return kps[idx]


def build_skin_sdf_mask(
    roi_shape: Tuple[int, int],
    kps_xy: Sequence[Sequence[float]],
    offset_xy: Tuple[int, int] = (0, 0),
    landmark_spec: str = "auto",
    forehead_pad_frac: float = 0.10,
    edge_width_px: float = 18.0,
    inner_bias_px: float = 0.0,
    gamma: float = 1.0,
    min_hull_points: int = 3,
) -> np.ndarray:
    """Construct a smooth facial skin mask using a signed-distance field."""
    H, W = roi_shape
    if H <= 0 or W <= 0:
        return np.zeros((max(H, 1), max(W, 1)), dtype=np.float32)

    kps = np.asarray(kps_xy, dtype=np.float32)
    if kps.ndim != 2 or kps.shape[1] != 2 or kps.shape[0] < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    ox, oy = map(float, offset_xy)
    pts = kps.copy()
    pts[:, 0] -= ox
    pts[:, 1] -= oy

    margin = 8.0
    in_roi = (
        (pts[:, 0] >= -margin)
        & (pts[:, 0] <= W + margin)
        & (pts[:, 1] >= -margin)
        & (pts[:, 1] <= H + margin)
    )
    pts = pts[in_roi]
    if len(pts) < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    skin_pts = _select_skin_points(pts, spec=landmark_spec)
    if len(skin_pts) < min_hull_points:
        skin_pts = pts

    x1, y1 = np.min(skin_pts, axis=0)
    x2, y2 = np.max(skin_pts, axis=0)
    if forehead_pad_frac > 1e-6:
        pad = (y2 - y1) * float(forehead_pad_frac)
        top_mask = skin_pts[:, 1] <= (y1 + 0.35 * (y2 - y1))
        skin_pts[top_mask, 1] = np.maximum(0.0, skin_pts[top_mask, 1] - pad)

    skin_pts_i = np.round(skin_pts).astype(np.int32)
    if len(skin_pts_i) < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    hull = cv2.convexHull(skin_pts_i)
    if hull is None or len(hull) < min_hull_points:
        return np.zeros((H, W), dtype=np.float32)

    inside = np.zeros((H, W), dtype=np.uint8)
    cv2.fillConvexPoly(inside, hull, 1)

    dist_in = cv2.distanceTransform(inside, distanceType=cv2.DIST_L2, maskSize=3)
    dist_out = cv2.distanceTransform(1 - inside, distanceType=cv2.DIST_L2, maskSize=3)

    sdf = dist_in - dist_out

    width = max(1e-3, float(edge_width_px))
    mask = 1.0 / (1.0 + np.exp(-(sdf - float(inner_bias_px)) / width))

    if gamma != 1.0:
        mask = np.clip(mask, 0.0, 1.0) ** float(gamma)

    return mask.astype(np.float32)
