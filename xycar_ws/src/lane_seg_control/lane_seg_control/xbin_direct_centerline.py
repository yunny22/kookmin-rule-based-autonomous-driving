"""Build a metric centerline directly from Xbin mask-adapter output."""

from __future__ import annotations

import cv2
import numpy as np


def extract_anchor_centers(
    yellow_mask: np.ndarray,
    yellow_probability: np.ndarray,
    *,
    anchor_stride_px: int = 4,
    anchor_offset_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model-image ``[x, y]`` centers and confidence per anchor row."""
    if yellow_mask.ndim != 2:
        raise ValueError("yellow_mask must be a 2-D array")
    if yellow_probability.shape != yellow_mask.shape:
        raise ValueError("yellow probability and mask shapes must match")

    stride = max(1, int(anchor_stride_px))
    offset = int(anchor_offset_px) % stride
    points: list[tuple[float, float]] = []
    confidences: list[float] = []
    for row in range(offset, yellow_mask.shape[0], stride):
        columns = np.flatnonzero(yellow_mask[row] > 0)
        if columns.size == 0:
            continue
        weights = yellow_probability[row, columns].astype(np.float64)
        weight_sum = float(weights.sum())
        if weight_sum > 1.0e-9:
            center_x = float(np.dot(columns, weights) / weight_sum)
        else:
            center_x = float(np.median(columns))
        points.append((center_x, float(row)))
        confidences.append(float(np.max(weights)) if weights.size else 0.0)

    if not points:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(confidences, dtype=np.float32),
    )


def project_points_to_metric(
    camera_points: np.ndarray,
    homography: np.ndarray,
    *,
    bev_width: int,
    bev_height: int,
    lateral_m_per_px: float,
    forward_m_per_px: float,
    minimum_forward_m: float,
    maximum_forward_m: float,
    maximum_abs_lateral_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project image points without rendering a BEV and return valid indices."""
    points = np.asarray(camera_points, dtype=np.float32)
    if points.size == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("camera_points must have shape Nx2")
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape 3x3")

    projected = cv2.perspectiveTransform(
        points.reshape((-1, 1, 2)), matrix
    ).reshape((-1, 2))
    forward = (
        float(bev_height - 1) - projected[:, 1]
    ) * float(forward_m_per_px)
    lateral = (
        0.5 * float(bev_width) - projected[:, 0]
    ) * float(lateral_m_per_px)
    metric = np.column_stack((forward, lateral))
    valid = (
        np.all(np.isfinite(metric), axis=1)
        & (forward >= float(minimum_forward_m))
        & (forward <= float(maximum_forward_m))
        & (np.abs(lateral) <= float(maximum_abs_lateral_m))
    )
    valid_indices = np.flatnonzero(valid)
    metric = metric[valid]
    if metric.shape[0] == 0:
        return metric.astype(np.float64), valid_indices
    order = np.argsort(metric[:, 0])
    return metric[order].astype(np.float64), valid_indices[order]
