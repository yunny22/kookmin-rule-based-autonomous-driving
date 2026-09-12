#!/usr/bin/env python3
"""Canonical yellow-centerline driver using fused Stanley and Pure Pursuit."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
try:
    from kaiev26_msgs.msg import Centerline
except ModuleNotFoundError:
    # Retain importability of the controller's pure functions without copying
    # the separate team message package into this scoped public release.
    Centerline = None
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from xycar_rule_drive.lane_rule_driver import (
    clamp,
    interpolate_clamped,
    inverse_lookup_table,
    make_point,
)


@dataclass(frozen=True)
class ConnectedPath:
    points: np.ndarray
    observation_count: int
    forward_span_m: float
    max_observation_gap_m: float
    fit_residual_m: float


@dataclass(frozen=True)
class SteeringTerms:
    pure_pursuit_rad: float
    stanley_rad: float
    fused_rad: float
    cross_track_error_m: float
    heading_error_rad: float
    target_x_m: float
    target_y_m: float


def command_during_lane_loss(
    *,
    has_valid_command: bool,
    last_angle_command: float,
    last_speed_command: float,
    lane_loss_speed_command: float,
    hold_last_steering: bool,
    hold_last_speed: bool,
) -> tuple[float, float]:
    """Apply the timeless lane-loss contract after the first valid command."""
    if not has_valid_command:
        return 0.0, 0.0
    angle = float(last_angle_command) if hold_last_steering else 0.0
    speed = (
        float(last_speed_command)
        if hold_last_speed
        else float(lane_loss_speed_command)
    )
    return angle, speed


def latency_compensated_lookahead(
    base_lookahead_m: float,
    speed_mps: float,
    control_latency_sec: float,
) -> float:
    """Preview where the vehicle reaches by the time steering takes effect."""
    return max(
        0.05,
        float(base_lookahead_m)
        + max(0.0, float(speed_mps))
        * max(0.0, float(control_latency_sec)),
    )


def predict_path_in_delayed_vehicle_frame(
    path: np.ndarray,
    *,
    speed_mps: float,
    curvature_per_m: float,
    latency_sec: float,
) -> np.ndarray:
    """Transform a current path into the predicted delayed vehicle frame."""
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return points.copy()
    latency = max(0.0, float(latency_sec))
    speed = max(0.0, float(speed_mps))
    if latency <= 0.0 or speed <= 0.0:
        return points.copy()

    curvature = float(curvature_per_m)
    heading_delta = speed * curvature * latency
    if abs(curvature) <= 1.0e-6:
        displacement_x = speed * latency
        displacement_y = 0.0
    else:
        displacement_x = math.sin(heading_delta) / curvature
        displacement_y = (1.0 - math.cos(heading_delta)) / curvature

    translated = points - np.asarray(
        [displacement_x, displacement_y],
        dtype=np.float64,
    )
    cosine = math.cos(heading_delta)
    sine = math.sin(heading_delta)
    predicted = np.column_stack(
        (
            cosine * translated[:, 0] + sine * translated[:, 1],
            -sine * translated[:, 0] + cosine * translated[:, 1],
        )
    )
    visible = predicted[:, 0] >= 0.02
    if int(np.count_nonzero(visible)) < 3:
        return points.copy()
    return predicted[visible]


def lead_compensated_steering_command(
    raw_command: float,
    previous_raw_command: float | None,
    dt: float,
    lead_time_sec: float,
    max_lead_command: float,
) -> float:
    """Lead curve release and true reversals without amplifying curve entry."""
    if previous_raw_command is None or dt <= 1.0e-4:
        return float(raw_command)
    raw = float(raw_command)
    previous = float(previous_raw_command)
    same_direction = raw * previous > 0.0
    if same_direction and abs(raw) >= abs(previous):
        return raw
    derivative = (
        raw - previous
    ) / float(dt)
    lead = derivative * max(0.0, float(lead_time_sec))
    limit = max(0.0, float(max_lead_command))
    compensated = raw + clamp(lead, -limit, limit)
    if same_direction and compensated * raw < 0.0:
        return 0.0
    return compensated


def pursuit_requests_command_reversal(
    last_angle_command: float,
    pure_pursuit_rad: float,
    activation_rad: float,
) -> bool:
    """Return true when pursuit asks for the opposite Xycar steering sign."""
    pursuit = float(pure_pursuit_rad)
    if abs(pursuit) < max(0.0, float(activation_rad)):
        return False
    desired_command_sign = -pursuit
    return float(last_angle_command) * desired_command_sign < 0.0


def steering_term_requests_command_reversal(
    last_angle_command: float,
    steering_rad: float,
    activation_rad: float,
) -> bool:
    """Return true for a strong term opposing the active Xycar command."""
    steering = float(steering_rad)
    if abs(steering) < max(0.0, float(activation_rad)):
        return False
    return float(last_angle_command) * (-steering) < 0.0


def update_heading_recovery_latch(
    recovery_command_sign: float,
    miss_count: int,
    *,
    trigger_command_sign: float,
    pure_pursuit_rad: float,
    stanley_rad: float,
    support_activation_rad: float,
    release_frames: int,
) -> tuple[float, int, bool]:
    """Latch a real steering reversal until path terms stop supporting it."""
    recovery = math.copysign(1.0, trigger_command_sign) if trigger_command_sign else (
        math.copysign(1.0, recovery_command_sign)
        if recovery_command_sign
        else 0.0
    )
    if recovery == 0.0:
        return 0.0, 0, False
    activation = max(0.0, float(support_activation_rad))
    pursuit_supports = (
        abs(float(pure_pursuit_rad)) >= activation
        and (-float(pure_pursuit_rad)) * recovery > 0.0
    )
    stanley_supports = (
        abs(float(stanley_rad)) >= activation
        and (-float(stanley_rad)) * recovery > 0.0
    )
    if pursuit_supports or stanley_supports:
        return recovery, 0, True
    misses = int(miss_count) + 1
    if misses >= max(1, int(release_frames)):
        return 0.0, 0, False
    return recovery, misses, False


def apply_turn_transition_recovery(
    raw_command: float,
    *,
    now: float,
    armed_turn_sign: float,
    recovery_sign: float,
    recovery_until: float,
    trigger_previous_command: float,
    trigger_new_command: float,
    minimum_recovery_command: float,
    hold_sec: float,
    cancel_opposed_command: float,
) -> tuple[float, float, float, float]:
    """Hold a confirmed opposite correction through steering actuator delay."""
    raw = float(raw_command)
    armed = (
        math.copysign(1.0, armed_turn_sign)
        if armed_turn_sign
        else 0.0
    )
    sign = float(recovery_sign)
    until = float(recovery_until)
    if sign and float(now) < until:
        if (
            raw * sign < 0.0
            and abs(raw) >= max(0.0, float(cancel_opposed_command))
        ):
            return raw, 0.0, 0.0, math.copysign(1.0, raw)
        minimum = max(0.0, float(minimum_recovery_command))
        return sign * max(minimum, raw * sign), sign, until, armed
    if abs(raw) >= max(0.0, float(trigger_previous_command)):
        armed = math.copysign(1.0, raw)
    if (
        armed
        and raw * armed < 0.0
        and abs(raw) >= max(0.0, float(trigger_new_command))
    ):
        sign = math.copysign(1.0, raw)
        until = float(now) + max(0.0, float(hold_sec))
        armed = 0.0
    if sign and float(now) < until:
        minimum = max(0.0, float(minimum_recovery_command))
        return sign * max(minimum, raw * sign), sign, until, armed
    return raw, 0.0, 0.0, armed


def adaptive_smooth_steering_command(
    *,
    last_command: float,
    raw_command: float,
    dt: float,
    straight_current_weight: float,
    curve_current_weight: float,
    straight_rate_limit: float,
    curve_rate_limit: float,
    curve_activation_command: float,
    curve_full_command: float,
) -> float:
    """Smooth straights while retaining full response in a tight curve."""
    activation = max(0.0, float(curve_activation_command))
    full = max(activation + 1.0e-6, float(curve_full_command))
    steering_level = max(abs(float(last_command)), abs(float(raw_command)))
    curve_fraction = clamp(
        (steering_level - activation) / (full - activation),
        0.0,
        1.0,
    )
    # A tiny sign change on a straight is normal 7 Hz perception noise. Do not
    # treat it as a curve reversal and bypass the straight-line damping.
    if (
        float(last_command) * float(raw_command) < 0.0
        and steering_level >= activation
    ):
        reversal_fraction = clamp(
            abs(float(raw_command) - float(last_command)) / full,
            0.0,
            1.0,
        )
        curve_fraction = max(curve_fraction, reversal_fraction)
    rate_limit = (
        (1.0 - curve_fraction) * max(0.0, float(straight_rate_limit))
        + curve_fraction * max(0.0, float(curve_rate_limit))
    )
    limited = float(last_command) + clamp(
        float(raw_command) - float(last_command),
        -rate_limit * max(0.0, float(dt)),
        rate_limit * max(0.0, float(dt)),
    )
    current_weight = (
        (1.0 - curve_fraction)
        * clamp(float(straight_current_weight), 0.0, 1.0)
        + curve_fraction * clamp(float(curve_current_weight), 0.0, 1.0)
    )
    return (
        (1.0 - current_weight) * float(last_command)
        + current_weight * limited
    )


def offset_path_right(path: np.ndarray, right_offset_m: float) -> np.ndarray:
    """Shift each BEV forward station toward the vehicle's right side."""
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    offset = max(0.0, float(right_offset_m))
    if offset <= 0.0:
        return points.copy()
    result = points.copy()
    result[:, 1] -= offset
    return result


def offset_path_left(path: np.ndarray, left_offset_m: float) -> np.ndarray:
    """Shift each BEV forward station toward the vehicle's left side."""
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    offset = max(0.0, float(left_offset_m))
    if offset <= 0.0:
        return points.copy()
    result = points.copy()
    result[:, 1] += offset
    return result


def white_boundary_to_target_offset(
    lane_half_width_m: float,
    target_right_offset_m: float,
) -> float:
    """Return the left shift from the outer white line to the target path."""
    yellow_to_white_m = 2.0 * max(0.0, float(lane_half_width_m))
    return max(
        0.0,
        yellow_to_white_m - max(0.0, float(target_right_offset_m)),
    )


def fuse_lane_center_paths(
    yellow_target: np.ndarray,
    white_target: np.ndarray,
    *,
    yellow_weight: float,
    point_count: int,
    extend_to_union: bool = True,
) -> np.ndarray:
    """Fuse yellow- and outer-white-derived estimates of the same lane center."""
    yellow = np.asarray(yellow_target, dtype=np.float64)
    white = np.asarray(white_target, dtype=np.float64)
    if yellow.ndim != 2 or yellow.shape[0] < 2 or yellow.shape[1] != 2:
        return white.copy()
    if white.ndim != 2 or white.shape[0] < 2 or white.shape[1] != 2:
        return yellow.copy()
    yellow = yellow[np.argsort(yellow[:, 0])]
    white = white[np.argsort(white[:, 0])]
    if extend_to_union:
        near_x = min(float(yellow[0, 0]), float(white[0, 0]))
        far_x = max(float(yellow[-1, 0]), float(white[-1, 0]))
    else:
        near_x = float(yellow[0, 0])
        far_x = float(yellow[-1, 0])
    sample_x = np.linspace(near_x, far_x, max(3, int(point_count)))
    yellow_y = np.interp(sample_x, yellow[:, 0], yellow[:, 1])
    white_y = np.interp(sample_x, white[:, 0], white[:, 1])
    yellow_available = (
        (sample_x >= float(yellow[0, 0]))
        & (sample_x <= float(yellow[-1, 0]))
    )
    white_available = (
        (sample_x >= float(white[0, 0]))
        & (sample_x <= float(white[-1, 0]))
    )
    weight = clamp(float(yellow_weight), 0.0, 1.0)
    sample_y = np.where(
        yellow_available & white_available,
        weight * yellow_y + (1.0 - weight) * white_y,
        np.where(yellow_available, yellow_y, white_y),
    )
    return np.column_stack((sample_x, sample_y))


def smooth_target_path(
    target_path: np.ndarray,
    previous_path: np.ndarray | None,
    previous_weight: float,
) -> np.ndarray:
    """Blend the final target after yellow/white source arbitration."""
    target = np.asarray(target_path, dtype=np.float64)
    previous = np.asarray(previous_path, dtype=np.float64)
    weight = clamp(float(previous_weight), 0.0, 0.95)
    if (
        weight <= 0.0
        or target.ndim != 2
        or target.shape[0] < 2
        or target.shape[1] != 2
        or previous.ndim != 2
        or previous.shape[0] < 2
        or previous.shape[1] != 2
    ):
        return target.copy()
    previous = previous[np.argsort(previous[:, 0])]
    result = target.copy()
    previous_y = np.interp(
        result[:, 0],
        previous[:, 0],
        previous[:, 1],
    )
    result[:, 1] = (
        weight * previous_y + (1.0 - weight) * result[:, 1]
    )
    return result


def median_path_lateral_difference(
    left_path: np.ndarray,
    right_path: np.ndarray,
    *,
    minimum_overlap_m: float,
    sample_count: int = 16,
) -> float | None:
    """Measure left-minus-right separation over the paths' shared forward span."""
    left = np.asarray(left_path, dtype=np.float64)
    right = np.asarray(right_path, dtype=np.float64)
    if (
        left.ndim != 2
        or left.shape[0] < 2
        or left.shape[1] != 2
        or right.ndim != 2
        or right.shape[0] < 2
        or right.shape[1] != 2
    ):
        return None
    left = left[np.argsort(left[:, 0])]
    right = right[np.argsort(right[:, 0])]
    near_x = max(float(left[0, 0]), float(right[0, 0]))
    far_x = min(float(left[-1, 0]), float(right[-1, 0]))
    if far_x - near_x < max(0.0, float(minimum_overlap_m)):
        return None
    sample_x = np.linspace(near_x, far_x, max(3, int(sample_count)))
    left_y = np.interp(sample_x, left[:, 0], left[:, 1])
    right_y = np.interp(sample_x, right[:, 0], right[:, 1])
    return float(np.median(left_y - right_y))


def outer_white_is_consistent(
    white_path: np.ndarray,
    *,
    yellow_path: np.ndarray | None,
    previous_white_path: np.ndarray | None,
    expected_yellow_to_white_m: float,
    yellow_tolerance_m: float,
    temporal_jump_m: float,
    minimum_overlap_m: float,
) -> bool:
    """Reject an inner / adjacent-road white line before it can steer the car."""
    if yellow_path is not None:
        separation = median_path_lateral_difference(
            yellow_path,
            white_path,
            minimum_overlap_m=minimum_overlap_m,
        )
        return (
            separation is not None
            and abs(
                separation - max(0.0, float(expected_yellow_to_white_m))
            )
            <= max(0.0, float(yellow_tolerance_m))
        )
    if previous_white_path is None:
        return False
    temporal_difference = median_path_lateral_difference(
        previous_white_path,
        white_path,
        minimum_overlap_m=minimum_overlap_m,
    )
    return (
        temporal_difference is not None
        and abs(temporal_difference) <= max(0.0, float(temporal_jump_m))
    )


def canonical_class_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read exact white/yellow classes from the shared canonical BGR contract."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("canonical road image must be HxWx3 BGR")
    blue, green, red = cv2.split(image)
    white = (
        (blue >= 200) & (green >= 200) & (red >= 200)
    ).astype(np.uint8) * 255
    yellow = (
        (blue <= 40) & (green >= 150) & (red >= 180)
    ).astype(np.uint8) * 255
    return white, yellow


def _split_runs(xs: np.ndarray) -> list[np.ndarray]:
    if xs.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(xs) > 1) + 1
    return [run for run in np.split(xs, breaks) if run.size]


def yellow_row_observations(
    mask: np.ndarray,
    *,
    lateral_range_m: float,
    forward_range_m: float,
    row_step_px: int = 2,
    min_run_width_px: int = 2,
    previous_path: np.ndarray | None = None,
    untracked_selection: str = "center",
) -> np.ndarray:
    """Return one centerline observation per sampled row in vehicle metres."""
    if mask.ndim != 2:
        raise ValueError("yellow mask must be single-channel")
    height, width = mask.shape
    observations: list[tuple[float, float]] = []
    previous = np.asarray(previous_path, dtype=np.float64)
    use_previous = (
        previous.ndim == 2
        and previous.shape[0] >= 2
        and previous.shape[1] == 2
    )
    if use_previous:
        previous = previous[np.argsort(previous[:, 0])]

    for row in range(height - 1, -1, -max(1, int(row_step_px))):
        runs = [
            run
            for run in _split_runs(np.flatnonzero(mask[row] > 0))
            if run.size >= max(1, int(min_run_width_px))
        ]
        if not runs:
            continue
        forward_m = (height - 1 - row) * float(forward_range_m) / max(
            1, height - 1
        )
        candidate_lateral = np.asarray(
            [
                (width * 0.5 - float(np.median(run)))
                * float(lateral_range_m)
                / float(width)
                for run in runs
            ],
            dtype=np.float64,
        )
        if use_previous:
            expected = float(
                np.interp(
                    forward_m,
                    previous[:, 0],
                    previous[:, 1],
                    left=previous[0, 1],
                    right=previous[-1, 1],
                )
            )
            selected = int(np.argmin(np.abs(candidate_lateral - expected)))
        elif untracked_selection == "right":
            selected = int(np.argmin(candidate_lateral))
        elif untracked_selection == "left":
            selected = int(np.argmax(candidate_lateral))
        else:
            selected = int(np.argmin(np.abs(candidate_lateral)))
        lateral_m = float(candidate_lateral[selected])
        observations.append((forward_m, lateral_m))
    return np.asarray(observations, dtype=np.float64).reshape(-1, 2)


def _longest_gap_connected_group(
    observations: np.ndarray,
    max_gap_m: float,
) -> tuple[np.ndarray, float]:
    if observations.shape[0] < 2:
        return observations, 0.0
    ordered = observations[np.argsort(observations[:, 0])]
    gaps = np.diff(ordered[:, 0])
    split_indices = np.flatnonzero(gaps > float(max_gap_m)) + 1
    groups = np.split(ordered, split_indices)
    group = max(
        groups,
        key=lambda item: (
            float(item[-1, 0] - item[0, 0]) if item.shape[0] >= 2 else 0.0,
            item.shape[0],
            -float(item[0, 0]) if item.size else -float("inf"),
        ),
    )
    group_gaps = np.diff(group[:, 0]) if group.shape[0] >= 2 else np.asarray([])
    return group, float(np.max(group_gaps)) if group_gaps.size else 0.0


def connect_yellow_centerline(
    yellow_mask: np.ndarray,
    *,
    lateral_range_m: float = 1.4,
    forward_range_m: float = 1.5,
    max_gap_m: float = 0.38,
    min_observations: int = 5,
    min_span_m: float = 0.10,
    max_fit_residual_m: float = 0.08,
    path_point_count: int = 32,
    near_extrapolation_m: float = 0.05,
    previous_path: np.ndarray | None = None,
    previous_weight: float = 0.25,
    untracked_selection: str = "center",
) -> ConnectedPath | None:
    """Connect visible yellow dashes with a robust continuous metric curve."""
    observations = yellow_row_observations(
        yellow_mask,
        lateral_range_m=lateral_range_m,
        forward_range_m=forward_range_m,
        previous_path=previous_path,
        untracked_selection=untracked_selection,
    )
    if observations.shape[0] < int(min_observations):
        return None
    observations, max_observation_gap = _longest_gap_connected_group(
        observations, max_gap_m
    )
    if observations.shape[0] < int(min_observations):
        return None

    forward_span = float(
        np.max(observations[:, 0]) - np.min(observations[:, 0])
    )
    if forward_span < float(min_span_m):
        return None

    degree = 3 if forward_span >= 0.85 and observations.shape[0] >= 10 else 2
    degree = min(degree, observations.shape[0] - 1)
    keep = np.ones(observations.shape[0], dtype=bool)
    coefficients = None
    for _ in range(4):
        if int(np.count_nonzero(keep)) <= degree:
            return None
        coefficients = np.polyfit(
            observations[keep, 0], observations[keep, 1], degree
        )
        residuals = np.abs(
            observations[:, 1]
            - np.polyval(coefficients, observations[:, 0])
        )
        median = float(np.median(residuals[keep]))
        robust_limit = min(
            float(max_fit_residual_m),
            max(0.025, 3.0 * median),
        )
        next_keep = residuals <= robust_limit
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    if coefficients is None or int(np.count_nonzero(keep)) < int(min_observations):
        return None

    residuals = np.abs(
        observations[keep, 1]
        - np.polyval(coefficients, observations[keep, 0])
    )
    fit_residual = float(np.sqrt(np.mean(residuals * residuals)))
    if fit_residual > float(max_fit_residual_m):
        return None

    far_x = float(np.max(observations[keep, 0]))
    observed_near_x = float(np.min(observations[keep, 0]))
    near_x = max(
        0.0,
        observed_near_x - max(0.0, float(near_extrapolation_m)),
    )
    sample_x = np.linspace(
        max(0.0, near_x),
        min(float(forward_range_m), far_x),
        max(3, int(path_point_count)),
    )
    sample_y = np.polyval(coefficients, sample_x)
    path = np.column_stack((sample_x, sample_y)).astype(np.float64)

    previous = np.asarray(previous_path, dtype=np.float64)
    if (
        previous.ndim == 2
        and previous.shape[0] >= 2
        and previous.shape[1] == 2
        and previous_weight > 0.0
    ):
        previous = previous[np.argsort(previous[:, 0])]
        previous_y = np.interp(
            sample_x,
            previous[:, 0],
            previous[:, 1],
            left=previous[0, 1],
            right=previous[-1, 1],
        )
        weight = clamp(float(previous_weight), 0.0, 0.95)
        path[:, 1] = weight * previous_y + (1.0 - weight) * path[:, 1]

    path[:, 1] = np.clip(
        path[:, 1],
        -float(lateral_range_m) * 0.5,
        float(lateral_range_m) * 0.5,
    )
    return ConnectedPath(
        points=path,
        observation_count=int(np.count_nonzero(keep)),
        forward_span_m=forward_span,
        max_observation_gap_m=max_observation_gap,
        fit_residual_m=fit_residual,
    )


def _path_lateral_at(path: np.ndarray, forward_x_m: float) -> float:
    ordered = path[np.argsort(path[:, 0])]
    return float(
        np.interp(
            forward_x_m,
            ordered[:, 0],
            ordered[:, 1],
            left=ordered[0, 1],
            right=ordered[-1, 1],
        )
    )


def _path_heading_at(
    path: np.ndarray,
    forward_x_m: float,
    window_m: float = 0.16,
) -> float:
    near_x = float(np.min(path[:, 0]))
    far_x = float(np.max(path[:, 0]))
    half_window = max(1.0e-4, float(window_m) * 0.5)
    evaluation_x = clamp(float(forward_x_m), near_x, far_x)
    start_x = max(near_x, evaluation_x - half_window)
    end_x = min(far_x, evaluation_x + half_window)
    if end_x - start_x < half_window:
        if evaluation_x <= near_x + half_window:
            end_x = min(far_x, near_x + 2.0 * half_window)
        else:
            start_x = max(near_x, far_x - 2.0 * half_window)
    if end_x - start_x < 1.0e-4:
        return 0.0
    start_y = _path_lateral_at(path, start_x)
    end_y = _path_lateral_at(path, end_x)
    return math.atan2(end_y - start_y, end_x - start_x)


def compute_departure_guard_pure_pursuit_weight(
    base_weight: float,
    *,
    cross_track_error_m: float,
    heading_error_rad: float,
    lateral_start_m: float,
    lateral_full_m: float,
    heading_start_rad: float,
    heading_full_rad: float,
    guarded_weight: float,
) -> tuple[float, float]:
    """Increase Stanley authority before the vehicle reaches a lane boundary."""

    def normalized_risk(value: float, start: float, full: float) -> float:
        lower = max(0.0, float(start))
        upper = max(lower + 1.0e-6, float(full))
        return clamp(
            (abs(float(value)) - lower) / (upper - lower),
            0.0,
            1.0,
        )

    risk = max(
        normalized_risk(
            cross_track_error_m,
            lateral_start_m,
            lateral_full_m,
        ),
        normalized_risk(
            heading_error_rad,
            heading_start_rad,
            heading_full_rad,
        ),
    )
    base = clamp(float(base_weight), 0.0, 1.0)
    guarded = min(base, clamp(float(guarded_weight), 0.0, 1.0))
    return (1.0 - risk) * base + risk * guarded, risk


def blend_pursuit_stanley(
    pure_pursuit_rad: float,
    stanley_rad: float,
    *,
    pure_pursuit_weight: float,
    opposed_stanley_weight: float,
) -> float:
    """Blend heading feedback without letting a noisy term reverse pursuit."""
    pursuit = float(pure_pursuit_rad)
    stanley = float(stanley_rad)
    weight = clamp(float(pure_pursuit_weight), 0.0, 1.0)
    if pursuit * stanley < 0.0:
        if abs(stanley) <= abs(pursuit):
            stanley_weight = clamp(float(opposed_stanley_weight), 0.0, 1.0)
            return (1.0 - stanley_weight) * pursuit + stanley_weight * stanley
        return pursuit
    return weight * pursuit + (1.0 - weight) * stanley


def path_heading_change_per_m(
    path: np.ndarray,
    *,
    near_x_m: float,
    far_x_m: float,
) -> float:
    """Estimate path curvature from the heading change across the visible path."""
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return float("inf")
    minimum_x = float(np.min(points[:, 0]))
    maximum_x = float(np.max(points[:, 0]))
    near_x = clamp(float(near_x_m), minimum_x, maximum_x)
    far_x = clamp(float(far_x_m), minimum_x, maximum_x)
    if far_x < near_x:
        near_x, far_x = far_x, near_x
    span = far_x - near_x
    if span <= 0.10:
        return float("inf")
    near_heading = _path_heading_at(points, near_x)
    far_heading = _path_heading_at(points, far_x)
    heading_delta = math.atan2(
        math.sin(far_heading - near_heading),
        math.cos(far_heading - near_heading),
    )
    return abs(heading_delta) / span


def anticipatory_center_corridor_error(
    cross_track_error_m: float,
    heading_error_rad: float,
    *,
    boundary_m: float,
    minimum_approach_heading_rad: float,
) -> float:
    """Reverse lateral correction inside a boundary while approaching center."""
    error = float(cross_track_error_m)
    heading = float(heading_error_rad)
    boundary = max(0.0, float(boundary_m))
    if (
        boundary <= 0.0
        or abs(error) > boundary
        or abs(heading) < max(0.0, float(minimum_approach_heading_rad))
        or error * heading >= 0.0
    ):
        return error
    return error - math.copysign(boundary, error)


def fused_stanley_pursuit(
    path: np.ndarray,
    *,
    lookahead_m: float,
    wheelbase_m: float,
    pure_pursuit_control_x_m: float,
    stanley_control_x_m: float,
    speed_mps: float,
    stanley_gain: float,
    stanley_softening_mps: float,
    pure_pursuit_weight: float,
    opposed_stanley_weight: float = 0.70,
    departure_guard_enabled: bool = False,
    departure_guard_lateral_start_m: float = 0.06,
    departure_guard_lateral_full_m: float = 0.16,
    departure_guard_heading_start_rad: float = 0.10,
    departure_guard_heading_full_rad: float = 0.35,
    departure_guard_pure_pursuit_weight: float = 0.35,
    straight_stanley_enabled: bool = False,
    straight_path_curvature_threshold: float = 0.16,
    straight_pure_pursuit_weight: float = 0.10,
    straight_stanley_gain: float = 0.65,
    straight_stanley_softening_mps: float = 0.65,
    straight_center_anticipation_m: float = 0.10,
    straight_center_minimum_approach_heading_rad: float = 0.02,
    straight_center_steering_scale: float = 0.70,
    straight_center_scale_heading_limit_rad: float = 0.20,
) -> SteeringTerms:
    """Return steering angles; positive radians mean a left turn."""
    if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] != 2:
        raise ValueError("path must contain at least three [forward, lateral] points")
    target_x = min(
        max(float(lookahead_m), float(np.min(path[:, 0]))),
        float(np.max(path[:, 0])),
    )
    target_y = _path_lateral_at(path, target_x)
    relative_x = target_x - float(pure_pursuit_control_x_m)
    distance_sq = max(0.01, relative_x * relative_x + target_y * target_y)
    pp_curvature = 2.0 * target_y / distance_sq
    pure_pursuit = math.atan(float(wheelbase_m) * pp_curvature)

    cross_track_error = _path_lateral_at(path, float(stanley_control_x_m))
    heading_error = _path_heading_at(path, float(stanley_control_x_m))
    path_curvature = path_heading_change_per_m(
        path,
        near_x_m=max(float(stanley_control_x_m), float(np.min(path[:, 0]))),
        far_x_m=min(float(lookahead_m), float(np.max(path[:, 0]))),
    )
    straight_stanley_active = (
        bool(straight_stanley_enabled)
        and path_curvature
        <= max(0.0, float(straight_path_curvature_threshold))
    )
    stanley_cross_track_error = cross_track_error
    if straight_stanley_active:
        stanley_cross_track_error = anticipatory_center_corridor_error(
            cross_track_error,
            heading_error,
            boundary_m=straight_center_anticipation_m,
            minimum_approach_heading_rad=(
                straight_center_minimum_approach_heading_rad
            ),
        )
    active_stanley_gain = (
        float(straight_stanley_gain)
        if straight_stanley_active
        else float(stanley_gain)
    )
    active_softening = (
        float(straight_stanley_softening_mps)
        if straight_stanley_active
        else float(stanley_softening_mps)
    )
    stanley = heading_error + math.atan2(
        active_stanley_gain * stanley_cross_track_error,
        max(0.0, float(speed_mps)) + max(1.0e-3, active_softening),
    )
    effective_weight = clamp(float(pure_pursuit_weight), 0.0, 1.0)
    guard_risk = 0.0
    if departure_guard_enabled:
        effective_weight, guard_risk = (
            compute_departure_guard_pure_pursuit_weight(
                effective_weight,
                cross_track_error_m=cross_track_error,
                heading_error_rad=heading_error,
                lateral_start_m=departure_guard_lateral_start_m,
                lateral_full_m=departure_guard_lateral_full_m,
                heading_start_rad=departure_guard_heading_start_rad,
                heading_full_rad=departure_guard_heading_full_rad,
                guarded_weight=departure_guard_pure_pursuit_weight,
            )
        )
    if straight_stanley_active:
        # On a straight, direct blending is intentional: the generic opposed
        # term guard can otherwise discard a dominant Stanley correction.
        effective_weight = min(
            effective_weight,
            clamp(float(straight_pure_pursuit_weight), 0.0, 1.0),
        )
        fused = (
            effective_weight * pure_pursuit
            + (1.0 - effective_weight) * stanley
        )
    elif guard_risk > 0.0:
        fused = (
            effective_weight * pure_pursuit
            + (1.0 - effective_weight) * stanley
        )
    else:
        fused = blend_pursuit_stanley(
            pure_pursuit,
            stanley,
            pure_pursuit_weight=effective_weight,
            opposed_stanley_weight=opposed_stanley_weight,
        )
    if straight_stanley_active:
        boundary = max(1.0e-6, abs(float(straight_center_anticipation_m)))
        lateral_ratio = clamp(abs(cross_track_error) / boundary, 0.0, 1.0)
        lateral_edge = lateral_ratio * lateral_ratio * (
            3.0 - 2.0 * lateral_ratio
        )
        heading_limit = max(
            1.0e-6,
            abs(float(straight_center_scale_heading_limit_rad)),
        )
        heading_ratio = clamp(abs(heading_error) / heading_limit, 0.0, 1.0)
        heading_edge = heading_ratio * heading_ratio * (
            3.0 - 2.0 * heading_ratio
        )
        centered_alignment = (1.0 - lateral_edge) * (1.0 - heading_edge)
        minimum_scale = clamp(float(straight_center_steering_scale), 0.0, 1.0)
        fused *= 1.0 - centered_alignment * (1.0 - minimum_scale)
    return SteeringTerms(
        pure_pursuit_rad=pure_pursuit,
        stanley_rad=stanley,
        fused_rad=fused,
        cross_track_error_m=cross_track_error,
        heading_error_rad=heading_error,
        target_x_m=target_x,
        target_y_m=target_y,
    )


class CanonicalStanleyPursuitDriver(Node):
    def __init__(
        self,
        node_name: str = "canonical_stanley_pursuit_driver",
    ) -> None:
        if Centerline is None:
            raise RuntimeError(
                "kaiev26_msgs is required to run the canonical driver; install "
                "the team message-interface package in the ROS environment"
            )
        super().__init__(node_name)
        self.declare_parameter(
            "canonical_topic", "/perception/canonical_road_image"
        )
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("shadow_motor_topic", "/xycar_motor_shadow")
        self.declare_parameter(
            "target_path_topic", "/rule_drive/connected_yellow_path"
        )
        self.declare_parameter(
            "debug_image_topic", "/rule_drive/canonical_debug_image"
        )
        self.declare_parameter(
            "debug_markers_topic", "/rule_drive/debug_markers"
        )
        self.declare_parameter("diagnostics_topic", "/rule_drive/diagnostics")
        self.declare_parameter("action_trace_topic", "/rl/action_applied")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("steering_only", False)
        self.declare_parameter("canonical_lateral_range_m", 1.4)
        self.declare_parameter("canonical_forward_range_m", 1.5)
        self.declare_parameter("yellow_max_gap_m", 0.38)
        self.declare_parameter("yellow_min_observations", 5)
        self.declare_parameter("yellow_min_span_m", 0.10)
        self.declare_parameter("yellow_max_fit_residual_m", 0.08)
        self.declare_parameter("path_point_count", 32)
        self.declare_parameter("path_previous_weight", 0.10)
        self.declare_parameter("min_lane_pixels", 8)
        self.declare_parameter("target_right_offset_m", 0.10)
        self.declare_parameter("white_fallback_enabled", True)
        self.declare_parameter("lane_half_width_m", 0.20)
        self.declare_parameter("white_fallback_max_gap_m", 0.25)
        self.declare_parameter("white_fallback_min_span_m", 0.18)
        self.declare_parameter("white_consistency_filter_enabled", True)
        self.declare_parameter("white_yellow_separation_tolerance_m", 0.16)
        self.declare_parameter("white_temporal_max_lateral_jump_m", 0.18)
        self.declare_parameter("white_consistency_min_overlap_m", 0.12)
        self.declare_parameter("yellow_prefer_min_span_m", 0.70)
        self.declare_parameter("white_fallback_span_advantage_m", 0.15)
        self.declare_parameter("yellow_fusion_weight", 1.0)
        self.declare_parameter("short_yellow_fusion_weight", 1.0)
        self.declare_parameter("extend_fused_path_to_white", False)
        self.declare_parameter("target_path_previous_weight", 0.10)
        self.declare_parameter("temporal_path_ego_compensation_enabled", False)
        self.declare_parameter("wheel_base_m", 0.32)
        self.declare_parameter("pure_pursuit_control_x_m", -0.08)
        self.declare_parameter("stanley_control_x_m", 0.16)
        self.declare_parameter("lookahead_distance_m", 1.50)
        self.declare_parameter("control_latency_preview_sec", 0.10)
        self.declare_parameter("stanley_gain", 1.15)
        self.declare_parameter("stanley_softening_mps", 0.35)
        self.declare_parameter("pure_pursuit_weight", 0.95)
        self.declare_parameter("straight_stanley_enabled", True)
        self.declare_parameter("straight_path_curvature_threshold", 0.16)
        self.declare_parameter("straight_pure_pursuit_weight", 0.10)
        self.declare_parameter("straight_stanley_gain", 0.65)
        self.declare_parameter("straight_stanley_softening_mps", 0.65)
        self.declare_parameter("straight_center_anticipation_m", 0.10)
        self.declare_parameter(
            "straight_center_minimum_approach_heading_rad", 0.02
        )
        self.declare_parameter("straight_center_steering_scale", 0.70)
        self.declare_parameter(
            "straight_center_scale_heading_limit_rad", 0.20
        )
        self.declare_parameter("departure_guard_enabled", True)
        self.declare_parameter("departure_guard_lateral_start_m", 0.06)
        self.declare_parameter("departure_guard_lateral_full_m", 0.16)
        self.declare_parameter("departure_guard_heading_start_rad", 0.10)
        self.declare_parameter("departure_guard_heading_full_rad", 0.35)
        self.declare_parameter(
            "departure_guard_pure_pursuit_weight", 0.65
        )
        self.declare_parameter("reversal_pure_pursuit_weight", 0.45)
        self.declare_parameter("reversal_activation_rad", 10.0)
        self.declare_parameter("stanley_reversal_activation_rad", 10.0)
        self.declare_parameter(
            "stanley_reversal_pure_pursuit_weight", 0.30
        )
        self.declare_parameter("heading_recovery_support_rad", 10.0)
        self.declare_parameter("heading_recovery_release_frames", 3)
        self.declare_parameter("turn_transition_hold_sec", 0.25)
        self.declare_parameter(
            "turn_transition_minimum_recovery_command", 18.0
        )
        self.declare_parameter(
            "turn_transition_trigger_previous_command", 12.0
        )
        self.declare_parameter("turn_transition_trigger_new_command", 2.0)
        self.declare_parameter(
            "turn_transition_cancel_opposed_command", 24.0
        )
        self.declare_parameter("opposed_stanley_weight", 0.70)
        self.declare_parameter("steering_current_weight", 0.25)
        self.declare_parameter("steering_rate_limit_cmd_per_sec", 180.0)
        self.declare_parameter("steering_curve_current_weight", 0.55)
        self.declare_parameter(
            "steering_curve_rate_limit_cmd_per_sec", 300.0
        )
        self.declare_parameter("steering_curve_activation_command", 12.0)
        self.declare_parameter("steering_curve_full_command", 24.0)
        self.declare_parameter("steering_lead_time_sec", 0.0)
        self.declare_parameter("steering_max_lead_command", 0.0)
        self.declare_parameter("speed_gain_mps_per_cmd", 0.080612)
        self.declare_parameter("cruise_speed_command", 20.0)
        self.declare_parameter("minimum_speed_command", 17.0)
        self.declare_parameter("curve_slowdown_angle_command", 24.0)
        self.declare_parameter("command_rate_hz", 7.0)
        self.declare_parameter("command_on_canonical", True)
        self.declare_parameter("hold_last_steering_on_lane_loss", True)
        self.declare_parameter("hold_last_speed_on_lane_loss", False)
        self.declare_parameter("lane_loss_speed_command", 4.0)
        # Competition steering calibration is intentionally omitted from the
        # public source. These neutral placeholders keep shadow mode safe;
        # a locally measured configuration is required for vehicle output.
        self.declare_parameter("angle_command_min", -1.0)
        self.declare_parameter("angle_command_max", 1.0)
        self.declare_parameter(
            "steering_map_commands",
            [-1.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "steering_map_curvatures",
            [1.0, 0.0, -1.0],
        )

        self.bridge = CvBridge()
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.drive_enabled = bool(self.get_parameter("drive_enabled").value)
        self.steering_only = bool(self.get_parameter("steering_only").value)
        self.lateral_range_m = float(
            self.get_parameter("canonical_lateral_range_m").value
        )
        self.forward_range_m = float(
            self.get_parameter("canonical_forward_range_m").value
        )
        self.angle_command_min = float(
            self.get_parameter("angle_command_min").value
        )
        self.angle_command_max = float(
            self.get_parameter("angle_command_max").value
        )
        commands = [
            float(value)
            for value in self.get_parameter("steering_map_commands").value
        ]
        curvatures = [
            float(value)
            for value in self.get_parameter("steering_map_curvatures").value
        ]
        self.command_inputs = commands
        self.command_curvatures = curvatures
        self.curvature_inputs, self.curvature_commands = inverse_lookup_table(
            commands, curvatures
        )

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.motor_pub = None
        if self.drive_enabled:
            self.motor_pub = self.create_publisher(
                Float32MultiArray,
                str(self.get_parameter("motor_topic").value),
                10,
            )
        self.shadow_motor_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("shadow_motor_topic").value),
            10,
        )
        self.path_pub = self.create_publisher(
            Centerline,
            str(self.get_parameter("target_path_topic").value),
            10,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            output_qos,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("debug_markers_topic").value),
            10,
        )
        self.diagnostics_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("diagnostics_topic").value),
            10,
        )
        self.action_trace_pub = self.create_publisher(
            TwistStamped,
            str(self.get_parameter("action_trace_topic").value),
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("canonical_topic").value),
            self.on_canonical,
            output_qos,
        )

        rate_hz = max(1.0, float(self.get_parameter("command_rate_hz").value))
        self.command_period_sec = 1.0 / rate_hz
        self.create_timer(1.0 / rate_hz, self.on_timer)
        self.latest_path: np.ndarray | None = None
        self.latest_yellow_path: np.ndarray | None = None
        self.latest_white_path: np.ndarray | None = None
        self.latest_header = None
        self.latest_terms: SteeringTerms | None = None
        self.latest_path_info: ConnectedPath | None = None
        self.latest_path_source = "none"
        self.last_tracking_update_time: float | None = None
        self.last_angle_command = 0.0
        self.last_speed_command = 0.0
        self.last_command_time: float | None = None
        self.last_raw_angle_command: float | None = None
        self.last_raw_command_time: float | None = None
        self.heading_recovery_command_sign = 0.0
        self.heading_recovery_miss_count = 0
        self.turn_transition_recovery_sign = 0.0
        self.turn_transition_recovery_until = 0.0
        self.turn_transition_armed_sign = 0.0
        self.last_canonical_command_time: float | None = None
        self.has_valid_command = False
        self.lane_visible = False
        self.path_valid = False
        self.loss_announced = False
        mode = "AUTO" if self.drive_enabled else "SHADOW"
        self.get_logger().info(
            "canonical Stanley/Pure Pursuit driver ready: "
            f"{mode}, 7Hz, speed={float(self.get_parameter('minimum_speed_command').value):.1f}"
            f"..{float(self.get_parameter('cruise_speed_command').value):.1f}, "
            f"yellow_gap={float(self.get_parameter('yellow_max_gap_m').value):.2f}m"
        )

    def on_canonical(self, message: Image) -> None:
        now = time.monotonic()
        self.advance_tracked_paths(now)
        try:
            image = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="bgr8"
            )
        except Exception as exc:
            self.get_logger().error(f"canonical image conversion failed: {exc}")
            return
        white, yellow = canonical_class_masks(image)
        lane_pixels = int(np.count_nonzero(white) + np.count_nonzero(yellow))
        self.lane_visible = lane_pixels >= int(
            self.get_parameter("min_lane_pixels").value
        )
        yellow_connected = connect_yellow_centerline(
            yellow,
            lateral_range_m=self.lateral_range_m,
            forward_range_m=self.forward_range_m,
            max_gap_m=float(self.get_parameter("yellow_max_gap_m").value),
            min_observations=int(
                self.get_parameter("yellow_min_observations").value
            ),
            min_span_m=float(self.get_parameter("yellow_min_span_m").value),
            max_fit_residual_m=float(
                self.get_parameter("yellow_max_fit_residual_m").value
            ),
            path_point_count=int(
                self.get_parameter("path_point_count").value
            ),
            previous_path=self.latest_yellow_path,
            previous_weight=float(
                self.get_parameter("path_previous_weight").value
            ),
        )
        white_candidate = None
        if (
            self.lane_visible
            and bool(self.get_parameter("white_fallback_enabled").value)
            and int(np.count_nonzero(white))
            >= int(self.get_parameter("min_lane_pixels").value)
        ):
            lane_half_width = float(
                self.get_parameter("lane_half_width_m").value
            )
            white_candidate = connect_yellow_centerline(
                white,
                lateral_range_m=self.lateral_range_m,
                forward_range_m=self.forward_range_m,
                max_gap_m=float(
                    self.get_parameter("white_fallback_max_gap_m").value
                ),
                min_observations=int(
                    self.get_parameter("yellow_min_observations").value
                ),
                min_span_m=float(
                    self.get_parameter("white_fallback_min_span_m").value
                ),
                max_fit_residual_m=float(
                    self.get_parameter("yellow_max_fit_residual_m").value
                ),
                path_point_count=int(
                    self.get_parameter("path_point_count").value
                ),
                previous_path=self.latest_white_path,
                previous_weight=float(
                    self.get_parameter("path_previous_weight").value
                ),
                untracked_selection="right",
            )
        white_connected = None
        if white_candidate is not None:
            consistency_required = bool(
                self.get_parameter("white_consistency_filter_enabled").value
            )
            if not consistency_required or outer_white_is_consistent(
                white_candidate.points,
                yellow_path=(
                    yellow_connected.points
                    if yellow_connected is not None
                    else None
                ),
                previous_white_path=self.latest_white_path,
                expected_yellow_to_white_m=2.0 * lane_half_width,
                yellow_tolerance_m=float(
                    self.get_parameter(
                        "white_yellow_separation_tolerance_m"
                    ).value
                ),
                temporal_jump_m=float(
                    self.get_parameter(
                        "white_temporal_max_lateral_jump_m"
                    ).value
                ),
                minimum_overlap_m=float(
                    self.get_parameter(
                        "white_consistency_min_overlap_m"
                    ).value
                ),
            ):
                white_connected = white_candidate
        if yellow_connected is not None:
            self.latest_yellow_path = yellow_connected.points
        if white_connected is not None:
            self.latest_white_path = white_connected.points

        yellow_target = (
            offset_path_right(
                yellow_connected.points,
                float(self.get_parameter("target_right_offset_m").value),
            )
            if yellow_connected is not None
            else None
        )
        white_target = (
            offset_path_left(
                white_connected.points,
                white_boundary_to_target_offset(
                    float(self.get_parameter("lane_half_width_m").value),
                    float(
                        self.get_parameter("target_right_offset_m").value
                    ),
                ),
            )
            if white_connected is not None
            else None
        )
        if yellow_target is not None and white_target is not None:
            minimum_span = float(
                self.get_parameter("yellow_prefer_min_span_m").value
            )
            yellow_weight = float(
                self.get_parameter(
                    "yellow_fusion_weight"
                    if yellow_connected.forward_span_m >= minimum_span
                    else "short_yellow_fusion_weight"
                ).value
            )
            # Yellow remains the geometric reference while it is visible.
            # White takes over only when yellow disappears completely.
            target_path = fuse_lane_center_paths(
                yellow_target,
                white_target,
                yellow_weight=yellow_weight,
                point_count=int(
                    self.get_parameter("path_point_count").value
                ),
                extend_to_union=bool(
                    self.get_parameter(
                        "extend_fused_path_to_white"
                    ).value
                ),
            )
            connected = yellow_connected
            path_source = "fused"
        elif yellow_target is not None:
            target_path = yellow_target
            connected = yellow_connected
            path_source = "yellow"
        elif white_target is not None:
            target_path = white_target
            connected = white_connected
            path_source = "white"
        else:
            target_path = None
            connected = None
            path_source = "none"

        self.path_valid = target_path is not None
        self.latest_path_info = connected
        self.latest_path_source = path_source
        if target_path is not None:
            target_path = smooth_target_path(
                target_path,
                self.latest_path,
                float(
                    self.get_parameter(
                        "target_path_previous_weight"
                    ).value
                ),
            )
            self.latest_path = target_path
            self.latest_header = message.header
            self.loss_announced = False
            self.publish_path(message.header, target_path, path_source)
        self.publish_debug_image(
            message,
            image,
            white,
            yellow,
            connected,
            self.latest_path if connected is not None else None,
            path_source,
        )
        if bool(self.get_parameter("command_on_canonical").value):
            self.last_canonical_command_time = now
            self.issue_command(now)

    def advance_tracked_paths(self, now: float) -> None:
        """Express temporal lane state in the current estimated vehicle frame."""
        if self.last_tracking_update_time is None:
            self.last_tracking_update_time = float(now)
            return
        dt = clamp(float(now) - self.last_tracking_update_time, 0.0, 0.30)
        self.last_tracking_update_time = float(now)
        if (
            not bool(
                self.get_parameter(
                    "temporal_path_ego_compensation_enabled"
                ).value
            )
            or not self.has_valid_command
            or dt <= 1.0e-4
        ):
            return
        speed_mps = max(
            0.0,
            self.last_speed_command
            * float(self.get_parameter("speed_gain_mps_per_cmd").value),
        )
        curvature = interpolate_clamped(
            self.last_angle_command,
            self.command_inputs,
            self.command_curvatures,
        )

        def advance(path: np.ndarray | None) -> np.ndarray | None:
            if path is None:
                return None
            return predict_path_in_delayed_vehicle_frame(
                path,
                speed_mps=speed_mps,
                curvature_per_m=curvature,
                latency_sec=dt,
            )

        self.latest_yellow_path = advance(self.latest_yellow_path)
        self.latest_white_path = advance(self.latest_white_path)
        self.latest_path = advance(self.latest_path)

    def steering_command_for_path(
        self, path: np.ndarray
    ) -> tuple[float, SteeringTerms]:
        speed_mps = max(
            0.0,
            self.last_speed_command
            * float(self.get_parameter("speed_gain_mps_per_cmd").value),
        )
        if speed_mps <= 0.0:
            speed_mps = (
                float(self.get_parameter("minimum_speed_command").value)
                * float(self.get_parameter("speed_gain_mps_per_cmd").value)
            )
        delayed_path = path
        latency_sec = float(
            self.get_parameter("control_latency_preview_sec").value
        )
        if self.has_valid_command and latency_sec > 0.0:
            active_curvature = interpolate_clamped(
                self.last_angle_command,
                self.command_inputs,
                self.command_curvatures,
            )
            delayed_path = predict_path_in_delayed_vehicle_frame(
                path,
                speed_mps=speed_mps,
                curvature_per_m=active_curvature,
                latency_sec=latency_sec,
            )
        terms = fused_stanley_pursuit(
            delayed_path,
            lookahead_m=float(
                self.get_parameter("lookahead_distance_m").value
            ),
            wheelbase_m=float(self.get_parameter("wheel_base_m").value),
            pure_pursuit_control_x_m=float(
                self.get_parameter("pure_pursuit_control_x_m").value
            ),
            stanley_control_x_m=float(
                self.get_parameter("stanley_control_x_m").value
            ),
            speed_mps=speed_mps,
            stanley_gain=float(self.get_parameter("stanley_gain").value),
            stanley_softening_mps=float(
                self.get_parameter("stanley_softening_mps").value
            ),
            pure_pursuit_weight=float(
                self.get_parameter("pure_pursuit_weight").value
            ),
            straight_stanley_enabled=bool(
                self.get_parameter("straight_stanley_enabled").value
            ),
            straight_path_curvature_threshold=float(
                self.get_parameter(
                    "straight_path_curvature_threshold"
                ).value
            ),
            straight_pure_pursuit_weight=float(
                self.get_parameter(
                    "straight_pure_pursuit_weight"
                ).value
            ),
            straight_stanley_gain=float(
                self.get_parameter("straight_stanley_gain").value
            ),
            straight_stanley_softening_mps=float(
                self.get_parameter(
                    "straight_stanley_softening_mps"
                ).value
            ),
            straight_center_anticipation_m=float(
                self.get_parameter(
                    "straight_center_anticipation_m"
                ).value
            ),
            straight_center_minimum_approach_heading_rad=float(
                self.get_parameter(
                    "straight_center_minimum_approach_heading_rad"
                ).value
            ),
            straight_center_steering_scale=float(
                self.get_parameter(
                    "straight_center_steering_scale"
                ).value
            ),
            straight_center_scale_heading_limit_rad=float(
                self.get_parameter(
                    "straight_center_scale_heading_limit_rad"
                ).value
            ),
            opposed_stanley_weight=float(
                self.get_parameter("opposed_stanley_weight").value
            ),
            departure_guard_enabled=bool(
                self.get_parameter("departure_guard_enabled").value
            ),
            departure_guard_lateral_start_m=float(
                self.get_parameter(
                    "departure_guard_lateral_start_m"
                ).value
            ),
            departure_guard_lateral_full_m=float(
                self.get_parameter("departure_guard_lateral_full_m").value
            ),
            departure_guard_heading_start_rad=float(
                self.get_parameter(
                    "departure_guard_heading_start_rad"
                ).value
            ),
            departure_guard_heading_full_rad=float(
                self.get_parameter(
                    "departure_guard_heading_full_rad"
                ).value
            ),
            departure_guard_pure_pursuit_weight=float(
                self.get_parameter(
                    "departure_guard_pure_pursuit_weight"
                ).value
            ),
        )
        pursuit_reversal = (
            self.has_valid_command
            and pursuit_requests_command_reversal(
                self.last_angle_command,
                terms.pure_pursuit_rad,
                float(
                    self.get_parameter("reversal_activation_rad").value
                ),
            )
        )
        stanley_reversal = (
            self.has_valid_command
            and steering_term_requests_command_reversal(
                self.last_angle_command,
                terms.stanley_rad,
                float(
                    self.get_parameter(
                        "stanley_reversal_activation_rad"
                    ).value
                ),
            )
        )
        trigger_sign = 0.0
        if stanley_reversal:
            trigger_sign = -terms.stanley_rad
        elif pursuit_reversal:
            trigger_sign = -terms.pure_pursuit_rad
        (
            self.heading_recovery_command_sign,
            self.heading_recovery_miss_count,
            use_heading_recovery,
        ) = update_heading_recovery_latch(
            self.heading_recovery_command_sign,
            self.heading_recovery_miss_count,
            trigger_command_sign=trigger_sign,
            pure_pursuit_rad=terms.pure_pursuit_rad,
            stanley_rad=terms.stanley_rad,
            support_activation_rad=float(
                self.get_parameter("heading_recovery_support_rad").value
            ),
            release_frames=int(
                self.get_parameter(
                    "heading_recovery_release_frames"
                ).value
            ),
        )
        if use_heading_recovery:
            pursuit_weight = clamp(
                float(
                    self.get_parameter(
                        "stanley_reversal_pure_pursuit_weight"
                    ).value
                ),
                0.0,
                1.0,
            )
            recovery_fused = (
                pursuit_weight * terms.pure_pursuit_rad
                + (1.0 - pursuit_weight) * terms.stanley_rad
            )
            terms = SteeringTerms(
                pure_pursuit_rad=terms.pure_pursuit_rad,
                stanley_rad=terms.stanley_rad,
                fused_rad=recovery_fused,
                cross_track_error_m=terms.cross_track_error_m,
                heading_error_rad=terms.heading_error_rad,
                target_x_m=terms.target_x_m,
                target_y_m=terms.target_y_m,
            )
        elif pursuit_reversal:
            reversal_fused = blend_pursuit_stanley(
                terms.pure_pursuit_rad,
                terms.stanley_rad,
                pure_pursuit_weight=float(
                    self.get_parameter(
                        "reversal_pure_pursuit_weight"
                    ).value
                ),
                opposed_stanley_weight=float(
                    self.get_parameter("opposed_stanley_weight").value
                ),
            )
            terms = SteeringTerms(
                pure_pursuit_rad=terms.pure_pursuit_rad,
                stanley_rad=terms.stanley_rad,
                fused_rad=reversal_fused,
                cross_track_error_m=terms.cross_track_error_m,
                heading_error_rad=terms.heading_error_rad,
                target_x_m=terms.target_x_m,
                target_y_m=terms.target_y_m,
            )
        elif stanley_reversal:
            pursuit_weight = clamp(
                float(
                    self.get_parameter(
                        "stanley_reversal_pure_pursuit_weight"
                    ).value
                ),
                0.0,
                1.0,
            )
            reversal_fused = (
                pursuit_weight * terms.pure_pursuit_rad
                + (1.0 - pursuit_weight) * terms.stanley_rad
            )
            terms = SteeringTerms(
                pure_pursuit_rad=terms.pure_pursuit_rad,
                stanley_rad=terms.stanley_rad,
                fused_rad=reversal_fused,
                cross_track_error_m=terms.cross_track_error_m,
                heading_error_rad=terms.heading_error_rad,
                target_x_m=terms.target_x_m,
                target_y_m=terms.target_y_m,
            )
        wheelbase = max(
            1.0e-3, float(self.get_parameter("wheel_base_m").value)
        )
        curvature = math.tan(terms.fused_rad) / wheelbase
        raw_command = interpolate_clamped(
            curvature, self.curvature_inputs, self.curvature_commands
        )
        return (
            clamp(
                raw_command, self.angle_command_min, self.angle_command_max
            ),
            terms,
        )

    def smooth_steering(self, raw_command: float, now: float) -> float:
        if not self.has_valid_command or self.last_command_time is None:
            return raw_command
        dt = max(0.0, now - self.last_command_time)
        return adaptive_smooth_steering_command(
            last_command=self.last_angle_command,
            raw_command=raw_command,
            dt=dt,
            straight_current_weight=float(
                self.get_parameter("steering_current_weight").value
            ),
            curve_current_weight=float(
                self.get_parameter("steering_curve_current_weight").value
            ),
            straight_rate_limit=float(
                self.get_parameter(
                    "steering_rate_limit_cmd_per_sec"
                ).value
            ),
            curve_rate_limit=float(
                self.get_parameter(
                    "steering_curve_rate_limit_cmd_per_sec"
                ).value
            ),
            curve_activation_command=float(
                self.get_parameter(
                    "steering_curve_activation_command"
                ).value
            ),
            curve_full_command=float(
                self.get_parameter("steering_curve_full_command").value
            ),
        )

    def speed_for_steering(self, angle_command: float) -> float:
        cruise = float(self.get_parameter("cruise_speed_command").value)
        minimum = float(self.get_parameter("minimum_speed_command").value)
        slowdown_angle = max(
            1.0,
            float(
                self.get_parameter("curve_slowdown_angle_command").value
            ),
        )
        curve_fraction = clamp(abs(angle_command) / slowdown_angle, 0.0, 1.0)
        return cruise + curve_fraction * (minimum - cruise)

    def on_timer(self) -> None:
        now = time.monotonic()
        if (
            bool(self.get_parameter("command_on_canonical").value)
            and self.last_canonical_command_time is not None
            and now - self.last_canonical_command_time
            < 1.5 * self.command_period_sec
        ):
            return
        self.issue_command(now)

    def issue_command(self, now: float) -> None:
        if self.path_valid and self.latest_path is not None:
            raw_angle, terms = self.steering_command_for_path(self.latest_path)
            controller_raw_angle = raw_angle
            (
                raw_angle,
                self.turn_transition_recovery_sign,
                self.turn_transition_recovery_until,
                self.turn_transition_armed_sign,
            ) = apply_turn_transition_recovery(
                raw_angle,
                now=now,
                armed_turn_sign=self.turn_transition_armed_sign,
                recovery_sign=self.turn_transition_recovery_sign,
                recovery_until=self.turn_transition_recovery_until,
                trigger_previous_command=float(
                    self.get_parameter(
                        "turn_transition_trigger_previous_command"
                    ).value
                ),
                trigger_new_command=float(
                    self.get_parameter(
                        "turn_transition_trigger_new_command"
                    ).value
                ),
                minimum_recovery_command=float(
                    self.get_parameter(
                        "turn_transition_minimum_recovery_command"
                    ).value
                ),
                hold_sec=float(
                    self.get_parameter("turn_transition_hold_sec").value
                ),
                cancel_opposed_command=float(
                    self.get_parameter(
                        "turn_transition_cancel_opposed_command"
                    ).value
                ),
            )
            raw_dt = (
                0.0
                if self.last_raw_command_time is None
                else max(0.0, now - self.last_raw_command_time)
            )
            compensated_angle = lead_compensated_steering_command(
                raw_angle,
                self.last_raw_angle_command,
                raw_dt,
                float(
                    self.get_parameter("steering_lead_time_sec").value
                ),
                float(
                    self.get_parameter(
                        "steering_max_lead_command"
                    ).value
                ),
            )
            compensated_angle = clamp(
                compensated_angle,
                self.angle_command_min,
                self.angle_command_max,
            )
            self.last_raw_angle_command = controller_raw_angle
            self.last_raw_command_time = now
            angle = self.smooth_steering(compensated_angle, now)
            speed = self.speed_for_steering(angle)
            self.latest_terms = terms
            self.has_valid_command = True
        elif (
            self.has_valid_command
            and bool(
                self.get_parameter(
                    "hold_last_steering_on_lane_loss"
                ).value
            )
        ):
            angle, speed = command_during_lane_loss(
                has_valid_command=self.has_valid_command,
                last_angle_command=self.last_angle_command,
                last_speed_command=self.last_speed_command,
                lane_loss_speed_command=float(
                    self.get_parameter("lane_loss_speed_command").value
                ),
                hold_last_steering=True,
                hold_last_speed=bool(
                    self.get_parameter("hold_last_speed_on_lane_loss").value
                ),
            )
            if not self.loss_announced:
                self.get_logger().warn(
                    "lane unavailable: holding the last steering command "
                    "without a time limit"
                )
                self.loss_announced = True
        else:
            angle = 0.0
            speed = 0.0

        if self.steering_only:
            speed = 0.0
        self.last_angle_command = float(angle)
        self.last_speed_command = float(speed)
        self.last_command_time = now
        self.publish_motor(angle, speed)
        self.publish_diagnostics()
        if self.latest_header is not None and self.latest_path is not None:
            self.publish_markers(self.latest_header, self.latest_path)

    def publish_motor(self, angle: float, speed: float) -> None:
        message = Float32MultiArray()
        message.data = [float(angle), float(speed)]
        self.shadow_motor_pub.publish(message)
        if self.motor_pub is not None:
            self.motor_pub.publish(message)
        if self.latest_header is not None:
            trace = TwistStamped()
            trace.header = self.latest_header
            trace.twist.angular.z = float(angle)
            trace.twist.linear.x = float(speed)
            self.action_trace_pub.publish(trace)

    def publish_path(self, header, path: np.ndarray, source: str) -> None:
        message = Centerline()
        message.header = header
        message.header.frame_id = self.base_frame_id
        message.detection_id = 0
        message.track_id = 0
        message.points = [
            make_point(float(forward), float(lateral), 0.0)
            for forward, lateral in path
        ]
        message.confidence = 1.0
        message.source = f"connected_{source}_reference"
        self.path_pub.publish(message)

    def publish_debug_image(
        self,
        source_message: Image,
        image: np.ndarray,
        white: np.ndarray,
        yellow: np.ndarray,
        connected: ConnectedPath | None,
        target_path: np.ndarray | None,
        path_source: str,
    ) -> None:
        if self.debug_image_pub.get_subscription_count() <= 0:
            return
        debug = image.copy()
        debug[white > 0] = (255, 255, 255)
        debug[yellow > 0] = (0, 220, 255)
        if connected is not None:
            height, width = debug.shape[:2]
            yellow_pixels = []
            for forward, lateral in connected.points:
                column = int(
                    round(
                        width * 0.5
                        - float(lateral) * width / self.lateral_range_m
                    )
                )
                row = int(
                    round(
                        height
                        - 1
                        - float(forward)
                        * (height - 1)
                        / self.forward_range_m
                    )
                )
                yellow_pixels.append((column, row))
            if len(yellow_pixels) >= 2:
                cv2.polylines(
                    debug,
                    [np.asarray(yellow_pixels, dtype=np.int32)],
                    False,
                    (0, 255, 80),
                    2,
                    cv2.LINE_AA,
                )
            target_pixels = []
            if target_path is not None:
                for forward, lateral in target_path:
                    column = int(
                        round(
                            width * 0.5
                            - float(lateral) * width / self.lateral_range_m
                        )
                    )
                    row = int(
                        round(
                            height
                            - 1
                            - float(forward)
                            * (height - 1)
                            / self.forward_range_m
                        )
                    )
                    target_pixels.append((column, row))
            if len(target_pixels) >= 2:
                cv2.polylines(
                    debug,
                    [np.asarray(target_pixels, dtype=np.int32)],
                    False,
                    (255, 180, 0),
                    2,
                    cv2.LINE_AA,
                )
        status = path_source.upper() if connected is not None else "HOLD"
        cv2.putText(
            debug,
            f"{status} steer={self.last_angle_command:.1f} "
            f"speed={self.last_speed_command:.1f}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 180, 0),
            1,
            cv2.LINE_AA,
        )
        output = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        output.header = source_message.header
        self.debug_image_pub.publish(output)

    def publish_diagnostics(self) -> None:
        info = self.latest_path_info
        terms = self.latest_terms
        message = Float32MultiArray()
        message.data = [
            1.0 if self.lane_visible else 0.0,
            1.0 if self.path_valid else 0.0,
            float(info.observation_count if info is not None else 0),
            float(info.forward_span_m if info is not None else 0.0),
            float(info.max_observation_gap_m if info is not None else 0.0),
            float(info.fit_residual_m if info is not None else -1.0),
            float(terms.pure_pursuit_rad if terms is not None else 0.0),
            float(terms.stanley_rad if terms is not None else 0.0),
            float(terms.fused_rad if terms is not None else 0.0),
            float(self.last_angle_command),
            float(self.last_speed_command),
            {"yellow": 1.0, "white": 2.0, "fused": 3.0}.get(
                self.latest_path_source, 0.0
            ),
        ]
        self.diagnostics_pub.publish(message)

    def publish_markers(self, header, path: np.ndarray) -> None:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header = header
        delete_all.header.frame_id = self.base_frame_id
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        line = Marker()
        line.header = delete_all.header
        line.ns = "connected_yellow_path"
        line.id = 1
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.035
        line.color.a = 1.0
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.25
        line.points = [
            make_point(float(forward), float(lateral), 0.02)
            for forward, lateral in path
        ]
        markers.markers.append(line)

        if self.latest_terms is not None:
            target = Marker()
            target.header = delete_all.header
            target.ns = "fused_lookahead"
            target.id = 2
            target.type = Marker.SPHERE
            target.action = Marker.ADD
            target.pose.orientation.w = 1.0
            target.pose.position = make_point(
                self.latest_terms.target_x_m,
                self.latest_terms.target_y_m,
                0.04,
            )
            target.scale.x = 0.08
            target.scale.y = 0.08
            target.scale.z = 0.08
            target.color.a = 1.0
            target.color.r = 0.1
            target.color.g = 0.55
            target.color.b = 1.0
            markers.markers.append(target)
        self.marker_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CanonicalStanleyPursuitDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_motor(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
