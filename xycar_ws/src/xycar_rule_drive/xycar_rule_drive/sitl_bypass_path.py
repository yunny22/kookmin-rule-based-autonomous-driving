"""Local bypass path ported from ``gazebo_sitl`` PlanningCore.

The original planner blends the nominal path into the opposite lane before an
obstacle, holds that lane through the obstacle, and blends back afterwards.
This module keeps that geometry and scales only its vehicle-sized distances
from the 2.473 m SITL chassis to the 0.55 m Xycar.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SitlBypassConfig:
    vehicle_length_m: float = 0.55
    vehicle_width_m: float = 0.28
    sitl_vehicle_length_m: float = 2.473
    lane_center_separation_m: float = 0.40
    estimated_obstacle_length_m: float = 0.55
    minimum_obstacle_width_m: float = 0.28

    @property
    def longitudinal_scale(self) -> float:
        return max(1.0e-3, self.vehicle_length_m) / max(
            1.0e-3, self.sitl_vehicle_length_m
        )

    @property
    def post_obstacle_margin_m(self) -> float:
        return 1.5 * self.longitudinal_scale

    @property
    def return_distance_m(self) -> float:
        return 5.0 * self.longitudinal_scale

    @property
    def memory_clearance_m(self) -> float:
        return 5.0 * self.longitudinal_scale


@dataclass(frozen=True)
class RememberedObstacle:
    x: float
    y: float
    length: float
    width: float
    bypass_side: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


def smoothstep(value: float) -> float:
    t = clamp(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bypass_weight(
    x: float,
    approach_start: float,
    hold_start: float,
    hold_end: float,
    return_end: float,
) -> float:
    """The same approach/hold/return blend used by SITL PlanningCore."""
    if x < approach_start or x > return_end:
        return 0.0
    if x < hold_start:
        return smoothstep(
            (x - approach_start) / max(1.0e-6, hold_start - approach_start)
        )
    if x <= hold_end:
        return 1.0
    return 1.0 - smoothstep(
        (x - hold_end) / max(1.0e-6, return_end - hold_end)
    )


def bypass_transition_window(
    obstacle_front_x: float,
    config: SitlBypassConfig,
) -> tuple[float, float]:
    """Scale the SITL 6/1/0.8/1.5/1.2 m window by chassis length."""
    scale = config.longitudinal_scale
    near_threshold = 6.0 * scale
    if obstacle_front_x <= near_threshold:
        hold_start = max(1.0 * scale, obstacle_front_x - 0.8 * scale)
        return 0.0, hold_start

    approach_start = max(0.0, obstacle_front_x - 6.0 * scale)
    hold_start = max(
        approach_start + 1.5 * scale,
        obstacle_front_x - 1.2 * scale,
    )
    return approach_start, hold_start


def _offset_path_left(path: np.ndarray, offset_m: float) -> np.ndarray:
    """Offset a local path along its left normal, including curved sections."""
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return points.copy()
    tangent = np.gradient(points, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    norm = np.maximum(norm, 1.0e-6)
    left_normal = np.column_stack((-tangent[:, 1] / norm, tangent[:, 0] / norm))
    return points + float(offset_m) * left_normal


def blend_to_opposite_lane(
    base_path: np.ndarray,
    obstacle: RememberedObstacle,
    config: SitlBypassConfig,
) -> np.ndarray:
    """Build the SITL lane-center bypass for an Xycar-local target path."""
    base = np.asarray(base_path, dtype=np.float64)
    if base.ndim != 2 or base.shape[0] < 2 or base.shape[1] != 2:
        return base.copy()

    length = max(0.01, float(obstacle.length))
    obstacle_front_x = max(0.0, float(obstacle.x) - 0.5 * length)
    obstacle_rear_x = float(obstacle.x) + 0.5 * length
    approach_start, hold_start = bypass_transition_window(
        obstacle_front_x,
        config,
    )
    hold_end = obstacle_rear_x + config.post_obstacle_margin_m
    return_end = hold_end + config.return_distance_m

    side = 1.0 if float(obstacle.bypass_side) >= 0.0 else -1.0
    target_lane = _offset_path_left(
        base,
        side * config.lane_center_separation_m,
    )
    target_lane = target_lane[np.argsort(target_lane[:, 0])]
    target_y = np.interp(
        base[:, 0],
        target_lane[:, 0],
        target_lane[:, 1],
    )
    result = base.copy()
    for index, point in enumerate(base):
        weight = bypass_weight(
            float(point[0]),
            approach_start,
            hold_start,
            hold_end,
            return_end,
        )
        result[index, 1] = float(point[1]) + (
            float(target_y[index]) - float(point[1])
        ) * weight
    return result


class SitlBypassPathPlanner:
    """Remember an obstacle in the ego frame and generate its bypass path."""

    def __init__(self, config: SitlBypassConfig) -> None:
        self.config = config
        self.remembered_obstacle: RememberedObstacle | None = None

    def reset(self) -> None:
        self.remembered_obstacle = None

    def observe(
        self,
        *,
        active: bool,
        observation_valid: bool,
        bypass_side: float,
        obstacle_x: float,
        obstacle_y: float,
        obstacle_length: float,
        obstacle_width: float,
    ) -> None:
        if not active:
            self.reset()
            return
        if not observation_valid or not math.isfinite(float(obstacle_x)):
            return

        length = max(
            self.config.estimated_obstacle_length_m,
            float(obstacle_length),
        )
        width = max(
            self.config.minimum_obstacle_width_m,
            float(obstacle_width),
        )
        if float(obstacle_y) > 0.0:
            side = -1.0
        elif float(obstacle_y) < 0.0:
            side = 1.0
        else:
            side = 1.0 if float(bypass_side) >= 0.0 else -1.0
        self.remembered_obstacle = RememberedObstacle(
            x=max(0.0, float(obstacle_x)),
            y=float(obstacle_y),
            length=length,
            width=width,
            bypass_side=side,
        )

    def advance(self, distance_m: float) -> None:
        obstacle = self.remembered_obstacle
        if obstacle is None:
            return
        updated_x = float(obstacle.x) - max(0.0, float(distance_m))
        if updated_x < -(
            float(obstacle.length) + self.config.memory_clearance_m
        ):
            self.reset()
            return
        self.remembered_obstacle = RememberedObstacle(
            x=updated_x,
            y=obstacle.y,
            length=obstacle.length,
            width=obstacle.width,
            bypass_side=obstacle.bypass_side,
        )

    def make_path(self, base_path: np.ndarray) -> np.ndarray:
        obstacle = self.remembered_obstacle
        if obstacle is None:
            return np.asarray(base_path, dtype=np.float64).copy()
        return blend_to_opposite_lane(base_path, obstacle, self.config)

    @property
    def active(self) -> bool:
        return self.remembered_obstacle is not None
