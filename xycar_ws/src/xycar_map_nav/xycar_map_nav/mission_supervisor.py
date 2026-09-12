"""Sensor-gated mission arbitration for global-path driving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence


AngleSector = tuple[float, float]


def angle_in_sector(
    angle_rad: float,
    sector: AngleSector,
    *,
    padding_rad: float = 0.0,
) -> bool:
    """Return whether an angle lies inside a non-wrapping camera sector."""
    padding = max(0.0, float(padding_rad))
    return (
        float(sector[0]) - padding
        <= float(angle_rad)
        <= float(sector[1]) + padding
    )


def lidar_target_matches_camera_sector(
    *,
    target_x_m: float,
    target_y_m: float,
    camera_sector: AngleSector | None,
    camera_sector_distance_m: float,
    excluded_sectors: Sequence[AngleSector] = (),
    angle_margin_rad: float = 0.0,
    distance_tolerance_m: float = 0.35,
) -> bool:
    """Associate a planar LiDAR target with one semantic camera target.

    Excluded sectors have priority. This is used to guarantee that a traffic
    light return cannot be promoted to a vehicle obstacle merely because it is
    the closest cluster on the driving path.
    """
    if camera_sector is None:
        return False
    x = float(target_x_m)
    y = float(target_y_m)
    if not math.isfinite(x) or not math.isfinite(y) or x <= 0.0:
        return False
    bearing = math.atan2(y, x)
    if any(angle_in_sector(bearing, sector) for sector in excluded_sectors):
        return False
    if not angle_in_sector(
        bearing,
        camera_sector,
        padding_rad=angle_margin_rad,
    ):
        return False
    expected_distance = float(camera_sector_distance_m)
    if not math.isfinite(expected_distance):
        return False
    target_distance = math.hypot(x, y)
    return abs(target_distance - expected_distance) <= max(
        0.0,
        float(distance_tolerance_m),
    )


def camera_box_lidar_sector(
    *,
    xmin: float,
    xmax: float,
    image_width: int,
    horizontal_fov_deg: float,
    padding_deg: float,
) -> tuple[float, float]:
    """Project a rectified camera box to a planar LiDAR angle interval."""
    if int(image_width) <= 0:
        raise ValueError("image_width must be positive")
    half_fov = 0.5 * math.radians(float(horizontal_fov_deg))
    padding = math.radians(max(0.0, float(padding_deg)))

    def pixel_to_angle(pixel_x: float) -> float:
        ratio = float(pixel_x) / float(image_width)
        return (0.5 - ratio) * (2.0 * half_fov)

    first = pixel_to_angle(float(xmin))
    second = pixel_to_angle(float(xmax))
    return min(first, second) - padding, max(first, second) + padding


def scan_sector_distance(
    *,
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    sector_min_angle: float,
    sector_max_angle: float,
    minimum_points: int,
    excluded_sectors: Sequence[AngleSector] = (),
) -> float:
    """Return the mean of the closest valid points in a LiDAR sector."""
    values = []
    for index, raw_value in enumerate(ranges):
        distance = float(raw_value)
        if (
            not math.isfinite(distance)
            or distance < float(range_min)
            or distance > float(range_max)
        ):
            continue
        angle = float(angle_min) + index * float(angle_increment)
        if (
            float(sector_min_angle) <= angle <= float(sector_max_angle)
            and not any(
                angle_in_sector(angle, sector)
                for sector in excluded_sectors
            )
        ):
            values.append(distance)
    required = max(1, int(minimum_points))
    if len(values) < required:
        return float("inf")
    values.sort()
    return sum(values[:required]) / required


class MissionMode(str, Enum):
    GLOBAL_PATH = "GLOBAL_PATH"
    TRAFFIC_STOP = "TRAFFIC_STOP"
    DYNAMIC_VEHICLE_RULE = "DYNAMIC_VEHICLE_RULE"
    CONE_RULE = "CONE_RULE"
    LIDAR_OBSTACLE_RULE = "LIDAR_OBSTACLE_RULE"
    LANE_INTERVENTION = "LANE_INTERVENTION"


@dataclass(frozen=True)
class MissionSupervisorConfig:
    semantic_timeout_sec: float = 0.75
    lidar_timeout_sec: float = 0.50
    cone_camera_required_frames: int = 2
    cone_camera_min_count: int = 1
    cone_camera_min_confidence: float = 0.50
    cone_lidar_min_count: int = 1
    cone_entry_distance_m: float = 0.50
    cone_minimum_duration_sec: float = 1.0
    cone_clear_hold_sec: float = 0.70
    cone_processing_hold_sec: float = 1.50
    vehicle_camera_required_frames: int = 2
    vehicle_camera_min_count: int = 1
    vehicle_camera_min_confidence: float = 0.45
    vehicle_entry_distance_m: float = 2.40
    vehicle_minimum_duration_sec: float = 0.50
    vehicle_clear_hold_sec: float = 0.50
    traffic_required_frames: int = 2
    traffic_control_enabled: bool = True
    lane_intervention_enabled: bool = False


@dataclass(frozen=True)
class MissionDecision:
    mode: MissionMode
    reason: str


class MissionSupervisor:
    """Choose the only active controller from asynchronous sensor evidence."""

    def __init__(self, config: MissionSupervisorConfig) -> None:
        self.config = config
        self.mode = MissionMode.GLOBAL_PATH
        self.mode_started_sec = 0.0

        self.cone_camera_frames = 0
        self.cone_camera_time = float("-inf")
        self.cone_camera_count = 0
        self.cone_camera_confidence = 0.0
        self.cone_lidar_time = float("-inf")
        self.cone_lidar_count = 0
        self.cone_lidar_distance_m = float("inf")
        self.cone_clear_started_sec: float | None = None

        self.vehicle_camera_frames = 0
        self.vehicle_camera_time = float("-inf")
        self.vehicle_camera_count = 0
        self.vehicle_camera_confidence = 0.0
        self.vehicle_lidar_distance_m = float("inf")
        self.vehicle_clear_started_sec: float | None = None

        self.traffic_stop_frames = 0
        self.traffic_green_frames = 0
        self.traffic_stop_latched = False
        self.traffic_color = "unknown"
        self.traffic_time = float("-inf")

        self.lane_risk = False
        self.lane_risk_time = float("-inf")

    def observe_objects(
        self,
        *,
        now_sec: float,
        cone_count: int,
        cone_max_confidence: float,
        vehicle_count: int,
        vehicle_max_confidence: float,
        vehicle_lidar_distance_m: float,
        traffic_color: str,
    ) -> None:
        now = float(now_sec)
        self.cone_camera_count = max(0, int(cone_count))
        self.cone_camera_confidence = max(
            0.0, float(cone_max_confidence)
        )
        cone_seen = (
            self.cone_camera_count
            >= self.config.cone_camera_min_count
            and self.cone_camera_confidence
            >= self.config.cone_camera_min_confidence
        )
        self.cone_camera_frames = (
            self.cone_camera_frames + 1 if cone_seen else 0
        )
        if cone_seen:
            self.cone_camera_time = now

        self.vehicle_camera_count = max(0, int(vehicle_count))
        self.vehicle_camera_confidence = max(
            0.0, float(vehicle_max_confidence)
        )
        self.vehicle_lidar_distance_m = float(vehicle_lidar_distance_m)
        vehicle_seen = (
            self.vehicle_camera_count
            >= self.config.vehicle_camera_min_count
            and self.vehicle_camera_confidence
            >= self.config.vehicle_camera_min_confidence
            and math.isfinite(self.vehicle_lidar_distance_m)
        )
        self.vehicle_camera_frames = (
            self.vehicle_camera_frames + 1 if vehicle_seen else 0
        )
        if vehicle_seen:
            self.vehicle_camera_time = now

        self._observe_traffic(now, traffic_color)

    def observe_cone_lidar(
        self,
        *,
        now_sec: float,
        count: int,
        nearest_distance_m: float,
    ) -> None:
        self.cone_lidar_time = float(now_sec)
        self.cone_lidar_count = max(0, int(count))
        self.cone_lidar_distance_m = float(nearest_distance_m)

    def cone_processing_requested(self, now_sec: float) -> bool:
        """Keep the expensive cone planner asleep until camera evidence exists."""
        camera_recent = (
            float(now_sec) - self.cone_camera_time
            <= self.config.cone_processing_hold_sec
        )
        return self.mode == MissionMode.CONE_RULE or camera_recent

    def observe_lane_risk(self, *, now_sec: float, risky: bool) -> None:
        self.lane_risk = bool(risky)
        self.lane_risk_time = float(now_sec)

    def _observe_traffic(self, now_sec: float, color: str) -> None:
        normalized = str(color).strip().lower()
        if normalized not in {"red", "yellow", "green"}:
            normalized = "unknown"
        self.traffic_color = normalized
        self.traffic_time = now_sec
        if normalized in {"red", "yellow"}:
            self.traffic_stop_frames += 1
            self.traffic_green_frames = 0
            if (
                self.traffic_stop_frames
                >= self.config.traffic_required_frames
            ):
                self.traffic_stop_latched = True
        elif normalized == "green":
            self.traffic_green_frames += 1
            self.traffic_stop_frames = 0
            if (
                self.traffic_green_frames
                >= self.config.traffic_required_frames
            ):
                self.traffic_stop_latched = False
        else:
            self.traffic_stop_frames = 0
            self.traffic_green_frames = 0

    def decide(
        self,
        *,
        now_sec: float,
        cone_command_ready: bool,
        dynamic_rule_mode: str,
        lidar_obstacle_rule_mode: str = "NORMAL",
    ) -> MissionDecision:
        now = float(now_sec)
        if (
            self.config.traffic_control_enabled
            and self.traffic_stop_latched
        ):
            return self._transition(
                MissionMode.TRAFFIC_STOP,
                now,
                f"traffic_{self.traffic_color}",
            )

        dynamic_ready = self._dynamic_entry_ready(now)
        dynamic_active = (
            self.mode == MissionMode.DYNAMIC_VEHICLE_RULE
            and not self._dynamic_exit_ready(now, dynamic_rule_mode)
        )
        if dynamic_ready or dynamic_active:
            reason = (
                "camera_vehicle_and_lidar_distance"
                if dynamic_ready
                else f"dynamic_rule_{dynamic_rule_mode.lower()}"
            )
            return self._transition(
                MissionMode.DYNAMIC_VEHICLE_RULE,
                now,
                reason,
            )

        cone_ready = (
            bool(cone_command_ready) and self._cone_entry_ready(now)
        )
        cone_active = (
            self.mode == MissionMode.CONE_RULE
            and not self._cone_exit_ready(now)
        )
        if cone_ready or cone_active:
            reason = (
                "camera_cone_and_lidar_distance"
                if cone_ready
                else "cone_evidence_hysteresis"
            )
            return self._transition(MissionMode.CONE_RULE, now, reason)

        if str(lidar_obstacle_rule_mode).upper() != "NORMAL":
            return self._transition(
                MissionMode.LIDAR_OBSTACLE_RULE,
                now,
                f"lidar_path_obstacle_{lidar_obstacle_rule_mode.lower()}",
            )

        lane_fresh = (
            now - self.lane_risk_time
            <= self.config.semantic_timeout_sec
        )
        if (
            self.config.lane_intervention_enabled
            and lane_fresh
            and self.lane_risk
        ):
            return self._transition(
                MissionMode.LANE_INTERVENTION,
                now,
                "predicted_two_wheel_lane_departure",
            )

        return self._transition(
            MissionMode.GLOBAL_PATH,
            now,
            "no_mission_override",
        )

    def _cone_entry_ready(self, now_sec: float) -> bool:
        camera_fresh = (
            now_sec - self.cone_camera_time
            <= self.config.semantic_timeout_sec
        )
        lidar_fresh = (
            now_sec - self.cone_lidar_time
            <= self.config.lidar_timeout_sec
        )
        return (
            camera_fresh
            and lidar_fresh
            and self.cone_camera_frames
            >= self.config.cone_camera_required_frames
            and self.cone_lidar_count >= self.config.cone_lidar_min_count
            and math.isfinite(self.cone_lidar_distance_m)
            and self.cone_lidar_distance_m
            <= self.config.cone_entry_distance_m
        )

    def _cone_exit_ready(self, now_sec: float) -> bool:
        if (
            now_sec - self.mode_started_sec
            < self.config.cone_minimum_duration_sec
        ):
            self.cone_clear_started_sec = None
            return False
        camera_clear = (
            now_sec - self.cone_camera_time
            > self.config.semantic_timeout_sec
        )
        lidar_clear = (
            now_sec - self.cone_lidar_time
            > self.config.lidar_timeout_sec
            or self.cone_lidar_count < self.config.cone_lidar_min_count
        )
        if not (camera_clear and lidar_clear):
            self.cone_clear_started_sec = None
            return False
        if self.cone_clear_started_sec is None:
            self.cone_clear_started_sec = now_sec
            return False
        return (
            now_sec - self.cone_clear_started_sec
            >= self.config.cone_clear_hold_sec
        )

    def _dynamic_entry_ready(self, now_sec: float) -> bool:
        return (
            now_sec - self.vehicle_camera_time
            <= self.config.semantic_timeout_sec
            and self.vehicle_camera_frames
            >= self.config.vehicle_camera_required_frames
            and math.isfinite(self.vehicle_lidar_distance_m)
            and self.vehicle_lidar_distance_m
            <= self.config.vehicle_entry_distance_m
        )

    def _dynamic_exit_ready(
        self, now_sec: float, dynamic_rule_mode: str
    ) -> bool:
        if (
            now_sec - self.mode_started_sec
            < self.config.vehicle_minimum_duration_sec
        ):
            self.vehicle_clear_started_sec = None
            return False
        camera_clear = (
            now_sec - self.vehicle_camera_time
            > self.config.semantic_timeout_sec
        )
        rule_clear = str(dynamic_rule_mode).upper() == "NORMAL"
        if not (camera_clear and rule_clear):
            self.vehicle_clear_started_sec = None
            return False
        if self.vehicle_clear_started_sec is None:
            self.vehicle_clear_started_sec = now_sec
            return False
        return (
            now_sec - self.vehicle_clear_started_sec
            >= self.config.vehicle_clear_hold_sec
        )

    def _transition(
        self,
        mode: MissionMode,
        now_sec: float,
        reason: str,
    ) -> MissionDecision:
        if mode != self.mode:
            self.mode = mode
            self.mode_started_sec = now_sec
            if mode != MissionMode.CONE_RULE:
                self.cone_clear_started_sec = None
            if mode != MissionMode.DYNAMIC_VEHICLE_RULE:
                self.vehicle_clear_started_sec = None
        return MissionDecision(mode=mode, reason=reason)
