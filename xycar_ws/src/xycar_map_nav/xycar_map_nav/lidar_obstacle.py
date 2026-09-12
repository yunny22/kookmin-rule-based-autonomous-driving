"""Route-aware LiDAR obstacle detection and temporary bypass control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Sequence


Point2 = tuple[float, float]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(maximum), float(value)))


@dataclass(frozen=True)
class LidarPathObstacleConfig:
    detect_distance_m: float = 1.50
    minimum_distance_m: float = 0.18
    path_corridor_half_width_m: float = 0.18
    minimum_cluster_points: int = 3
    maximum_scan_index_gap: int = 2
    maximum_cluster_gap_m: float = 0.16
    minimum_cluster_width_m: float = 0.09
    maximum_cluster_width_m: float = 0.70
    side_probe_inner_m: float = 0.18
    side_probe_outer_m: float = 0.55
    lidar_x_m: float = 0.065
    lidar_y_m: float = 0.0
    lidar_yaw_rad: float = 0.0


@dataclass(frozen=True)
class LidarPathObstacle:
    distance_m: float
    lateral_m: float
    width_m: float
    point_count: int
    x_vehicle_m: float
    y_vehicle_m: float
    left_clearance_m: float
    right_clearance_m: float


@dataclass(frozen=True)
class LidarBypassConfig:
    required_frames: int = 2
    left_offset_m: float = 0.28
    right_offset_m: float = 0.31
    offset_rate_mps: float = 0.45
    speed_limit_command: float = 4.0
    estimated_obstacle_length_m: float = 0.35
    post_obstacle_margin_m: float = 0.45
    clear_hold_sec: float = 0.25
    return_deadband_m: float = 0.02
    centered_lateral_deadband_m: float = 0.04


@dataclass(frozen=True)
class LidarBypassState:
    mode: str
    lateral_offset_m: float
    speed_limit_command: float | None
    remaining_clear_distance_m: float


@dataclass(frozen=True)
class _ProjectedPoint:
    scan_index: int
    x: float
    y: float
    route_s: float
    route_lateral: float


def _vehicle_point(
    point: Point2,
    *,
    vehicle_x: float,
    vehicle_y: float,
    vehicle_yaw: float,
) -> Point2:
    dx = float(point[0]) - float(vehicle_x)
    dy = float(point[1]) - float(vehicle_y)
    cosine = math.cos(float(vehicle_yaw))
    sine = math.sin(float(vehicle_yaw))
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def _local_route(
    points: Sequence[Point2],
    nearest_index: int,
    *,
    vehicle_x: float,
    vehicle_y: float,
    vehicle_yaw: float,
    maximum_distance_m: float,
    closed: bool,
) -> list[Point2]:
    if len(points) < 2:
        return []
    count = len(points)
    index = max(0, min(int(nearest_index), count - 1))
    local = [
        _vehicle_point(
            points[index],
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
    ]
    distance = 0.0
    for step in range(1, count + 1):
        next_index = index + step
        if not closed and next_index >= count:
            break
        next_index %= count
        previous = local[-1]
        current = _vehicle_point(
            points[next_index],
            vehicle_x=vehicle_x,
            vehicle_y=vehicle_y,
            vehicle_yaw=vehicle_yaw,
        )
        distance += math.hypot(
            current[0] - previous[0],
            current[1] - previous[1],
        )
        local.append(current)
        if distance >= float(maximum_distance_m):
            break
    return local


def _project_to_polyline(
    point: Point2,
    polyline: Sequence[Point2],
) -> tuple[float, float, float]:
    best_distance = float("inf")
    best_s = 0.0
    best_lateral = 0.0
    accumulated = 0.0
    for first, second in zip(polyline, polyline[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if length <= 1.0e-6:
            continue
        px = point[0] - first[0]
        py = point[1] - first[1]
        ratio = clamp((px * dx + py * dy) / (length * length), 0.0, 1.0)
        projection_x = first[0] + ratio * dx
        projection_y = first[1] + ratio * dy
        error_x = point[0] - projection_x
        error_y = point[1] - projection_y
        distance = math.hypot(error_x, error_y)
        if distance < best_distance:
            best_distance = distance
            best_s = accumulated + ratio * length
            best_lateral = (dx * py - dy * px) / length
        accumulated += length
    return best_s, best_lateral, best_distance


def _cluster_points(
    points: Sequence[_ProjectedPoint],
    config: LidarPathObstacleConfig,
) -> list[list[_ProjectedPoint]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda item: item.scan_index)
    groups: list[list[_ProjectedPoint]] = [[ordered[0]]]
    for point in ordered[1:]:
        previous = groups[-1][-1]
        index_gap = point.scan_index - previous.scan_index
        spatial_gap = math.hypot(
            point.x - previous.x,
            point.y - previous.y,
        )
        if (
            index_gap <= config.maximum_scan_index_gap
            and spatial_gap <= config.maximum_cluster_gap_m
        ):
            groups[-1].append(point)
        else:
            groups.append([point])
    return groups


def _cluster_width(points: Sequence[_ProjectedPoint]) -> float:
    if len(points) < 2:
        return 0.0
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def detect_path_obstacle(
    *,
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    route_points: Sequence[Point2],
    nearest_index: int,
    vehicle_x: float,
    vehicle_y: float,
    vehicle_yaw: float,
    closed: bool,
    config: LidarPathObstacleConfig,
) -> LidarPathObstacle | None:
    """Find the nearest multi-ray obstacle intersecting the global route."""
    local_route = _local_route(
        route_points,
        nearest_index,
        vehicle_x=vehicle_x,
        vehicle_y=vehicle_y,
        vehicle_yaw=vehicle_yaw,
        maximum_distance_m=config.detect_distance_m + 0.50,
        closed=closed,
    )
    if len(local_route) < 2:
        return None

    cosine = math.cos(config.lidar_yaw_rad)
    sine = math.sin(config.lidar_yaw_rad)
    candidates: list[_ProjectedPoint] = []
    all_front_points: list[Point2] = []
    for index, raw_range in enumerate(ranges):
        distance = float(raw_range)
        if (
            not math.isfinite(distance)
            or distance < max(float(range_min), config.minimum_distance_m)
            or distance > min(float(range_max), config.detect_distance_m + 0.75)
        ):
            continue
        angle = float(angle_min) + index * float(angle_increment)
        lidar_x = distance * math.cos(angle)
        lidar_y = distance * math.sin(angle)
        x = config.lidar_x_m + cosine * lidar_x - sine * lidar_y
        y = config.lidar_y_m + sine * lidar_x + cosine * lidar_y
        if x <= 0.0:
            continue
        all_front_points.append((x, y))
        route_s, lateral, path_distance = _project_to_polyline(
            (x, y), local_route
        )
        if (
            route_s < config.minimum_distance_m
            or route_s > config.detect_distance_m
            or path_distance > config.path_corridor_half_width_m
        ):
            continue
        candidates.append(
            _ProjectedPoint(
                scan_index=index,
                x=x,
                y=y,
                route_s=route_s,
                route_lateral=lateral,
            )
        )

    valid_clusters = []
    for cluster in _cluster_points(candidates, config):
        width = _cluster_width(cluster)
        if (
            len(cluster) >= config.minimum_cluster_points
            and config.minimum_cluster_width_m
            <= width
            <= config.maximum_cluster_width_m
        ):
            valid_clusters.append((min(p.route_s for p in cluster), width, cluster))
    if not valid_clusters:
        return None

    distance_m, width_m, cluster = min(
        valid_clusters, key=lambda item: item[0]
    )
    left_clearance = min(
        (
            x
            for x, y in all_front_points
            if config.side_probe_inner_m
            <= y
            <= config.side_probe_outer_m
        ),
        default=config.detect_distance_m,
    )
    right_clearance = min(
        (
            x
            for x, y in all_front_points
            if -config.side_probe_outer_m
            <= y
            <= -config.side_probe_inner_m
        ),
        default=config.detect_distance_m,
    )
    return LidarPathObstacle(
        distance_m=float(distance_m),
        lateral_m=float(median(p.route_lateral for p in cluster)),
        width_m=float(width_m),
        point_count=len(cluster),
        x_vehicle_m=float(median(p.x for p in cluster)),
        y_vehicle_m=float(median(p.y for p in cluster)),
        left_clearance_m=float(left_clearance),
        right_clearance_m=float(right_clearance),
    )


def forward_route_distance(
    points: Sequence[Point2],
    start_index: int,
    end_index: int,
    *,
    closed: bool,
) -> float:
    """Return forward arc distance while rejecting small backward index jitter."""
    if len(points) < 2:
        return 0.0
    count = len(points)
    start = max(0, min(int(start_index), count - 1))
    end = max(0, min(int(end_index), count - 1))
    if start == end:
        return 0.0
    if not closed and end < start:
        return 0.0

    distance = 0.0
    index = start
    for _ in range(count):
        if index == end:
            break
        next_index = index + 1
        if next_index >= count:
            if not closed:
                break
            next_index = 0
        distance += math.hypot(
            points[next_index][0] - points[index][0],
            points[next_index][1] - points[index][1],
        )
        index = next_index
    if closed:
        total = sum(
            math.hypot(
                points[(index + 1) % count][0] - points[index][0],
                points[(index + 1) % count][1] - points[index][1],
            )
            for index in range(count)
        )
        if distance > 0.5 * total:
            return 0.0
    return distance


class LidarObstacleBypassRule:
    """Remember an obstacle by route progress and blend back after passing it."""

    def __init__(self, config: LidarBypassConfig) -> None:
        self.config = config
        self.mode = "NORMAL"
        self.current_offset_m = 0.0
        self.seen_frames = 0
        self.entry_path_index: int | None = None
        self.clear_distance_m = 0.0
        self.last_seen_sec = float("-inf")
        self.last_update_sec: float | None = None

    def reset(self) -> None:
        self.mode = "NORMAL"
        self.current_offset_m = 0.0
        self.seen_frames = 0
        self.entry_path_index = None
        self.clear_distance_m = 0.0
        self.last_seen_sec = float("-inf")

    def update(
        self,
        *,
        now_sec: float,
        observation: LidarPathObstacle | None,
        path_index: int,
        route_points: Sequence[Point2],
        closed: bool,
        enabled: bool,
    ) -> LidarBypassState:
        now = float(now_sec)
        dt = (
            0.0
            if self.last_update_sec is None
            else clamp(now - self.last_update_sec, 0.0, 0.2)
        )
        self.last_update_sec = now
        if not enabled:
            self.reset()
            return self.state()

        if observation is None:
            self.seen_frames = 0
        else:
            self.seen_frames += 1
            self.last_seen_sec = now

        if (
            self.mode == "NORMAL"
            and observation is not None
            and self.seen_frames >= self.config.required_frames
        ):
            self.mode = self._choose_bypass_mode(observation)
            self.entry_path_index = int(path_index)
            self.clear_distance_m = self._clear_target(observation)

        travelled = 0.0
        if self.entry_path_index is not None:
            travelled = forward_route_distance(
                route_points,
                self.entry_path_index,
                path_index,
                closed=closed,
            )
        if self.mode in {"BYPASS_LEFT", "BYPASS_RIGHT"}:
            if observation is not None:
                self.clear_distance_m = max(
                    self.clear_distance_m,
                    travelled + self._clear_target(observation),
                )
            clear_by_progress = travelled >= self.clear_distance_m
            clear_by_sensor = (
                now - self.last_seen_sec >= self.config.clear_hold_sec
            )
            if clear_by_progress and clear_by_sensor:
                self.mode = "RETURN_CENTER"
                self.entry_path_index = None

        if self.mode == "BYPASS_LEFT":
            target_offset = self.config.left_offset_m
        elif self.mode == "BYPASS_RIGHT":
            target_offset = -self.config.right_offset_m
        else:
            target_offset = 0.0
        maximum_change = max(0.0, self.config.offset_rate_mps * dt)
        self.current_offset_m += clamp(
            target_offset - self.current_offset_m,
            -maximum_change,
            maximum_change,
        )
        if (
            self.mode == "RETURN_CENTER"
            and abs(self.current_offset_m)
            <= self.config.return_deadband_m
        ):
            self.mode = "NORMAL"
            self.current_offset_m = 0.0
            self.clear_distance_m = 0.0
        return self.state(travelled)

    def state(self, travelled_m: float = 0.0) -> LidarBypassState:
        active = self.mode != "NORMAL"
        return LidarBypassState(
            mode=self.mode,
            lateral_offset_m=self.current_offset_m,
            speed_limit_command=(
                self.config.speed_limit_command if active else None
            ),
            remaining_clear_distance_m=max(
                0.0, self.clear_distance_m - float(travelled_m)
            ),
        )

    def _choose_bypass_mode(
        self, observation: LidarPathObstacle
    ) -> str:
        deadband = self.config.centered_lateral_deadband_m
        if observation.lateral_m > deadband:
            return "BYPASS_RIGHT"
        if observation.lateral_m < -deadband:
            return "BYPASS_LEFT"
        if observation.left_clearance_m >= observation.right_clearance_m:
            return "BYPASS_LEFT"
        return "BYPASS_RIGHT"

    def _clear_target(self, observation: LidarPathObstacle) -> float:
        obstacle_length = max(
            self.config.estimated_obstacle_length_m,
            observation.width_m,
        )
        return (
            max(0.0, observation.distance_m)
            + obstacle_length
            + self.config.post_obstacle_margin_m
        )
