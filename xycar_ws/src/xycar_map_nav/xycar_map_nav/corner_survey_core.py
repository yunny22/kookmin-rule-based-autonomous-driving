"""Geometry and LaserScan helpers for corner waypoint surveying."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


Point = tuple[float, float]


@dataclass(frozen=True)
class SectorMeasurement:
    distance_m: float
    point_count: int
    angular_span_rad: float

    @property
    def available(self) -> bool:
        return math.isfinite(self.distance_m) and self.point_count > 0


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def measure_sector(
    ranges: Sequence[float],
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    center_angle_rad: float,
    half_angle_rad: float,
    distance_quantile: float = 0.20,
) -> SectorMeasurement:
    """Return a robust near-distance and angular support for one scan sector."""
    samples: list[tuple[float, float]] = []
    half_angle = max(0.0, float(half_angle_rad))
    for index, raw_distance in enumerate(ranges):
        distance = float(raw_distance)
        if not math.isfinite(distance):
            continue
        if distance < float(range_min_m) or distance > float(range_max_m):
            continue
        angle = float(angle_min_rad) + index * float(angle_increment_rad)
        relative = normalize_angle(angle - float(center_angle_rad))
        if abs(relative) <= half_angle:
            samples.append((relative, distance))

    if not samples:
        return SectorMeasurement(float("inf"), 0, 0.0)

    distances = sorted(sample[1] for sample in samples)
    quantile = min(max(float(distance_quantile), 0.0), 1.0)
    quantile_index = int(round(quantile * (len(distances) - 1)))
    angles = [sample[0] for sample in samples]
    return SectorMeasurement(
        distance_m=distances[quantile_index],
        point_count=len(samples),
        angular_span_rad=max(angles) - min(angles),
    )


def waypoint_distances(
    points: Sequence[Point],
    *,
    closed: bool = False,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Return per-point incoming distance, cumulative distance, and total."""
    if not points:
        return (), (), 0.0

    incoming = [0.0]
    cumulative = [0.0]
    for previous, current in zip(points, points[1:]):
        distance = math.hypot(
            float(current[0]) - float(previous[0]),
            float(current[1]) - float(previous[1]),
        )
        incoming.append(distance)
        cumulative.append(cumulative[-1] + distance)

    total = cumulative[-1]
    if closed and len(points) > 1:
        total += math.hypot(
            float(points[0][0]) - float(points[-1][0]),
            float(points[0][1]) - float(points[-1][1]),
        )
    return tuple(incoming), tuple(cumulative), total
