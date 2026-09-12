"""Sensor-presence latch for cone driving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class ConeModeEvent(str, Enum):
    NONE = "NONE"
    STARTED = "STARTED"
    FINISHED = "FINISHED"


class ConeReentryEvent(str, Enum):
    NONE = "NONE"
    STARTED = "STARTED"
    S_CURVE_REACHED = "S_CURVE_REACHED"


@dataclass(frozen=True)
class ConeModeConfig:
    entry_confidence: float = 0.35
    entry_frames: int = 3
    exit_frames: int = 1
    exit_absence_sec: float = 0.25
    # Forward x distance to the nearest camera-confirmed cone cluster.
    entry_distance_m: float = 0.95


class ConeModeLatch:
    """Enter on confirmed cone sensors and release only after both disappear.

    ``lidar_distance_m`` is the longitudinal (forward x) distance.  Keeping
    this gate longitudinal avoids granting steering authority too early for a
    cone that is laterally far from the vehicle.
    """

    def __init__(self, config: ConeModeConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.entry_streak = 0
        self.exit_streak = 0
        self.absence_started_sec: float | None = None

    def observe_command(
        self,
        *,
        confidence: float,
        speed_command: float,
        yolo_confirmed: bool,
        lidar_distance_m: float,
    ) -> ConeModeEvent:
        valid_entry = (
            bool(yolo_confirmed)
            and math.isfinite(float(lidar_distance_m))
            and float(lidar_distance_m) <= self.config.entry_distance_m
            and float(confidence) >= self.config.entry_confidence
            and float(speed_command) > 0.0
        )
        if self.active:
            return ConeModeEvent.NONE
        self.entry_streak = self.entry_streak + 1 if valid_entry else 0
        if self.entry_streak < max(1, int(self.config.entry_frames)):
            return ConeModeEvent.NONE
        self.active = True
        self.entry_streak = 0
        self.exit_streak = 0
        self.absence_started_sec = None
        return ConeModeEvent.STARTED

    def update_presence(
        self,
        *,
        sensor_present: bool,
        now_sec: float | None = None,
    ) -> ConeModeEvent:
        if not self.active:
            return ConeModeEvent.NONE
        if bool(sensor_present):
            self.exit_streak = 0
            self.absence_started_sec = None
            return ConeModeEvent.NONE

        self.exit_streak += 1
        if now_sec is not None and self.absence_started_sec is None:
            self.absence_started_sec = float(now_sec)
        if self.exit_streak < max(1, int(self.config.exit_frames)):
            return ConeModeEvent.NONE
        if now_sec is not None:
            now = float(now_sec)
            absence_sec = max(0.0, now - self.absence_started_sec)
            if absence_sec < max(0.0, float(self.config.exit_absence_sec)):
                return ConeModeEvent.NONE

        self.active = False
        self.exit_streak = 0
        self.absence_started_sec = None
        return ConeModeEvent.FINISHED


@dataclass(frozen=True)
class ConeReentrySuppressionConfig:
    enabled: bool = True
    straight_max_abs_angle_command: float = 5.0
    straight_confirmation_frames: int = 3
    s_curve_left_angle_command: float = -8.0
    s_curve_confirmation_frames: int = 3


class ConeReentrySuppression:
    """Block a second cone mission until normal RULE reaches the S curve."""

    def __init__(self, config: ConeReentrySuppressionConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.straight_ready = False
        self.straight_frames = 0
        self.curve_frames = 0

    def start(self) -> ConeReentryEvent:
        self.reset()
        if not self.config.enabled:
            return ConeReentryEvent.NONE
        self.active = True
        return ConeReentryEvent.STARTED

    def update(
        self,
        *,
        rule_controls_vehicle: bool,
        rule_command_fresh: bool,
        rule_angle_command: float,
    ) -> ConeReentryEvent:
        if not self.active:
            return ConeReentryEvent.NONE
        if not rule_controls_vehicle or not rule_command_fresh:
            self.curve_frames = 0
            return ConeReentryEvent.NONE

        angle = float(rule_angle_command)
        if not math.isfinite(angle):
            self.curve_frames = 0
            return ConeReentryEvent.NONE

        if not self.straight_ready:
            straight = abs(angle) <= max(
                0.0,
                float(self.config.straight_max_abs_angle_command),
            )
            self.straight_frames = self.straight_frames + 1 if straight else 0
            if self.straight_frames >= max(
                1,
                int(self.config.straight_confirmation_frames),
            ):
                self.straight_ready = True
            return ConeReentryEvent.NONE

        left_threshold = min(
            0.0,
            float(self.config.s_curve_left_angle_command),
        )
        self.curve_frames = self.curve_frames + 1 if angle <= left_threshold else 0
        if self.curve_frames < max(
            1,
            int(self.config.s_curve_confirmation_frames),
        ):
            return ConeReentryEvent.NONE

        self.reset()
        return ConeReentryEvent.S_CURVE_REACHED
