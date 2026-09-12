"""Mission-scoped speed and steering guard for the first S-curve entry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class SCurveEntryTrigger(str, Enum):
    NONE = "none"
    SHORTCUT_EXIT = "shortcut_exit"
    GREEN_CAR_EXIT = "green_car_exit"
    RED_CAR_EXIT = "red_car_exit"


class SCurveEntryEvent(str, Enum):
    NONE = "none"
    STARTED = "started"
    CURVE_HANDOFF = "curve_handoff"
    RESET = "reset"


class PostRedTurnWindowEvent(str, Enum):
    NONE = "none"
    STARTED = "started"
    GREEN_CAR_STARTED = "green_car_started"
    TIMED_OUT = "timed_out"
    RESET = "reset"


@dataclass(frozen=True)
class PostRedTurnWindowState:
    active: bool = False
    started_at: float = float("-inf")


def update_post_red_turn_window(
    state: PostRedTurnWindowState,
    *,
    now: float,
    drive_armed: bool,
    red_return_active: bool,
    red_avoidance_completed: bool,
    green_avoidance_active: bool,
    maximum_duration_sec: float,
) -> tuple[PostRedTurnWindowState, PostRedTurnWindowEvent]:
    """Bound the mission window used by the post-red steering release."""
    if not bool(drive_armed):
        return PostRedTurnWindowState(), (
            PostRedTurnWindowEvent.RESET
            if state.active
            else PostRedTurnWindowEvent.NONE
        )
    if state.active and bool(green_avoidance_active):
        return PostRedTurnWindowState(), PostRedTurnWindowEvent.GREEN_CAR_STARTED
    if state.active and float(now) - float(state.started_at) >= max(
        0.0,
        float(maximum_duration_sec),
    ):
        return PostRedTurnWindowState(), PostRedTurnWindowEvent.TIMED_OUT
    if state.active:
        return state, PostRedTurnWindowEvent.NONE
    if bool(red_return_active) or bool(red_avoidance_completed):
        return (
            PostRedTurnWindowState(active=True, started_at=float(now)),
            PostRedTurnWindowEvent.STARTED,
        )
    return state, PostRedTurnWindowEvent.NONE


@dataclass(frozen=True)
class SCurveEntryGuardConfig:
    enabled: bool = True
    speed_cap_command: float = 20.0
    red_car_speed_cap_command: float = 13.0
    speed_cap_start_distance_m: float = 8.0
    straight_max_abs_angle_command: float = 5.0
    straight_confirmation_frames: int = 3
    minimum_curve_distance_m: float = 1.50
    curve_left_angle_command: float = -8.0
    curve_speed_margin_command: float = 1.00
    curve_confirmation_frames: int = 3
    overdue_distance_m: float = 4.50


@dataclass(frozen=True)
class SCurveEntryGuardState:
    active: bool
    trigger: SCurveEntryTrigger
    distance_m: float
    straight_ready: bool
    straight_frames: int
    curve_frames: int
    overdue: bool
    speed_cap_active: bool


def _vehicle_avoidance_completed(
    *,
    previous_controls_vehicle: bool,
    previous_target_class_name: str,
    controls_vehicle: bool,
    target_class_name: str,
) -> bool:
    """Detect a class-specific avoidance handoff back to RULE."""
    normalized = (
        str(previous_target_class_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return bool(
        previous_controls_vehicle
        and not controls_vehicle
        and normalized == str(target_class_name)
    )


def green_car_avoidance_completed(
    *,
    previous_controls_vehicle: bool,
    previous_target_class_name: str,
    controls_vehicle: bool,
) -> bool:
    """Detect only a real green-car avoidance handoff back to RULE."""
    return _vehicle_avoidance_completed(
        previous_controls_vehicle=previous_controls_vehicle,
        previous_target_class_name=previous_target_class_name,
        controls_vehicle=controls_vehicle,
        target_class_name="green_car",
    )


def red_car_avoidance_completed(
    *,
    previous_controls_vehicle: bool,
    previous_target_class_name: str,
    controls_vehicle: bool,
) -> bool:
    """Detect only a real red-car avoidance handoff back to RULE."""
    return _vehicle_avoidance_completed(
        previous_controls_vehicle=previous_controls_vehicle,
        previous_target_class_name=previous_target_class_name,
        controls_vehicle=controls_vehicle,
        target_class_name="red_car",
    )


class SCurveEntryGuard:
    """Keep post-mission RULE driving slow until the first left curve owns it.

    The shortcut exits while steering left, so a residual left command alone
    must never release this guard.  It first waits for a stable straight RULE
    candidate, then for the later S-entry left curve at the measured distance.
    """

    def __init__(self, config: SCurveEntryGuardConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> SCurveEntryEvent:
        was_active = bool(getattr(self, "active", False))
        self.active = False
        self.trigger = SCurveEntryTrigger.NONE
        self.distance_m = 0.0
        self.straight_ready = False
        self.straight_frames = 0
        self.curve_frames = 0
        self.overdue = False
        self.speed_cap_active = False
        return SCurveEntryEvent.RESET if was_active else SCurveEntryEvent.NONE

    def start(self, trigger: SCurveEntryTrigger) -> SCurveEntryEvent:
        if not self.config.enabled or trigger == SCurveEntryTrigger.NONE:
            return SCurveEntryEvent.NONE
        self.active = True
        self.trigger = trigger
        self.distance_m = 0.0
        self.straight_ready = False
        self.straight_frames = 0
        self.curve_frames = 0
        self.overdue = False
        self.speed_cap_active = trigger == SCurveEntryTrigger.RED_CAR_EXIT
        return SCurveEntryEvent.STARTED

    def update(
        self,
        *,
        dt_sec: float,
        vehicle_speed_mps: float,
        vehicle_speed_fresh: bool,
        rule_angle_command: float,
        rule_speed_command: float,
        rule_command_fresh: bool,
    ) -> SCurveEntryEvent:
        if not self.active:
            return SCurveEntryEvent.NONE

        dt = max(0.0, min(0.25, float(dt_sec)))
        if (
            vehicle_speed_fresh
            and math.isfinite(float(vehicle_speed_mps))
        ):
            self.distance_m += max(0.0, float(vehicle_speed_mps)) * dt
        self.overdue = self.distance_m >= max(
            0.0, float(self.config.overdue_distance_m)
        )

        speed_cap_command = self.active_speed_cap_command()
        curve_speed_limit = speed_cap_command + max(
            0.0, float(self.config.curve_speed_margin_command)
        )
        if not self.speed_cap_active:
            curve_speed_detected = bool(
                rule_command_fresh
                and math.isfinite(float(rule_speed_command))
                and 0.0 < float(rule_speed_command) <= curve_speed_limit
            )
            distance_gate_reached = self.distance_m >= max(
                0.0, float(self.config.speed_cap_start_distance_m)
            )
            self.speed_cap_active = bool(
                curve_speed_detected or distance_gate_reached
            )
        straight_evidence = bool(
            rule_command_fresh
            and math.isfinite(float(rule_angle_command))
            and math.isfinite(float(rule_speed_command))
            and abs(float(rule_angle_command))
            <= max(0.0, float(self.config.straight_max_abs_angle_command))
            and float(rule_speed_command)
            > speed_cap_command
            + max(0.0, float(self.config.curve_speed_margin_command))
        )
        if not self.straight_ready:
            self.straight_frames = (
                self.straight_frames + 1 if straight_evidence else 0
            )
            if self.straight_frames >= max(
                1, int(self.config.straight_confirmation_frames)
            ):
                self.straight_ready = True

        left_threshold = min(
            0.0, float(self.config.curve_left_angle_command)
        )
        red_car_exit = self.trigger == SCurveEntryTrigger.RED_CAR_EXIT
        curve_evidence = bool(
            self.straight_ready
            and (
                red_car_exit
                or self.distance_m
                >= max(0.0, float(self.config.minimum_curve_distance_m))
            )
            and rule_command_fresh
            and math.isfinite(float(rule_angle_command))
            and math.isfinite(float(rule_speed_command))
            and float(rule_angle_command) <= left_threshold
            and (
                red_car_exit
                or 0.0 < float(rule_speed_command) <= curve_speed_limit
            )
        )
        self.curve_frames = self.curve_frames + 1 if curve_evidence else 0
        if self.curve_frames < max(
            1, int(self.config.curve_confirmation_frames)
        ):
            return SCurveEntryEvent.NONE

        self.active = False
        return SCurveEntryEvent.CURVE_HANDOFF

    def limit_command(
        self,
        *,
        angle_command: float,
        speed_command: float,
    ) -> tuple[float, float]:
        """Apply only the mission-scoped speed cap; preserve RULE steering."""
        angle = float(angle_command)
        speed = float(speed_command)
        if not self.active:
            return angle, speed

        if self.speed_cap_active and speed > 0.0:
            speed = min(
                speed,
                self.active_speed_cap_command(),
            )
        return angle, speed

    def active_speed_cap_command(self) -> float:
        if self.trigger == SCurveEntryTrigger.RED_CAR_EXIT:
            return max(0.0, float(self.config.red_car_speed_cap_command))
        return max(0.0, float(self.config.speed_cap_command))

    def state(self) -> SCurveEntryGuardState:
        return SCurveEntryGuardState(
            active=self.active,
            trigger=self.trigger,
            distance_m=self.distance_m,
            straight_ready=self.straight_ready,
            straight_frames=self.straight_frames,
            curve_frames=self.curve_frames,
            overdue=self.overdue,
            speed_cap_active=self.speed_cap_active,
        )
