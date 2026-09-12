"""Three-lap traffic and shortcut policy for the competition course."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RaceLapEvent(str, Enum):
    NONE = "NONE"
    RACE_STARTED = "RACE_STARTED"
    NEXT_LAP_ARMED = "NEXT_LAP_ARMED"
    LAP_STARTED = "LAP_STARTED"
    SHORTCUT_COMPLETED = "SHORTCUT_COMPLETED"


@dataclass(frozen=True)
class RaceLapPolicyConfig:
    enabled: bool = False
    total_laps: int = 3
    signal_release_frames: int = 2


class RaceLapPolicy:
    """Track lap policy without relying on SLAM or odometry.

    The common S-curve handoff arms one lap transition.  The next new
    green/left traffic-light session consumes that transition, which avoids
    counting every detector frame as another lap.
    """

    def __init__(self, config: RaceLapPolicyConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.race_started = not bool(self.config.enabled)
        self.current_lap = 1
        self.next_lap_armed = False
        self.shortcut_completed = False
        self.signal_session_active = False
        self.signal_absence_frames = 0

    def observe_start_gate(self, *, green_release_confirmed: bool) -> RaceLapEvent:
        if self.race_started or not bool(green_release_confirmed):
            return RaceLapEvent.NONE
        self.race_started = True
        return RaceLapEvent.RACE_STARTED

    def observe_direction_signal(self, *, present: bool) -> RaceLapEvent:
        if bool(present):
            self.signal_absence_frames = 0
            if self.signal_session_active:
                return RaceLapEvent.NONE
            self.signal_session_active = True
            if not self.race_started or not self.next_lap_armed:
                return RaceLapEvent.NONE
            self.current_lap = min(
                max(1, int(self.config.total_laps)),
                self.current_lap + 1,
            )
            self.next_lap_armed = False
            return RaceLapEvent.LAP_STARTED

        if not self.signal_session_active:
            return RaceLapEvent.NONE
        self.signal_absence_frames += 1
        if self.signal_absence_frames >= max(
            1, int(self.config.signal_release_frames)
        ):
            self.signal_session_active = False
            self.signal_absence_frames = 0
        return RaceLapEvent.NONE

    def arm_next_lap(self) -> RaceLapEvent:
        if (
            not self.race_started
            or self.next_lap_armed
            or self.current_lap >= max(1, int(self.config.total_laps))
        ):
            return RaceLapEvent.NONE
        self.next_lap_armed = True
        return RaceLapEvent.NEXT_LAP_ARMED

    def mark_shortcut_completed(self) -> RaceLapEvent:
        if self.shortcut_completed:
            return RaceLapEvent.NONE
        self.shortcut_completed = True
        return RaceLapEvent.SHORTCUT_COMPLETED

    @property
    def ignore_stop_signals(self) -> bool:
        return bool(self.config.enabled and self.race_started)

    @property
    def shortcut_allowed(self) -> bool:
        if not self.config.enabled:
            return True
        return bool(
            self.race_started
            and self.current_lap >= 2
            and not self.shortcut_completed
        )

    @property
    def force_straight(self) -> bool:
        return not self.shortcut_allowed

    @property
    def force_straight_reason(self) -> str:
        if not self.config.enabled:
            return "lap policy disabled"
        if not self.race_started:
            return "waiting for initial green start"
        if self.current_lap == 1:
            return "lap 1 always runs straight"
        if self.shortcut_completed:
            return "shortcut already completed on an earlier lap"
        return "shortcut perception allowed"
