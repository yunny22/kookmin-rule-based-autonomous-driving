"""Mission-scoped turn-transition bypass after the static red car."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class PostRedTurnExitEvent(str, Enum):
    NONE = "none"
    LEFT_TURN_CONFIRMED = "left_turn_confirmed"
    BYPASS_STARTED = "bypass_started"
    STRAIGHT_CONFIRMED = "straight_confirmed"
    BYPASS_TIMEOUT = "bypass_timeout"
    RESET = "reset"


@dataclass(frozen=True)
class PostRedTurnExitConfig:
    enabled: bool = True
    left_entry_command: float = -20.0
    left_entry_frames: int = 2
    exit_raw_minimum_command: float = -4.0
    exit_max_abs_curvature_per_m: float = 0.20
    minimum_path_points: int = 10
    minimum_path_confidence: float = 0.12
    exit_confirmation_frames: int = 2
    bypass_duration_sec: float = 0.50
    straight_max_abs_command: float = 5.0
    straight_confirmation_frames: int = 3


@dataclass(frozen=True)
class PostRedTurnExitState:
    window_active: bool
    left_turn_confirmed: bool
    left_frames: int
    exit_frames: int
    bypass_active: bool
    bypass_until: float
    straight_frames: int
    completed: bool


class PostRedTurnExitBypass:
    """Release only the stale left-turn latch as the 90-degree turn ends."""

    def __init__(self, config: PostRedTurnExitConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.window_active = False
        self.left_turn_confirmed = False
        self.left_frames = 0
        self.exit_frames = 0
        self.bypass_active = False
        self.bypass_until = 0.0
        self.straight_frames = 0
        self.completed = False

    def begin_cycle(
        self,
        *,
        now: float,
        window_active: bool,
        new_path_frame: bool,
        path_valid: bool,
        path_point_count: int,
        path_confidence: float,
        raw_command: float,
        curve_detection_per_m: float,
    ) -> tuple[bool, PostRedTurnExitEvent]:
        if not self.config.enabled or not bool(window_active):
            had_state = bool(
                self.window_active
                or self.left_turn_confirmed
                or self.bypass_active
                or self.completed
            )
            self.reset()
            return False, (
                PostRedTurnExitEvent.RESET
                if had_state
                else PostRedTurnExitEvent.NONE
            )

        if not self.window_active:
            self.reset()
            self.window_active = True

        if self.completed:
            return False, PostRedTurnExitEvent.NONE

        if self.bypass_active:
            if float(now) >= self.bypass_until:
                self.bypass_active = False
                self.completed = True
                return False, PostRedTurnExitEvent.BYPASS_TIMEOUT
            return True, PostRedTurnExitEvent.NONE

        if not bool(new_path_frame):
            return False, PostRedTurnExitEvent.NONE

        quality_valid = bool(
            path_valid
            and int(path_point_count)
            >= max(1, int(self.config.minimum_path_points))
            and math.isfinite(float(path_confidence))
            and float(path_confidence)
            >= max(0.0, float(self.config.minimum_path_confidence))
            and math.isfinite(float(raw_command))
            and math.isfinite(float(curve_detection_per_m))
        )

        event = PostRedTurnExitEvent.NONE
        if not self.left_turn_confirmed:
            left_evidence = bool(
                quality_valid
                and float(raw_command)
                <= min(0.0, float(self.config.left_entry_command))
            )
            self.left_frames = self.left_frames + 1 if left_evidence else 0
            if self.left_frames >= max(1, int(self.config.left_entry_frames)):
                self.left_turn_confirmed = True
                event = PostRedTurnExitEvent.LEFT_TURN_CONFIRMED

        exit_evidence = bool(
            self.left_turn_confirmed
            and quality_valid
            and float(raw_command)
            >= float(self.config.exit_raw_minimum_command)
            and abs(float(curve_detection_per_m))
            <= max(
                0.0,
                float(self.config.exit_max_abs_curvature_per_m),
            )
        )
        self.exit_frames = self.exit_frames + 1 if exit_evidence else 0
        if self.exit_frames >= max(
            1,
            int(self.config.exit_confirmation_frames),
        ):
            self.bypass_active = True
            self.bypass_until = float(now) + max(
                0.0,
                float(self.config.bypass_duration_sec),
            )
            self.straight_frames = 0
            return True, PostRedTurnExitEvent.BYPASS_STARTED
        return False, event

    def observe_output(
        self,
        *,
        new_path_frame: bool,
        path_valid: bool,
        output_command: float,
    ) -> PostRedTurnExitEvent:
        if not self.bypass_active or not bool(new_path_frame):
            return PostRedTurnExitEvent.NONE
        straight = bool(
            path_valid
            and math.isfinite(float(output_command))
            and abs(float(output_command))
            <= max(0.0, float(self.config.straight_max_abs_command))
        )
        self.straight_frames = self.straight_frames + 1 if straight else 0
        if self.straight_frames < max(
            1,
            int(self.config.straight_confirmation_frames),
        ):
            return PostRedTurnExitEvent.NONE
        self.bypass_active = False
        self.completed = True
        return PostRedTurnExitEvent.STRAIGHT_CONFIRMED

    def state(self) -> PostRedTurnExitState:
        return PostRedTurnExitState(
            window_active=self.window_active,
            left_turn_confirmed=self.left_turn_confirmed,
            left_frames=self.left_frames,
            exit_frames=self.exit_frames,
            bypass_active=self.bypass_active,
            bypass_until=self.bypass_until,
            straight_frames=self.straight_frames,
            completed=self.completed,
        )
