"""Fixed-source command selector used by the integrated real-vehicle drive."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CandidateSource(str, Enum):
    RL = "RL"
    RULE = "RULE"


class HybridState(str, Enum):
    START_DELAY = "START_DELAY"
    RUNNING = "RUNNING"
    SENSOR_STOP = "SENSOR_STOP"


@dataclass(frozen=True)
class SequentialHybridConfig:
    initial_source: CandidateSource = CandidateSource.RULE
    start_delay_sec: float = 3.0
    candidate_fresh_sec: float = 0.35
    candidate_hold_sec: float = 0.75
    minimum_speed_command: float = 3.0
    maximum_speed_command: float = 30.0
    maximum_abs_angle_command: float = 42.0

    def __post_init__(self) -> None:
        if self.candidate_fresh_sec < 0.0:
            raise ValueError("candidate fresh timeout must be non-negative")
        if self.candidate_hold_sec < self.candidate_fresh_sec:
            raise ValueError("candidate hold must cover the fresh timeout")
        if self.maximum_speed_command < self.minimum_speed_command:
            raise ValueError("speed command range is reversed")


@dataclass(frozen=True)
class SequentialHybridInput:
    scan_fresh: bool
    rl_command_age_sec: float
    rl_angle_command: float
    rl_speed_command: float
    rule_command_age_sec: float
    rule_angle_command: float
    rule_speed_command: float


@dataclass(frozen=True)
class SequentialHybridOutput:
    state: HybridState
    source: CandidateSource
    angle_command: float
    speed_command: float
    reason: str


class SequentialHybridController:
    """Select one fixed base source; mission overrides live in the ROS node."""

    def __init__(self, config: SequentialHybridConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = HybridState.START_DELAY
        self.source = self.config.initial_source
        self.state_elapsed_sec = 0.0
        self.last_angle_command = 0.0
        self.last_speed_command = 0.0

    def _candidate(
        self,
        inputs: SequentialHybridInput,
    ) -> tuple[float, float, float]:
        if self.source == CandidateSource.RL:
            return (
                float(inputs.rl_angle_command),
                float(inputs.rl_speed_command),
                float(inputs.rl_command_age_sec),
            )
        return (
            float(inputs.rule_angle_command),
            float(inputs.rule_speed_command),
            float(inputs.rule_command_age_sec),
        )

    def _limit(self, angle: float, speed: float) -> tuple[float, float]:
        angle = min(
            max(float(angle), -self.config.maximum_abs_angle_command),
            self.config.maximum_abs_angle_command,
        )
        speed = float(speed)
        if speed > 0.0:
            speed = min(
                max(speed, self.config.minimum_speed_command),
                self.config.maximum_speed_command,
            )
        return angle, speed

    def _output(
        self,
        angle: float,
        speed: float,
        reason: str,
    ) -> SequentialHybridOutput:
        self.last_angle_command = float(angle)
        self.last_speed_command = float(speed)
        return SequentialHybridOutput(
            state=self.state,
            source=self.source,
            angle_command=float(angle),
            speed_command=float(speed),
            reason=reason,
        )

    def step(
        self,
        inputs: SequentialHybridInput,
        *,
        dt_sec: float,
    ) -> SequentialHybridOutput:
        dt = max(0.0, min(0.25, float(dt_sec)))
        self.state_elapsed_sec += dt

        if not inputs.scan_fresh:
            self.state = HybridState.SENSOR_STOP
            return self._output(0.0, 0.0, "LiDAR scan stale")

        angle, speed, candidate_age = self._candidate(inputs)
        if candidate_age > self.config.candidate_hold_sec:
            self.state = HybridState.SENSOR_STOP
            return self._output(0.0, 0.0, f"{self.source.value} command stale")
        if candidate_age > self.config.candidate_fresh_sec:
            return self._output(
                self.last_angle_command,
                self.last_speed_command,
                f"holding last {self.source.value} command",
            )

        if self.state == HybridState.START_DELAY:
            if self.state_elapsed_sec < self.config.start_delay_sec:
                return self._output(0.0, 0.0, "start delay")
            self.state = HybridState.RUNNING
            self.state_elapsed_sec = 0.0
        elif self.state == HybridState.SENSOR_STOP:
            self.state = HybridState.RUNNING
            self.state_elapsed_sec = 0.0

        angle, speed = self._limit(angle, speed)
        return self._output(angle, speed, self.source.value)
