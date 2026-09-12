"""Pure traffic-light mission state for the integrated drive selector.

The object detector supplies only monocular 2-D boxes.  Box area divided by
image area is therefore used as a configurable proximity proxy; it is not a
metric distance estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrafficLightAction(str, Enum):
    CLEAR = "CLEAR"
    STOP = "STOP"
    LEFT_APPROACH = "LEFT_APPROACH"


@dataclass(frozen=True)
class SignalObservation:
    confidence: float = 0.0
    box_area_ratio: float = 0.0

    def qualifies(
        self,
        *,
        minimum_confidence: float,
        minimum_box_area_ratio: float = 0.0,
    ) -> bool:
        return (
            float(self.confidence) >= float(minimum_confidence)
            and float(self.box_area_ratio)
            >= float(minimum_box_area_ratio)
        )


@dataclass(frozen=True)
class TrafficLightFrame:
    red: SignalObservation = SignalObservation()
    yellow: SignalObservation = SignalObservation()
    green: SignalObservation = SignalObservation()
    left: SignalObservation = SignalObservation()


@dataclass(frozen=True)
class TrafficLightConfig:
    minimum_confidence: float = 0.50
    left_minimum_confidence: float = 0.50
    stop_min_box_area_ratio: float = 0.0
    go_min_box_area_ratio: float = 0.0
    stop_required_frames: int = 2
    go_required_frames: int = 2
    left_required_frames: int = 2
    left_absence_frames: int = 1
    left_start_delay_sec: float = 0.75

    def __post_init__(self) -> None:
        for value, label in (
            (self.minimum_confidence, "traffic-light confidence"),
            (self.left_minimum_confidence, "left_4 confidence"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1]")
        for value, label in (
            (self.stop_min_box_area_ratio, "stop box area"),
            (self.go_min_box_area_ratio, "go box area"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} ratio must be in [0, 1]")
        for value, label in (
            (self.stop_required_frames, "stop frames"),
            (self.go_required_frames, "go frames"),
            (self.left_required_frames, "left frames"),
            (self.left_absence_frames, "left absence frames"),
        ):
            if int(value) < 1:
                raise ValueError(f"{label} must be positive")
        if self.left_start_delay_sec < 0.0:
            raise ValueError("left start delay must be non-negative")


@dataclass(frozen=True)
class TrafficLightDecision:
    action: TrafficLightAction
    shortcut_start: bool
    signal_name: str
    box_area_ratio: float
    reason: str
    direction_finalized: bool = False
    cancel_shortcut: bool = False


def left_signal_approach_speed_limit(
    *,
    left_detection_frames: int,
    decision_action: TrafficLightAction,
    maximum_speed_command: float,
    blocked_by_active_mission: bool,
) -> float | None:
    """Cap RULE speed from the first valid left-arrow frame onward."""
    if blocked_by_active_mission:
        return None
    if (
        int(left_detection_frames) <= 0
        and decision_action != TrafficLightAction.LEFT_APPROACH
    ):
        return None
    return max(0.0, float(maximum_speed_command))


class TrafficLightController:
    """Latch stop signals and finalize turn direction at detector dropout.

    A close red or yellow remains latched through detector dropouts.  It is
    released only by consecutive close green or left-arrow observations.
    Green and left detections form one direction-observation session.  The
    latest qualifying YOLO class is retained while that class fluctuates, and
    the session is finalized only after consecutive detector dropouts.  A
    final left starts the shortcut sequence; a final green cancels any
    provisional shortcut preparation.
    """

    def __init__(self, config: TrafficLightConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.stop_latched = False
        self.stop_signal = ""
        self.stop_frames = 0
        self.go_frames = 0
        self.left_frames = 0
        self.left_confirmed = False
        self.left_absence_frames = 0
        self.left_last_seen_sec = float("-inf")
        self.left_disappeared_sec: float | None = None
        self.left_confidence = 0.0
        self.left_triggered = False
        self.direction_session_active = False
        self.direction_seen_frames = 0
        self.last_direction_signal = ""
        self.last_direction_box_area_ratio = 0.0
        self.direction_finalized = False
        self.latest_decision = TrafficLightDecision(
            action=TrafficLightAction.CLEAR,
            shortcut_start=False,
            signal_name="",
            box_area_ratio=0.0,
            reason="no traffic-light restriction",
        )

    def shortcut_finished(self) -> None:
        """Require a new left-arrow observation sequence for another run."""
        self.left_frames = 0
        self.left_confirmed = False
        self.left_absence_frames = 0
        self.left_last_seen_sec = float("-inf")
        self.left_disappeared_sec = None
        self.left_confidence = 0.0
        self.left_triggered = False
        self.direction_session_active = False
        self.direction_seen_frames = 0
        self.last_direction_signal = ""
        self.last_direction_box_area_ratio = 0.0
        self.direction_finalized = False
        self.latest_decision = TrafficLightDecision(
            action=TrafficLightAction.CLEAR,
            shortcut_start=False,
            signal_name="",
            box_area_ratio=0.0,
            reason="shortcut finished; normal arbitration restored",
        )

    def observe(
        self,
        *,
        now_sec: float,
        frame: TrafficLightFrame,
        shortcut_active: bool = False,
    ) -> TrafficLightDecision:
        # Once the left mission owns the vehicle, ShortcutCore remains latched
        # until its explicit T-exit completion.  Do not let a later background
        # traffic-light box partially advance or interrupt that timed mission.
        if shortcut_active:
            return self._decision(
                TrafficLightAction.CLEAR,
                signal_name="left_4",
                box_area_ratio=frame.left.box_area_ratio,
                reason="left_4 shortcut mission active",
            )

        red_close = frame.red.qualifies(
            minimum_confidence=self.config.minimum_confidence,
            minimum_box_area_ratio=self.config.stop_min_box_area_ratio,
        )
        yellow_close = frame.yellow.qualifies(
            minimum_confidence=self.config.minimum_confidence,
            minimum_box_area_ratio=self.config.stop_min_box_area_ratio,
        )
        green_close = frame.green.qualifies(
            minimum_confidence=self.config.minimum_confidence,
            minimum_box_area_ratio=self.config.go_min_box_area_ratio,
        )
        left_seen = frame.left.qualifies(
            minimum_confidence=self.config.left_minimum_confidence,
        )
        direction_signal = ""
        direction_observation = SignalObservation()
        if left_seen and green_close:
            left_score = (
                float(frame.left.confidence),
                float(frame.left.box_area_ratio),
            )
            green_score = (
                float(frame.green.confidence),
                float(frame.green.box_area_ratio),
            )
            if left_score >= green_score:
                direction_signal = "left_4"
                direction_observation = frame.left
            else:
                direction_signal = "green_4"
                direction_observation = frame.green
        elif left_seen:
            direction_signal = "left_4"
            direction_observation = frame.left
        elif green_close:
            direction_signal = "green_4"
            direction_observation = frame.green

        stop_seen = red_close or yellow_close
        if stop_seen:
            self.stop_frames += 1
            self.go_frames = 0
            candidate_name = "red_4" if red_close else "yellow_4"
            if self.stop_frames >= self.config.stop_required_frames:
                self.stop_latched = True
                self.stop_signal = candidate_name
        else:
            self.stop_frames = 0
            self.go_frames = self.go_frames + 1 if green_close else 0

        # Red/yellow in the same detector frame always wins, so a green release
        # happens only without simultaneous stop evidence.
        if (
            self.stop_latched
            and not stop_seen
            and self.go_frames >= self.config.go_required_frames
        ):
            self.stop_latched = False
            self.stop_signal = ""

        if stop_seen:
            # Do not arm a turn from contradictory simultaneous detections.
            self._clear_direction_session(clear_trigger=True)
        elif direction_signal:
            if not self.direction_session_active:
                self.direction_session_active = True
                self.direction_seen_frames = 0
                self.left_frames = 0
                self.left_confirmed = False
                self.left_triggered = False
            self.direction_seen_frames += 1
            self.last_direction_signal = direction_signal
            self.direction_finalized = False
            self.last_direction_box_area_ratio = float(
                direction_observation.box_area_ratio
            )
            self.left_absence_frames = 0
            self.left_disappeared_sec = None
            if direction_signal == "left_4":
                self.left_last_seen_sec = float(now_sec)
                self.left_confidence = float(frame.left.confidence)
                self.left_frames += 1
                if self.left_frames >= self.config.left_required_frames:
                    self.left_confirmed = True
                    self.stop_latched = False
                    self.stop_signal = ""
        elif self.direction_session_active:
            self.left_absence_frames += 1
            if self.left_absence_frames == 1:
                self.left_disappeared_sec = float(now_sec)

        if self.stop_latched:
            observed = frame.red if self.stop_signal == "red_4" else frame.yellow
            return self._decision(
                TrafficLightAction.STOP,
                signal_name=self.stop_signal,
                box_area_ratio=observed.box_area_ratio,
                reason=f"close {self.stop_signal} latched",
            )
        if self.direction_session_active and direction_signal:
            return self._decision(
                TrafficLightAction.LEFT_APPROACH,
                signal_name=direction_signal,
                box_area_ratio=direction_observation.box_area_ratio,
                reason=(
                    "traffic direction provisional; final YOLO class="
                    f"{direction_signal}"
                ),
            )
        finalized_left_now = False
        if (
            self.direction_session_active
            and self.left_absence_frames >= self.config.left_absence_frames
        ):
            if self.last_direction_signal == "green_4":
                area = self.last_direction_box_area_ratio
                self._clear_direction_session(clear_trigger=True)
                return self._decision(
                    TrafficLightAction.CLEAR,
                    signal_name="green_4",
                    box_area_ratio=area,
                    reason=(
                        "final YOLO traffic direction green_4; "
                        "shortcut cancelled"
                    ),
                    direction_finalized=True,
                    cancel_shortcut=True,
                )
            finalized_left_now = not self.direction_finalized
            self.direction_finalized = True
            if not self.left_confirmed:
                signal_name = self.last_direction_signal
                area = self.last_direction_box_area_ratio
                self._clear_direction_session(clear_trigger=True)
                return self._decision(
                    TrafficLightAction.CLEAR,
                    signal_name=signal_name,
                    box_area_ratio=area,
                    reason=(
                        "final YOLO traffic direction lacked two left_4 "
                        "observations; shortcut cancelled"
                    ),
                    direction_finalized=True,
                    cancel_shortcut=True,
                )
        if self.left_confirmed:
            return self._left_approach_decision(
                now_sec=float(now_sec),
                box_area_ratio=self.last_direction_box_area_ratio,
                direction_finalized=finalized_left_now,
            )
        if green_close:
            return self._decision(
                TrafficLightAction.CLEAR,
                signal_name="green_4",
                box_area_ratio=frame.green.box_area_ratio,
                reason="green_4 permits normal arbitration",
            )
        return self._decision(
            TrafficLightAction.CLEAR,
            reason="no traffic-light restriction",
        )

    def update(
        self,
        *,
        now_sec: float,
        shortcut_active: bool = False,
    ) -> TrafficLightDecision:
        """Advance only the post-disappearance timer at control-loop rate."""
        if shortcut_active:
            return self._decision(
                TrafficLightAction.CLEAR,
                signal_name="left_4",
                reason="left_4 shortcut mission active",
            )
        if self.stop_latched:
            return self._decision(
                TrafficLightAction.STOP,
                signal_name=self.stop_signal,
                reason=f"close {self.stop_signal} latched",
            )
        if self.left_confirmed:
            return self._left_approach_decision(now_sec=float(now_sec))
        if self.latest_decision.shortcut_start:
            return self._decision(
                self.latest_decision.action,
                signal_name=self.latest_decision.signal_name,
                box_area_ratio=self.latest_decision.box_area_ratio,
                reason=self.latest_decision.reason,
            )
        return self.latest_decision

    def shortcut_start_rejected(self) -> None:
        """Allow the next control tick to retry a ready start request."""
        self.left_triggered = False

    def _clear_direction_session(self, *, clear_trigger: bool) -> None:
        self.left_frames = 0
        self.left_confirmed = False
        self.left_absence_frames = 0
        self.left_last_seen_sec = float("-inf")
        self.left_disappeared_sec = None
        self.left_confidence = 0.0
        if clear_trigger:
            self.left_triggered = False
        self.direction_session_active = False
        self.direction_seen_frames = 0
        self.last_direction_signal = ""
        self.last_direction_box_area_ratio = 0.0
        self.direction_finalized = False

    def _left_approach_decision(
        self,
        *,
        now_sec: float,
        box_area_ratio: float = 0.0,
        direction_finalized: bool = False,
    ) -> TrafficLightDecision:
        shortcut_start = bool(
            not self.stop_latched
            and self.left_confirmed
            and not self.left_triggered
            and self.left_disappeared_sec is not None
            and self.left_absence_frames
            >= self.config.left_absence_frames
            and float(now_sec) - self.left_disappeared_sec
            >= self.config.left_start_delay_sec
        )
        if shortcut_start:
            self.left_triggered = True
            reason = "left_4 disappearance delay elapsed; shortcut requested"
        elif self.left_disappeared_sec is None:
            reason = "left_4 confirmed; waiting for disappearance"
        elif self.left_absence_frames < self.config.left_absence_frames:
            reason = (
                "left_4 disappearance candidate; confirming "
                f"{self.left_absence_frames}/"
                f"{self.config.left_absence_frames}"
            )
        else:
            elapsed = max(0.0, float(now_sec) - self.left_disappeared_sec)
            remaining = max(0.0, self.config.left_start_delay_sec - elapsed)
            reason = (
                "left_4 disappeared; straight approach "
                f"remaining={remaining:.2f}s"
            )
        return self._decision(
            TrafficLightAction.LEFT_APPROACH,
            shortcut_start=shortcut_start,
            signal_name="left_4",
            box_area_ratio=box_area_ratio,
            reason=reason,
            direction_finalized=direction_finalized,
        )

    def _decision(
        self,
        action: TrafficLightAction,
        *,
        shortcut_start: bool = False,
        signal_name: str = "",
        box_area_ratio: float = 0.0,
        reason: str,
        direction_finalized: bool = False,
        cancel_shortcut: bool = False,
    ) -> TrafficLightDecision:
        self.latest_decision = TrafficLightDecision(
            action=action,
            shortcut_start=bool(shortcut_start),
            signal_name=str(signal_name),
            box_area_ratio=float(box_area_ratio),
            reason=str(reason),
            direction_finalized=bool(direction_finalized),
            cancel_shortcut=bool(cancel_shortcut),
        )
        return self.latest_decision
