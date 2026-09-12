"""One-shot YOLO trigger for the integrated shortcut override."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ShortcutModeEvent(str, Enum):
    NONE = "NONE"
    STARTED = "STARTED"
    FINISHED = "FINISHED"
    REARMED = "REARMED"


@dataclass(frozen=True)
class ShortcutModeConfig:
    minimum_confidence: float = 0.50
    required_frames: int = 2
    rearm_absence_sec: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("shortcut confidence must be in [0, 1]")
        if self.required_frames < 1:
            raise ValueError("shortcut required frames must be positive")
        if self.rearm_absence_sec < 0.0:
            raise ValueError("shortcut rearm absence must be non-negative")


class ShortcutModeLatch:
    """Latch one shortcut run until the candidate reports completion."""

    def __init__(self, config: ShortcutModeConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.armed = True
        self.detection_frames = 0
        self.last_detection_time = float("-inf")
        self.confidence = 0.0

    def observe(
        self,
        *,
        now_sec: float,
        detected: bool,
        confidence: float,
    ) -> ShortcutModeEvent:
        seen = bool(
            detected
            and float(confidence) >= self.config.minimum_confidence
        )
        if seen:
            self.last_detection_time = float(now_sec)
            self.confidence = float(confidence)
            if self.armed and not self.active:
                self.detection_frames += 1
        else:
            self.detection_frames = 0

        if (
            self.armed
            and not self.active
            and self.detection_frames >= self.config.required_frames
        ):
            self.active = True
            self.armed = False
            self.detection_frames = 0
            return ShortcutModeEvent.STARTED
        return self.update(now_sec=now_sec)

    def start(
        self,
        *,
        now_sec: float,
        confidence: float = 0.0,
    ) -> ShortcutModeEvent:
        """Start after an external traffic-light sequencer confirms entry.

        The original ``observe`` API remains available for standalone tests,
        but the integrated selector uses this explicit event only after
        left_4 has first been confirmed and then disappeared.
        """
        if self.active or not self.armed:
            return ShortcutModeEvent.NONE
        self.active = True
        self.armed = False
        self.detection_frames = 0
        self.last_detection_time = float(now_sec)
        self.confidence = float(confidence)
        return ShortcutModeEvent.STARTED

    def update(self, *, now_sec: float) -> ShortcutModeEvent:
        if (
            not self.active
            and not self.armed
            and float(now_sec) - self.last_detection_time
            >= self.config.rearm_absence_sec
        ):
            self.armed = True
            self.confidence = 0.0
            return ShortcutModeEvent.REARMED
        return ShortcutModeEvent.NONE

    def finish(self) -> ShortcutModeEvent:
        if not self.active:
            return ShortcutModeEvent.NONE
        self.active = False
        self.detection_frames = 0
        return ShortcutModeEvent.FINISHED
