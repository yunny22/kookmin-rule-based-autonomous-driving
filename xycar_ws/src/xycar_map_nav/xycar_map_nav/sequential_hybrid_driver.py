"""ROS node for ordered LiDAR-gate RL/rule command selection."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from lane_seg_control.white_lane_fitter import fit_yellow_centerline_reference
from my_rule_msgs.msg import ObjectDetectionArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger
from xycar_msgs.msg import XycarVescState

from .corner_survey_core import measure_sector, SectorMeasurement
from .cone_mode_latch import ConeModeConfig
from .cone_mode_latch import ConeModeEvent
from .cone_mode_latch import ConeModeLatch
from .cone_mode_latch import ConeReentryEvent
from .cone_mode_latch import ConeReentrySuppression
from .cone_mode_latch import ConeReentrySuppressionConfig
from .lidar_obstacle import detect_path_obstacle
from .lidar_obstacle import LidarPathObstacle
from .lidar_obstacle import LidarPathObstacleConfig
from .mission_supervisor import camera_box_lidar_sector
from .mission_supervisor import lidar_target_matches_camera_sector
from .mission_supervisor import scan_sector_distance
from .race_lap_policy import RaceLapEvent
from .race_lap_policy import RaceLapPolicy
from .race_lap_policy import RaceLapPolicyConfig
from .sequential_hybrid_core import CandidateSource
from .sequential_hybrid_core import HybridState
from .sequential_hybrid_core import SequentialHybridConfig
from .sequential_hybrid_core import SequentialHybridController
from .sequential_hybrid_core import SequentialHybridInput
from .shortcut_mode_latch import ShortcutModeConfig
from .shortcut_mode_latch import ShortcutModeEvent
from .shortcut_mode_latch import ShortcutModeLatch
from .s_curve_entry_guard import green_car_avoidance_completed
from .s_curve_entry_guard import PostRedTurnWindowEvent
from .s_curve_entry_guard import PostRedTurnWindowState
from .s_curve_entry_guard import red_car_avoidance_completed
from .s_curve_entry_guard import SCurveEntryEvent
from .s_curve_entry_guard import SCurveEntryGuard
from .s_curve_entry_guard import SCurveEntryGuardConfig
from .s_curve_entry_guard import SCurveEntryTrigger
from .s_curve_entry_guard import update_post_red_turn_window
from .traffic_light_control import SignalObservation
from .traffic_light_control import TrafficLightAction
from .traffic_light_control import TrafficLightConfig
from .traffic_light_control import TrafficLightController
from .traffic_light_control import TrafficLightFrame
from .traffic_light_control import left_signal_approach_speed_limit
from .space_drive_gate_core import GateCandidateMode
from .yolo_lidar_avoidance import ShortcutAvoidanceSuppression
from .yolo_lidar_avoidance import ShortcutAvoidanceSuppressionConfig
from .yolo_lidar_avoidance import green_car_retrigger_suppressed
from .yolo_lidar_avoidance import preferred_avoidance_mode_from_image_center
from .yolo_lidar_avoidance import update_green_car_retrigger_confirmation
from .yolo_lidar_avoidance import YoloLidarAvoidanceConfig
from .yolo_lidar_avoidance import YoloLidarAvoidanceController
from .yolo_lidar_avoidance import YoloLidarAvoidanceMode


STATE_CODES = {state: float(index) for index, state in enumerate(HybridState)}
SOURCE_CODES = {
    CandidateSource.RL: 0.0,
    CandidateSource.RULE: 1.0,
}
ANSI_YELLOW = "\033[93m"
ANSI_GREEN = "\033[92m"
ANSI_BLUE = "\033[94m"
ANSI_MAGENTA = "\033[95m"
ANSI_RESET = "\033[0m"


def cone_disarm_hold_active(
    *,
    now_sec: float,
    disarmed_since_sec: float,
    hold_sec: float,
) -> bool:
    """Return whether cone mission state should survive a SPACE stop."""
    if not math.isfinite(float(disarmed_since_sec)):
        return False
    elapsed = max(0.0, float(now_sec) - float(disarmed_since_sec))
    return elapsed <= max(0.0, float(hold_sec))


def command_timestamp_is_fresh(
    *,
    now_sec: float,
    command_time_sec: float,
    timeout_sec: float,
) -> bool:
    """Return whether a command timestamp is finite, ordered, and recent."""
    if not math.isfinite(float(command_time_sec)):
        return False
    age_sec = float(now_sec) - float(command_time_sec)
    return 0.0 <= age_sec <= max(0.0, float(timeout_sec))


def update_return_center_confirmation_frames(
    current_frames: int,
    *,
    return_center_active: bool,
    path_valid: bool,
    cross_track_error_m: float,
    maximum_abs_error_m: float,
) -> int:
    """Count consecutive valid RULE frames near the physical lane center."""
    valid = (
        bool(return_center_active)
        and bool(path_valid)
        and math.isfinite(float(cross_track_error_m))
        and abs(float(cross_track_error_m))
        <= max(0.0, float(maximum_abs_error_m))
    )
    return int(current_frames) + 1 if valid else 0


def cone_processing_requested(
    *,
    shortcut_active: bool,
    cone_active: bool,
    yolo_age_sec: float,
    yolo_timeout_sec: float,
    reentry_suppressed: bool = False,
) -> bool:
    """Allow camera-confirmed cone planning before the motor gate is armed."""
    return bool(
        not shortcut_active
        and not reentry_suppressed
        and (
            cone_active
            or 0.0 <= float(yolo_age_sec) <= max(
                0.0, float(yolo_timeout_sec)
            )
        )
    )


def shortcut_search_should_yield_to_cone(
    *,
    shortcut_entry_search_active: bool,
    shortcut_active: bool,
    cone_event: ConeModeEvent,
) -> bool:
    """Yield only a failed W1 search to a confirmed cone-course entry."""
    return bool(
        shortcut_entry_search_active
        and not shortcut_active
        and cone_event == ConeModeEvent.STARTED
    )


def selected_external_lateral_offset(
    *,
    shortcut_left_lane_active: bool,
    shortcut_left_lane_offset_m: float,
    avoidance_offset_m: float,
) -> float:
    """Give shortcut pre-positioning priority over vehicle avoidance offset."""
    if shortcut_left_lane_active:
        return max(0.0, float(shortcut_left_lane_offset_m))
    return float(avoidance_offset_m)


def shortcut_preposition_requested(
    *,
    action: TrafficLightAction,
    signal_name: str,
    shortcut_class_name: str,
    green_class_name: str,
    suppress_initial_green: bool,
) -> bool:
    """Preserve signal pre-positioning except during the first green release."""
    signal = str(signal_name)
    return bool(
        action == TrafficLightAction.LEFT_APPROACH
        and signal in {str(shortcut_class_name), str(green_class_name)}
        and not bool(suppress_initial_green)
    )


def cone_approach_speed_limit(
    *,
    enabled: bool,
    cone_active: bool,
    shortcut_active: bool,
    yolo_frames: int,
    yolo_age_sec: float,
    yolo_timeout_sec: float,
    confirmed_yolo_frames: int,
    yolo_confidence: float,
    strong_yolo_confidence: float,
    cluster_count: int,
    cluster_age_sec: float,
    cluster_timeout_sec: float,
    cluster_forward_distance_m: float,
    confirmed_distance_m: float,
    first_yolo_speed_command: float,
    confirmed_speed_command: float,
) -> tuple[float | None, str]:
    """Return a speed-only cone-entry cap before CONE takes steering.

    The first central camera detection starts an early deceleration while RULE
    continues to steer. The validated cone-course speed is withheld until a
    camera-gated LiDAR cluster is near the measured entry boundary. Steering
    authority remains protected by the independent cone latch.
    """
    if not enabled or cone_active or shortcut_active:
        return None, "none"
    yolo_fresh = bool(
        int(yolo_frames) > 0
        and 0.0 <= float(yolo_age_sec) <= max(0.0, float(yolo_timeout_sec))
    )
    cluster_fresh = bool(
        int(cluster_count) > 0
        and 0.0
        <= float(cluster_age_sec)
        <= max(0.0, float(cluster_timeout_sec))
    )
    cluster_near = bool(
        cluster_fresh
        and math.isfinite(float(cluster_forward_distance_m))
        and 0.0 < float(cluster_forward_distance_m)
        <= max(0.0, float(confirmed_distance_m))
    )
    if cluster_near:
        return max(0.0, float(confirmed_speed_command)), "near_cluster"
    if yolo_fresh:
        return max(0.0, float(first_yolo_speed_command)), "first_yolo"
    return None, "none"


def cone_approach_brake_decision(
    *,
    enabled: bool,
    cone_active: bool,
    shortcut_active: bool,
    cluster_count: int,
    cluster_age_sec: float,
    cluster_timeout_sec: float,
    cluster_forward_distance_m: float,
    vehicle_speed_mps: float,
    vehicle_speed_age_sec: float,
    vehicle_speed_timeout_sec: float,
    target_speed_mps: float,
    deceleration_mps2: float,
    response_time_sec: float,
    distance_margin_m: float,
    hard_stop_distance_m: float,
    stale_speed_stop_distance_m: float,
) -> tuple[bool, float, str]:
    """Decide whether pre-entry braking must be latched."""
    if not enabled or cone_active or shortcut_active:
        return False, 0.0, "inactive"
    cluster_fresh = bool(
        int(cluster_count) > 0
        and 0.0
        <= float(cluster_age_sec)
        <= max(0.0, float(cluster_timeout_sec))
        and math.isfinite(float(cluster_forward_distance_m))
    )
    if not cluster_fresh:
        return False, 0.0, "no_fresh_cluster"
    forward = max(0.0, float(cluster_forward_distance_m))
    hard_stop = max(0.0, float(hard_stop_distance_m))
    if forward <= hard_stop:
        return True, hard_stop, "hard_distance"
    speed_fresh = bool(
        math.isfinite(float(vehicle_speed_mps))
        and 0.0
        <= float(vehicle_speed_age_sec)
        <= max(0.0, float(vehicle_speed_timeout_sec))
    )
    if not speed_fresh:
        fallback = max(0.0, float(stale_speed_stop_distance_m))
        return forward <= fallback, fallback, "speed_stale"
    speed = abs(float(vehicle_speed_mps))
    target = max(0.0, float(target_speed_mps))
    deceleration = max(1.0e-6, float(deceleration_mps2))
    braking_distance = 0.0
    if speed > target:
        braking_distance = (speed * speed - target * target) / (
            2.0 * deceleration
        )
    required = (
        speed * max(0.0, float(response_time_sec))
        + braking_distance
        + max(0.0, float(distance_margin_m))
    )
    return forward <= required, required, "braking_distance"


def cone_brake_hold_release_ready(
    *,
    cone_active: bool,
    cone_command_fresh: bool,
    vehicle_speed_mps: float,
    vehicle_speed_age_sec: float,
    vehicle_speed_timeout_sec: float,
    release_speed_mps: float,
) -> bool:
    """Release a latched entry brake only with steering and safe speed."""
    return bool(
        cone_active
        and cone_command_fresh
        and math.isfinite(float(vehicle_speed_mps))
        and 0.0
        <= float(vehicle_speed_age_sec)
        <= max(0.0, float(vehicle_speed_timeout_sec))
        and abs(float(vehicle_speed_mps))
        <= max(0.0, float(release_speed_mps))
    )


def interpolate_command(
    target_angle_deg: float,
    actual_angles_deg: Sequence[float],
    commands: Sequence[float],
) -> float:
    """Convert a physical wheel angle to the existing Xycar servo command."""
    actual = np.asarray(actual_angles_deg, dtype=np.float64)
    command = np.asarray(commands, dtype=np.float64)
    if actual.size < 2 or actual.size != command.size:
        raise ValueError("steering conversion tables must have equal length >= 2")
    if np.any(np.diff(actual) <= 0.0):
        raise ValueError("physical steering angles must be strictly increasing")
    sign = -1.0 if float(target_angle_deg) < 0.0 else 1.0
    magnitude = abs(float(target_angle_deg))
    mapped = float(np.interp(magnitude, actual, command))
    return sign * mapped


def is_avoidance_detection(
    *,
    class_name: str,
    confidence: float,
    vehicle_names: set[str],
    vehicle_min_confidence: float,
    cone_as_vehicle_obstacle: bool,
    cone_min_confidence: float,
) -> bool:
    if class_name == "cone" and cone_as_vehicle_obstacle:
        return float(confidence) >= float(cone_min_confidence)
    return (
        class_name in vehicle_names
        and float(confidence) >= float(vehicle_min_confidence)
    )


def avoidance_speed_limit_for_vehicle_class(
    class_name: str,
    *,
    default_speed_limit_command: float,
    red_car_speed_limit_command: float,
    green_car_speed_limit_command: float,
) -> float:
    """Select the speed limit without merging the model's car classes."""
    normalized = (
        str(class_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if normalized == "red_car":
        return max(0.0, float(red_car_speed_limit_command))
    if normalized == "green_car":
        return max(0.0, float(green_car_speed_limit_command))
    return max(0.0, float(default_speed_limit_command))


def right_avoidance_settling_speed_limit(
    *,
    mode: YoloLidarAvoidanceMode,
    lateral_offset_m: float,
    right_offset_m: float,
    tolerance_m: float,
    speed_limit_command: float,
) -> float | None:
    """Cap AVOID_RIGHT until the requested lane offset is established."""
    if mode != YoloLidarAvoidanceMode.AVOID_RIGHT:
        return None
    target_offset_m = -max(0.0, float(right_offset_m))
    if abs(float(lateral_offset_m) - target_offset_m) <= max(
        0.0,
        float(tolerance_m),
    ):
        return None
    return max(0.0, float(speed_limit_command))


def shortcut_suppresses_vehicle_class(
    class_name: str,
    *,
    suppression_active: bool,
) -> bool:
    """Suppress every class that has already qualified as an obstacle."""
    del class_name
    return bool(suppression_active)


def preferred_avoidance_mode_from_yellow_reference(
    *,
    object_x: float,
    object_y: float,
    yellow_x_by_y,
    deadband_px: float,
) -> YoloLidarAvoidanceMode | None:
    if yellow_x_by_y is None or len(yellow_x_by_y) == 0:
        return None
    row = max(0, min(len(yellow_x_by_y) - 1, int(round(object_y))))
    divider_x = float(yellow_x_by_y[row])
    if not math.isfinite(divider_x):
        return None
    deadband = max(0.0, float(deadband_px))
    if float(object_x) < divider_x - deadband:
        return YoloLidarAvoidanceMode.AVOID_RIGHT
    if float(object_x) > divider_x + deadband:
        return YoloLidarAvoidanceMode.AVOID_LEFT
    return None


def straight_road_side_decision_allowed(
    *,
    rule_angle_command: float,
    rule_command_age_sec: float,
    yellow_reference_age_sec: float,
    yellow_reference_rmse_px: float,
    yellow_reference_span_ratio: float,
    maximum_rule_angle_command: float,
    rule_command_timeout_sec: float,
    yellow_reference_timeout_sec: float,
    maximum_yellow_rmse_px: float,
    minimum_yellow_span_ratio: float,
) -> bool:
    """Require fresh, nearly straight lane geometry before choosing a side."""
    values = (
        rule_angle_command,
        rule_command_age_sec,
        yellow_reference_age_sec,
        yellow_reference_rmse_px,
        yellow_reference_span_ratio,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return False
    return (
        abs(float(rule_angle_command))
        <= max(0.0, float(maximum_rule_angle_command))
        and 0.0 <= float(rule_command_age_sec)
        <= max(0.0, float(rule_command_timeout_sec))
        and 0.0 <= float(yellow_reference_age_sec)
        <= max(0.0, float(yellow_reference_timeout_sec))
        and float(yellow_reference_rmse_px)
        <= max(0.0, float(maximum_yellow_rmse_px))
        and float(yellow_reference_span_ratio)
        >= max(0.0, float(minimum_yellow_span_ratio))
    )


class SequentialHybridDriver(Node):
    def __init__(self) -> None:
        super().__init__("sequential_hybrid_driver")
        self._declare_parameters()
        self.drive_enabled = bool(self.get_parameter("drive_enabled").value)
        self.gate_arming_required = bool(
            self.get_parameter("gate_arming_required").value
        )
        self.force_rule_only = bool(
            self.get_parameter("force_rule_only").value
        )
        self.drive_armed = not self.gate_arming_required
        self.gate_disarmed_time = float("-inf")
        self.sector_centers_deg = self._float_array("sector_centers_deg")
        self.controller = SequentialHybridController(self._config())
        if self.force_rule_only:
            self.controller.source = CandidateSource.RULE
        self.rl_command = (0.0, 0.0)
        self.rule_command = (0.0, 0.0)
        self.rl_command_time = float("-inf")
        self.rule_command_time = float("-inf")
        self.rule_path_valid = False
        self.rule_cross_track_error_m = float("inf")
        self.rule_diagnostics_time = float("-inf")
        self.return_center_confirmation_frames = 0
        self.post_red_turn_window = PostRedTurnWindowState()
        self.shortcut_command = (0.0, 0.0, 0.0)
        self.shortcut_command_time = float("-inf")
        self.shortcut_phase_code = 0.0
        self.shortcut_entry_search_active = False
        self.shortcut_entry_search_started_time = float("-inf")
        self.shortcut_entry_ready = False
        self.shortcut_latch = ShortcutModeLatch(
            ShortcutModeConfig(
                minimum_confidence=float(
                    self.get_parameter("shortcut_yolo_min_confidence").value
                ),
                required_frames=int(
                    self.get_parameter("shortcut_yolo_required_frames").value
                ),
                rearm_absence_sec=float(
                    self.get_parameter("shortcut_rearm_absence_sec").value
                ),
            )
        )
        self.shortcut_avoidance_suppression = ShortcutAvoidanceSuppression(
            ShortcutAvoidanceSuppressionConfig(
                enabled=bool(
                    self.get_parameter(
                        "vehicle_avoidance_shortcut_suppression_enabled"
                    ).value
                ),
                release_left_angle_command=float(
                    self.get_parameter(
                        "vehicle_avoidance_shortcut_release_left_angle_command"
                    ).value
                ),
                release_required_frames=int(
                    self.get_parameter(
                        "vehicle_avoidance_shortcut_release_required_frames"
                    ).value
                ),
            )
        )
        self.green_car_retrigger_blocked = False
        self.green_car_retrigger_frames = 0
        self.s_curve_entry_guard = SCurveEntryGuard(
            SCurveEntryGuardConfig(
                enabled=bool(
                    self.get_parameter("s_curve_entry_guard_enabled").value
                ),
                speed_cap_command=float(
                    self.get_parameter(
                        "s_curve_entry_speed_cap_command"
                    ).value
                ),
                red_car_speed_cap_command=float(
                    self.get_parameter(
                        "s_curve_entry_red_car_speed_cap_command"
                    ).value
                ),
                speed_cap_start_distance_m=float(
                    self.get_parameter(
                        "s_curve_entry_speed_cap_start_distance_m"
                    ).value
                ),
                straight_max_abs_angle_command=float(
                    self.get_parameter(
                        "s_curve_entry_straight_max_abs_angle_command"
                    ).value
                ),
                straight_confirmation_frames=int(
                    self.get_parameter(
                        "s_curve_entry_straight_confirmation_frames"
                    ).value
                ),
                minimum_curve_distance_m=float(
                    self.get_parameter(
                        "s_curve_entry_minimum_curve_distance_m"
                    ).value
                ),
                curve_left_angle_command=float(
                    self.get_parameter(
                        "s_curve_entry_left_angle_command"
                    ).value
                ),
                curve_speed_margin_command=float(
                    self.get_parameter(
                        "s_curve_entry_curve_speed_margin_command"
                    ).value
                ),
                curve_confirmation_frames=int(
                    self.get_parameter(
                        "s_curve_entry_curve_confirmation_frames"
                    ).value
                ),
                overdue_distance_m=float(
                    self.get_parameter(
                        "s_curve_entry_overdue_distance_m"
                    ).value
                ),
            )
        )
        self.traffic_light_controller = TrafficLightController(
            TrafficLightConfig(
                minimum_confidence=float(
                    self.get_parameter("traffic_light_min_confidence").value
                ),
                left_minimum_confidence=float(
                    self.get_parameter("shortcut_yolo_min_confidence").value
                ),
                stop_min_box_area_ratio=float(
                    self.get_parameter(
                        "traffic_light_stop_min_box_area_ratio"
                    ).value
                ),
                go_min_box_area_ratio=float(
                    self.get_parameter(
                        "traffic_light_go_min_box_area_ratio"
                    ).value
                ),
                stop_required_frames=int(
                    self.get_parameter(
                        "traffic_light_stop_required_frames"
                    ).value
                ),
                go_required_frames=int(
                    self.get_parameter(
                        "traffic_light_go_required_frames"
                    ).value
                ),
                left_required_frames=int(
                    self.get_parameter("shortcut_yolo_required_frames").value
                ),
                left_absence_frames=int(
                    self.get_parameter(
                        "shortcut_yolo_absence_frames"
                    ).value
                ),
                # S starts perception immediately after two absent detector
                # frames. Steering still waits for the metric spatial gate.
                left_start_delay_sec=0.0,
            )
        )
        self.traffic_light_decision = (
            self.traffic_light_controller.latest_decision
        )
        self.race_lap_policy = RaceLapPolicy(
            RaceLapPolicyConfig(
                enabled=bool(
                    self.get_parameter("race_lap_policy_enabled").value
                ),
                total_laps=int(
                    self.get_parameter("race_total_laps").value
                ),
                signal_release_frames=int(
                    self.get_parameter(
                        "race_signal_session_release_frames"
                    ).value
                ),
            )
        )
        self.latest_traffic_light_frame = TrafficLightFrame()
        self.last_left_detect_count = 0
        self.last_left_absence_count = 0
        self.shortcut_left_lane_offset_active = False
        self.initial_green_offset_suppression_active = False
        self.initial_green_offset_suppression_completed = False
        self.cone_command = (0.0, 0.0, 0.0)
        self.last_valid_cone_command = (0.0, 0.0)
        self.cone_command_time = float("-inf")
        self.last_valid_cone_command_time = float("-inf")
        self.cone_bypass = ConeModeLatch(
            ConeModeConfig(
                entry_confidence=float(
                    self.get_parameter("cone_entry_confidence").value
                ),
                entry_frames=int(
                    self.get_parameter("cone_entry_frames").value
                ),
                exit_frames=int(
                    self.get_parameter("cone_exit_frames").value
                ),
                exit_absence_sec=float(
                    self.get_parameter("cone_exit_absence_sec").value
                ),
                entry_distance_m=float(
                    self.get_parameter("cone_entry_distance_m").value
                ),
            )
        )
        self.cone_reentry_suppression = ConeReentrySuppression(
            ConeReentrySuppressionConfig(
                enabled=bool(
                    self.get_parameter(
                        "cone_reentry_suppression_until_s_curve"
                    ).value
                ),
                straight_max_abs_angle_command=float(
                    self.get_parameter(
                        "s_curve_entry_straight_max_abs_angle_command"
                    ).value
                ),
                straight_confirmation_frames=int(
                    self.get_parameter(
                        "s_curve_entry_straight_confirmation_frames"
                    ).value
                ),
                s_curve_left_angle_command=float(
                    self.get_parameter(
                        "s_curve_entry_left_angle_command"
                    ).value
                ),
                s_curve_confirmation_frames=int(
                    self.get_parameter(
                        "s_curve_entry_curve_confirmation_frames"
                    ).value
                ),
            )
        )
        self.cone_yolo_frames = 0
        self.cone_yolo_time = float("-inf")
        self.cone_yolo_confidence = 0.0
        self.cone_approach_yolo_frames = 0
        self.cone_approach_yolo_time = float("-inf")
        self.cone_approach_yolo_confidence = 0.0
        self.cone_lidar_distance_m = float("inf")
        self.cone_lidar_forward_distance_m = float("inf")
        self.cone_cluster_count = 0
        self.cone_cluster_time = float("-inf")
        self.vehicle_speed_mps = float("inf")
        self.vehicle_speed_time = float("-inf")
        self.cone_approach_brake_hold_active = False
        self.cone_approach_brake_release_frames = 0
        self.cone_approach_brake_required_distance_m = 0.0
        self.cone_approach_brake_reason = "none"
        self.latest_scan: LaserScan | None = None
        self.tracked_vehicle_sector: tuple[float, float] | None = None
        self.tracked_vehicle_sector_time = float("-inf")
        self.tracked_vehicle_distance_m = float("inf")
        self.tracked_vehicle_distance_time = float("-inf")
        self.yellow_reference_x_by_y = None
        self.yellow_reference_width = 0
        self.yellow_reference_height = 0
        self.yellow_reference_time = float("-inf")
        self.yellow_reference_rmse_px = float("inf")
        self.yellow_reference_span_ratio = 0.0
        self.avoidance_side_basis_code = 0.0
        self.traffic_light_sectors: tuple[tuple[float, float], ...] = ()
        self.traffic_light_sector_time = float("-inf")
        avoidance_yolo_min_confidence = float(
            self.get_parameter("vehicle_yolo_min_confidence").value
        )
        if bool(self.get_parameter("cone_as_vehicle_obstacle").value):
            avoidance_yolo_min_confidence = min(
                avoidance_yolo_min_confidence,
                float(
                    self.get_parameter(
                        "cone_as_vehicle_min_confidence"
                    ).value
                ),
            )
        self.avoidance_controller = YoloLidarAvoidanceController(
            YoloLidarAvoidanceConfig(
                yolo_min_confidence=avoidance_yolo_min_confidence,
                yolo_required_frames=int(
                    self.get_parameter("vehicle_yolo_required_frames").value
                ),
                yolo_timeout_sec=float(
                    self.get_parameter("vehicle_yolo_timeout_sec").value
                ),
                red_car_yolo_timeout_sec=float(
                    self.get_parameter(
                        "vehicle_red_car_yolo_timeout_sec"
                    ).value
                ),
                entry_distance_m=float(
                    self.get_parameter(
                        "vehicle_avoidance_entry_distance_m"
                    ).value
                ),
                minimum_side_clearance_m=float(
                    self.get_parameter(
                        "vehicle_minimum_side_clearance_m"
                    ).value
                ),
                left_offset_m=float(
                    self.get_parameter("vehicle_left_offset_m").value
                ),
                right_offset_m=float(
                    self.get_parameter("vehicle_right_offset_m").value
                ),
                offset_rate_mps=float(
                    self.get_parameter("vehicle_offset_rate_mps").value
                ),
                speed_limit_command=float(
                    self.get_parameter(
                        "vehicle_avoidance_speed_limit_command"
                    ).value
                ),
                minimum_avoid_sec=float(
                    self.get_parameter("vehicle_minimum_avoid_sec").value
                ),
                clear_hold_sec=float(
                    self.get_parameter("vehicle_clear_hold_sec").value
                ),
                green_car_clear_hold_sec=float(
                    self.get_parameter(
                        "vehicle_green_car_clear_hold_sec"
                    ).value
                ),
                return_hold_sec=float(
                    self.get_parameter("vehicle_return_hold_sec").value
                ),
                return_deadband_m=float(
                    self.get_parameter("vehicle_return_deadband_m").value
                ),
                immediate_on_yolo=bool(
                    self.get_parameter(
                        "vehicle_avoidance_immediate_on_yolo"
                    ).value
                ),
                preferred_side_required_frames=int(
                    self.get_parameter(
                        "vehicle_preferred_side_required_frames"
                    ).value
                ),
                active_side_reselection_required_frames=int(
                    self.get_parameter(
                        "vehicle_active_side_reselection_required_frames"
                    ).value
                ),
                active_side_reselection_offset_rate_mps=float(
                    self.get_parameter(
                        "vehicle_active_side_reselection_offset_rate_mps"
                    ).value
                ),
            )
        )
        self.avoidance_state = self.avoidance_controller.state()
        self.local_obstacle_route = tuple(
            (0.02 * index, 0.0) for index in range(251)
        )
        self.lidar_obstacle_config = LidarPathObstacleConfig(
            detect_distance_m=float(
                self.get_parameter("lidar_obstacle_detect_distance_m").value
            ),
            minimum_distance_m=float(
                self.get_parameter("lidar_obstacle_minimum_distance_m").value
            ),
            path_corridor_half_width_m=float(
                self.get_parameter("lidar_obstacle_path_half_width_m").value
            ),
            minimum_cluster_points=int(
                self.get_parameter("lidar_obstacle_minimum_cluster_points").value
            ),
            maximum_scan_index_gap=int(
                self.get_parameter("lidar_obstacle_maximum_scan_index_gap").value
            ),
            maximum_cluster_gap_m=float(
                self.get_parameter("lidar_obstacle_maximum_cluster_gap_m").value
            ),
            minimum_cluster_width_m=float(
                self.get_parameter("lidar_obstacle_minimum_cluster_width_m").value
            ),
            maximum_cluster_width_m=float(
                self.get_parameter("lidar_obstacle_maximum_cluster_width_m").value
            ),
            side_probe_inner_m=float(
                self.get_parameter("lidar_obstacle_side_probe_inner_m").value
            ),
            side_probe_outer_m=float(
                self.get_parameter("lidar_obstacle_side_probe_outer_m").value
            ),
            lidar_x_m=float(
                self.get_parameter("lidar_obstacle_lidar_x_m").value
            ),
            lidar_y_m=float(
                self.get_parameter("lidar_obstacle_lidar_y_m").value
            ),
            lidar_yaw_rad=math.radians(
                float(
                    self.get_parameter("lidar_obstacle_lidar_yaw_deg").value
                )
            ),
        )
        self.latest_lidar_obstacle: LidarPathObstacle | None = None
        self.latest_vehicle_lidar_obstacle: LidarPathObstacle | None = None
        self.scan_time = float("-inf")
        self.sectors = tuple(
            SectorMeasurement(float("inf"), 0, 0.0)
            for _ in self.sector_centers_deg
        )
        self.last_update_time = time.monotonic()
        self.last_status_time = 0.0
        self.last_status_key = ""

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("rl_command_topic").value),
            self._on_rl_command,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("rule_command_topic").value),
            self._on_rule_command,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("rule_diagnostics_topic").value),
            self._on_rule_diagnostics,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("cone_command_topic").value),
            self._on_cone_command,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("shortcut_command_topic").value),
            self._on_shortcut_command,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._on_scan,
            sensor_qos,
        )
        self.create_subscription(
            ObjectDetectionArray,
            str(self.get_parameter("object_detections_topic").value),
            self._on_object_detections,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("yellow_mask_topic").value),
            self._on_yellow_mask,
            sensor_qos,
        )
        self.create_subscription(
            PoseArray,
            str(self.get_parameter("cone_cluster_topic").value),
            self._on_cone_clusters,
            sensor_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("drive_armed_topic").value),
            self._on_drive_armed,
            10,
        )
        self.create_subscription(
            XycarVescState,
            str(self.get_parameter("vesc_state_topic").value),
            self._on_vesc_state,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("shortcut_entry_ready_topic").value),
            self._on_shortcut_entry_ready,
            10,
        )
        self.shadow_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("shadow_motor_topic").value),
            10,
        )
        self.motor_pub = (
            self.create_publisher(
                Float32MultiArray,
                str(self.get_parameter("motor_topic").value),
                10,
            )
            if self.drive_enabled
            else None
        )
        self.mode_pub = self.create_publisher(
            String,
            str(self.get_parameter("mode_topic").value),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.traffic_light_status_pub = self.create_publisher(
            String,
            str(self.get_parameter("traffic_light_status_topic").value),
            10,
        )
        self.diagnostics_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("diagnostics_topic").value),
            10,
        )
        self.avoidance_offset_pub = self.create_publisher(
            Float32,
            str(self.get_parameter("avoidance_offset_topic").value),
            10,
        )
        self.avoidance_debug_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("avoidance_debug_topic").value),
            10,
        )
        self.avoidance_active_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("avoidance_active_topic").value),
            10,
        )
        self.avoidance_active_pub.publish(Bool(data=False))
        self.avoidance_return_active_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("avoidance_return_active_topic").value),
            10,
        )
        self.post_red_turn_exit_window_pub = self.create_publisher(
            Bool,
            str(
                self.get_parameter(
                    "post_red_turn_exit_window_topic"
                ).value
            ),
            10,
        )
        self.post_red_turn_exit_window_pub.publish(Bool(data=False))
        self.avoidance_path_request_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("avoidance_path_request_topic").value),
            10,
        )
        cone_gate_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cone_processing_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("cone_processing_enabled_topic").value),
            cone_gate_qos,
        )
        self.cone_processing_pub.publish(Bool(data=False))
        self.shortcut_processing_pub = self.create_publisher(
            Bool,
            str(
                self.get_parameter(
                    "shortcut_processing_enabled_topic"
                ).value
            ),
            cone_gate_qos,
        )
        self.shortcut_processing_pub.publish(Bool(data=False))
        self.create_service(Trigger, "~/reset", self._reset)
        rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self.control_timer = self.create_timer(1.0 / rate_hz, self._control_step)
        self.get_logger().info(
            "INTEGRATED RULE DRIVE: waypoint/map/pose/SLAM are not used; "
            f"drive_enabled={self.drive_enabled}, "
            f"gate_arming_required={self.gate_arming_required}, "
            f"scan={self.get_parameter('scan_topic').value}, "
            "cone_freshness=LiDAR-derived-command, "
            "cone_precompute=enabled-while-stopped"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("gate_arming_required", False)
        self.declare_parameter("gate_disarm_cone_hold_sec", 5.0)
        self.declare_parameter("force_rule_only", True)
        self.declare_parameter("drive_armed_topic", "/hybrid_gate/drive_armed")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("rl_command_topic", "/rl/policy_motor_shadow")
        self.declare_parameter("rule_command_topic", "/hybrid/rule_candidate")
        self.declare_parameter(
            "rule_diagnostics_topic", "/rule_drive/diagnostics"
        )
        self.declare_parameter("cone_command_topic", "/my_rule/cone_cmd")
        self.declare_parameter(
            "shortcut_command_topic", "/hybrid/shortcut_candidate"
        )
        self.declare_parameter(
            "shortcut_processing_enabled_topic",
            "/hybrid/shortcut_processing_enabled",
        )
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter(
            "shadow_motor_topic", "/hybrid_gate/xycar_motor_shadow"
        )
        self.declare_parameter("mode_topic", "/hybrid_gate/mode")
        self.declare_parameter("status_topic", "/hybrid_gate/status")
        self.declare_parameter(
            "traffic_light_status_topic", "/hybrid/traffic_light_status"
        )
        self.declare_parameter("race_lap_policy_enabled", False)
        self.declare_parameter("race_total_laps", 3)
        self.declare_parameter("race_signal_session_release_frames", 2)
        self.declare_parameter(
            "diagnostics_topic", "/hybrid_gate/diagnostics"
        )
        self.declare_parameter(
            "sector_centers_deg", [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]
        )
        self.declare_parameter("sector_half_angle_deg", 8.0)
        self.declare_parameter("sector_distance_quantile", 0.20)
        self.declare_parameter("minimum_sector_points", 4)
        self.declare_parameter("minimum_sector_span_deg", 4.0)
        self.declare_parameter("scan_timeout_sec", 0.35)
        self.declare_parameter("initial_source", "RULE")
        self.declare_parameter("start_delay_sec", 3.0)
        self.declare_parameter("candidate_fresh_sec", 0.35)
        self.declare_parameter("candidate_hold_sec", 0.75)
        self.declare_parameter("minimum_speed_command", 3.0)
        self.declare_parameter("maximum_speed_command", 30.0)
        self.declare_parameter("maximum_abs_angle_command", 42.0)
        self.declare_parameter("cone_entry_confidence", 0.35)
        self.declare_parameter("cone_exit_confidence", 0.20)
        self.declare_parameter("cone_entry_frames", 3)
        self.declare_parameter("cone_exit_frames", 1)
        self.declare_parameter("cone_exit_absence_sec", 0.25)
        self.declare_parameter(
            "cone_reentry_suppression_until_s_curve", True
        )
        self.declare_parameter("cone_max_target_angle_deg", 42.0)
        self.declare_parameter(
            "cone_steering_actual_deg", [0.0, 4.0, 10.0, 16.0, 26.0]
        )
        self.declare_parameter(
            "cone_steering_command", [0.0, 10.0, 20.0, 30.0, 42.0]
        )
        self.declare_parameter("cone_yolo_min_confidence", 0.50)
        self.declare_parameter("cone_yolo_required_frames", 2)
        self.declare_parameter("cone_yolo_timeout_sec", 0.75)
        self.declare_parameter("cone_approach_slowdown_enabled", True)
        self.declare_parameter("cone_approach_yolo_min_confidence", 0.40)
        self.declare_parameter("cone_approach_center_min_ratio", 0.20)
        self.declare_parameter("cone_approach_center_max_ratio", 0.80)
        self.declare_parameter("cone_approach_first_speed_command", 15.0)
        self.declare_parameter("cone_approach_confirmed_speed_command", 8.0)
        self.declare_parameter("cone_approach_confirmed_distance_m", 1.15)
        self.declare_parameter("cone_approach_target_speed_mps", 0.65)
        self.declare_parameter("cone_approach_deceleration_mps2", 1.50)
        self.declare_parameter("cone_approach_response_time_sec", 0.10)
        self.declare_parameter("cone_approach_brake_margin_m", 0.10)
        self.declare_parameter("cone_approach_hard_stop_distance_m", 0.35)
        self.declare_parameter(
            "cone_approach_stale_speed_stop_distance_m", 0.80
        )
        self.declare_parameter("cone_approach_release_speed_mps", 0.80)
        self.declare_parameter("cone_approach_release_frames", 2)
        self.declare_parameter("vehicle_speed_timeout_sec", 0.25)
        self.declare_parameter("vesc_state_topic", "/vehicle/vesc_state")
        self.declare_parameter("s_curve_entry_guard_enabled", True)
        self.declare_parameter("s_curve_entry_speed_cap_command", 20.0)
        self.declare_parameter(
            "s_curve_entry_red_car_speed_cap_command", 13.0
        )
        self.declare_parameter(
            "s_curve_entry_speed_cap_start_distance_m", 8.0
        )
        self.declare_parameter(
            "s_curve_entry_straight_max_abs_angle_command", 5.0
        )
        self.declare_parameter(
            "s_curve_entry_straight_confirmation_frames", 3
        )
        self.declare_parameter(
            "s_curve_entry_minimum_curve_distance_m", 1.50
        )
        self.declare_parameter("s_curve_entry_left_angle_command", -8.0)
        self.declare_parameter(
            "s_curve_entry_curve_speed_margin_command", 1.00
        )
        self.declare_parameter(
            "s_curve_entry_curve_confirmation_frames", 3
        )
        self.declare_parameter("s_curve_entry_overdue_distance_m", 4.50)
        # 2026-08-20 competition-course bags: the successful entry first
        # produced three valid cone paths at forward x=0.883/0.797/0.701 m.
        # A 0.95 m gate preserves that sequence while rejecting the premature
        # hand-pushed transition observed at x=1.320 m.
        self.declare_parameter("cone_entry_distance_m", 0.95)
        self.declare_parameter("cone_cluster_timeout_sec", 0.50)
        self.declare_parameter("cone_command_timeout_sec", 0.35)
        self.declare_parameter("cone_sensor_presence_timeout_sec", 0.25)
        self.declare_parameter("cone_cluster_topic", "/my_rule/cone_clusters")
        self.declare_parameter(
            "cone_processing_enabled_topic",
            "/my_rule/cone_processing_enabled",
        )
        self.declare_parameter(
            "object_detections_topic", "/my_rule/object_detections"
        )
        self.declare_parameter("shortcut_enabled", True)
        self.declare_parameter("shortcut_class_name", "left_4")
        self.declare_parameter("shortcut_yolo_min_confidence", 0.40)
        self.declare_parameter("shortcut_yolo_required_frames", 2)
        self.declare_parameter("shortcut_yolo_absence_frames", 1)
        self.declare_parameter("shortcut_entry_speed_command", 9.0)
        self.declare_parameter("shortcut_left_lane_offset_m", 0.03)
        self.declare_parameter("shortcut_wait_for_entry_ready", True)
        self.declare_parameter(
            "shortcut_entry_ready_topic", "/shortcut/entry/ready"
        )
        self.declare_parameter("shortcut_entry_search_timeout_sec", 12.0)
        self.declare_parameter("shortcut_candidate_timeout_sec", 0.35)
        self.declare_parameter("shortcut_rearm_absence_sec", 1.0)
        self.declare_parameter(
            "vehicle_class_names",
            ["obstacle_vehicle", "red_car", "green_car"],
        )
        self.declare_parameter(
            "traffic_light_class_names",
            [
                "traffic_light",
                "traffic_signal",
                "red",
                "yellow",
                "green",
                "red_4",
                "yellow_4",
                "green_4",
                "left_4",
            ],
        )
        self.declare_parameter("traffic_light_control_enabled", True)
        self.declare_parameter("traffic_light_red_class_name", "red_4")
        self.declare_parameter(
            "traffic_light_yellow_class_name", "yellow_4"
        )
        self.declare_parameter("traffic_light_green_class_name", "green_4")
        self.declare_parameter("traffic_light_min_confidence", 0.50)
        self.declare_parameter(
            "traffic_light_stop_min_box_area_ratio", 0.0
        )
        self.declare_parameter(
            "traffic_light_go_min_box_area_ratio", 0.0
        )
        self.declare_parameter("traffic_light_stop_required_frames", 2)
        self.declare_parameter("traffic_light_go_required_frames", 2)
        self.declare_parameter("traffic_light_sector_memory_sec", 0.75)
        self.declare_parameter("traffic_light_lidar_padding_deg", 2.0)
        self.declare_parameter("vehicle_yolo_min_confidence", 0.45)
        self.declare_parameter("cone_as_vehicle_obstacle", False)
        self.declare_parameter("cone_as_vehicle_min_confidence", 0.50)
        self.declare_parameter("vehicle_yolo_required_frames", 1)
        self.declare_parameter("vehicle_yolo_timeout_sec", 0.50)
        self.declare_parameter("vehicle_red_car_yolo_timeout_sec", 0.50)
        self.declare_parameter("vehicle_camera_lidar_hfov_deg", 60.0)
        self.declare_parameter("vehicle_camera_lidar_padding_deg", 3.0)
        self.declare_parameter(
            "yellow_mask_topic", "/lane_seg/yellow_centerline_mask"
        )
        self.declare_parameter("yellow_reference_timeout_sec", 0.40)
        self.declare_parameter("yellow_reference_min_pixels", 3)
        self.declare_parameter("yellow_reference_residual_px", 6.0)
        self.declare_parameter("yellow_side_deadband_px", 6.0)
        self.declare_parameter("vehicle_side_decision_straight_only", False)
        self.declare_parameter(
            "vehicle_side_decision_max_rule_angle_command", 8.0
        )
        self.declare_parameter(
            "vehicle_side_decision_rule_timeout_sec", 0.25
        )
        self.declare_parameter("yellow_straight_max_rmse_px", 3.0)
        self.declare_parameter("yellow_straight_min_span_ratio", 0.20)
        self.declare_parameter("vehicle_preferred_side_required_frames", 1)
        self.declare_parameter(
            "vehicle_active_side_reselection_required_frames", 2
        )
        self.declare_parameter(
            "vehicle_green_car_retrigger_required_frames", 2
        )
        self.declare_parameter(
            "vehicle_active_side_reselection_offset_rate_mps", 1.30
        )
        self.declare_parameter("vehicle_lidar_min_points", 2)
        self.declare_parameter("vehicle_lidar_sector_memory_sec", 0.5)
        self.declare_parameter(
            "vehicle_lidar_association_angle_margin_deg", 2.0
        )
        self.declare_parameter(
            "vehicle_lidar_association_distance_tolerance_m", 0.35
        )
        self.declare_parameter("vehicle_avoidance_enabled", True)
        self.declare_parameter(
            "vehicle_avoidance_shortcut_suppression_enabled", True
        )
        self.declare_parameter(
            "vehicle_avoidance_shortcut_release_left_angle_command", -8.0
        )
        self.declare_parameter(
            "vehicle_avoidance_shortcut_release_required_frames", 2
        )
        self.declare_parameter(
            "vehicle_avoidance_immediate_on_yolo", True
        )
        self.declare_parameter("vehicle_avoidance_entry_distance_m", 1.20)
        self.declare_parameter("vehicle_minimum_side_clearance_m", 0.70)
        self.declare_parameter("vehicle_left_offset_m", 0.13)
        self.declare_parameter("vehicle_right_offset_m", 0.13)
        self.declare_parameter("vehicle_offset_rate_mps", 0.65)
        self.declare_parameter("vehicle_avoidance_speed_limit_command", 20.0)
        self.declare_parameter(
            "vehicle_avoidance_right_settle_speed_limit_command", 14.0
        )
        self.declare_parameter(
            "vehicle_avoidance_right_settle_tolerance_m", 0.02
        )
        self.declare_parameter(
            "vehicle_red_car_avoidance_speed_limit_command", 20.0
        )
        self.declare_parameter(
            "vehicle_green_car_avoidance_speed_limit_command", 20.0
        )
        self.declare_parameter("vehicle_minimum_avoid_sec", 0.50)
        self.declare_parameter("vehicle_clear_hold_sec", 0.50)
        self.declare_parameter("vehicle_green_car_clear_hold_sec", 0.60)
        self.declare_parameter("vehicle_return_hold_sec", 0.30)
        self.declare_parameter("vehicle_return_deadband_m", 0.02)
        self.declare_parameter("vehicle_return_cross_track_error_m", 0.08)
        self.declare_parameter("vehicle_return_required_frames", 3)
        self.declare_parameter(
            "vehicle_return_diagnostics_timeout_sec", 0.30
        )
        self.declare_parameter(
            "avoidance_offset_topic", "/hybrid/avoidance_lateral_offset"
        )
        self.declare_parameter(
            "avoidance_debug_topic", "/hybrid/avoidance_debug"
        )
        self.declare_parameter(
            "avoidance_active_topic", "/hybrid/avoidance_active"
        )
        self.declare_parameter(
            "avoidance_return_active_topic",
            "/hybrid/avoidance_return_active",
        )
        self.declare_parameter(
            "post_red_turn_exit_window_topic",
            "/hybrid/post_red_turn_exit_window",
        )
        self.declare_parameter(
            "post_red_turn_exit_window_maximum_sec", 6.0
        )
        self.declare_parameter(
            "avoidance_path_request_topic",
            "/hybrid/avoidance_path_request",
        )
        self.declare_parameter("vehicle_body_length_m", 0.55)
        self.declare_parameter("vehicle_body_width_m", 0.28)
        self.declare_parameter("lidar_obstacle_detect_distance_m", 1.50)
        self.declare_parameter("lidar_obstacle_minimum_distance_m", 0.18)
        self.declare_parameter("lidar_obstacle_path_half_width_m", 0.18)
        self.declare_parameter("lidar_obstacle_minimum_cluster_points", 3)
        self.declare_parameter("lidar_obstacle_maximum_scan_index_gap", 2)
        self.declare_parameter("lidar_obstacle_maximum_cluster_gap_m", 0.16)
        self.declare_parameter("lidar_obstacle_minimum_cluster_width_m", 0.09)
        self.declare_parameter("lidar_obstacle_maximum_cluster_width_m", 0.70)
        self.declare_parameter("lidar_obstacle_side_probe_inner_m", 0.18)
        self.declare_parameter("lidar_obstacle_side_probe_outer_m", 0.55)
        self.declare_parameter("lidar_obstacle_lidar_x_m", 0.065)
        self.declare_parameter("lidar_obstacle_lidar_y_m", 0.0)
        self.declare_parameter("lidar_obstacle_lidar_yaw_deg", 0.0)

    def _float_array(self, name: str) -> tuple[float, ...]:
        return tuple(float(value) for value in self.get_parameter(name).value)

    def _config(self) -> SequentialHybridConfig:
        return SequentialHybridConfig(
            initial_source=CandidateSource(
                str(self.get_parameter("initial_source").value).upper()
            ),
            start_delay_sec=float(self.get_parameter("start_delay_sec").value),
            candidate_fresh_sec=float(
                self.get_parameter("candidate_fresh_sec").value
            ),
            candidate_hold_sec=float(
                self.get_parameter("candidate_hold_sec").value
            ),
            minimum_speed_command=float(
                self.get_parameter("minimum_speed_command").value
            ),
            maximum_speed_command=float(
                self.get_parameter("maximum_speed_command").value
            ),
            maximum_abs_angle_command=float(
                self.get_parameter("maximum_abs_angle_command").value
            ),
        )

    def _on_rl_command(self, message: Float32MultiArray) -> None:
        if len(message.data) < 2:
            return
        self.rl_command = (float(message.data[0]), float(message.data[1]))
        self.rl_command_time = time.monotonic()

    def _on_rule_command(self, message: Float32MultiArray) -> None:
        if len(message.data) < 2:
            return
        self.rule_command = (float(message.data[0]), float(message.data[1]))
        self.rule_command_time = time.monotonic()

    def _on_rule_diagnostics(self, message: Float32MultiArray) -> None:
        if len(message.data) <= 12:
            return
        self.rule_path_valid = float(message.data[1]) > 0.5
        self.rule_cross_track_error_m = float(message.data[12])
        self.rule_diagnostics_time = time.monotonic()
        previous_frames = self.return_center_confirmation_frames
        self.return_center_confirmation_frames = (
            update_return_center_confirmation_frames(
                self.return_center_confirmation_frames,
                return_center_active=(
                    self.avoidance_state.mode
                    == YoloLidarAvoidanceMode.RETURN_CENTER
                ),
                path_valid=self.rule_path_valid,
                cross_track_error_m=self.rule_cross_track_error_m,
                maximum_abs_error_m=float(
                    self.get_parameter(
                        "vehicle_return_cross_track_error_m"
                    ).value
                ),
            )
        )
        required_frames = max(
            1,
            int(
                self.get_parameter(
                    "vehicle_return_required_frames"
                ).value
            ),
        )
        if (
            previous_frames < required_frames
            and self.return_center_confirmation_frames >= required_frames
        ):
            self.get_logger().info(
                "[AVOIDANCE RETURN] lane center confirmed: "
                f"CTE={self.rule_cross_track_error_m:+.3f}m, "
                f"frames={self.return_center_confirmation_frames}/"
                f"{required_frames}"
            )

    def _on_vesc_state(self, message: XycarVescState) -> None:
        speed = abs(float(message.speed_mps))
        if not math.isfinite(speed):
            return
        self.vehicle_speed_mps = speed
        self.vehicle_speed_time = time.monotonic()

    def _reset_cone_approach_brake(self) -> None:
        self.cone_approach_brake_hold_active = False
        self.cone_approach_brake_release_frames = 0
        self.cone_approach_brake_required_distance_m = 0.0
        self.cone_approach_brake_reason = "none"

    def _reset_cone_detection_state(self) -> None:
        self.cone_yolo_frames = 0
        self.cone_yolo_time = float("-inf")
        self.cone_yolo_confidence = 0.0
        self.cone_approach_yolo_frames = 0
        self.cone_approach_yolo_time = float("-inf")
        self.cone_approach_yolo_confidence = 0.0

    def _update_cone_reentry_suppression(self, output, *, now: float) -> None:
        event = self.cone_reentry_suppression.update(
            rule_controls_vehicle=bool(
                output.state == HybridState.RUNNING
                and output.source == CandidateSource.RULE
                and not self.shortcut_latch.active
                and not self.shortcut_entry_search_active
                and not self.cone_bypass.active
                and not self.avoidance_state.controls_vehicle
                and self.traffic_light_decision.action
                != TrafficLightAction.STOP
            ),
            rule_command_fresh=bool(
                now - self.rule_command_time
                <= self.controller.config.candidate_hold_sec
            ),
            rule_angle_command=self.rule_command[0],
        )
        if event == ConeReentryEvent.S_CURVE_REACHED:
            lap_event = self.race_lap_policy.arm_next_lap()
            if lap_event == RaceLapEvent.NEXT_LAP_ARMED:
                self.get_logger().warning(
                    f"{ANSI_BLUE}[LAP] LAP"
                    f"{self.race_lap_policy.current_lap} S-CURVE HANDOFF "
                    "(CONE SEQUENCE); NEXT TRAFFIC SESSION ARMED"
                    f"{ANSI_RESET}"
                )
            self._reset_cone_detection_state()
            self.get_logger().warning(
                f"{ANSI_GREEN}[CONE] REENTRY ENABLED AT S-CURVE ENTRY"
                f"{ANSI_RESET}"
            )

    def _start_s_curve_entry_guard(
        self, trigger: SCurveEntryTrigger
    ) -> None:
        event = self.s_curve_entry_guard.start(trigger)
        if event != SCurveEntryEvent.STARTED:
            return
        self.get_logger().warning(
            f"{ANSI_BLUE}[S_ENTRY] START trigger={trigger.value}; "
            "speed_cap="
            f"{self.s_curve_entry_guard.active_speed_cap_command():.1f}; "
            "RULE steering retained"
            f"{ANSI_RESET}"
        )

    def _reset_s_curve_entry_guard(self, reason: str) -> None:
        event = self.s_curve_entry_guard.reset()
        if event == SCurveEntryEvent.RESET:
            self.get_logger().info(f"[S_ENTRY] RESET reason={reason}")

    def _update_post_red_turn_exit_window(
        self,
        *,
        now: float,
        red_avoidance_completed_now: bool,
    ) -> None:
        target = self._normalize_class_name(
            self.avoidance_state.target_class_name
        )
        red_return_active = bool(
            self.avoidance_state.mode
            == YoloLidarAvoidanceMode.RETURN_CENTER
            and target == "red_car"
        )
        green_avoidance_active = bool(
            self.avoidance_state.controls_vehicle
            and target == "green_car"
        )
        self.post_red_turn_window, event = update_post_red_turn_window(
            self.post_red_turn_window,
            now=now,
            drive_armed=self.drive_armed,
            red_return_active=red_return_active,
            red_avoidance_completed=red_avoidance_completed_now,
            green_avoidance_active=green_avoidance_active,
            maximum_duration_sec=float(
                self.get_parameter(
                    "post_red_turn_exit_window_maximum_sec"
                ).value
            ),
        )
        self.post_red_turn_exit_window_pub.publish(
            Bool(data=self.post_red_turn_window.active)
        )
        if event == PostRedTurnWindowEvent.STARTED:
            self.get_logger().warning(
                f"{ANSI_BLUE}[POST_RED_TURN] WINDOW START; "
                "watching 90-degree left-turn exit"
                f"{ANSI_RESET}"
            )
        elif event != PostRedTurnWindowEvent.NONE:
            self.get_logger().info(
                "[POST_RED_TURN] WINDOW END reason="
                f"{event.value}"
            )

    def _reset_post_red_turn_exit_window(self) -> None:
        self.post_red_turn_window = PostRedTurnWindowState()
        self.post_red_turn_exit_window_pub.publish(Bool(data=False))

    def _apply_s_curve_entry_guard(self, output, *, now: float, dt: float):
        event = self.s_curve_entry_guard.update(
            dt_sec=dt,
            vehicle_speed_mps=self.vehicle_speed_mps,
            vehicle_speed_fresh=(
                now - self.vehicle_speed_time
                <= float(
                    self.get_parameter("vehicle_speed_timeout_sec").value
                )
            ),
            rule_angle_command=self.rule_command[0],
            rule_speed_command=self.rule_command[1],
            rule_command_fresh=(
                now - self.rule_command_time
                <= self.controller.config.candidate_hold_sec
            ),
        )
        if event == SCurveEntryEvent.CURVE_HANDOFF:
            state = self.s_curve_entry_guard.state()
            lap_event = self.race_lap_policy.arm_next_lap()
            if lap_event == RaceLapEvent.NEXT_LAP_ARMED:
                self.get_logger().warning(
                    f"{ANSI_BLUE}[LAP] LAP{self.race_lap_policy.current_lap} "
                    "S-CURVE HANDOFF; NEXT TRAFFIC SESSION ARMED"
                    f"{ANSI_RESET}"
                )
            if self.green_car_retrigger_blocked:
                self.green_car_retrigger_blocked = False
                self.green_car_retrigger_frames = 0
                self.get_logger().warning(
                    f"{ANSI_BLUE}[AVOIDANCE] GREEN_CAR RETRIGGER "
                    f"RE-ENABLED AT S-CURVE ENTRY{ANSI_RESET}"
                )
            avoidance_restored = self.shortcut_avoidance_suppression.release()
            self.get_logger().warning(
                f"{ANSI_GREEN}[S_ENTRY] NORMAL CURVE HANDOFF; "
                f"trigger={state.trigger.value}; "
                f"distance={state.distance_m:.2f}m; "
                f"RULE=[{self.rule_command[0]:.1f},"
                f"{self.rule_command[1]:.1f}]{ANSI_RESET}"
            )
            if avoidance_restored:
                self.get_logger().warning(
                    f"{ANSI_BLUE}[MISSION] ALL VEHICLE AVOIDANCE RESTORED "
                    f"AT S-CURVE ENTRY{ANSI_RESET}"
                )
            return output

        state = self.s_curve_entry_guard.state()
        s_entry_controls = bool(
            state.active
            and output.state == HybridState.RUNNING
            and output.source == CandidateSource.RULE
            and not self.shortcut_latch.active
            and not self.shortcut_entry_search_active
            and not self.cone_bypass.active
            and not self.avoidance_state.controls_vehicle
            and self.traffic_light_decision.action
            != TrafficLightAction.STOP
        )
        if not s_entry_controls:
            return output

        angle, speed = self.s_curve_entry_guard.limit_command(
            angle_command=output.angle_command,
            speed_command=output.speed_command,
        )
        state = self.s_curve_entry_guard.state()
        detail = "rule"
        if state.overdue:
            detail += "+overdue"
        cap_detail = (
            f"{self.s_curve_entry_guard.active_speed_cap_command():.1f}"
            if state.speed_cap_active
            else "waiting"
        )
        return replace(
            output,
            angle_command=angle,
            speed_command=speed,
            reason=(
                f"S-entry guard {state.trigger.value}; "
                f"d={state.distance_m:.2f}m; "
                f"straight={int(state.straight_ready)}; "
                f"steering={detail}; cap={cap_detail}"
            ),
        )

    def _on_drive_armed(self, message: Bool) -> None:
        armed = bool(message.data)
        if not self.gate_arming_required:
            self.drive_armed = True
            return
        if self.drive_armed and not armed:
            # SPACE remains an immediate motor-output stop in space_drive_gate.
            # Preserve only the cone mission state for a short operator pause;
            # otherwise re-arming in the middle of the course briefly selects
            # RULE before camera/LiDAR entry confirmation accumulates again.
            self.gate_disarmed_time = time.monotonic()
            # Shortcut and traffic-light state follows the operational yellow
            # branch and is reset immediately on operator disarm. Cone state is
            # intentionally retained for gate_disarm_cone_hold_sec.
            self._reset_shortcut_state()
            self.race_lap_policy.reset()
            self.initial_green_offset_suppression_active = False
            self.initial_green_offset_suppression_completed = False
            self._reset_s_curve_entry_guard("SPACE disarmed")
            self._reset_post_red_turn_exit_window()
            self.green_car_retrigger_blocked = False
            self.green_car_retrigger_frames = 0
            self.avoidance_controller.reset()
            self.avoidance_state = self.avoidance_controller.state()
            self.avoidance_offset_pub.publish(Float32(data=0.0))
            self._publish_avoidance_path_request(
                now=time.monotonic(),
                scan_fresh=False,
                force_inactive=True,
            )
            if not self.cone_bypass.active:
                self._reset_cone_approach_brake()
        elif not self.drive_armed and armed:
            self.gate_disarmed_time = float("-inf")
        self.drive_armed = armed

    def _reset_shortcut_state(self) -> None:
        self.shortcut_latch.reset()
        self.shortcut_avoidance_suppression.reset()
        self.traffic_light_controller.reset()
        self.traffic_light_decision = (
            self.traffic_light_controller.latest_decision
        )
        self.latest_traffic_light_frame = TrafficLightFrame()
        self.shortcut_command = (0.0, 0.0, 0.0)
        self.shortcut_command_time = float("-inf")
        self.shortcut_phase_code = 0.0
        self.shortcut_entry_search_active = False
        self.shortcut_entry_search_started_time = float("-inf")
        self.shortcut_entry_ready = False
        self.last_left_detect_count = 0
        self.last_left_absence_count = 0
        self._set_shortcut_left_lane_offset_active(
            False,
            reason="shortcut state reset",
        )
        self.shortcut_processing_pub.publish(Bool(data=False))

    def _set_shortcut_left_lane_offset_active(
        self,
        active: bool,
        *,
        reason: str,
    ) -> None:
        requested = bool(active)
        if requested == self.shortcut_left_lane_offset_active:
            return
        self.shortcut_left_lane_offset_active = requested
        offset_m = (
            max(
                0.0,
                float(
                    self.get_parameter("shortcut_left_lane_offset_m").value
                ),
            )
            if requested
            else 0.0
        )
        self.avoidance_offset_pub.publish(Float32(data=offset_m))
        state = "ACTIVE" if requested else "RELEASED"
        self.get_logger().warning(
            f"{ANSI_MAGENTA}[MISSION] SHORTCUT LEFT-LANE PREPOSITION "
            f"{state}: offset={offset_m:+.2f}m reason={reason}{ANSI_RESET}"
        )

    def _on_shortcut_entry_ready(self, message: Bool) -> None:
        """Transfer authority only when the metric W1 spatial gate opens."""
        self.shortcut_entry_ready = bool(message.data)
        if (
            not self.shortcut_entry_ready
            or not self.shortcut_entry_search_active
            or self.shortcut_latch.active
        ):
            return
        now = time.monotonic()
        if (
            not self.drive_armed
            or self.traffic_light_decision.action == TrafficLightAction.STOP
        ):
            return
        if self._shortcut_entry_search_expired(now):
            self._cancel_shortcut_entry_search(
                "[MISSION] W1 spatial gate arrived after search timeout"
            )
            return
        if now - self.shortcut_command_time > float(
            self.get_parameter("shortcut_candidate_timeout_sec").value
        ):
            # The ready signal and mux candidate arrive on separate topics.
            # Keep RULE authority until the first spatially gated candidate is
            # cached instead of producing a one-cycle motor stop.
            return
        event = self.shortcut_latch.start(
            now_sec=now,
            confidence=self.traffic_light_controller.left_confidence,
        )
        if event != ShortcutModeEvent.NONE:
            self.shortcut_entry_search_active = False
            self.shortcut_entry_search_started_time = float("-inf")
            self._handle_shortcut_event(event)

    def _shortcut_entry_search_expired(self, now: float) -> bool:
        return bool(
            self.shortcut_entry_search_active
            and float(now) - self.shortcut_entry_search_started_time
            > float(
                self.get_parameter("shortcut_entry_search_timeout_sec").value
            )
        )

    def _cancel_shortcut_entry_search(
        self, reason: str, *, report_as_error: bool = True
    ) -> None:
        if not self.shortcut_entry_search_active:
            return
        self.shortcut_entry_search_active = False
        self.shortcut_entry_search_started_time = float("-inf")
        self.shortcut_entry_ready = False
        self.traffic_light_controller.shortcut_finished()
        self.traffic_light_decision = (
            self.traffic_light_controller.latest_decision
        )
        self.shortcut_processing_pub.publish(Bool(data=False))
        self._set_shortcut_left_lane_offset_active(False, reason=reason)
        self.shortcut_avoidance_suppression.reset()
        self.avoidance_controller.reset()
        self.avoidance_state = self.avoidance_controller.state()
        self.avoidance_offset_pub.publish(Float32(data=0.0))
        if report_as_error:
            self.get_logger().error(reason)
        else:
            self.get_logger().warning(reason)

    def _cancel_shortcut_for_final_straight(self) -> None:
        reason = (
            "[TRAFFIC] FINAL=green_4; provisional left/shortcut control "
            "cancelled, returning to yellow Xbin RULE"
        )
        if self.shortcut_latch.active:
            self._reset_shortcut_state()
        elif self.shortcut_entry_search_active:
            self._cancel_shortcut_entry_search(
                reason,
                report_as_error=False,
            )
        else:
            self.shortcut_processing_pub.publish(Bool(data=False))
            self._set_shortcut_left_lane_offset_active(
                False,
                reason="final YOLO direction green_4",
            )
            self.shortcut_avoidance_suppression.reset()
        self.get_logger().warning(f"{ANSI_GREEN}{reason}{ANSI_RESET}")

    def _handle_shortcut_event(self, event: ShortcutModeEvent) -> None:
        if event == ShortcutModeEvent.STARTED:
            self._reset_s_curve_entry_guard("new shortcut started")
            self.shortcut_avoidance_suppression.start_shortcut()
            self.cone_bypass.reset()
            self.avoidance_controller.reset()
            self.avoidance_state = self.avoidance_controller.state()
            self.avoidance_offset_pub.publish(
                Float32(
                    data=selected_external_lateral_offset(
                        shortcut_left_lane_active=(
                            self.shortcut_left_lane_offset_active
                        ),
                        shortcut_left_lane_offset_m=float(
                            self.get_parameter(
                                "shortcut_left_lane_offset_m"
                            ).value
                        ),
                        avoidance_offset_m=0.0,
                    )
                )
            )
            self.shortcut_processing_pub.publish(Bool(data=True))
            self.get_logger().warning(
                "[MISSION] SHORTCUT ENTRY CONTROL ACTIVE"
            )
        elif event == ShortcutModeEvent.FINISHED:
            lap_event = self.race_lap_policy.mark_shortcut_completed()
            if lap_event == RaceLapEvent.SHORTCUT_COMPLETED:
                self.get_logger().warning(
                    f"{ANSI_MAGENTA}[LAP] LAP"
                    f"{self.race_lap_policy.current_lap} SHORTCUT COMPLETED; "
                    "FOLLOWING LAPS FORCE STRAIGHT"
                    f"{ANSI_RESET}"
                )
            self.shortcut_avoidance_suppression.start_rule_handoff()
            self.shortcut_entry_search_active = False
            self.shortcut_entry_search_started_time = float("-inf")
            self.shortcut_entry_ready = False
            self.traffic_light_controller.shortcut_finished()
            self.traffic_light_decision = (
                self.traffic_light_controller.latest_decision
            )
            self.shortcut_processing_pub.publish(Bool(data=False))
            self._set_shortcut_left_lane_offset_active(
                False,
                reason="yellow Xbin RULE handoff",
            )
            self._start_s_curve_entry_guard(
                SCurveEntryTrigger.SHORTCUT_EXIT
            )
            self.get_logger().warning(
                f"{ANSI_GREEN}[MISSION] CONTROL SWITCHED -> "
                f"YELLOW XBIN RULE{ANSI_RESET}"
            )
        elif event == ShortcutModeEvent.REARMED:
            self.get_logger().info("[MISSION] left_4 trigger rearmed")

    def _handle_traffic_shortcut_request(self, now: float) -> None:
        if (
            not self.traffic_light_decision.shortcut_start
            or not bool(self.get_parameter("shortcut_enabled").value)
            or not self.race_lap_policy.shortcut_allowed
            or self.shortcut_entry_search_active
            or self.shortcut_latch.active
            or self.cone_bypass.active
        ):
            return
        self.shortcut_entry_search_active = True
        self.shortcut_entry_search_started_time = float(now)
        self.shortcut_entry_ready = False
        self.shortcut_command = (0.0, 0.0, 0.0)
        self.shortcut_command_time = float("-inf")
        self.shortcut_phase_code = 0.0
        self.shortcut_avoidance_suppression.start_shortcut()
        self.avoidance_controller.reset()
        self.avoidance_state = self.avoidance_controller.state()
        self.avoidance_offset_pub.publish(
            Float32(
                data=selected_external_lateral_offset(
                    shortcut_left_lane_active=(
                        self.shortcut_left_lane_offset_active
                    ),
                    shortcut_left_lane_offset_m=float(
                        self.get_parameter(
                            "shortcut_left_lane_offset_m"
                        ).value
                    ),
                    avoidance_offset_m=0.0,
                )
            )
        )
        self.shortcut_processing_pub.publish(Bool(data=True))
        self.get_logger().warning(
            "[MISSION] SHORTCUT ENTRY PERCEPTION START; "
            "ALL VEHICLE AVOIDANCE SUPPRESSED UNTIL S-CURVE ENTRY"
        )

    def _on_shortcut_command(self, message: Float32MultiArray) -> None:
        if len(message.data) < 3 or not (
            self.shortcut_entry_search_active or self.shortcut_latch.active
        ):
            return
        self.shortcut_command = (
            float(message.data[0]),
            float(message.data[1]),
            float(message.data[2]),
        )
        self.shortcut_phase_code = (
            float(message.data[3]) if len(message.data) >= 4 else 0.0
        )
        self.shortcut_command_time = time.monotonic()
        if self.shortcut_latch.active and self.shortcut_command[2] >= 0.5:
            self._handle_shortcut_event(self.shortcut_latch.finish())

    @staticmethod
    def _normalize_class_name(value: str) -> str:
        return (
            str(value)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _scan_distance_for_sector(
        self,
        sector: tuple[float, float],
        *,
        excluded_sectors: tuple[tuple[float, float], ...] = (),
    ) -> float:
        scan = self.latest_scan
        if scan is None:
            return float("inf")
        return scan_sector_distance(
            ranges=scan.ranges,
            angle_min=float(scan.angle_min),
            angle_increment=float(scan.angle_increment),
            range_min=float(scan.range_min),
            range_max=float(scan.range_max),
            sector_min_angle=float(sector[0]),
            sector_max_angle=float(sector[1]),
            minimum_points=int(
                self.get_parameter("vehicle_lidar_min_points").value
            ),
            excluded_sectors=excluded_sectors,
        )

    def _fresh_traffic_light_sectors(
        self,
        now: float,
    ) -> tuple[tuple[float, float], ...]:
        if (
            float(now) - self.traffic_light_sector_time
            <= float(
                self.get_parameter(
                    "traffic_light_sector_memory_sec"
                ).value
            )
        ):
            return self.traffic_light_sectors
        return ()

    def _signal_observation(
        self,
        message: ObjectDetectionArray,
        class_name: str,
    ) -> SignalObservation:
        target = self._normalize_class_name(class_name)
        image_area = max(
            1, int(message.image_width) * int(message.image_height)
        )
        candidates = []
        for item in message.detections:
            if self._normalize_class_name(item.class_name) != target:
                continue
            width = max(0, int(item.xmax) - int(item.xmin))
            height = max(0, int(item.ymax) - int(item.ymin))
            candidates.append(
                SignalObservation(
                    confidence=float(item.confidence),
                    box_area_ratio=width * height / float(image_area),
                )
            )
        return (
            max(
                candidates,
                key=lambda item: (item.box_area_ratio, item.confidence),
            )
            if candidates
            else SignalObservation()
        )

    def _on_object_detections(
        self,
        message: ObjectDetectionArray,
    ) -> None:
        now = time.monotonic()
        image_width = max(0, int(message.image_width))
        traffic_frame = TrafficLightFrame(
            red=self._signal_observation(
                message,
                str(
                    self.get_parameter(
                        "traffic_light_red_class_name"
                    ).value
                ),
            ),
            yellow=self._signal_observation(
                message,
                str(
                    self.get_parameter(
                        "traffic_light_yellow_class_name"
                    ).value
                ),
            ),
            green=self._signal_observation(
                message,
                str(
                    self.get_parameter(
                        "traffic_light_green_class_name"
                    ).value
                ),
            ),
            left=self._signal_observation(
                message,
                str(self.get_parameter("shortcut_class_name").value),
            ),
        )
        self.latest_traffic_light_frame = traffic_frame
        direction_present = bool(
            traffic_frame.green.qualifies(
                minimum_confidence=float(
                    self.get_parameter("traffic_light_min_confidence").value
                ),
                minimum_box_area_ratio=float(
                    self.get_parameter(
                        "traffic_light_go_min_box_area_ratio"
                    ).value
                ),
            )
            or traffic_frame.left.qualifies(
                minimum_confidence=float(
                    self.get_parameter("shortcut_yolo_min_confidence").value
                )
            )
        )
        lap_event = self.race_lap_policy.observe_direction_signal(
            present=direction_present
        )
        if lap_event == RaceLapEvent.LAP_STARTED:
            policy = (
                "FORCE_STRAIGHT"
                if self.race_lap_policy.force_straight
                else "YOLO_DIRECTION"
            )
            self.get_logger().warning(
                f"{ANSI_BLUE}[LAP] LAP"
                f"{self.race_lap_policy.current_lap} START; "
                f"policy={policy}; shortcut_completed="
                f"{int(self.race_lap_policy.shortcut_completed)}"
                f"{ANSI_RESET}"
            )
        control_traffic_frame = traffic_frame
        if self.race_lap_policy.ignore_stop_signals:
            control_traffic_frame = replace(
                control_traffic_frame,
                red=SignalObservation(),
                yellow=SignalObservation(),
            )
        if self.race_lap_policy.force_straight:
            control_traffic_frame = replace(
                control_traffic_frame,
                left=SignalObservation(),
            )
        suppress_initial_green_offset_this_frame = (
            self.initial_green_offset_suppression_active
        )
        if (
            self.drive_armed
            and bool(
                self.get_parameter("traffic_light_control_enabled").value
            )
        ):
            stop_was_latched = self.traffic_light_controller.stop_latched
            self.traffic_light_decision = (
                self.traffic_light_controller.observe(
                    now_sec=now,
                    frame=control_traffic_frame,
                    # Signal STOP remains above an already-active shortcut.
                    shortcut_active=False,
                )
            )
            initial_green_released = bool(
                not self.initial_green_offset_suppression_completed
                and stop_was_latched
                and not self.traffic_light_controller.stop_latched
                and traffic_frame.green.qualifies(
                    minimum_confidence=float(
                        self.get_parameter("traffic_light_min_confidence").value
                    ),
                    minimum_box_area_ratio=float(
                        self.get_parameter(
                            "traffic_light_go_min_box_area_ratio"
                        ).value
                    ),
                )
            )
            if initial_green_released:
                self.initial_green_offset_suppression_active = True
                suppress_initial_green_offset_this_frame = True
                self._set_shortcut_left_lane_offset_active(
                    False,
                    reason="initial red-to-green departure",
                )
                self.get_logger().warning(
                    f"{ANSI_GREEN}[TRAFFIC] INITIAL GREEN OFFSET SUPPRESSED"
                    f"{ANSI_RESET}"
                )
            start_event = self.race_lap_policy.observe_start_gate(
                green_release_confirmed=bool(
                    not self.traffic_light_controller.stop_latched
                    and self.traffic_light_controller.go_frames
                    >= int(
                        self.get_parameter(
                            "traffic_light_go_required_frames"
                        ).value
                    )
                )
            )
            if start_event == RaceLapEvent.RACE_STARTED:
                self.get_logger().warning(
                    f"{ANSI_GREEN}[RACE] INITIAL GREEN RELEASED -> "
                    "LAP1 FORCE_STRAIGHT; LATER RED/YELLOW STOP DISABLED"
                    f"{ANSI_RESET}"
                )
        else:
            self.traffic_light_controller.reset()
            self.traffic_light_decision = (
                self.traffic_light_controller.latest_decision
            )

        if self.traffic_light_decision.direction_finalized:
            self.get_logger().warning(
                f"{ANSI_GREEN}[TRAFFIC] FINAL YOLO DIRECTION="
                f"{self.traffic_light_decision.signal_name or 'none'}; "
                f"cancel_shortcut="
                f"{int(self.traffic_light_decision.cancel_shortcut)}"
                f"{ANSI_RESET}"
            )
            if self.initial_green_offset_suppression_active:
                self.initial_green_offset_suppression_active = False
                self.initial_green_offset_suppression_completed = True
                self.get_logger().warning(
                    f"{ANSI_GREEN}[TRAFFIC] INITIAL GREEN SESSION COMPLETE; "
                    f"GREEN OFFSET RESTORED{ANSI_RESET}"
                )
        if self.traffic_light_decision.cancel_shortcut:
            self._cancel_shortcut_for_final_straight()

        left_seen = traffic_frame.left.qualifies(
            minimum_confidence=float(
                self.get_parameter("shortcut_yolo_min_confidence").value
            )
        )
        if left_seen:
            count = min(
                int(self.get_parameter("shortcut_yolo_required_frames").value),
                self.traffic_light_controller.left_frames,
            )
            if count != self.last_left_detect_count:
                self.get_logger().info(
                    f"{ANSI_YELLOW}[MISSION] left_4 DETECTED "
                    f"confidence={traffic_frame.left.confidence:.2f} "
                    f"{count}/"
                    f"{int(self.get_parameter('shortcut_yolo_required_frames').value)}"
                    f"{ANSI_RESET}"
                )
                self.last_left_detect_count = count
            self.last_left_absence_count = 0
        elif self.traffic_light_controller.left_confirmed:
            count = min(
                int(self.get_parameter("shortcut_yolo_absence_frames").value),
                self.traffic_light_controller.left_absence_frames,
            )
            if count != self.last_left_absence_count:
                self.get_logger().info(
                    f"{ANSI_YELLOW}[MISSION] left_4 ABSENT "
                    f"{count}/"
                    f"{int(self.get_parameter('shortcut_yolo_absence_frames').value)}"
                    f"{ANSI_RESET}"
                )
                self.last_left_absence_count = count
        if (
            shortcut_preposition_requested(
                action=self.traffic_light_decision.action,
                signal_name=self.traffic_light_decision.signal_name,
                shortcut_class_name=str(
                    self.get_parameter("shortcut_class_name").value
                ),
                green_class_name=str(
                    self.get_parameter("traffic_light_green_class_name").value
                ),
                suppress_initial_green=(
                    suppress_initial_green_offset_this_frame
                ),
            )
            and bool(self.get_parameter("shortcut_enabled").value)
            and self.race_lap_policy.shortcut_allowed
            and self.shortcut_latch.armed
            and not self.cone_bypass.active
        ):
            self._set_shortcut_left_lane_offset_active(
                True,
                reason=(
                    "provisional traffic direction; latest YOLO="
                    f"{self.traffic_light_decision.signal_name or 'unknown'}"
                ),
            )
        self._handle_traffic_shortcut_request(now)
        cone_confidences = [
            float(item.confidence)
            for item in message.detections
            if self._normalize_class_name(item.class_name) == "cone"
        ]
        cone_confidence = max(cone_confidences, default=0.0)
        cone_seen = cone_confidence >= float(
            self.get_parameter("cone_yolo_min_confidence").value
        )
        self.cone_yolo_frames = (
            self.cone_yolo_frames + 1 if cone_seen else 0
        )
        if cone_seen:
            self.cone_yolo_time = now
            self.cone_yolo_confidence = cone_confidence
        center_min = float(
            self.get_parameter("cone_approach_center_min_ratio").value
        )
        center_max = float(
            self.get_parameter("cone_approach_center_max_ratio").value
        )
        approach_confidences = (
            [
                float(item.confidence)
                for item in message.detections
                if (
                    self._normalize_class_name(item.class_name) == "cone"
                    and min(center_min, center_max)
                    <= (float(item.xmin) + float(item.xmax))
                    / (2.0 * float(image_width))
                    <= max(center_min, center_max)
                )
            ]
            if image_width > 0
            else []
        )
        approach_confidence = max(approach_confidences, default=0.0)
        central_cone_seen = bool(
            approach_confidence
            >= float(
                self.get_parameter(
                    "cone_approach_yolo_min_confidence"
                ).value
            )
        )
        self.cone_approach_yolo_frames = (
            self.cone_approach_yolo_frames + 1
            if central_cone_seen
            else 0
        )
        if central_cone_seen:
            self.cone_approach_yolo_time = now
            self.cone_approach_yolo_confidence = approach_confidence

        traffic_light_names = {
            self._normalize_class_name(value)
            for value in self.get_parameter(
                "traffic_light_class_names"
            ).value
        }
        traffic_light_sectors = []
        if image_width > 0:
            for item in message.detections:
                if (
                    self._normalize_class_name(item.class_name)
                    not in traffic_light_names
                    or float(item.confidence)
                    < float(
                        self.get_parameter(
                            "traffic_light_min_confidence"
                        ).value
                    )
                ):
                    continue
                traffic_light_sectors.append(
                    camera_box_lidar_sector(
                        xmin=float(item.xmin),
                        xmax=float(item.xmax),
                        image_width=image_width,
                        horizontal_fov_deg=float(
                            self.get_parameter(
                                "vehicle_camera_lidar_hfov_deg"
                            ).value
                        ),
                        padding_deg=float(
                            self.get_parameter(
                                "traffic_light_lidar_padding_deg"
                            ).value
                        ),
                    )
                )
        self.traffic_light_sectors = tuple(traffic_light_sectors)
        self.traffic_light_sector_time = now

        vehicle_names = {
            self._normalize_class_name(value)
            for value in self.get_parameter("vehicle_class_names").value
        }
        cone_as_vehicle_obstacle = bool(
            self.get_parameter("cone_as_vehicle_obstacle").value
        )
        vehicle_candidates = [
            item
            for item in message.detections
            if is_avoidance_detection(
                class_name=self._normalize_class_name(item.class_name),
                confidence=float(item.confidence),
                vehicle_names=vehicle_names,
                vehicle_min_confidence=float(
                    self.get_parameter("vehicle_yolo_min_confidence").value
                ),
                cone_as_vehicle_obstacle=cone_as_vehicle_obstacle,
                cone_min_confidence=float(
                    self.get_parameter(
                        "cone_as_vehicle_min_confidence"
                    ).value
                ),
            )
            and not shortcut_suppresses_vehicle_class(
                self._normalize_class_name(item.class_name),
                suppression_active=(
                    self.shortcut_avoidance_suppression.active
                ),
            )
        ]
        green_car_detected = any(
            self._normalize_class_name(item.class_name) == "green_car"
            for item in vehicle_candidates
        )
        (
            self.green_car_retrigger_frames,
            green_car_retrigger_confirmed,
        ) = update_green_car_retrigger_confirmation(
            self.green_car_retrigger_frames,
            green_car_detected=green_car_detected,
            blocked_until_s_curve=self.green_car_retrigger_blocked,
            s_curve_entry_guard_active=self.s_curve_entry_guard.state().active,
            required_frames=int(
                self.get_parameter(
                    "vehicle_green_car_retrigger_required_frames"
                ).value
            ),
        )
        if green_car_retrigger_confirmed:
            self.green_car_retrigger_blocked = False
            self.green_car_retrigger_frames = 0
            self._reset_s_curve_entry_guard(
                "green_car re-detected for two frames"
            )
            self.get_logger().warning(
                f"{ANSI_BLUE}[AVOIDANCE] GREEN_CAR RE-DETECTED; "
                f"S-ENTRY GUARD RELEASED -> IMMEDIATE AVOIDANCE{ANSI_RESET}"
            )
        vehicles = [
            item
            for item in vehicle_candidates
            if not green_car_retrigger_suppressed(
                self._normalize_class_name(item.class_name),
                blocked_until_s_curve=(
                    self.green_car_retrigger_blocked
                ),
            )
        ]
        selected = None
        selected_sector = None
        selected_distance = float("inf")
        for vehicle in vehicles:
            if image_width <= 0:
                continue
            sector = camera_box_lidar_sector(
                xmin=float(vehicle.xmin),
                xmax=float(vehicle.xmax),
                image_width=image_width,
                horizontal_fov_deg=float(
                    self.get_parameter(
                        "vehicle_camera_lidar_hfov_deg"
                    ).value
                ),
                padding_deg=float(
                    self.get_parameter(
                        "vehicle_camera_lidar_padding_deg"
                    ).value
                ),
            )
            distance = self._scan_distance_for_sector(
                sector,
                excluded_sectors=self.traffic_light_sectors,
            )
            if selected is None or distance < selected_distance:
                selected = vehicle
                selected_sector = sector
                selected_distance = distance
        if selected is None and vehicles:
            selected = max(vehicles, key=lambda item: float(item.confidence))
        if selected_sector is not None:
            self.tracked_vehicle_sector = selected_sector
            self.tracked_vehicle_sector_time = now
            self.tracked_vehicle_distance_m = selected_distance
            if math.isfinite(selected_distance):
                self.tracked_vehicle_distance_time = now
        preferred_mode = None
        selected_class_name = (
            self._normalize_class_name(selected.class_name)
            if selected is not None
            else ""
        )
        if (
            self.avoidance_controller.mode == YoloLidarAvoidanceMode.IDLE
            and self.avoidance_controller.preferred_mode is None
        ):
            self.avoidance_side_basis_code = 0.0
        if selected is not None and image_width > 0:
            box_center_x = 0.5 * (
                float(selected.xmin) + float(selected.xmax)
            )
            yellow_fresh = (
                now - self.yellow_reference_time
                <= float(
                    self.get_parameter("yellow_reference_timeout_sec").value
                )
            )
            side_decision_allowed = not bool(
                self.get_parameter(
                    "vehicle_side_decision_straight_only"
                ).value
            ) or straight_road_side_decision_allowed(
                rule_angle_command=self.rule_command[0],
                rule_command_age_sec=now - self.rule_command_time,
                yellow_reference_age_sec=now - self.yellow_reference_time,
                yellow_reference_rmse_px=self.yellow_reference_rmse_px,
                yellow_reference_span_ratio=(
                    self.yellow_reference_span_ratio
                ),
                maximum_rule_angle_command=float(
                    self.get_parameter(
                        "vehicle_side_decision_max_rule_angle_command"
                    ).value
                ),
                rule_command_timeout_sec=float(
                    self.get_parameter(
                        "vehicle_side_decision_rule_timeout_sec"
                    ).value
                ),
                yellow_reference_timeout_sec=float(
                    self.get_parameter("yellow_reference_timeout_sec").value
                ),
                maximum_yellow_rmse_px=float(
                    self.get_parameter("yellow_straight_max_rmse_px").value
                ),
                minimum_yellow_span_ratio=float(
                    self.get_parameter(
                        "yellow_straight_min_span_ratio"
                    ).value
                ),
            )
            if (
                side_decision_allowed
                and yellow_fresh
                and self.yellow_reference_x_by_y is not None
                and self.yellow_reference_width > 0
                and self.yellow_reference_height > 0
                and int(message.image_height) > 0
            ):
                object_x = (
                    box_center_x
                    * self.yellow_reference_width
                    / float(image_width)
                )
                object_y = (
                    float(selected.ymax)
                    * self.yellow_reference_height
                    / float(message.image_height)
                )
                preferred_mode = (
                    preferred_avoidance_mode_from_yellow_reference(
                        object_x=object_x,
                        object_y=object_y,
                        yellow_x_by_y=self.yellow_reference_x_by_y,
                        deadband_px=float(
                            self.get_parameter(
                                "yellow_side_deadband_px"
                            ).value
                        ),
                    )
                )
                if preferred_mode is not None:
                    self.avoidance_side_basis_code = 1.0
            if (
                preferred_mode is None
                and side_decision_allowed
                and selected_class_name == "green_car"
            ):
                preferred_mode = preferred_avoidance_mode_from_image_center(
                    object_center_x=box_center_x,
                    image_width=float(image_width),
                )
                if preferred_mode is not None:
                    self.avoidance_side_basis_code = 2.0
            if not side_decision_allowed:
                self.avoidance_side_basis_code = -1.0
        else:
            side_decision_allowed = False
        self.avoidance_controller.observe_yolo(
            now_sec=now,
            detected=selected is not None,
            confidence=(float(selected.confidence) if selected else 0.0),
            lidar_distance_m=selected_distance,
            preferred_mode=preferred_mode,
            side_decision_allowed=side_decision_allowed,
            target_class_name=selected_class_name,
            speed_limit_command=(
                avoidance_speed_limit_for_vehicle_class(
                    selected_class_name,
                    default_speed_limit_command=float(
                        self.get_parameter(
                            "vehicle_avoidance_speed_limit_command"
                        ).value
                    ),
                    red_car_speed_limit_command=float(
                        self.get_parameter(
                            "vehicle_red_car_avoidance_speed_limit_command"
                        ).value
                    ),
                    green_car_speed_limit_command=float(
                        self.get_parameter(
                            "vehicle_green_car_avoidance_speed_limit_command"
                        ).value
                    ),
                )
                if selected is not None
                else None
            ),
        )
        if self.cone_reentry_suppression.active:
            self._reset_cone_detection_state()
        self._publish_cone_processing_gate(now)

    def _on_yellow_mask(self, message: Image) -> None:
        if str(message.encoding).lower() not in {"mono8", "8uc1"}:
            return
        width = int(message.width)
        height = int(message.height)
        step = int(message.step)
        if width <= 0 or height <= 0 or step < width:
            return
        raw = np.frombuffer(message.data, dtype=np.uint8)
        if raw.size < height * step:
            return
        mask = raw[: height * step].reshape(height, step)[:, :width]
        reference = fit_yellow_centerline_reference(
            mask,
            min_pixels=int(
                self.get_parameter("yellow_reference_min_pixels").value
            ),
            residual_threshold_px=float(
                self.get_parameter("yellow_reference_residual_px").value
            ),
            line_width_px=1,
        )
        if not reference.valid:
            return
        occupied_rows = np.nonzero(mask > 0)[0]
        row_span_px = (
            float(np.ptp(occupied_rows)) if occupied_rows.size >= 2 else 0.0
        )
        self.yellow_reference_x_by_y = reference.x_by_y
        self.yellow_reference_width = width
        self.yellow_reference_height = height
        self.yellow_reference_time = time.monotonic()
        self.yellow_reference_rmse_px = float(reference.rmse_px)
        self.yellow_reference_span_ratio = row_span_px / max(
            1.0, float(height - 1)
        )

    def _on_cone_clusters(self, message: PoseArray) -> None:
        points = [
            (
                float(pose.position.x),
                float(pose.position.y),
            )
            for pose in message.poses
            if (
                float(pose.position.x) > 0.0
                and math.isfinite(float(pose.position.x))
                and math.isfinite(float(pose.position.y))
            )
        ]
        distances = [math.hypot(x, y) for x, y in points]
        self.cone_lidar_distance_m = min(distances, default=float("inf"))
        self.cone_lidar_forward_distance_m = min(
            (x for x, _y in points),
            default=float("inf"),
        )
        self.cone_cluster_count = len(distances)
        self.cone_cluster_time = time.monotonic()

    def _cone_yolo_confirmed(self, now: float) -> bool:
        return bool(
            self.cone_yolo_frames
            >= int(self.get_parameter("cone_yolo_required_frames").value)
            and now - self.cone_yolo_time
            <= float(self.get_parameter("cone_yolo_timeout_sec").value)
        )

    def _cone_sensor_present(self, now: float) -> bool:
        timeout = float(
            self.get_parameter("cone_sensor_presence_timeout_sec").value
        )
        yolo_present = now - self.cone_yolo_time <= timeout
        lidar_present = bool(
            self.cone_cluster_count > 0
            and now - self.cone_cluster_time <= timeout
        )
        return yolo_present or lidar_present

    def _cone_target_to_command(self, angle_deg: float) -> float:
        limit = float(self.get_parameter("cone_max_target_angle_deg").value)
        target = float(np.clip(angle_deg, -limit, limit))
        return interpolate_command(
            target,
            self.get_parameter("cone_steering_actual_deg").value,
            self.get_parameter("cone_steering_command").value,
        )

    def _publish_cone_processing_gate(self, now: float) -> None:
        requested = cone_processing_requested(
            shortcut_active=self.shortcut_latch.active,
            cone_active=self.cone_bypass.active,
            yolo_age_sec=now - self.cone_approach_yolo_time,
            yolo_timeout_sec=float(
                self.get_parameter("cone_yolo_timeout_sec").value
            ),
            reentry_suppressed=self.cone_reentry_suppression.active,
        )
        self.cone_processing_pub.publish(Bool(data=requested))

    def _cone_disarm_hold_active(self, now: float) -> bool:
        return bool(
            self.gate_arming_required
            and not self.drive_armed
            and self.cone_bypass.active
            and cone_disarm_hold_active(
                now_sec=now,
                disarmed_since_sec=self.gate_disarmed_time,
                hold_sec=float(
                    self.get_parameter("gate_disarm_cone_hold_sec").value
                ),
            )
        )

    def _handle_cone_event(self, event: ConeModeEvent) -> None:
        if event == ConeModeEvent.STARTED:
            self.green_car_retrigger_blocked = False
            self.green_car_retrigger_frames = 0
            self._set_shortcut_left_lane_offset_active(
                False,
                reason="confirmed cone-course entry",
            )
            if shortcut_search_should_yield_to_cone(
                shortcut_entry_search_active=(
                    self.shortcut_entry_search_active
                ),
                shortcut_active=self.shortcut_latch.active,
                cone_event=event,
            ):
                self._cancel_shortcut_entry_search(
                    "[MISSION] SHORTCUT W1 SEARCH ABORTED -> CONE_RULE; "
                    "confirmed cone-course entry takes control",
                    report_as_error=False,
                )
            self._reset_s_curve_entry_guard("cone mission started")
            self.get_logger().warning(
                "CONE_RULE START; YOLO+LiDAR cone confirmed; "
                f"forward={self.cone_lidar_forward_distance_m:.3f}m, "
                "gate="
                f"{float(self.get_parameter('cone_entry_distance_m').value):.3f}m"
            )
        elif event == ConeModeEvent.FINISHED:
            self._reset_cone_approach_brake()
            self._reset_cone_detection_state()
            reentry_event = self.cone_reentry_suppression.start()
            self.get_logger().warning(
                "CONE_RULE FINISHED; YOLO and LiDAR cones both disappeared"
            )
            if reentry_event == ConeReentryEvent.STARTED:
                self.get_logger().warning(
                    f"{ANSI_BLUE}[CONE] REENTRY SUPPRESSED UNTIL "
                    f"S-CURVE ENTRY{ANSI_RESET}"
                )

    def _on_cone_command(self, message: Float32MultiArray) -> None:
        if len(message.data) < 3:
            return
        if self.cone_reentry_suppression.active:
            self.cone_bypass.reset()
            return
        angle = float(message.data[0])
        speed = float(message.data[1])
        confidence = float(message.data[2])
        now = time.monotonic()
        self.cone_command = (angle, speed, confidence)
        self.cone_command_time = now
        if (
            confidence
            > float(self.get_parameter("cone_exit_confidence").value)
            and speed > 0.0
        ):
            self.last_valid_cone_command = (angle, speed)
            self.last_valid_cone_command_time = now
        event = self.cone_bypass.observe_command(
            confidence=confidence,
            speed_command=speed,
            yolo_confirmed=self._cone_yolo_confirmed(now),
            lidar_distance_m=(
                self.cone_lidar_forward_distance_m
                if now - self.cone_cluster_time
                <= float(
                    self.get_parameter("cone_cluster_timeout_sec").value
                )
                else float("inf")
            ),
        )
        self._handle_cone_event(event)

    def _on_scan(self, message: LaserScan) -> None:
        self.latest_scan = message
        maximum = float(message.range_max)
        if maximum <= 0.0 or not math.isfinite(maximum):
            maximum = 20.0
        half_angle = math.radians(
            float(self.get_parameter("sector_half_angle_deg").value)
        )
        quantile = float(
            self.get_parameter("sector_distance_quantile").value
        )
        minimum_points = int(
            self.get_parameter("minimum_sector_points").value
        )
        minimum_span = math.radians(
            float(self.get_parameter("minimum_sector_span_deg").value)
        )
        sectors = []
        for center_deg in self.sector_centers_deg:
            measurement = measure_sector(
                message.ranges,
                angle_min_rad=float(message.angle_min),
                angle_increment_rad=float(message.angle_increment),
                range_min_m=max(0.0, float(message.range_min)),
                range_max_m=maximum,
                center_angle_rad=math.radians(center_deg),
                half_angle_rad=half_angle,
                distance_quantile=quantile,
            )
            if (
                measurement.point_count < minimum_points
                or measurement.angular_span_rad < minimum_span
            ):
                measurement = SectorMeasurement(float("inf"), 0, 0.0)
            sectors.append(measurement)
        self.sectors = tuple(sectors)
        self.latest_lidar_obstacle = detect_path_obstacle(
            ranges=message.ranges,
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=max(0.0, float(message.range_min)),
            range_max=maximum,
            route_points=self.local_obstacle_route,
            nearest_index=0,
            vehicle_x=0.0,
            vehicle_y=0.0,
            vehicle_yaw=0.0,
            closed=False,
            config=self.lidar_obstacle_config,
        )
        now = time.monotonic()
        excluded_sectors = self._fresh_traffic_light_sectors(now)
        vehicle_sector_fresh = bool(
            self.tracked_vehicle_sector is not None
            and now - self.tracked_vehicle_sector_time
            <= float(
                self.get_parameter(
                    "vehicle_lidar_sector_memory_sec"
                ).value
            )
        )
        if vehicle_sector_fresh:
            distance = self._scan_distance_for_sector(
                self.tracked_vehicle_sector,
                excluded_sectors=excluded_sectors,
            )
            if math.isfinite(distance):
                self.tracked_vehicle_distance_m = distance
                self.tracked_vehicle_distance_time = now
                self.avoidance_controller.update_lidar_distance(distance)
        obstacle = self.latest_lidar_obstacle
        if (
            vehicle_sector_fresh
            and obstacle is not None
            and lidar_target_matches_camera_sector(
                target_x_m=float(obstacle.x_vehicle_m),
                target_y_m=float(obstacle.y_vehicle_m),
                camera_sector=self.tracked_vehicle_sector,
                camera_sector_distance_m=self.tracked_vehicle_distance_m,
                excluded_sectors=excluded_sectors,
                angle_margin_rad=math.radians(
                    float(
                        self.get_parameter(
                            "vehicle_lidar_association_angle_margin_deg"
                        ).value
                    )
                ),
                distance_tolerance_m=float(
                    self.get_parameter(
                        "vehicle_lidar_association_distance_tolerance_m"
                    ).value
                ),
            )
        ):
            self.latest_vehicle_lidar_obstacle = obstacle
        else:
            self.latest_vehicle_lidar_obstacle = None
        self.scan_time = now

    def _reset(self, _request, response):
        self.controller.reset()
        if self.force_rule_only:
            self.controller.source = CandidateSource.RULE
        self._reset_shortcut_state()
        self.race_lap_policy.reset()
        self.initial_green_offset_suppression_active = False
        self.initial_green_offset_suppression_completed = False
        self.cone_bypass.reset()
        self.cone_reentry_suppression.reset()
        self.cone_yolo_frames = 0
        self.cone_yolo_time = float("-inf")
        self.cone_yolo_confidence = 0.0
        self.cone_approach_yolo_frames = 0
        self.cone_approach_yolo_time = float("-inf")
        self.cone_approach_yolo_confidence = 0.0
        self.cone_lidar_distance_m = float("inf")
        self.cone_lidar_forward_distance_m = float("inf")
        self.cone_cluster_count = 0
        self.cone_cluster_time = float("-inf")
        self._reset_cone_approach_brake()
        self._reset_s_curve_entry_guard("integrated reset service")
        self._reset_post_red_turn_exit_window()
        self.avoidance_controller.reset()
        self.avoidance_state = self.avoidance_controller.state()
        self.avoidance_offset_pub.publish(Float32(data=0.0))
        self._publish_avoidance_path_request(
            now=time.monotonic(),
            scan_fresh=False,
            force_inactive=True,
        )
        response.success = True
        response.message = "integrated rule drive reset"
        return response

    def _publish_traffic_light_status(self) -> None:
        decision = self.traffic_light_decision
        frame = self.latest_traffic_light_frame
        self.traffic_light_status_pub.publish(
            String(
                data=(
                    f"action={decision.action.value} "
                    f"signal={decision.signal_name or 'none'} "
                    f"box_area_ratio={decision.box_area_ratio:.6f} "
                    f"stop_frames={self.traffic_light_controller.stop_frames} "
                    f"green_frames={self.traffic_light_controller.go_frames} "
                    f"left_frames={self.traffic_light_controller.left_frames} "
                    "left_absence_frames="
                    f"{self.traffic_light_controller.left_absence_frames} "
                    f"red_box={frame.red.box_area_ratio:.6f} "
                    f"yellow_box={frame.yellow.box_area_ratio:.6f} "
                    f"green_box={frame.green.box_area_ratio:.6f} "
                    f"left_box={frame.left.box_area_ratio:.6f} "
                    f"lap={self.race_lap_policy.current_lap} "
                    "race_started="
                    f"{int(self.race_lap_policy.race_started)} "
                    "shortcut_completed="
                    f"{int(self.race_lap_policy.shortcut_completed)} "
                    "shortcut_allowed="
                    f"{int(self.race_lap_policy.shortcut_allowed)} "
                    f"reason={decision.reason}"
                )
            )
        )

    def _publish_avoidance_path_request(
        self,
        *,
        now: float,
        scan_fresh: bool,
        force_inactive: bool = False,
    ) -> None:
        """Publish the metric obstacle consumed by the SITL bypass planner.

        Layout: active, observation_valid, bypass_side, obstacle_center_x,
        obstacle_y, obstacle_length, obstacle_width.
        """
        mode = self.avoidance_state.mode
        active = (
            not force_inactive
            and mode
            in {
                YoloLidarAvoidanceMode.AVOID_LEFT,
                YoloLidarAvoidanceMode.AVOID_RIGHT,
                YoloLidarAvoidanceMode.RETURN_CENTER,
            }
        )
        if mode == YoloLidarAvoidanceMode.AVOID_LEFT:
            bypass_side = 1.0
        elif mode == YoloLidarAvoidanceMode.AVOID_RIGHT:
            bypass_side = -1.0
        elif self.avoidance_state.lateral_offset_m > 0.0:
            bypass_side = 1.0
        elif self.avoidance_state.lateral_offset_m < 0.0:
            bypass_side = -1.0
        elif (
            self.avoidance_controller.preferred_mode
            == YoloLidarAvoidanceMode.AVOID_RIGHT
        ):
            bypass_side = -1.0
        else:
            bypass_side = 1.0

        obstacle_length = float(
            self.get_parameter("vehicle_body_length_m").value
        )
        obstacle_width = float(
            self.get_parameter("vehicle_body_width_m").value
        )
        obstacle_x = float("inf")
        obstacle_y = 0.0
        observation_valid = False
        obstacle = self.latest_vehicle_lidar_obstacle
        if scan_fresh and obstacle is not None:
            obstacle_x = max(
                0.0,
                float(obstacle.x_vehicle_m) + 0.5 * obstacle_length,
            )
            obstacle_y = float(obstacle.y_vehicle_m)
            obstacle_width = max(obstacle_width, float(obstacle.width_m))
            observation_valid = True
        elif (
            scan_fresh
            and math.isfinite(self.tracked_vehicle_distance_m)
            and now - self.tracked_vehicle_distance_time
            <= float(self.get_parameter("scan_timeout_sec").value)
        ):
            obstacle_x = max(
                0.0,
                self.tracked_vehicle_distance_m + 0.5 * obstacle_length,
            )
            observation_valid = True

        self.avoidance_path_request_pub.publish(
            Float32MultiArray(
                data=[
                    1.0 if active else 0.0,
                    1.0 if observation_valid else 0.0,
                    bypass_side,
                    obstacle_x,
                    obstacle_y,
                    obstacle_length,
                    obstacle_width,
                ]
            )
        )

    def _publish_status(self, output, now: float) -> None:
        if self.traffic_light_decision.action == TrafficLightAction.STOP:
            source_value = "TRAFFIC_LIGHT"
            source_label = self.traffic_light_decision.signal_name.upper()
        elif self.shortcut_latch.active:
            source_value = "SHORTCUT"
            phase_names = {
                0: "IDLE",
                1: "ENTER",
                2: "CRUISE",
                3: "EXIT",
                4: "DONE",
                11: "W1_ONLY_ENTRY",
                12: "Y1_LOCKED_ENTRY",
                13: "W1_Y1_ENTRY",
                14: "SEMANTIC_HANDOFF",
                15: "YELLOW_COUNT_FORCE",
            }
            source_label = "SHORTCUT_" + phase_names.get(
                int(round(self.shortcut_phase_code)), "UNKNOWN"
            )
        elif self.cone_bypass.active:
            source_value = "CONE_RULE"
            source_label = "CONE_RULE"
        elif self.avoidance_state.controls_vehicle:
            source_value = "YOLO_LIDAR_AVOIDANCE"
            source_label = self.avoidance_state.mode.value
        elif (
            self.avoidance_state.mode
            == YoloLidarAvoidanceMode.YOLO_TRACKING
        ):
            source_value = output.source.value
            source_label = "YOLO_WAIT_SIDE"
        elif self.shortcut_entry_search_active:
            source_value = "SHORTCUT"
            source_label = "SHORTCUT_W1_SEARCH"
        else:
            source_value = output.source.value
            source_label = (
                "RL/IMITATION"
                if output.source == CandidateSource.RL
                else "RULE"
            )
        status = (
            f"state={output.state.value} source={source_value} "
            f"mode_label={source_label} "
            f"cmd=[{output.angle_command:.1f},{output.speed_command:.1f}] "
            f"reason={output.reason}"
        )
        self.mode_pub.publish(String(data=source_value))
        self.status_pub.publish(String(data=status))
        status_key = (
            f"{output.state.value}:{source_value}:{output.reason}"
        )
        if status_key != self.last_status_key or now - self.last_status_time >= 0.5:
            self.get_logger().info(status)
            self.last_status_key = status_key
            self.last_status_time = now

    def _control_step(self) -> None:
        now = time.monotonic()
        dt = max(0.0, min(0.25, now - self.last_update_time))
        self.last_update_time = now
        scan_fresh = (
            now - self.scan_time
            <= float(self.get_parameter("scan_timeout_sec").value)
        )
        cone_hold_active = self._cone_disarm_hold_active(now)
        cone_reset_after_hold = bool(
            self.gate_arming_required
            and not self.drive_armed
            and self.cone_bypass.active
            and math.isfinite(self.gate_disarmed_time)
            and not cone_hold_active
        )
        if cone_reset_after_hold:
            self.cone_bypass.reset()
            # The retained mission state has now been cleared once. Return to
            # normal stationary precomputation instead of resetting every
            # control cycle while fresh cones remain in view.
            self.gate_disarmed_time = float("-inf")
            self.get_logger().warning(
                "CONE_RULE reset after SPACE hold timeout"
            )
        self._handle_shortcut_event(
            self.shortcut_latch.update(now_sec=now)
        )
        if self._shortcut_entry_search_expired(now):
            self._cancel_shortcut_entry_search(
                "[MISSION] W1 search timeout; LR-ASPP disabled, RULE retained"
            )
        if (
            self.drive_armed
            and bool(
                self.get_parameter("traffic_light_control_enabled").value
            )
        ):
            self.traffic_light_decision = (
                self.traffic_light_controller.update(
                    now_sec=now,
                    shortcut_active=False,
                )
            )
            self._handle_traffic_shortcut_request(now)
        self.shortcut_processing_pub.publish(
            Bool(
                data=bool(
                    self.shortcut_latch.active
                    or self.shortcut_entry_search_active
                )
            )
        )
        if not cone_hold_active and not cone_reset_after_hold:
            self._handle_cone_event(
                self.cone_bypass.update_presence(
                    sensor_present=self._cone_sensor_present(now),
                    now_sec=now,
                )
            )
        self._publish_cone_processing_gate(now)
        previous_avoidance_state = self.avoidance_state
        if self.shortcut_avoidance_suppression.active:
            self.avoidance_controller.reset()
            self.avoidance_state = self.avoidance_controller.state()
        elif (
            self.drive_armed
            and bool(self.get_parameter("vehicle_avoidance_enabled").value)
        ):
            return_diagnostics_fresh = command_timestamp_is_fresh(
                now_sec=now,
                command_time_sec=self.rule_diagnostics_time,
                timeout_sec=float(
                    self.get_parameter(
                        "vehicle_return_diagnostics_timeout_sec"
                    ).value
                ),
            )
            return_center_confirmed = (
                return_diagnostics_fresh
                and self.return_center_confirmation_frames
                >= max(
                    1,
                    int(
                        self.get_parameter(
                            "vehicle_return_required_frames"
                        ).value
                    ),
                )
            )
            self.avoidance_state = self.avoidance_controller.step(
                now_sec=now,
                dt_sec=dt,
                obstacle=self.latest_vehicle_lidar_obstacle,
                cone_active=self.cone_bypass.active,
                return_center_confirmed=return_center_confirmed,
            )
        else:
            self.avoidance_controller.reset()
            self.avoidance_state = self.avoidance_controller.state()
        if self.avoidance_state.mode != YoloLidarAvoidanceMode.RETURN_CENTER:
            self.return_center_confirmation_frames = 0
        return_center_active = (
            self.avoidance_state.mode
            == YoloLidarAvoidanceMode.RETURN_CENTER
        )
        avoidance_active = self.avoidance_state.mode in (
            YoloLidarAvoidanceMode.AVOID_LEFT,
            YoloLidarAvoidanceMode.AVOID_RIGHT,
        )
        self.avoidance_active_pub.publish(Bool(data=avoidance_active))
        self.avoidance_return_active_pub.publish(
            Bool(data=return_center_active)
        )
        avoidance_exit_trigger = SCurveEntryTrigger.NONE
        if green_car_avoidance_completed(
            previous_controls_vehicle=(
                previous_avoidance_state.controls_vehicle
            ),
            previous_target_class_name=(
                previous_avoidance_state.target_class_name
            ),
            controls_vehicle=self.avoidance_state.controls_vehicle,
        ):
            self.green_car_retrigger_blocked = True
            self.green_car_retrigger_frames = 0
            self.get_logger().warning(
                f"{ANSI_BLUE}[AVOIDANCE] GREEN_CAR COMPLETE; "
                f"RETRIGGER BLOCKED UNTIL S-CURVE ENTRY{ANSI_RESET}"
            )
            avoidance_exit_trigger = SCurveEntryTrigger.GREEN_CAR_EXIT
        elif red_car_avoidance_completed(
            previous_controls_vehicle=(
                previous_avoidance_state.controls_vehicle
            ),
            previous_target_class_name=(
                previous_avoidance_state.target_class_name
            ),
            controls_vehicle=self.avoidance_state.controls_vehicle,
        ):
            self.green_car_retrigger_blocked = False
            self.green_car_retrigger_frames = 0
            avoidance_exit_trigger = SCurveEntryTrigger.RED_CAR_EXIT
        if (
            self.drive_armed
            and not self.shortcut_latch.active
            and not self.cone_bypass.active
            and avoidance_exit_trigger != SCurveEntryTrigger.NONE
        ):
            self._start_s_curve_entry_guard(avoidance_exit_trigger)
        elif (
            self.avoidance_state.controls_vehicle
            and self.s_curve_entry_guard.state().active
        ):
            self._reset_s_curve_entry_guard("new vehicle avoidance started")
        self._update_post_red_turn_exit_window(
            now=now,
            red_avoidance_completed_now=(
                avoidance_exit_trigger == SCurveEntryTrigger.RED_CAR_EXIT
            ),
        )
        # Camera/yellow-side avoidance must move the rule target even when a
        # strict LiDAR obstacle cluster is unavailable. The controller ramps
        # this offset, so the path moves without a one-frame steering jump.
        external_lateral_offset_m = selected_external_lateral_offset(
            shortcut_left_lane_active=(
                self.shortcut_left_lane_offset_active
            ),
            shortcut_left_lane_offset_m=float(
                self.get_parameter("shortcut_left_lane_offset_m").value
            ),
            avoidance_offset_m=float(
                self.avoidance_state.lateral_offset_m
            ),
        )
        self.avoidance_offset_pub.publish(
            Float32(data=external_lateral_offset_m)
        )
        self._publish_avoidance_path_request(
            now=now,
            scan_fresh=scan_fresh,
        )
        obstacle = self.latest_vehicle_lidar_obstacle
        avoidance_mode_code = float(
            list(YoloLidarAvoidanceMode).index(
                self.avoidance_state.mode
            )
        )
        preferred_mode = self.avoidance_controller.preferred_mode
        preferred_side_code = (
            1.0
            if preferred_mode == YoloLidarAvoidanceMode.AVOID_LEFT
            else -1.0
            if preferred_mode == YoloLidarAvoidanceMode.AVOID_RIGHT
            else 0.0
        )
        self.avoidance_debug_pub.publish(
            Float32MultiArray(
                data=[
                    avoidance_mode_code,
                    float(self.avoidance_state.lateral_offset_m),
                    float(self.avoidance_state.tracked_distance_m),
                    float(self.avoidance_state.yolo_confidence),
                    float(obstacle.left_clearance_m)
                    if obstacle
                    else float("inf"),
                    float(obstacle.right_clearance_m)
                    if obstacle
                    else float("inf"),
                    float(obstacle.x_vehicle_m)
                    if obstacle
                    else float("inf"),
                    float(obstacle.y_vehicle_m)
                    if obstacle
                    else 0.0,
                    float(obstacle.width_m)
                    if obstacle
                    else 0.0,
                    preferred_side_code,
                    0.0,
                    self.avoidance_side_basis_code,
                    float(self.rule_cross_track_error_m),
                    float(self.return_center_confirmation_frames),
                    1.0 if return_center_active else 0.0,
                ]
            )
        )
        output = self.controller.step(
            SequentialHybridInput(
                scan_fresh=(scan_fresh or self.force_rule_only),
                rl_command_age_sec=now - self.rl_command_time,
                rl_angle_command=self.rl_command[0],
                rl_speed_command=self.rl_command[1],
                rule_command_age_sec=now - self.rule_command_time,
                rule_angle_command=self.rule_command[0],
                rule_speed_command=self.rule_command[1],
            ),
            dt_sec=dt,
        )
        left_approach_limit = left_signal_approach_speed_limit(
            left_detection_frames=(
                self.traffic_light_controller.left_frames
            ),
            decision_action=self.traffic_light_decision.action,
            maximum_speed_command=float(
                self.get_parameter("shortcut_entry_speed_command").value
            ),
            blocked_by_active_mission=bool(
                self.shortcut_latch.active
                or self.shortcut_entry_search_active
                or self.cone_bypass.active
                or self.avoidance_state.controls_vehicle
            ),
        )
        if self.traffic_light_decision.action == TrafficLightAction.STOP:
            output = replace(
                output,
                state=HybridState.SENSOR_STOP,
                angle_command=0.0,
                speed_command=0.0,
                reason=self.traffic_light_decision.reason,
            )
        elif (
            left_approach_limit is not None
            and output.state == HybridState.RUNNING
        ):
            output = replace(
                output,
                speed_command=(
                    min(float(output.speed_command), left_approach_limit)
                    if float(output.speed_command) > 0.0
                    else float(output.speed_command)
                ),
                reason=(
                    "left_4 approach; RULE steering retained; speed cap="
                    f"{left_approach_limit:.1f}"
                ),
            )
        elif self.shortcut_latch.active:
            shortcut_age = now - self.shortcut_command_time
            if (
                output.state == HybridState.RUNNING
                and shortcut_age
                <= float(
                    self.get_parameter(
                        "shortcut_candidate_timeout_sec"
                    ).value
                )
                and now - self.rule_command_time
                <= self.controller.config.candidate_hold_sec
            ):
                maximum_angle = (
                    self.controller.config.maximum_abs_angle_command
                )
                output = replace(
                    output,
                    state=HybridState.RUNNING,
                    angle_command=float(
                        np.clip(
                            self.shortcut_command[0],
                            -maximum_angle,
                            maximum_angle,
                        )
                    ),
                    # The shortcut candidate may apply an entry-only speed cap.
                    speed_command=float(
                        np.clip(
                            self.shortcut_command[1],
                            0.0,
                            self.controller.config.maximum_speed_command,
                        )
                    ),
                    reason=(
                        "shortcut override; entry speed cap applied; "
                        f"phase={int(round(self.shortcut_phase_code))}"
                    ),
                )
            else:
                output = replace(
                    output,
                    state=HybridState.SENSOR_STOP,
                    angle_command=0.0,
                    speed_command=0.0,
                    reason="shortcut candidate or RULE speed stale",
                )
        elif self.shortcut_entry_search_active:
            shortcut_age = now - self.shortcut_command_time
            if (
                output.state == HybridState.RUNNING
                and shortcut_age
                <= float(
                    self.get_parameter(
                        "shortcut_candidate_timeout_sec"
                    ).value
                )
                and now - self.rule_command_time
                <= self.controller.config.candidate_hold_sec
            ):
                maximum_angle = (
                    self.controller.config.maximum_abs_angle_command
                )
                output = replace(
                    output,
                    state=HybridState.RUNNING,
                    angle_command=float(
                        np.clip(
                            self.shortcut_command[0],
                            -maximum_angle,
                            maximum_angle,
                        )
                    ),
                    speed_command=float(
                        np.clip(
                            self.shortcut_command[1],
                            0.0,
                            self.controller.config.maximum_speed_command,
                        )
                    ),
                    reason=(
                        "shortcut pre-entry; entry speed cap applied; "
                        f"phase={int(round(self.shortcut_phase_code))}"
                    ),
                )
            # While the camera gate is starting, retain the existing RULE
            # output instead of inserting a motor stop.
        elif self.cone_bypass.active:
            if output.state == HybridState.RUNNING and scan_fresh:
                output = replace(
                    output,
                    state=HybridState.RUNNING,
                    angle_command=self._cone_target_to_command(
                        self.last_valid_cone_command[0]
                    ),
                    speed_command=float(self.last_valid_cone_command[1]),
                    reason="cone rule override",
                )
            else:
                output = replace(
                    output,
                    state=HybridState.SENSOR_STOP,
                    angle_command=0.0,
                    speed_command=0.0,
                    reason="cone LiDAR scan stale",
                )
        elif (
            self.avoidance_state.controls_vehicle
            and output.state == HybridState.RUNNING
        ):
            if (
                self.avoidance_state.mode
                == YoloLidarAvoidanceMode.WAIT_SIDE_CLEAR
            ):
                reason = "YOLO obstacle close; both lanes blocked"
                if self.avoidance_controller.config.immediate_on_yolo:
                    reason = (
                        "YOLO obstacle; yellow centerline side unavailable"
                    )
                if self.avoidance_side_basis_code < 0.0:
                    reason = "YOLO obstacle; curved-road side decision blocked"
                output = replace(
                    output,
                    angle_command=0.0,
                    speed_command=0.0,
                    reason=reason,
                )
            elif (
                now - self.rule_command_time
                <= self.controller.config.candidate_hold_sec
            ):
                speed = float(output.speed_command)
                speed_limit = self.avoidance_state.speed_limit_command
                if speed > 0.0 and speed_limit is not None:
                    speed = min(speed, float(speed_limit))
                right_settle_limit = right_avoidance_settling_speed_limit(
                    mode=self.avoidance_state.mode,
                    lateral_offset_m=self.avoidance_state.lateral_offset_m,
                    right_offset_m=self.avoidance_controller.config.right_offset_m,
                    tolerance_m=float(
                        self.get_parameter(
                            "vehicle_avoidance_right_settle_tolerance_m"
                        ).value
                    ),
                    speed_limit_command=float(
                        self.get_parameter(
                            "vehicle_avoidance_right_settle_speed_limit_command"
                        ).value
                    ),
                )
                if speed > 0.0 and right_settle_limit is not None:
                    speed = min(speed, right_settle_limit)
                settle_reason = ""
                if right_settle_limit is not None:
                    settle_reason = (
                        f", settle_cap={right_settle_limit:.1f}"
                    )
                output = replace(
                    output,
                    angle_command=float(self.rule_command[0]),
                    speed_command=speed,
                    reason=(
                        f"{self.avoidance_state.mode.value}; "
                        f"target={self.avoidance_state.target_class_name}; "
                        f"LiDAR={self.avoidance_state.tracked_distance_m:.2f}m, "
                        f"offset={self.avoidance_state.lateral_offset_m:+.2f}m"
                        f"{settle_reason}"
                    ),
                )
            else:
                output = replace(
                    output,
                    state=HybridState.SENSOR_STOP,
                    angle_command=0.0,
                    speed_command=0.0,
                    reason="avoidance lane-rule command stale",
                )
        self._update_cone_reentry_suppression(output, now=now)
        output = self._apply_s_curve_entry_guard(
            output,
            now=now,
            dt=dt,
        )
        approach_limit, approach_stage = cone_approach_speed_limit(
            enabled=bool(
                self.get_parameter("cone_approach_slowdown_enabled").value
            ),
            cone_active=self.cone_bypass.active,
            shortcut_active=bool(
                self.shortcut_latch.active
                or self.shortcut_entry_search_active
                or self.cone_reentry_suppression.active
            ),
            yolo_frames=self.cone_approach_yolo_frames,
            yolo_age_sec=now - self.cone_approach_yolo_time,
            yolo_timeout_sec=float(
                self.get_parameter("cone_yolo_timeout_sec").value
            ),
            confirmed_yolo_frames=int(
                self.get_parameter("cone_yolo_required_frames").value
            ),
            yolo_confidence=self.cone_approach_yolo_confidence,
            strong_yolo_confidence=float(
                self.get_parameter("cone_yolo_min_confidence").value
            ),
            cluster_count=self.cone_cluster_count,
            cluster_age_sec=now - self.cone_cluster_time,
            cluster_timeout_sec=float(
                self.get_parameter("cone_cluster_timeout_sec").value
            ),
            cluster_forward_distance_m=(
                self.cone_lidar_forward_distance_m
            ),
            confirmed_distance_m=float(
                self.get_parameter(
                    "cone_approach_confirmed_distance_m"
                ).value
            ),
            first_yolo_speed_command=float(
                self.get_parameter(
                    "cone_approach_first_speed_command"
                ).value
            ),
            confirmed_speed_command=float(
                self.get_parameter(
                    "cone_approach_confirmed_speed_command"
                ).value
            ),
        )
        if (
            approach_limit is not None
            and output.state == HybridState.RUNNING
            and output.speed_command > 0.0
        ):
            if output.speed_command > approach_limit:
                output = replace(
                    output,
                    speed_command=float(approach_limit),
                    reason=(
                        f"cone approach {approach_stage}; "
                        f"RULE steering retained; cap={approach_limit:.1f}"
                    ),
                )

        shortcut_controls = bool(
            self.shortcut_latch.active
            or self.shortcut_entry_search_active
        )
        if shortcut_controls:
            self._reset_cone_approach_brake()
        else:
            vehicle_speed_age_sec = now - self.vehicle_speed_time
            brake_required, required_distance_m, brake_reason = (
                cone_approach_brake_decision(
                    enabled=bool(
                        self.get_parameter(
                            "cone_approach_slowdown_enabled"
                        ).value
                    ),
                    cone_active=self.cone_bypass.active,
                    shortcut_active=shortcut_controls,
                    cluster_count=self.cone_cluster_count,
                    cluster_age_sec=now - self.cone_cluster_time,
                    cluster_timeout_sec=float(
                        self.get_parameter(
                            "cone_cluster_timeout_sec"
                        ).value
                    ),
                    cluster_forward_distance_m=(
                        self.cone_lidar_forward_distance_m
                    ),
                    vehicle_speed_mps=self.vehicle_speed_mps,
                    vehicle_speed_age_sec=vehicle_speed_age_sec,
                    vehicle_speed_timeout_sec=float(
                        self.get_parameter(
                            "vehicle_speed_timeout_sec"
                        ).value
                    ),
                    target_speed_mps=float(
                        self.get_parameter(
                            "cone_approach_target_speed_mps"
                        ).value
                    ),
                    deceleration_mps2=float(
                        self.get_parameter(
                            "cone_approach_deceleration_mps2"
                        ).value
                    ),
                    response_time_sec=float(
                        self.get_parameter(
                            "cone_approach_response_time_sec"
                        ).value
                    ),
                    distance_margin_m=float(
                        self.get_parameter(
                            "cone_approach_brake_margin_m"
                        ).value
                    ),
                    hard_stop_distance_m=float(
                        self.get_parameter(
                            "cone_approach_hard_stop_distance_m"
                        ).value
                    ),
                    stale_speed_stop_distance_m=float(
                        self.get_parameter(
                            "cone_approach_stale_speed_stop_distance_m"
                        ).value
                    ),
                )
            )
            if (
                not self.cone_approach_brake_hold_active
                and brake_required
                and output.state == HybridState.RUNNING
                and output.speed_command > 0.0
            ):
                self.cone_approach_brake_hold_active = True
                self.cone_approach_brake_release_frames = 0
                self.cone_approach_brake_required_distance_m = (
                    required_distance_m
                )
                self.cone_approach_brake_reason = brake_reason
                self.get_logger().warning(
                    "CONE ENTRY BRAKE START; "
                    f"reason={brake_reason}, "
                    f"forward={self.cone_lidar_forward_distance_m:.3f}m, "
                    f"required={required_distance_m:.3f}m, "
                    f"speed={self.vehicle_speed_mps:.3f}m/s"
                )

            if self.cone_approach_brake_hold_active:
                cone_command_fresh = command_timestamp_is_fresh(
                    now_sec=now,
                    command_time_sec=self.last_valid_cone_command_time,
                    timeout_sec=float(
                        self.get_parameter(
                            "cone_command_timeout_sec"
                        ).value
                    ),
                )
                release_ready = cone_brake_hold_release_ready(
                    cone_active=self.cone_bypass.active,
                    cone_command_fresh=cone_command_fresh,
                    vehicle_speed_mps=self.vehicle_speed_mps,
                    vehicle_speed_age_sec=vehicle_speed_age_sec,
                    vehicle_speed_timeout_sec=float(
                        self.get_parameter(
                            "vehicle_speed_timeout_sec"
                        ).value
                    ),
                    release_speed_mps=float(
                        self.get_parameter(
                            "cone_approach_release_speed_mps"
                        ).value
                    ),
                )
                self.cone_approach_brake_release_frames = (
                    self.cone_approach_brake_release_frames + 1
                    if release_ready
                    else 0
                )
                release_frames = max(
                    1,
                    int(
                        self.get_parameter(
                            "cone_approach_release_frames"
                        ).value
                    ),
                )
                if self.cone_approach_brake_release_frames >= release_frames:
                    self.get_logger().warning(
                        "CONE ENTRY BRAKE RELEASE; "
                        f"speed={self.vehicle_speed_mps:.3f}m/s, "
                        f"frames={self.cone_approach_brake_release_frames}"
                    )
                    self._reset_cone_approach_brake()
                else:
                    output = replace(
                        output,
                        state=HybridState.SENSOR_STOP,
                        angle_command=0.0,
                        speed_command=0.0,
                        reason=(
                            "cone entry brake hold; "
                            f"v={self.vehicle_speed_mps:.2f}m/s, "
                            "forward="
                            f"{self.cone_lidar_forward_distance_m:.2f}m, "
                            "required="
                            f"{self.cone_approach_brake_required_distance_m:.2f}m, "
                            f"trigger={self.cone_approach_brake_reason}"
                        ),
                    )
        if self.cone_bypass.active:
            candidate_mode = GateCandidateMode.CONE
        elif (
            not self.shortcut_latch.active
            and not self.shortcut_entry_search_active
            and not self.avoidance_state.controls_vehicle
            and output.source == CandidateSource.RULE
        ):
            candidate_mode = GateCandidateMode.RULE
        else:
            candidate_mode = GateCandidateMode.OTHER
        self.shadow_pub.publish(
            Float32MultiArray(
                data=[
                    output.angle_command,
                    output.speed_command,
                    float(candidate_mode),
                ]
            )
        )
        if self.motor_pub is not None:
            self.motor_pub.publish(
                Float32MultiArray(
                    data=[output.angle_command, output.speed_command]
                )
            )
        s_entry_state = self.s_curve_entry_guard.state()
        diagnostics = Float32MultiArray(
            data=[
                STATE_CODES[output.state],
                (
                    6.0
                    if self.traffic_light_decision.action
                    == TrafficLightAction.STOP
                    else 5.0
                    if self.shortcut_latch.active
                    else 2.0
                    if self.cone_bypass.active
                    else 3.0
                    if self.avoidance_state.controls_vehicle
                    else 4.0
                    if self.avoidance_state.mode
                    == YoloLidarAvoidanceMode.YOLO_TRACKING
                    else SOURCE_CODES[output.source]
                ),
                *[float(item.distance_m) for item in self.sectors],
                float(now - self.rl_command_time),
                float(now - self.rule_command_time),
                float(output.angle_command),
                float(output.speed_command),
                float(
                    list(TrafficLightAction).index(
                        self.traffic_light_decision.action
                    )
                ),
                float(self.traffic_light_decision.box_area_ratio),
                float(now - self.scan_time),
                float(now - self.last_valid_cone_command_time),
                float(self.vehicle_speed_mps),
                float(now - self.vehicle_speed_time),
                float(self.cone_lidar_forward_distance_m),
                float(self.cone_approach_brake_hold_active),
                float(self.cone_bypass.entry_streak),
                float(self.cone_approach_yolo_confidence),
                float(self.cone_approach_brake_required_distance_m),
                1.0 if s_entry_state.active else 0.0,
                float(list(SCurveEntryTrigger).index(s_entry_state.trigger)),
                float(s_entry_state.distance_m),
                1.0 if s_entry_state.straight_ready else 0.0,
                float(s_entry_state.curve_frames),
                1.0 if s_entry_state.overdue else 0.0,
                1.0 if self.post_red_turn_window.active else 0.0,
            ]
        )
        self.diagnostics_pub.publish(diagnostics)
        self._publish_traffic_light_status()
        self._publish_status(output, now)

    def stop(self) -> None:
        self.control_timer.cancel()
        self.s_curve_entry_guard.reset()
        self._reset_post_red_turn_exit_window()
        self.green_car_retrigger_blocked = False
        self.green_car_retrigger_frames = 0
        self.cone_reentry_suppression.reset()
        self.shortcut_entry_search_active = False
        self.shortcut_entry_search_started_time = float("-inf")
        self.shortcut_entry_ready = False
        self.avoidance_offset_pub.publish(Float32(data=0.0))
        self.avoidance_active_pub.publish(Bool(data=False))
        self.avoidance_return_active_pub.publish(Bool(data=False))
        self._publish_avoidance_path_request(
            now=time.monotonic(),
            scan_fresh=False,
            force_inactive=True,
        )
        self.cone_processing_pub.publish(Bool(data=False))
        self.shortcut_processing_pub.publish(Bool(data=False))
        command = Float32MultiArray(data=[0.0, 0.0])
        self.shadow_pub.publish(command)
        if self.motor_pub is not None:
            self.motor_pub.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = SequentialHybridDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        if rclpy.ok():
            node.stop()
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
