"""Interactive Space-key gate for integrated rule driving."""

from __future__ import annotations

from collections import deque
import math
import re
import select
import sys
import termios
import time
import tty
from typing import Mapping

from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Float32MultiArray, String

from .space_drive_gate_core import GateCandidateMode
from .space_drive_gate_core import RuleToConeSteeringBlend
from .space_drive_gate_core import SpaceDriveGateController


STATUS_PATTERN = re.compile(
    r"state=(?P<state>\S+)\s+source=(?P<source>\S+)\s+"
    r"mode_label=(?P<label>\S+).*?reason=(?P<reason>.*)$"
)
RULE_DIAGNOSTICS_CURVE_SPEED_MODE_INDEX = 37
RULE_DIAGNOSTICS_DEGRADED_SPEED_MODE_INDEX = 38
RULE_DIAGNOSTICS_CLASSIFIED_SPEED_INDEX = 39


def active_drive_mode(source: str, mode_label: str) -> str:
    avoidance_modes = {
        "AVOID_LEFT": "AVOIDANCE_LEFT",
        "AVOID_RIGHT": "AVOIDANCE_RIGHT",
        "WAIT_SIDE_CLEAR": "AVOIDANCE_BLOCKED",
        "RETURN_CENTER": "AVOIDANCE_RETURN",
    }
    if mode_label.startswith("YOLO_TRACKING("):
        return "AVOIDANCE_TRACK"
    if mode_label in avoidance_modes:
        return avoidance_modes[mode_label]
    if mode_label == "CONE_RULE" or source == "CONE_RULE":
        return "CONE"
    if source == "SHORTCUT":
        return "SHORTCUT"
    if source == "TRAFFIC_LIGHT":
        return "TRAFFIC_LIGHT"
    if source == "RULE":
        return "RULE"
    if source == "RL":
        return "MODEL"
    return "WAIT"


def classify_vesc_health(
    *,
    diagnostic_message: str,
    diagnostic_level: int,
    values: Mapping[str, str],
) -> tuple[str, str]:
    """Return a stable operator-facing VESC state and explanation."""
    guard_state = str(values.get("guard_state", "unknown"))
    voltage_text = str(values.get("voltage_input", "nan"))
    output_scale_text = str(values.get("voltage_output_scale", "nan"))
    fault_text = str(values.get("fault_code", "0"))
    try:
        voltage = float(voltage_text)
    except ValueError:
        voltage = float("nan")
    try:
        output_scale = float(output_scale_text)
    except ValueError:
        output_scale = float("nan")
    try:
        fault_code = int(fault_text)
    except ValueError:
        fault_code = 0

    voltage_label = f"{voltage:.2f}V" if math.isfinite(voltage) else "미확인"
    if guard_state == "low_voltage_limited":
        scale_label = (
            f"{100.0 * output_scale:.0f}%"
            if math.isfinite(output_scale)
            else "미확인"
        )
        return (
            "LOW_VOLTAGE_LIMITED",
            f"저전압 출력 제한: 전압={voltage_label}, 출력={scale_label}",
        )
    if guard_state == "fault_latched":
        if fault_code == 2 or (math.isfinite(voltage) and voltage <= 6.0):
            return (
                "LOW_VOLTAGE_STOP",
                f"저전압 보호 차단: 전압={voltage_label}, fault={fault_code}",
            )
        return (
            "VESC_FAULT_STOP",
            f"VESC 고장 차단: 전압={voltage_label}, fault={fault_code}",
        )
    message = str(diagnostic_message).strip()
    if diagnostic_level >= DiagnosticStatus.ERROR:
        return "VESC_ERROR_STOP", message or "VESC 진단 오류"
    if message in {
        "drive disabled",
        "motor command timeout",
        "VESC telemetry timeout",
    }:
        return "VESC_NOT_READY", message
    return "NORMAL", f"전압 정상: {voltage_label}"


def stop_reason_text(
    *,
    output_reason: str,
    selector_state: str,
    selector_reason: str,
    vesc_state: str,
    vesc_reason: str,
) -> str:
    if vesc_state in {
        "LOW_VOLTAGE_STOP",
        "VESC_FAULT_STOP",
        "VESC_ERROR_STOP",
        "VESC_NOT_READY",
    }:
        return vesc_reason
    if output_reason == "SPACE_STOP":
        return "SPACE 키로 정지"
    if output_reason == "CANDIDATE_STALE":
        return "통합 주행 후보 명령이 끊김"
    if output_reason == "SELECTOR_STOP":
        detail = selector_reason.strip()
        return detail if detail else "주행 선택기가 속도 0을 요청함"
    if selector_state not in {"UNKNOWN", "RUNNING"}:
        detail = selector_reason.strip()
        return detail if detail else f"주행 선택기 상태={selector_state}"
    return output_reason or "정지 원인 미확인"


def estimate_message_rate_hz(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    duration = float(timestamps[-1]) - float(timestamps[0])
    if duration <= 0.0:
        return 0.0
    return float(len(timestamps) - 1) / duration


def format_avoidance_basis(
    drive_mode: str,
    debug_data: list[float] | tuple[float, ...] | None,
    *,
    target_label: str,
    yolo_min_confidence: float,
    entry_distance_m: float,
    minimum_side_clearance_m: float,
) -> str:
    target_label = {"cone": "라바콘", "vehicle": "차량"}.get(
        target_label,
        target_label,
    )
    if debug_data is None or len(debug_data) < 9:
        return f"대상={target_label} | 회피 센서값 대기 중"

    distance = float(debug_data[2])
    confidence = float(debug_data[3])
    left = float(debug_data[4])
    right = float(debug_data[5])
    preferred_side = float(debug_data[9]) if len(debug_data) >= 10 else 0.0
    side_basis_code = float(debug_data[11]) if len(debug_data) >= 12 else 0.0
    yellow_side_available = (
        side_basis_code > 0.5 and abs(preferred_side) > 0.5
    )

    def obstacle_side_text() -> str:
        if preferred_side > 0.5 and yellow_side_available:
            obstacle_side = "오른쪽"
        elif preferred_side < -0.5 and yellow_side_available:
            obstacle_side = "왼쪽"
        else:
            return ""
        return f"노란 중앙선 기준 장애물={obstacle_side}"

    def distance_text(value: float) -> str:
        return f"{value:.2f}m" if math.isfinite(value) else "미확인"

    common = (
        f"대상={target_label} | YOLO={confidence:.2f}"
        f">={float(yolo_min_confidence):.2f} | "
        f"LiDAR={distance_text(distance)}"
    )
    if drive_mode == "AVOIDANCE_TRACK":
        side_text = obstacle_side_text()
        side_suffix = f" | {side_text}" if side_text else ""
        return (
            f"{common} | 진입기준={float(entry_distance_m):.2f}m | "
            f"회피 전 거리 추적 중{side_suffix}"
        )
    clearances = (
        f"좌측여유={distance_text(left)}, 우측여유={distance_text(right)}, "
        f"최소기준={float(minimum_side_clearance_m):.2f}m"
    )
    if drive_mode == "AVOIDANCE_LEFT":
        side_text = obstacle_side_text()
        if side_text:
            return (
                f"{common} | {side_text} | "
                "중앙선 기준 즉시 왼쪽으로 회피 (LiDAR 승인 미사용)"
            )
        return f"{common} | {clearances} | 좌측 공간이 더 넓어 좌회피"
    if drive_mode == "AVOIDANCE_RIGHT":
        side_text = obstacle_side_text()
        if side_text:
            return (
                f"{common} | {side_text} | "
                "중앙선 기준 즉시 오른쪽으로 회피 (LiDAR 승인 미사용)"
            )
        return f"{common} | {clearances} | 우측 공간이 더 넓어 우회피"
    if drive_mode == "AVOIDANCE_BLOCKED":
        side_text = obstacle_side_text()
        if not side_text:
            return f"{common} | 노란 중앙선 좌우판단 대기"
        if not math.isfinite(left) or not math.isfinite(right):
            return f"{common} | {side_text} | 회피 전환 대기"
        if preferred_side < -0.5 and right < minimum_side_clearance_m:
            return (
                f"{common} | {side_text} | {clearances} | "
                "오른쪽 공간 부족으로 대기"
            )
        if preferred_side > 0.5 and left < minimum_side_clearance_m:
            return (
                f"{common} | {side_text} | {clearances} | "
                "왼쪽 공간 부족으로 대기"
            )
        return f"{common} | {clearances} | 양쪽 공간 부족으로 대기"
    if drive_mode == "AVOIDANCE_RETURN":
        return f"{common} | 장애물 소실 확인 완료, 중앙 경로로 복귀"
    return f"{common} | 회피 비활성"


class SpaceDriveGate(Node):
    def __init__(self) -> None:
        super().__init__("space_drive_gate")
        self._declare_parameters()
        self.controller = SpaceDriveGateController(
            speed_command=float(self.get_parameter("speed_command").value),
            maximum_speed_command=float(
                self.get_parameter("maximum_speed_command").value
            ),
            maximum_abs_angle_command=float(
                self.get_parameter("maximum_abs_angle_command").value
            ),
            steering_only=bool(self.get_parameter("steering_only").value),
            adaptive_steering_speed_enabled=bool(
                self.get_parameter(
                    "adaptive_steering_speed_enabled"
                ).value
            ),
            turn_speed_command=float(
                self.get_parameter("turn_speed_command").value
            ),
            slowdown_start_angle_command=float(
                self.get_parameter(
                    "slowdown_start_angle_command"
                ).value
            ),
            full_slowdown_angle_command=float(
                self.get_parameter(
                    "full_slowdown_angle_command"
                ).value
            ),
        )
        self.rule_to_cone_blend = RuleToConeSteeringBlend(
            maximum_rate_command_per_sec=float(
                self.get_parameter(
                    "rule_to_cone_steering_rate_command_per_sec"
                ).value
            )
        )
        self.candidate = (0.0, 0.0)
        self.candidate_mode = GateCandidateMode.UNKNOWN
        self.candidate_time = float("-inf")
        self.selector_status = "waiting for selector status"
        self.source = "UNKNOWN"
        self.mode_label = "UNKNOWN"
        self.selector_state = "UNKNOWN"
        self.selector_reason = "waiting for selector status"
        self.quit_requested = False
        self.has_started = False
        self.last_display_key = ""
        self.vesc_state = "UNKNOWN"
        self.vesc_reason = "VESC 진단 대기"
        self.avoidance_debug: list[float] | None = None
        self.rule_path_speed_mode = "UNKNOWN"
        self.rule_path_speed_command = float("nan")
        self.rl_message_times: deque[float] = deque(maxlen=30)
        self.stdin_is_tty = sys.stdin.isatty()
        self.original_terminal_settings = None
        if self.stdin_is_tty:
            self.original_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        else:
            self.get_logger().error(
                "stdin is not a TTY; run this executable directly in a terminal"
            )

        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("candidate_topic").value),
            self._on_candidate,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("selector_status_topic").value),
            self._on_selector_status,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("rl_candidate_topic").value),
            self._on_rl_candidate,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("avoidance_debug_topic").value),
            self._on_avoidance_debug,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("rule_diagnostics_topic").value),
            self._on_rule_diagnostics,
            10,
        )
        self.create_subscription(
            DiagnosticArray,
            str(self.get_parameter("vesc_diagnostics_topic").value),
            self._on_diagnostics,
            10,
        )
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("motor_topic").value),
            10,
        )
        self.armed_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("drive_armed_topic").value),
            10,
        )
        self._publish_armed()
        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.timer = self.create_timer(1.0 / rate_hz, self._on_timer)
        self.get_logger().info(
            "READY; RULE->CONE steering blend="
            f"{self.rule_to_cone_blend.maximum_rate_command_per_sec:.1f} "
            "command/s"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter(
            "candidate_topic", "/hybrid_gate/xycar_motor_shadow"
        )
        self.declare_parameter(
            "selector_status_topic", "/hybrid_gate/status"
        )
        self.declare_parameter("rl_candidate_topic", "/rl/policy_motor_shadow")
        self.declare_parameter(
            "avoidance_debug_topic", "/hybrid/avoidance_debug"
        )
        self.declare_parameter(
            "rule_diagnostics_topic", "/rule_drive/diagnostics"
        )
        self.declare_parameter("vesc_diagnostics_topic", "/diagnostics")
        self.declare_parameter("vesc_diagnostic_name", "xycar_vesc_driver")
        self.declare_parameter("avoidance_target_label", "vehicle")
        self.declare_parameter("avoidance_yolo_min_confidence", 0.45)
        self.declare_parameter("avoidance_entry_distance_m", 1.20)
        self.declare_parameter("avoidance_minimum_side_clearance_m", 0.70)
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("drive_armed_topic", "/hybrid_gate/drive_armed")
        self.declare_parameter("speed_command", 3.0)
        self.declare_parameter("maximum_speed_command", 30.0)
        self.declare_parameter("maximum_abs_angle_command", 42.0)
        self.declare_parameter("steering_only", False)
        self.declare_parameter("adaptive_steering_speed_enabled", True)
        self.declare_parameter("turn_speed_command", 12.0)
        self.declare_parameter("slowdown_start_angle_command", 18.0)
        self.declare_parameter("full_slowdown_angle_command", 42.0)
        self.declare_parameter("candidate_timeout_sec", 0.40)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter(
            "rule_to_cone_steering_rate_command_per_sec", 180.0
        )

    def _on_candidate(self, message: Float32MultiArray) -> None:
        if len(message.data) < 2:
            return
        self.candidate = (float(message.data[0]), float(message.data[1]))
        self.candidate_mode = (
            RuleToConeSteeringBlend._mode(message.data[2])
            if len(message.data) >= 3
            else GateCandidateMode.UNKNOWN
        )
        self.candidate_time = time.monotonic()

    def _on_selector_status(self, message: String) -> None:
        self.selector_status = message.data
        match = STATUS_PATTERN.search(message.data)
        if match is None:
            return
        self.selector_state = match.group("state")
        self.source = match.group("source")
        self.mode_label = match.group("label")
        self.selector_reason = match.group("reason")

    def _on_rl_candidate(self, _message: Float32MultiArray) -> None:
        self.rl_message_times.append(time.monotonic())

    def _on_avoidance_debug(self, message: Float32MultiArray) -> None:
        self.avoidance_debug = [float(value) for value in message.data]

    def _on_rule_diagnostics(self, message: Float32MultiArray) -> None:
        if len(message.data) <= RULE_DIAGNOSTICS_CLASSIFIED_SPEED_INDEX:
            return
        if (
            float(
                message.data[RULE_DIAGNOSTICS_DEGRADED_SPEED_MODE_INDEX]
            )
            > 0.5
        ):
            mode = "DEGRADED"
        elif (
            float(message.data[RULE_DIAGNOSTICS_CURVE_SPEED_MODE_INDEX])
            > 0.5
        ):
            mode = "CURVE"
        else:
            mode = "STRAIGHT"
        self.rule_path_speed_mode = mode
        self.rule_path_speed_command = float(
            message.data[RULE_DIAGNOSTICS_CLASSIFIED_SPEED_INDEX]
        )

    def _on_diagnostics(self, message: DiagnosticArray) -> None:
        expected_name = str(
            self.get_parameter("vesc_diagnostic_name").value
        )
        for status in message.status:
            if status.name != expected_name and not status.name.endswith(
                "/" + expected_name
            ):
                continue
            values = {item.key: item.value for item in status.values}
            self.vesc_state, self.vesc_reason = classify_vesc_health(
                diagnostic_message=status.message,
                diagnostic_level=status.level,
                values=values,
            )
            return

    def _rl_rate_hz(self, now: float) -> float:
        if not self.rl_message_times or now - self.rl_message_times[-1] > 1.0:
            return 0.0
        return estimate_message_rate_hz(list(self.rl_message_times))

    def _read_key(self) -> str | None:
        if not self.stdin_is_tty:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)

    def _handle_key(self, key: str) -> None:
        if key == " ":
            running = self.controller.toggle()
            self.has_started = True
            self.last_display_key = ""
            if not running:
                self._publish_stop()
            self._publish_armed()
        elif key in {"q", "Q", "\x1b", "\x03"}:
            self.controller.stop()
            self._publish_stop()
            self.quit_requested = True

    def _display(self, output, now: float) -> None:
        if not self.has_started:
            return
        arm_state = "RUN" if self.controller.armed else "STOP"
        drive_mode = active_drive_mode(self.source, self.mode_label)
        vesc_stops_output = self.vesc_state in {
            "LOW_VOLTAGE_STOP",
            "VESC_FAULT_STOP",
            "VESC_ERROR_STOP",
            "VESC_NOT_READY",
        }
        stopped = (
            not self.controller.armed
            or output.reason != "SPACE_RUN"
            or vesc_stops_output
        )
        motion_state = "STOP" if stopped else "RUN"
        key = (
            f"{arm_state}:{motion_state}:{drive_mode}:{self.selector_state}:"
            f"{output.reason}:{self.vesc_state}"
        )
        if key == self.last_display_key:
            return
        model_hz = (
            f" | HZ={self._rl_rate_hz(now):.1f}"
            if drive_mode == "MODEL"
            else ""
        )
        path_speed = ""
        if drive_mode == "RULE" and self.rule_path_speed_mode != "UNKNOWN":
            path_speed = (
                f" | PATH={self.rule_path_speed_mode}"
                f"({self.rule_path_speed_command:.1f})"
            )
        power_status = (
            f" | 전원={self.vesc_state}({self.vesc_reason})"
            if self.vesc_state != "NORMAL"
            else ""
        )
        if stopped:
            reason = stop_reason_text(
                output_reason=output.reason,
                selector_state=self.selector_state,
                selector_reason=self.selector_reason,
                vesc_state=self.vesc_state,
                vesc_reason=self.vesc_reason,
            )
            self.get_logger().warning(
                f"[주행 상태 변경] MODE={drive_mode} | STATE=STOP | "
                f"이유={reason}{power_status}"
            )
        else:
            detail = ""
            if drive_mode.startswith("AVOIDANCE_"):
                detail = (
                    " | 판단="
                    + format_avoidance_basis(
                        drive_mode,
                        self.avoidance_debug,
                        target_label=str(
                            self.get_parameter("avoidance_target_label").value
                        ),
                        yolo_min_confidence=float(
                            self.get_parameter(
                                "avoidance_yolo_min_confidence"
                            ).value
                        ),
                        entry_distance_m=float(
                            self.get_parameter(
                                "avoidance_entry_distance_m"
                            ).value
                        ),
                        minimum_side_clearance_m=float(
                            self.get_parameter(
                                "avoidance_minimum_side_clearance_m"
                            ).value
                        ),
                    )
                )
            self.get_logger().info(
                f"[주행 상태 변경] MODE={drive_mode}{model_hz} | "
                f"STATE=RUN | SPEED={output.speed_command:.1f}"
                f"{path_speed}{power_status}{detail}"
            )
        self.last_display_key = key

    def _on_timer(self) -> None:
        while True:
            key = self._read_key()
            if key is None:
                break
            self._handle_key(key)
        now = time.monotonic()
        self._publish_armed()
        candidate_fresh = (
            now - self.candidate_time
            <= float(self.get_parameter("candidate_timeout_sec").value)
        )
        blend_was_active = self.rule_to_cone_blend.active
        candidate_angle = self.rule_to_cone_blend.apply(
            self.candidate[0],
            candidate_mode=self.candidate_mode,
            now_sec=now,
            output_enabled=bool(
                self.controller.armed
                and candidate_fresh
                and (
                    self.controller.steering_only
                    or self.candidate[1] > 0.0
                )
            ),
        )
        if not blend_was_active and self.rule_to_cone_blend.active:
            self.get_logger().warning(
                "[조향 연결] RULE -> CONE 전환 완화 시작 "
                f"({self.rule_to_cone_blend.maximum_rate_command_per_sec:.1f} "
                "command/s)"
            )
        elif blend_was_active and not self.rule_to_cone_blend.active:
            self.get_logger().info("[조향 연결] CONE 목표각 연결 완료")
        output = self.controller.command(
            candidate_fresh=candidate_fresh,
            candidate_angle_command=candidate_angle,
            candidate_speed_command=self.candidate[1],
        )
        self.motor_pub.publish(
            Float32MultiArray(
                data=[output.angle_command, output.speed_command]
            )
        )
        self._display(output, now)
        if self.quit_requested:
            raise KeyboardInterrupt

    def _publish_stop(self) -> None:
        self.motor_pub.publish(Float32MultiArray(data=[0.0, 0.0]))

    def _publish_armed(self) -> None:
        self.armed_pub.publish(Bool(data=self.controller.armed))

    def restore_terminal(self) -> None:
        if self.original_terminal_settings is not None:
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                self.original_terminal_settings,
            )
            self.original_terminal_settings = None

    def stop(self) -> None:
        self.controller.stop()
        self._publish_armed()
        self.timer.cancel()
        for _ in range(3):
            self._publish_stop()
            rclpy.spin_once(self, timeout_sec=0.02)


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = SpaceDriveGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.restore_terminal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
