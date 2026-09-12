from __future__ import annotations

import math
import time
from bisect import bisect_right
from typing import Sequence

from geometry_msgs.msg import Point
try:
    from kaiev26_msgs.msg import Centerline, RoadSegment, RoadSegmentArray
except ModuleNotFoundError:
    # Keep pure geometry and interpolation utilities importable for review and
    # testing. Runtime use still requires the team's message-interface package.
    Centerline = RoadSegment = RoadSegmentArray = None
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def apply_steering_only(speed_command: float, steering_only: bool) -> float:
    return 0.0 if steering_only else float(speed_command)


def interpolate_clamped(
    value: float,
    inputs: Sequence[float],
    outputs: Sequence[float],
) -> float:
    if len(inputs) != len(outputs) or len(inputs) < 2:
        raise ValueError("lookup inputs and outputs must have the same length >= 2")
    if any(right <= left for left, right in zip(inputs, inputs[1:])):
        raise ValueError("lookup inputs must be strictly increasing")
    if value <= inputs[0]:
        return float(outputs[0])
    if value >= inputs[-1]:
        return float(outputs[-1])

    upper = bisect_right(inputs, value)
    lower = upper - 1
    ratio = (value - inputs[lower]) / (inputs[upper] - inputs[lower])
    return float(outputs[lower] + ratio * (outputs[upper] - outputs[lower]))


def inverse_lookup_table(
    inputs: Sequence[float],
    outputs: Sequence[float],
) -> tuple[list[float], list[float]]:
    if len(inputs) != len(outputs) or len(inputs) < 2:
        raise ValueError("lookup inputs and outputs must have the same length >= 2")
    inverse_pairs = sorted(zip(outputs, inputs))
    inverse_inputs = [float(value) for value, _ in inverse_pairs]
    inverse_outputs = [float(value) for _, value in inverse_pairs]
    interpolate_clamped(0.0, inverse_inputs, inverse_outputs)
    return inverse_inputs, inverse_outputs


def make_point(x: float, y: float, z: float = 0.0) -> Point:
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def pure_pursuit_curvature(
    target_x: float,
    target_y: float,
    control_point_x: float,
) -> float:
    relative_x = target_x - control_point_x
    distance_sq = max(0.05, relative_x * relative_x + target_y * target_y)
    return 2.0 * target_y / distance_sq


def midpoint_biased_toward_first(
    first_y: float,
    second_y: float,
    bias_m: float,
) -> float:
    midpoint = (first_y + second_y) * 0.5
    if abs(first_y - second_y) < 1.0e-6:
        return midpoint
    direction = 1.0 if first_y > second_y else -1.0
    return midpoint + direction * max(0.0, bias_m)


def offset_polyline(points: list[Point], right_offset_m: float) -> list[Point]:
    ordered = sorted(points, key=lambda point: point.x)
    if len(ordered) < 2:
        return []

    offset_points = []
    for index, point in enumerate(ordered):
        previous = ordered[max(0, index - 1)]
        following = ordered[min(len(ordered) - 1, index + 1)]
        tangent_x = float(following.x) - float(previous.x)
        tangent_y = float(following.y) - float(previous.y)
        tangent_length = math.hypot(tangent_x, tangent_y)
        if tangent_length < 1.0e-6:
            continue
        right_normal_x = tangent_y / tangent_length
        right_normal_y = -tangent_x / tangent_length
        offset_points.append(
            make_point(
                point.x + right_offset_m * right_normal_x,
                point.y + right_offset_m * right_normal_y,
                point.z,
            )
        )
    return sorted(offset_points, key=lambda point: point.x)


def propagate_path(
    points: list[Point],
    speed_mps: float,
    curvature: float,
    duration_sec: float,
) -> list[Point]:
    if duration_sec <= 0.0 or abs(speed_mps) < 1.0e-6:
        return [make_point(point.x, point.y, point.z) for point in points]

    distance = speed_mps * duration_sec
    yaw_delta = curvature * distance
    if abs(curvature) < 1.0e-6:
        translation_x = distance
        translation_y = 0.0
    else:
        translation_x = math.sin(yaw_delta) / curvature
        translation_y = (1.0 - math.cos(yaw_delta)) / curvature

    cos_yaw = math.cos(yaw_delta)
    sin_yaw = math.sin(yaw_delta)
    propagated = []
    for point in points:
        translated_x = float(point.x) - translation_x
        translated_y = float(point.y) - translation_y
        propagated.append(
            make_point(
                cos_yaw * translated_x + sin_yaw * translated_y,
                -sin_yaw * translated_x + cos_yaw * translated_y,
                point.z,
            )
        )
    return sorted(propagated, key=lambda point: point.x)


class LaneRuleDriver(Node):
    def __init__(self) -> None:
        if Centerline is None:
            raise RuntimeError(
                "kaiev26_msgs is required to run LaneRuleDriver; install the "
                "team message-interface package in the ROS environment"
            )
        super().__init__("xycar_lane_rule_driver")
        self.declare_parameter("road_segments_topic", "/perception/road_segments")
        self.declare_parameter("centerline_topic", "/perception/centerline")
        self.declare_parameter("centerline_fallback_enabled", True)
        self.declare_parameter("motor_topic", "/xycar_motor")
        self.declare_parameter("shadow_motor_topic", "/xycar_motor_shadow")
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("steering_only", False)
        self.declare_parameter("target_path_topic", "/rule_drive/target_path")
        self.declare_parameter("debug_markers_topic", "/rule_drive/debug_markers")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("lane_width_m", 0.80)
        self.declare_parameter("min_lane_width_m", 0.35)
        self.declare_parameter("max_lane_width_m", 1.20)
        self.declare_parameter("target_lane", "right")
        self.declare_parameter("yellow_offset_fallback_enabled", True)
        self.declare_parameter("lane_center_offset_m", 0.20)
        self.declare_parameter("path_bias_toward_yellow_m", 0.02)
        self.declare_parameter("min_segment_points", 4)
        self.declare_parameter("min_target_points", 3)
        self.declare_parameter("sample_count", 18)
        self.declare_parameter("wheel_base_m", 0.32)
        self.declare_parameter("control_point_x_m", -0.08)
        self.declare_parameter("steering_gain_rad_per_cmd", -0.0068)
        # The competition vehicle's measured steering map is intentionally
        # omitted from the public release. Supply a local calibration to use
        # real motor output.
        self.declare_parameter("use_measured_steering_map", False)
        self.declare_parameter(
            "steering_map_commands",
            [-1.0, 0.0, 1.0],
        )
        self.declare_parameter(
            "steering_map_curvatures",
            [1.0, 0.0, -1.0],
        )
        self.declare_parameter("angle_command_min", -1.0)
        self.declare_parameter("angle_command_max", 1.0)
        self.declare_parameter("lookahead_distance_m", 1.20)
        self.declare_parameter("min_lookahead_x_m", 0.45)
        self.declare_parameter("speed_command", 8.0)
        self.declare_parameter("min_speed_command", 5.0)
        self.declare_parameter("slow_down_angle_cmd", 18.0)
        self.declare_parameter("max_abs_angle_for_drive", 38.0)
        self.declare_parameter("perception_timeout_sec", 0.40)
        self.declare_parameter("command_rate_hz", 20.0)
        self.declare_parameter("hold_last_path_sec", 0.25)
        self.declare_parameter("prediction_enabled", True)
        self.declare_parameter("prediction_speed_command", 4.0)
        self.declare_parameter("speed_gain_mps_per_cmd", 0.080612)

        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.drive_enabled = bool(self.get_parameter("drive_enabled").value)
        self.steering_only = bool(self.get_parameter("steering_only").value)
        self.centerline_fallback_enabled = bool(
            self.get_parameter("centerline_fallback_enabled").value
        )
        self.lane_width_m = float(self.get_parameter("lane_width_m").value)
        self.min_lane_width_m = float(self.get_parameter("min_lane_width_m").value)
        self.max_lane_width_m = float(self.get_parameter("max_lane_width_m").value)
        self.target_lane = str(self.get_parameter("target_lane").value).lower()
        self.yellow_offset_fallback_enabled = bool(
            self.get_parameter("yellow_offset_fallback_enabled").value
        )
        self.lane_center_offset_m = float(
            self.get_parameter("lane_center_offset_m").value
        )
        self.path_bias_toward_yellow_m = float(
            self.get_parameter("path_bias_toward_yellow_m").value
        )
        self.min_segment_points = int(self.get_parameter("min_segment_points").value)
        self.min_target_points = int(self.get_parameter("min_target_points").value)
        self.sample_count = max(3, int(self.get_parameter("sample_count").value))
        self.wheel_base_m = float(self.get_parameter("wheel_base_m").value)
        self.control_point_x_m = float(self.get_parameter("control_point_x_m").value)
        self.steering_gain_rad_per_cmd = float(
            self.get_parameter("steering_gain_rad_per_cmd").value
        )
        self.use_measured_steering_map = bool(
            self.get_parameter("use_measured_steering_map").value
        )
        self.steering_map_commands = [
            float(value) for value in self.get_parameter("steering_map_commands").value
        ]
        self.steering_map_curvatures = [
            float(value) for value in self.get_parameter("steering_map_curvatures").value
        ]
        (
            self.measured_curvature_inputs,
            self.measured_curvature_commands,
        ) = inverse_lookup_table(
            self.steering_map_commands,
            self.steering_map_curvatures,
        )
        self.angle_command_min = float(self.get_parameter("angle_command_min").value)
        self.angle_command_max = float(self.get_parameter("angle_command_max").value)
        self.lookahead_distance_m = float(self.get_parameter("lookahead_distance_m").value)
        self.min_lookahead_x_m = float(self.get_parameter("min_lookahead_x_m").value)
        self.speed_command = float(self.get_parameter("speed_command").value)
        self.min_speed_command = float(self.get_parameter("min_speed_command").value)
        self.slow_down_angle_cmd = float(self.get_parameter("slow_down_angle_cmd").value)
        self.max_abs_angle_for_drive = float(self.get_parameter("max_abs_angle_for_drive").value)
        self.perception_timeout_sec = float(self.get_parameter("perception_timeout_sec").value)
        self.hold_last_path_sec = float(self.get_parameter("hold_last_path_sec").value)
        self.prediction_enabled = bool(self.get_parameter("prediction_enabled").value)
        self.prediction_speed_command = float(
            self.get_parameter("prediction_speed_command").value
        )
        self.speed_gain_mps_per_cmd = float(
            self.get_parameter("speed_gain_mps_per_cmd").value
        )

        self.motor_pub = None
        if self.drive_enabled:
            self.motor_pub = self.create_publisher(
                Float32MultiArray,
                str(self.get_parameter("motor_topic").value),
                10,
            )
        self.shadow_motor_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("shadow_motor_topic").value),
            10,
        )
        self.target_path_pub = self.create_publisher(
            Centerline,
            str(self.get_parameter("target_path_topic").value),
            10,
        )
        self.debug_markers_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter("debug_markers_topic").value),
            10,
        )
        self.create_subscription(
            RoadSegmentArray,
            str(self.get_parameter("road_segments_topic").value),
            self.on_road_segments,
            10,
        )
        self.create_subscription(
            Centerline,
            str(self.get_parameter("centerline_topic").value),
            self.on_centerline,
            10,
        )

        rate_hz = max(1.0, float(self.get_parameter("command_rate_hz").value))
        self.create_timer(1.0 / rate_hz, self.on_timer)

        self.last_segments_time = 0.0
        self.last_target_path: list[Point] = []
        self.last_header = None
        self.last_angle_command = 0.0
        self.last_speed_command = 0.0
        self.last_path_update_time = time.monotonic()
        self.prediction_active = False
        self.last_road_path_stamp: tuple[int, int] | None = None
        if not self.drive_enabled:
            mode = "SHADOW (motor output disabled)"
        elif self.steering_only:
            mode = "STEERING-ONLY (propulsion locked at zero)"
        else:
            mode = "AUTO"
        self.get_logger().info(f"lane rule driver ready: {mode}")

    def on_road_segments(self, msg: RoadSegmentArray) -> None:
        target_path = self.build_target_path(msg)
        if len(target_path) < self.min_target_points:
            return
        self.last_road_path_stamp = self.header_stamp_key(msg.header)
        self.accept_target_path(msg.header, target_path)

    def on_centerline(self, msg: Centerline) -> None:
        if not self.centerline_fallback_enabled:
            return
        if len(msg.points) < self.min_target_points:
            return
        if self.header_stamp_key(msg.header) == self.last_road_path_stamp:
            return
        self.get_logger().warn(
            "using perception centerline fallback",
            throttle_duration_sec=2.0,
        )
        self.accept_target_path(msg.header, list(msg.points))

    @staticmethod
    def header_stamp_key(header) -> tuple[int, int]:
        return int(header.stamp.sec), int(header.stamp.nanosec)

    def accept_target_path(self, header, target_path: list[Point]) -> None:
        self.last_target_path = target_path
        self.last_header = header
        now = time.monotonic()
        self.last_segments_time = now
        self.last_path_update_time = now
        self.prediction_active = False
        self.publish_target_path(header, target_path, predicted=False)
        self.publish_debug_markers(header, target_path, predicted=False)

    def build_target_path(self, msg: RoadSegmentArray) -> list[Point]:
        yellow_segments = [
            segment
            for segment in msg.segments
            if segment.type in {RoadSegment.TYPE_YESOL, RoadSegment.TYPE_YEDOT}
            and len(segment.points) >= self.min_segment_points
        ]
        white_segments = [
            segment
            for segment in msg.segments
            if segment.type in {RoadSegment.TYPE_WHSOL, RoadSegment.TYPE_WHDOT}
            and len(segment.points) >= self.min_segment_points
        ]
        if not yellow_segments:
            return []

        yellow = max(
            yellow_segments,
            key=lambda segment: float(segment.confidence) * max(1, len(segment.points)),
        )
        matching_white_segments = [
            segment
            for segment in white_segments
            if self.segment_matches_target_lane(yellow, segment)
        ]

        best_path: list[Point] = []
        best_score = -1.0
        for white in matching_white_segments:
            path, score = self.mid_path_between_segments(yellow, white)
            if score > best_score:
                best_path = path
                best_score = score
        if best_path:
            return best_path
        if not self.yellow_offset_fallback_enabled:
            return []

        side_sign = 1.0 if self.target_lane == "right" else -1.0
        fallback_offset_m = max(
            0.0,
            self.lane_center_offset_m - self.path_bias_toward_yellow_m,
        )
        fallback = offset_polyline(
            list(yellow.points),
            side_sign * fallback_offset_m,
        )
        if len(fallback) >= self.min_target_points:
            self.get_logger().warn(
                "using yellow-centerline offset fallback",
                throttle_duration_sec=2.0,
            )
            return fallback
        return []

    def segment_matches_target_lane(
        self,
        yellow: RoadSegment,
        white: RoadSegment,
    ) -> bool:
        if self.target_lane not in {"left", "right"}:
            return True
        bounds = self.overlap_x_bounds(yellow.points, white.points)
        if bounds is None:
            return False

        start_x, end_x = bounds
        lateral_offsets = []
        for index in range(5):
            x = start_x + (end_x - start_x) * index / 4.0
            yellow_y = self.y_at_x(yellow.points, x)
            white_y = self.y_at_x(white.points, x)
            if yellow_y is not None and white_y is not None:
                lateral_offsets.append(white_y - yellow_y)
        if not lateral_offsets:
            return False

        mean_offset = sum(lateral_offsets) / len(lateral_offsets)
        return mean_offset > 0.0 if self.target_lane == "left" else mean_offset < 0.0

    def mid_path_between_segments(
        self,
        yellow: RoadSegment,
        white: RoadSegment,
    ) -> tuple[list[Point], float]:
        bounds = self.overlap_x_bounds(yellow.points, white.points)
        if bounds is None:
            return [], -1.0
        start_x, end_x = bounds
        if end_x <= start_x:
            return [], -1.0

        path = []
        lane_width_errors = []
        for index in range(self.sample_count):
            ratio = index / float(self.sample_count - 1)
            x = start_x + (end_x - start_x) * ratio
            yellow_y = self.y_at_x(yellow.points, x)
            white_y = self.y_at_x(white.points, x)
            if yellow_y is None or white_y is None:
                continue
            width = abs(yellow_y - white_y)
            if width < self.min_lane_width_m or width > self.max_lane_width_m:
                continue
            path.append(
                make_point(
                    x,
                    midpoint_biased_toward_first(
                        yellow_y,
                        white_y,
                        self.path_bias_toward_yellow_m,
                    ),
                    0.0,
                )
            )
            lane_width_errors.append(abs(width - self.lane_width_m))

        if len(path) < self.min_target_points:
            return [], -1.0
        mean_error = sum(lane_width_errors) / max(1, len(lane_width_errors))
        score = len(path) - 3.0 * mean_error
        return path, score

    def on_timer(self) -> None:
        now = time.monotonic()
        age = now - self.last_segments_time
        if not self.last_target_path or age > self.perception_timeout_sec + self.hold_last_path_sec:
            self.stop_vehicle()
            return

        predicting = age > self.perception_timeout_sec
        if predicting:
            if not self.prediction_enabled:
                self.stop_vehicle()
                return
            self.propagate_last_path(now)

        target = self.lookahead_point(self.last_target_path)
        if target is None:
            if not predicting:
                self.stop_vehicle()
                return
            angle_command = self.last_angle_command
        else:
            angle_command = self.compute_angle_command(target)
        speed_command = self.compute_speed_command(
            angle_command,
            0.0 if predicting else age,
        )
        if predicting:
            speed_command = min(speed_command, self.prediction_speed_command)
            if not self.prediction_active:
                self.get_logger().warn(
                    "perception lost: propagating last lane path",
                    throttle_duration_sec=1.0,
                )
            self.prediction_active = True
            if self.last_header is not None:
                self.publish_target_path(
                    self.last_header,
                    self.last_target_path,
                    predicted=True,
                )
                self.publish_debug_markers(
                    self.last_header,
                    self.last_target_path,
                    predicted=True,
                )
        speed_command = apply_steering_only(speed_command, self.steering_only)
        self.last_angle_command = angle_command
        self.last_speed_command = speed_command
        self.publish_motor(angle_command, speed_command)

    def propagate_last_path(self, now: float) -> None:
        duration_sec = clamp(now - self.last_path_update_time, 0.0, 0.50)
        speed_mps = self.speed_gain_mps_per_cmd * self.last_speed_command
        curvature = self.curvature_for_command(self.last_angle_command)
        self.last_target_path = propagate_path(
            self.last_target_path,
            speed_mps,
            curvature,
            duration_sec,
        )
        self.last_path_update_time = now

    def curvature_for_command(self, angle_command: float) -> float:
        if self.use_measured_steering_map:
            return interpolate_clamped(
                angle_command,
                self.steering_map_commands,
                self.steering_map_curvatures,
            )
        if self.wheel_base_m <= 1.0e-6:
            return 0.0
        steering_rad = self.steering_gain_rad_per_cmd * angle_command
        return math.tan(steering_rad) / self.wheel_base_m

    def stop_vehicle(self) -> None:
        self.last_speed_command = 0.0
        self.publish_motor(0.0, 0.0)

    def lookahead_point(self, path: list[Point]) -> Point | None:
        forward = [point for point in path if point.x >= self.min_lookahead_x_m]
        if not forward:
            return None
        return min(forward, key=lambda point: abs(point.x - self.lookahead_distance_m))

    def compute_angle_command(self, target: Point) -> float:
        curvature = pure_pursuit_curvature(
            target.x,
            target.y,
            self.control_point_x_m,
        )
        if self.use_measured_steering_map:
            angle_command = interpolate_clamped(
                curvature,
                self.measured_curvature_inputs,
                self.measured_curvature_commands,
            )
            return clamp(angle_command, self.angle_command_min, self.angle_command_max)

        steering_rad = math.atan(self.wheel_base_m * curvature)
        if abs(self.steering_gain_rad_per_cmd) < 1.0e-6:
            return 0.0
        angle_command = steering_rad / self.steering_gain_rad_per_cmd
        return clamp(angle_command, self.angle_command_min, self.angle_command_max)

    def compute_speed_command(self, angle_command: float, perception_age: float) -> float:
        if perception_age > self.perception_timeout_sec:
            return 0.0
        abs_angle = abs(angle_command)
        if abs_angle >= self.max_abs_angle_for_drive:
            return 0.0
        if abs_angle <= self.slow_down_angle_cmd:
            return self.speed_command
        ratio = (abs_angle - self.slow_down_angle_cmd) / max(
            1.0,
            self.max_abs_angle_for_drive - self.slow_down_angle_cmd,
        )
        return self.speed_command + ratio * (self.min_speed_command - self.speed_command)

    def publish_motor(self, angle: float, speed: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(angle), float(speed)]
        self.shadow_motor_pub.publish(msg)
        if self.motor_pub is not None:
            self.motor_pub.publish(msg)

    def publish_target_path(self, header, path: list[Point], predicted: bool = False) -> None:
        msg = Centerline()
        msg.header = header
        if not msg.header.frame_id:
            msg.header.frame_id = self.base_frame_id
        msg.detection_id = 0
        msg.track_id = 0
        msg.points = path
        msg.confidence = 1.0
        msg.source = (
            "xycar_lane_rule_driver_predicted"
            if predicted
            else "xycar_lane_rule_driver"
        )
        self.target_path_pub.publish(msg)

    def publish_debug_markers(
        self,
        header,
        path: list[Point],
        predicted: bool = False,
    ) -> None:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.header = header
        if not delete_all.header.frame_id:
            delete_all.header.frame_id = self.base_frame_id
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        path_marker = self.line_marker(delete_all.header, 1, "rule_target_path", path)
        path_marker.color.r = 1.0 if predicted else 0.0
        path_marker.color.g = 0.45 if predicted else 1.0
        path_marker.color.b = 0.0 if predicted else 0.25
        path_marker.scale.x = 0.04
        markers.markers.append(path_marker)

        lookahead = self.lookahead_point(path)
        if lookahead is not None:
            point_marker = Marker()
            point_marker.header = delete_all.header
            point_marker.ns = "rule_lookahead"
            point_marker.id = 2
            point_marker.type = Marker.SPHERE
            point_marker.action = Marker.ADD
            point_marker.pose.position = lookahead
            point_marker.pose.orientation.w = 1.0
            point_marker.scale.x = 0.10
            point_marker.scale.y = 0.10
            point_marker.scale.z = 0.10
            point_marker.color.a = 1.0
            point_marker.color.r = 0.1
            point_marker.color.g = 0.45
            point_marker.color.b = 1.0
            markers.markers.append(point_marker)

        control_marker = Marker()
        control_marker.header = delete_all.header
        control_marker.ns = "rule_control_point"
        control_marker.id = 3
        control_marker.type = Marker.SPHERE
        control_marker.action = Marker.ADD
        control_marker.pose.position.x = self.control_point_x_m
        control_marker.pose.orientation.w = 1.0
        control_marker.scale.x = 0.08
        control_marker.scale.y = 0.08
        control_marker.scale.z = 0.08
        control_marker.color.a = 1.0
        control_marker.color.r = 1.0
        control_marker.color.g = 0.35
        control_marker.color.b = 0.0
        markers.markers.append(control_marker)
        self.debug_markers_pub.publish(markers)

    def line_marker(self, header, marker_id: int, namespace: str, points: list[Point]) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.03
        marker.color.a = 1.0
        marker.points = points
        return marker

    def overlap_x_bounds(self, first: list[Point], second: list[Point]) -> tuple[float, float] | None:
        if len(first) < 2 or len(second) < 2:
            return None
        start_x = max(min(point.x for point in first), min(point.x for point in second))
        end_x = min(max(point.x for point in first), max(point.x for point in second))
        return start_x, end_x

    def y_at_x(self, points: list[Point], target_x: float) -> float | None:
        if not points:
            return None
        sorted_points = sorted(points, key=lambda point: point.x)
        if target_x <= sorted_points[0].x:
            return float(sorted_points[0].y)
        if target_x >= sorted_points[-1].x:
            return float(sorted_points[-1].y)
        for start, end in zip(sorted_points, sorted_points[1:]):
            min_x = min(float(start.x), float(end.x))
            max_x = max(float(start.x), float(end.x))
            if target_x < min_x or target_x > max_x:
                continue
            dx = float(end.x) - float(start.x)
            if abs(dx) < 1.0e-6:
                return float(start.y)
            ratio = (target_x - float(start.x)) / dx
            return float(start.y) + ratio * (float(end.y) - float(start.y))
        return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneRuleDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_motor(0.0, 0.0)
        except Exception:
            # Launch shutdown can invalidate the ROS context between the check and publish.
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
