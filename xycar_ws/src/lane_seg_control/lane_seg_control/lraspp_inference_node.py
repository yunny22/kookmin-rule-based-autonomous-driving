#!/usr/bin/env python3
"""ROS 2 TorchScript lane perception without motor outputs."""

from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from kaiev26_msgs.msg import Centerline
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray, Header

from lane_seg_control.camera_input import (
    CameraRectifier,
    cv_image_to_message,
    decode_compressed_bgr,
    raw_image_to_bgr,
)
from lane_seg_control.canonical_adapter_node import (
    BevGeometry,
    CanonicalRenderConfig,
    build_bev_geometry,
    render_canonical_from_bev_masks,
    warp_semantic_masks_only,
)
from lane_seg_control.xbin_direct_centerline import (
    extract_anchor_centers,
    project_points_to_metric,
)


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
CameraMessage: TypeAlias = Image | CompressedImage


def prepare_model_input(
    frame: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Return a contiguous ImageNet-normalized RGB NCHW tensor array."""
    if frame.shape[:2] == (height, width):
        resized = frame
    else:
        resized = cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_AREA
        )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[np.newaxis, ...])


def masks_from_probabilities(
    probabilities: np.ndarray,
    *,
    white_class_id: int = 1,
    yellow_class_id: int = 2,
    white_confidence: float = 0.5,
    yellow_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert CxHxW semantic probabilities into disjoint lane masks."""
    if probabilities.ndim != 3:
        raise ValueError(
            f"semantic probabilities must be CxHxW, got {probabilities.shape}"
        )
    class_count = probabilities.shape[0]
    if not 0 <= white_class_id < class_count:
        raise ValueError(
            f"white class {white_class_id} is outside {class_count} classes"
        )
    if not 0 <= yellow_class_id < class_count:
        raise ValueError(
            f"yellow class {yellow_class_id} is outside {class_count} classes"
        )

    labels = np.argmax(probabilities, axis=0)
    confidence = np.max(probabilities, axis=0)
    white = (
        (labels == int(white_class_id))
        & (confidence >= float(white_confidence))
    ).astype(np.uint8) * 255
    yellow = (
        (labels == int(yellow_class_id))
        & (confidence >= float(yellow_confidence))
    ).astype(np.uint8) * 255
    return white, yellow


class LrasppInferenceNode(Node):
    """Publish audited Xbin lane outputs through the reusable TorchScript node."""

    def __init__(self) -> None:
        super().__init__("lane_seg_lraspp_inference")
        package_share = Path(get_package_share_directory("lane_seg_control"))
        self.declare_parameter(
            "model_path",
            str(
                package_share
                / "models"
                / "xbin_centerline_model.pt"
            ),
        )
        self.declare_parameter("image_topic", "/wide_camera/rect/image_raw")
        self.declare_parameter("use_compressed_image", False)
        self.declare_parameter("enable_rectify", False)
        self.declare_parameter("direct_model_rectify_enabled", False)
        self.declare_parameter("direct_model_rectify_oversample", 1)
        self.declare_parameter("camera_yaml", "")
        self.declare_parameter("rect_balance", 0.3)
        self.declare_parameter("max_input_age_sec", 0.0)
        self.declare_parameter("processed_image_topic", "/lane_seg/source_image")
        self.declare_parameter(
            "white_mask_topic", "/lane_seg/white_boundary_mask"
        )
        self.declare_parameter(
            "yellow_mask_topic", "/lane_seg/yellow_centerline_mask"
        )
        self.declare_parameter("debug_topic", "/lane_seg/debug_image")
        self.declare_parameter(
            "perception_debug_topic", "/perception/yolo_debug_image"
        )
        self.declare_parameter("diagnostics_topic", "/lane_seg/diagnostics")
        self.declare_parameter("input_width", 256)
        self.declare_parameter("input_height", 144)
        self.declare_parameter("white_class_id", 1)
        self.declare_parameter("yellow_class_id", 2)
        self.declare_parameter("white_confidence", 0.5)
        self.declare_parameter("yellow_confidence", 0.5)
        self.declare_parameter("cpu_threads", 4)
        self.declare_parameter("opencv_threads", 1)
        self.declare_parameter("output_qos_depth", 1)
        self.declare_parameter("debug_rate_hz", 1.0)
        self.declare_parameter("max_output_rate_hz", 0.0)
        self.declare_parameter("output_native_resolution", True)
        self.declare_parameter("publish_intermediate_topics", True)
        self.declare_parameter("diagnostics_component_counts_enabled", False)
        self.declare_parameter("direct_canonical_enabled", False)
        self.declare_parameter("direct_centerline_enabled", False)
        self.declare_parameter(
            "direct_centerline_topic", "/perception/xbin_direct_centerline"
        )
        self.declare_parameter("direct_centerline_anchor_stride_px", 4)
        self.declare_parameter("direct_centerline_anchor_offset_px", 2)
        self.declare_parameter("direct_centerline_minimum_points", 3)
        self.declare_parameter("direct_centerline_minimum_forward_m", 0.03)
        self.declare_parameter(
            "canonical_topic", "/perception/canonical_road_image"
        )
        self.declare_parameter(
            "canonical_white_topic", "/perception/canonical_white_mask"
        )
        self.declare_parameter(
            "canonical_yellow_topic", "/perception/canonical_yellow_mask"
        )
        self.declare_parameter(
            "canonical_valid_topic", "/perception/canonical_valid_mask"
        )
        self.declare_parameter("base_frame_id", "base_footprint")
        # Public placeholders only. A real vehicle needs a separately
        # measured camera/projective mapping.
        self.declare_parameter("src_tl_x_ratio", 0.35)
        self.declare_parameter("src_tl_y_ratio", 0.45)
        self.declare_parameter("src_tr_x_ratio", 0.65)
        self.declare_parameter("src_tr_y_ratio", 0.45)
        self.declare_parameter("src_br_x_ratio", 0.85)
        self.declare_parameter("src_br_y_ratio", 0.90)
        self.declare_parameter("src_bl_x_ratio", 0.15)
        self.declare_parameter("src_bl_y_ratio", 0.90)
        self.declare_parameter("dst_left_ratio", 0.25)
        self.declare_parameter("dst_right_ratio", 0.75)
        self.declare_parameter("dst_top_y_ratio", 0.0)
        self.declare_parameter("dst_bottom_y_ratio", 0.666666667)
        self.declare_parameter("bev_width", 640)
        self.declare_parameter("bev_height", 640)
        self.declare_parameter("bev_valid_lateral_margin_px", 0)
        self.declare_parameter("bev_valid_erode_px", 0)
        self.declare_parameter("bev_clip_to_source_polygon", False)
        self.declare_parameter("lateral_m_per_px", 2.0 / 640.0)
        self.declare_parameter("forward_m_per_px", 3.0 / 640.0)
        self.declare_parameter("canonical_width", 256)
        self.declare_parameter("canonical_height", 144)
        self.declare_parameter("canonical_lateral_range_m", 2.0)
        self.declare_parameter("canonical_forward_range_m", 3.0)
        self.declare_parameter("canonical_background_gray", 36)
        self.declare_parameter("canonical_line_width_px", 5)
        self.declare_parameter("canonical_white_fit_enabled", True)
        self.declare_parameter("canonical_white_fit_window_count", 9)
        self.declare_parameter("canonical_white_fit_margin_px", 24)
        self.declare_parameter("canonical_white_fit_min_pixels", 4)
        self.declare_parameter("canonical_white_fit_min_centers", 2)
        self.declare_parameter("canonical_white_fit_min_span_px", 8)
        self.declare_parameter("canonical_white_fit_residual_px", 6.0)
        self.declare_parameter("canonical_white_fit_line_width_px", 5)
        self.declare_parameter("canonical_yellow_divider_enabled", True)
        self.declare_parameter("canonical_yellow_divider_min_pixels", 3)
        self.declare_parameter("canonical_yellow_divider_residual_px", 6.0)
        self.declare_parameter("canonical_yellow_divider_line_width_px", 5)
        self.declare_parameter("canonical_yellow_normalize_enabled", False)
        self.declare_parameter("canonical_yellow_normalize_line_width_px", 5)
        self.declare_parameter("canonical_yellow_normalize_min_area_px", 3)
        self.declare_parameter("canonical_yellow_normalize_smoothing_rows", 5)

        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        self.white_class_id = int(self.get_parameter("white_class_id").value)
        self.yellow_class_id = int(self.get_parameter("yellow_class_id").value)
        self.white_confidence = float(
            self.get_parameter("white_confidence").value
        )
        self.yellow_confidence = float(
            self.get_parameter("yellow_confidence").value
        )
        self.debug_rate_hz = float(self.get_parameter("debug_rate_hz").value)
        self.max_output_rate_hz = max(
            0.0, float(self.get_parameter("max_output_rate_hz").value)
        )
        self.output_native_resolution = bool(
            self.get_parameter("output_native_resolution").value
        )
        self.use_compressed_image = bool(
            self.get_parameter("use_compressed_image").value
        )
        self.enable_rectify = bool(
            self.get_parameter("enable_rectify").value
        )
        self.direct_model_rectify_enabled = bool(
            self.get_parameter("direct_model_rectify_enabled").value
        )
        self.direct_model_rectify_oversample = max(
            1,
            int(
                self.get_parameter(
                    "direct_model_rectify_oversample"
                ).value
            ),
        )
        self.max_input_age_sec = max(
            0.0, float(self.get_parameter("max_input_age_sec").value)
        )
        self.publish_intermediate_topics = bool(
            self.get_parameter("publish_intermediate_topics").value
        )
        self.diagnostics_component_counts_enabled = bool(
            self.get_parameter(
                "diagnostics_component_counts_enabled"
            ).value
        )
        self.direct_canonical_enabled = bool(
            self.get_parameter("direct_canonical_enabled").value
        )
        self.direct_centerline_enabled = bool(
            self.get_parameter("direct_centerline_enabled").value
        )
        if not self.direct_canonical_enabled:
            self.publish_intermediate_topics = True
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.rectifier: CameraRectifier | None = None
        if self.enable_rectify:
            self.rectifier = CameraRectifier(
                str(self.get_parameter("camera_yaml").value),
                float(self.get_parameter("rect_balance").value),
            )
        if self.direct_model_rectify_enabled and self.rectifier is None:
            raise ValueError(
                "direct_model_rectify_enabled requires enable_rectify=true"
            )
        if self.direct_model_rectify_enabled and self.output_native_resolution:
            raise ValueError(
                "direct_model_rectify_enabled requires "
                "output_native_resolution=false"
            )
        self.geometry: BevGeometry | None = None
        self.geometry_input_size: tuple[int, int] | None = None
        self.bev_valid_mask: np.ndarray | None = None
        self.render_config = self.make_render_config()

        import torch

        self.torch = torch
        cpu_threads = max(1, int(self.get_parameter("cpu_threads").value))
        opencv_threads = max(1, int(self.get_parameter("opencv_threads").value))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cv2.setNumThreads(opencv_threads)

        model_path = Path(
            str(self.get_parameter("model_path").value)
        ).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Xbin centerline model not found: {model_path}"
            )
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.model = torch.jit.optimize_for_inference(model)
        warmup = torch.zeros(
            (1, 3, self.input_height, self.input_width), dtype=torch.float32
        )
        with torch.inference_mode():
            output = self.model(warmup)
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise RuntimeError(
                "Xbin centerline model must return an NxCxHxW tensor, "
                f"got {type(output)!r}"
            )
        if output.shape[1] <= max(self.white_class_id, self.yellow_class_id):
            raise RuntimeError(
                f"lane model has {output.shape[1]} classes but lane class IDs "
                f"are {self.white_class_id}/{self.yellow_class_id}"
            )

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(self.get_parameter("output_qos_depth").value)),
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(self.get_parameter("output_qos_depth").value)),
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.image_callback_group = MutuallyExclusiveCallbackGroup()
        self.inference_callback_group = MutuallyExclusiveCallbackGroup()
        self.processed_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("processed_image_topic").value),
            output_qos,
        )
        self.white_pub = self.create_publisher(
            Image,
            str(self.get_parameter("white_mask_topic").value),
            output_qos,
        )
        self.yellow_pub = self.create_publisher(
            Image,
            str(self.get_parameter("yellow_mask_topic").value),
            output_qos,
        )
        self.debug_pub = self.create_publisher(
            Image, str(self.get_parameter("debug_topic").value), input_qos
        )
        self.perception_debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("perception_debug_topic").value),
            input_qos,
        )
        self.diagnostics_pub = self.create_publisher(
            Float32MultiArray,
            str(self.get_parameter("diagnostics_topic").value),
            output_qos,
        )
        self.direct_centerline_pub = None
        self.direct_centerline_sequence = 0
        if self.direct_centerline_enabled:
            self.direct_centerline_pub = self.create_publisher(
                Centerline,
                str(self.get_parameter("direct_centerline_topic").value),
                10,
            )
        self.canonical_pub = None
        self.canonical_white_pub = None
        self.canonical_yellow_pub = None
        self.canonical_valid_pub = None
        if self.direct_canonical_enabled:
            self.canonical_pub = self.create_publisher(
                Image,
                str(self.get_parameter("canonical_topic").value),
                output_qos,
            )
            self.canonical_white_pub = self.create_publisher(
                Image,
                str(self.get_parameter("canonical_white_topic").value),
                output_qos,
            )
            self.canonical_yellow_pub = self.create_publisher(
                Image,
                str(self.get_parameter("canonical_yellow_topic").value),
                output_qos,
            )
            self.canonical_valid_pub = self.create_publisher(
                Image,
                str(self.get_parameter("canonical_valid_topic").value),
                output_qos,
            )
        message_type = CompressedImage if self.use_compressed_image else Image
        self.image_sub = self.create_subscription(
            message_type,
            str(self.get_parameter("image_topic").value),
            self.on_image,
            input_qos,
            callback_group=self.image_callback_group,
        )

        self.last_log_time = time.monotonic()
        self.last_debug_bucket: int | None = None
        self.latest_image: CameraMessage | None = None
        self.last_processed_stamp_ns: int | None = None
        self.received_image_count = 0
        self.stale_input_count = 0
        self.replaced_input_count = 0
        self.scheduler_tick_count = 0
        self.scheduler_stop = threading.Event()
        self.scheduler_thread: threading.Thread | None = None
        self.context.on_shutdown(self.stop_output_scheduler)
        if self.max_output_rate_hz > 0.0:
            self.scheduler_thread = threading.Thread(
                target=self.run_output_scheduler,
                name="lane_seg_output_scheduler",
                daemon=True,
            )
            self.scheduler_thread.start()
        self.get_logger().info(
            f"Xbin centerline perception ready: model={model_path}, "
            f"input={self.input_width}x{self.input_height}, classes="
            f"background/white/yellow=0/{self.white_class_id}/{self.yellow_class_id}, "
            f"thresholds={self.white_confidence:.2f}/{self.yellow_confidence:.2f}, "
            f"threads=torch:{cpu_threads},opencv:{opencv_threads}, "
            f"output={'native' if self.output_native_resolution else 'model'}, "
            f"source={'compressed' if self.use_compressed_image else 'raw'}, "
            f"rectify={self.enable_rectify}, direct_model_rectify="
            f"{self.direct_model_rectify_enabled}x"
            f"{self.direct_model_rectify_oversample}, direct_canonical="
            f"{self.direct_canonical_enabled}, direct_centerline="
            f"{self.direct_centerline_enabled}, "
            f"scheduler={'latest-frame timer' if self.max_output_rate_hz > 0.0 else 'input'}, "
            f"rate_limit={self.max_output_rate_hz:.1f}Hz"
        )

    def parameter_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def make_render_config(self) -> CanonicalRenderConfig:
        return CanonicalRenderConfig(
            lateral_m_per_px=self.parameter_float("lateral_m_per_px"),
            forward_m_per_px=self.parameter_float("forward_m_per_px"),
            lateral_range_m=self.parameter_float(
                "canonical_lateral_range_m"
            ),
            forward_range_m=self.parameter_float(
                "canonical_forward_range_m"
            ),
            output_width=int(self.get_parameter("canonical_width").value),
            output_height=int(self.get_parameter("canonical_height").value),
            background_gray=int(
                self.get_parameter("canonical_background_gray").value
            ),
            line_width_px=int(
                self.get_parameter("canonical_line_width_px").value
            ),
            white_fit_enabled=bool(
                self.get_parameter("canonical_white_fit_enabled").value
            ),
            white_fit_window_count=int(
                self.get_parameter("canonical_white_fit_window_count").value
            ),
            white_fit_margin_px=int(
                self.get_parameter("canonical_white_fit_margin_px").value
            ),
            white_fit_min_pixels=int(
                self.get_parameter("canonical_white_fit_min_pixels").value
            ),
            white_fit_min_centers=int(
                self.get_parameter("canonical_white_fit_min_centers").value
            ),
            white_fit_min_span_px=int(
                self.get_parameter("canonical_white_fit_min_span_px").value
            ),
            white_fit_residual_px=self.parameter_float(
                "canonical_white_fit_residual_px"
            ),
            white_fit_line_width_px=int(
                self.get_parameter(
                    "canonical_white_fit_line_width_px"
                ).value
            ),
            yellow_divider_enabled=bool(
                self.get_parameter("canonical_yellow_divider_enabled").value
            ),
            yellow_divider_min_pixels=int(
                self.get_parameter(
                    "canonical_yellow_divider_min_pixels"
                ).value
            ),
            yellow_divider_residual_px=self.parameter_float(
                "canonical_yellow_divider_residual_px"
            ),
            yellow_divider_line_width_px=int(
                self.get_parameter(
                    "canonical_yellow_divider_line_width_px"
                ).value
            ),
            yellow_normalize_enabled=bool(
                self.get_parameter(
                    "canonical_yellow_normalize_enabled"
                ).value
            ),
            yellow_normalize_line_width_px=int(
                self.get_parameter(
                    "canonical_yellow_normalize_line_width_px"
                ).value
            ),
            yellow_normalize_min_area_px=int(
                self.get_parameter(
                    "canonical_yellow_normalize_min_area_px"
                ).value
            ),
            yellow_normalize_smoothing_rows=int(
                self.get_parameter(
                    "canonical_yellow_normalize_smoothing_rows"
                ).value
            ),
        )

    def ensure_geometry(self, width: int, height: int) -> BevGeometry:
        if self.geometry is not None and self.geometry_input_size == (
            width,
            height,
        ):
            return self.geometry
        source = (
            self.parameter_float("src_tl_x_ratio"),
            self.parameter_float("src_tl_y_ratio"),
            self.parameter_float("src_tr_x_ratio"),
            self.parameter_float("src_tr_y_ratio"),
            self.parameter_float("src_br_x_ratio"),
            self.parameter_float("src_br_y_ratio"),
            self.parameter_float("src_bl_x_ratio"),
            self.parameter_float("src_bl_y_ratio"),
        )
        destination = (
            self.parameter_float("dst_left_ratio"),
            self.parameter_float("dst_right_ratio"),
            self.parameter_float("dst_top_y_ratio"),
            self.parameter_float("dst_bottom_y_ratio"),
        )
        self.geometry = build_bev_geometry(
            width,
            height,
            source_ratios=source,
            destination_ratios=destination,
            bev_width=int(self.get_parameter("bev_width").value),
            bev_height=int(self.get_parameter("bev_height").value),
        )
        self.geometry_input_size = (width, height)
        self.bev_valid_mask = None
        return self.geometry

    @staticmethod
    def message_stamp_ns(message: CameraMessage) -> int:
        return (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def input_age_sec(self, stamp_ns: int) -> float:
        if stamp_ns <= 0:
            return 0.0
        delta_ns = self.get_clock().now().nanoseconds - stamp_ns
        if delta_ns < 0 or delta_ns >= 60_000_000_000:
            return 0.0
        return delta_ns / 1.0e9

    def decode_frame(self, message: CameraMessage) -> np.ndarray:
        if isinstance(message, CompressedImage):
            frame = decode_compressed_bgr(message.data)
            if frame is None:
                raise ValueError("compressed camera payload is empty or invalid")
        else:
            frame = raw_image_to_bgr(message)
        if self.rectifier is not None:
            if self.direct_model_rectify_enabled:
                frame = self.rectifier.rectify_to_size(
                    frame,
                    self.input_width
                    * self.direct_model_rectify_oversample,
                    self.input_height
                    * self.direct_model_rectify_oversample,
                )
            else:
                frame = self.rectifier.rectify(frame)
        return frame

    def on_image(self, message: CameraMessage) -> None:
        self.received_image_count += 1
        if self.latest_image is not None:
            latest_stamp = self.message_stamp_ns(self.latest_image)
            if latest_stamp != self.last_processed_stamp_ns:
                self.replaced_input_count += 1
        self.latest_image = message
        if self.max_output_rate_hz <= 0.0:
            self.process_image(message)

    def on_output_timer(self) -> None:
        message = self.latest_image
        if message is None:
            return
        stamp_ns = self.message_stamp_ns(message)
        if self.last_processed_stamp_ns == stamp_ns:
            return
        if (
            self.last_processed_stamp_ns is not None
            and stamp_ns < self.last_processed_stamp_ns
        ):
            self.last_debug_bucket = None
        self.process_image(message)

    def run_output_scheduler(self) -> None:
        period = 1.0 / self.max_output_rate_hz
        deadline = time.monotonic() + period
        while not self.scheduler_stop.is_set():
            wait_sec = max(0.0, deadline - time.monotonic())
            if self.scheduler_stop.wait(wait_sec):
                break
            self.scheduler_tick_count += 1
            try:
                self.on_output_timer()
            except RuntimeError:
                if self.scheduler_stop.is_set() or not rclpy.ok(
                    context=self.context
                ):
                    break
                raise
            deadline += period
            if deadline < time.monotonic() - period:
                deadline = time.monotonic() + period

    def stop_output_scheduler(self) -> None:
        self.scheduler_stop.set()
        if (
            self.scheduler_thread is not None
            and self.scheduler_thread is not threading.current_thread()
        ):
            self.scheduler_thread.join(timeout=2.0)

    def destroy_node(self):
        self.stop_output_scheduler()
        return super().destroy_node()

    @staticmethod
    def output_header(message: CameraMessage, frame_id: str) -> Header:
        header = Header()
        header.stamp = message.header.stamp
        header.frame_id = frame_id
        return header

    def publish_image(
        self,
        publisher,
        frame: np.ndarray,
        encoding: str,
        header: Header,
    ) -> None:
        output = cv_image_to_message(frame, encoding, header)
        publisher.publish(output)

    def publish_direct_centerline(
        self,
        message: CameraMessage,
        yellow_mask: np.ndarray,
        yellow_probability: np.ndarray,
    ) -> tuple[int, float, float]:
        """Publish metric Xbin points without creating a BEV/canonical image."""
        if self.direct_centerline_pub is None:
            return 0, 0.0, 0.0
        started = time.perf_counter()
        anchor_stride = max(
            1,
            int(
                self.get_parameter(
                    "direct_centerline_anchor_stride_px"
                ).value
            ),
        )
        anchor_offset = int(
            self.get_parameter(
                "direct_centerline_anchor_offset_px"
            ).value
        ) % anchor_stride
        camera_points, confidences = extract_anchor_centers(
            yellow_mask,
            yellow_probability,
            anchor_stride_px=anchor_stride,
            anchor_offset_px=anchor_offset,
        )
        geometry = self.ensure_geometry(
            yellow_mask.shape[1], yellow_mask.shape[0]
        )
        metric_points, valid_indices = project_points_to_metric(
            camera_points,
            geometry.matrix,
            bev_width=geometry.width,
            bev_height=geometry.height,
            lateral_m_per_px=self.parameter_float("lateral_m_per_px"),
            forward_m_per_px=self.parameter_float("forward_m_per_px"),
            minimum_forward_m=self.parameter_float(
                "direct_centerline_minimum_forward_m"
            ),
            maximum_forward_m=self.parameter_float(
                "canonical_forward_range_m"
            ),
            maximum_abs_lateral_m=0.5
            * self.parameter_float("canonical_lateral_range_m"),
        )

        output = Centerline()
        output.header = self.output_header(message, self.base_frame_id)
        self.direct_centerline_sequence += 1
        output.detection_id = self.direct_centerline_sequence
        output.track_id = 0
        minimum_points = max(
            3,
            int(
                self.get_parameter(
                    "direct_centerline_minimum_points"
                ).value
            ),
        )
        if metric_points.shape[0] >= minimum_points:
            for forward_m, lateral_m in metric_points:
                point = Point()
                point.x = float(forward_m)
                point.y = float(lateral_m)
                point.z = 0.0
                output.points.append(point)
            valid_confidences = confidences[valid_indices]
            anchor_count = len(
                range(anchor_offset, yellow_mask.shape[0], anchor_stride)
            )
            coverage = metric_points.shape[0] / max(1, anchor_count)
            output.confidence = float(
                np.mean(valid_confidences) * coverage
            )
        else:
            output.confidence = 0.0
        output.source = "xbin_direct"
        self.direct_centerline_pub.publish(output)
        maximum_forward = (
            float(metric_points[-1, 0]) if metric_points.shape[0] else 0.0
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return len(output.points), maximum_forward, elapsed_ms

    def process_image(self, message: CameraMessage) -> None:
        callback_started = time.perf_counter()
        stamp_ns = self.message_stamp_ns(message)
        input_age_sec = self.input_age_sec(stamp_ns)
        if (
            self.max_input_age_sec > 0.0
            and input_age_sec > self.max_input_age_sec
        ):
            self.stale_input_count += 1
            self.last_processed_stamp_ns = stamp_ns
            return
        self.last_processed_stamp_ns = stamp_ns

        decode_started = time.perf_counter()
        try:
            frame = self.decode_frame(message)
        except Exception as exc:
            self.get_logger().error(f"camera conversion failed: {exc}")
            return
        decode_ms = (time.perf_counter() - decode_started) * 1000.0

        started = time.perf_counter()
        model_input = prepare_model_input(
            frame, self.input_width, self.input_height
        )
        try:
            tensor = self.torch.from_numpy(model_input)
            with self.torch.inference_mode():
                logits = self.model(tensor)
                probabilities = self.torch.softmax(logits, dim=1)[0].cpu().numpy()
        except Exception as exc:
            self.get_logger().error(f"lane model inference failed: {exc}")
            return

        white_small, yellow_small = masks_from_probabilities(
            probabilities,
            white_class_id=self.white_class_id,
            yellow_class_id=self.yellow_class_id,
            white_confidence=self.white_confidence,
            yellow_confidence=self.yellow_confidence,
        )
        debug_requested = (
            self.debug_pub.get_subscription_count() > 0
            or self.perception_debug_pub.get_subscription_count() > 0
        )
        source_requested = (
            self.publish_intermediate_topics
            and self.processed_image_pub.get_subscription_count() > 0
        )
        if self.output_native_resolution:
            output_frame = frame
            output_size = (frame.shape[1], frame.shape[0])
            white = cv2.resize(
                white_small, output_size, interpolation=cv2.INTER_NEAREST
            )
            yellow = cv2.resize(
                yellow_small, output_size, interpolation=cv2.INTER_NEAREST
            )
        else:
            output_frame = None
            if source_requested or debug_requested:
                output_frame = cv2.resize(
                    frame,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_AREA,
                )
            white = white_small
            yellow = yellow_small
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        direct_path_count, direct_path_forward_m, direct_path_ms = (
            self.publish_direct_centerline(
                message,
                yellow_small,
                probabilities[self.yellow_class_id],
            )
        )

        canonical_ms = 0.0
        if self.direct_canonical_enabled:
            canonical_started = time.perf_counter()
            geometry = self.ensure_geometry(white.shape[1], white.shape[0])
            bev_white, bev_yellow, valid = warp_semantic_masks_only(
                white,
                yellow,
                geometry,
                valid_lateral_margin_px=int(
                    self.get_parameter(
                        "bev_valid_lateral_margin_px"
                    ).value
                ),
                valid_erode_px=int(
                    self.get_parameter("bev_valid_erode_px").value
                ),
                clip_to_source_polygon=bool(
                    self.get_parameter(
                        "bev_clip_to_source_polygon"
                    ).value
                ),
                precomputed_valid=self.bev_valid_mask,
            )
            if self.bev_valid_mask is None:
                self.bev_valid_mask = valid
            result = render_canonical_from_bev_masks(
                bev_white,
                bev_yellow,
                valid,
                self.render_config,
            )
            canonical_header = self.output_header(
                message,
                self.base_frame_id,
            )
            self.publish_image(
                self.canonical_pub,
                result.stages.road_image,
                "bgr8",
                canonical_header,
            )
            for publisher, image in (
                (self.canonical_white_pub, result.stages.white_mask),
                (self.canonical_yellow_pub, result.stages.yellow_mask),
                (self.canonical_valid_pub, result.stages.valid_mask),
            ):
                if publisher.get_subscription_count() > 0:
                    self.publish_image(
                        publisher,
                        image,
                        "mono8",
                        canonical_header,
                    )
            canonical_ms = (
                time.perf_counter() - canonical_started
            ) * 1000.0

        camera_header = self.output_header(message, message.header.frame_id)
        if self.publish_intermediate_topics:
            if self.processed_image_pub.get_subscription_count() > 0:
                if output_frame is None:
                    raise RuntimeError("processed output frame was not prepared")
                self.publish_image(
                    self.processed_image_pub,
                    output_frame,
                    "bgr8",
                    camera_header,
                )
            if self.white_pub.get_subscription_count() > 0:
                self.publish_image(
                    self.white_pub,
                    white,
                    "mono8",
                    camera_header,
                )
            if self.yellow_pub.get_subscription_count() > 0:
                self.publish_image(
                    self.yellow_pub,
                    yellow,
                    "mono8",
                    camera_header,
                )

        debug_bucket = (
            int(stamp_ns * self.debug_rate_hz / 1_000_000_000)
            if stamp_ns > 0 and self.debug_rate_hz > 0.0
            else None
        )
        if (
            debug_requested
        ) and debug_bucket is not None and debug_bucket != self.last_debug_bucket:
            if output_frame is None:
                raise RuntimeError("debug output frame was not prepared")
            overlay = np.zeros_like(output_frame)
            overlay[white > 0] = (255, 255, 255)
            overlay[yellow > 0] = (0, 220, 255)
            selected = (white > 0) | (yellow > 0)
            debug = output_frame.copy()
            blended = cv2.addWeighted(
                output_frame, 0.45, overlay, 0.55, 0.0
            )
            debug[selected] = blended[selected]
            debug_message = cv_image_to_message(
                debug, "bgr8", camera_header
            )
            self.debug_pub.publish(debug_message)
            self.perception_debug_pub.publish(debug_message)
            self.last_debug_bucket = debug_bucket

        white_components = -1
        yellow_components = -1
        if self.diagnostics_component_counts_enabled:
            white_components = max(
                0, cv2.connectedComponents(white_small)[0] - 1
            )
            yellow_components = max(
                0, cv2.connectedComponents(yellow_small)[0] - 1
            )
        callback_elapsed_ms = (
            time.perf_counter() - callback_started
        ) * 1000.0
        small_area = float(max(1, white_small.size))
        if self.diagnostics_pub.get_subscription_count() > 0:
            diagnostics = Float32MultiArray()
            diagnostics.data = [
                float(elapsed_ms),
                float(1000.0 / elapsed_ms if elapsed_ms > 0.0 else 0.0),
                float(white_components),
                float(yellow_components),
                float(np.count_nonzero(white_small) / small_area),
                float(np.count_nonzero(yellow_small) / small_area),
                float(callback_elapsed_ms),
                float(self.received_image_count),
                float(self.scheduler_tick_count),
                float(decode_ms),
                float(canonical_ms),
                float(input_age_sec * 1000.0),
                float(self.replaced_input_count),
                float(self.stale_input_count),
                float(direct_path_count),
                float(direct_path_forward_m),
                float(direct_path_ms),
            ]
            self.diagnostics_pub.publish(diagnostics)

        now = time.monotonic()
        if now - self.last_log_time >= 5.0:
            self.get_logger().info(
                f"lane perception: {elapsed_ms:.1f}ms, "
                f"decode_rect={decode_ms:.1f}ms, canonical="
                f"{canonical_ms:.1f}ms, total={callback_elapsed_ms:.1f}ms, "
                f"source_age={input_age_sec * 1000.0:.1f}ms, "
                f"white_px={np.count_nonzero(white_small)}, "
                f"yellow_px={np.count_nonzero(yellow_small)}"
                f", direct_points={direct_path_count}, "
                f"direct_forward={direct_path_forward_m:.2f}m, "
                f"direct={direct_path_ms:.2f}ms"
            )
            self.last_log_time = now


def main() -> None:
    rclpy.init()
    node = LrasppInferenceNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
            node.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (KeyboardInterrupt, ExternalShutdownException):
                pass


if __name__ == "__main__":
    main()
