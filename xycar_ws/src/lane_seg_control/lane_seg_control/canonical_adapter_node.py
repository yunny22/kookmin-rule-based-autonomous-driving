#!/usr/bin/env python3
"""Convert lane_seg_control semantic masks to the shared canonical contract."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from lane_seg_control.white_lane_fitter import (
    WhiteLaneFitResult,
    YellowCenterlineReference,
    compose_fitted_canonical,
    fit_white_lane_boundaries,
    fit_yellow_centerline_reference,
    normalize_yellow_fragments,
    render_white_lane_fit_debug,
)


@dataclass(frozen=True)
class CanonicalRoadStages:
    """Intermediate canonical-road outputs retained by this package."""

    road_image: np.ndarray
    white_mask: np.ndarray
    yellow_mask: np.ndarray
    pre_geometry_white_mask: np.ndarray
    pre_geometry_yellow_mask: np.ndarray
    post_geometry_white_mask: np.ndarray
    post_geometry_yellow_mask: np.ndarray
    valid_mask: np.ndarray


def _metric_crop(
    image: np.ndarray,
    *,
    lateral_m_per_px: float,
    forward_m_per_px: float,
    lateral_range_m: float,
    forward_range_m: float,
) -> np.ndarray:
    """Crop a centered, metric extent from a BEV image with zero padding."""
    if lateral_m_per_px <= 0.0 or forward_m_per_px <= 0.0:
        raise ValueError("BEV metric scales must be positive")
    if lateral_range_m <= 0.0 or forward_range_m <= 0.0:
        raise ValueError("canonical metric ranges must be positive")

    source_height, source_width = image.shape[:2]
    crop_width = max(1, int(round(lateral_range_m / lateral_m_per_px)))
    crop_height = max(1, int(round(forward_range_m / forward_m_per_px)))
    center_x = source_width // 2
    left = center_x - crop_width // 2
    right = left + crop_width
    top = source_height - crop_height
    bottom = source_height

    source_left = max(0, left)
    source_right = min(source_width, right)
    source_top = max(0, top)
    source_bottom = min(source_height, bottom)
    cropped = image[source_top:source_bottom, source_left:source_right]
    pad_left = max(0, -left)
    pad_right = max(0, right - source_width)
    pad_top = max(0, -top)
    pad_bottom = max(0, bottom - source_height)
    if any((pad_left, pad_right, pad_top, pad_bottom)):
        border_value = 0 if image.ndim == 2 else [0] * image.shape[2]
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
    return cropped


def _fixed_width_mask(mask: np.ndarray, line_width_px: int) -> np.ndarray:
    """Skeletonize then redraw a binary mask with a fixed line width."""
    work = (mask > 0).astype(np.uint8) * 255
    if not np.any(work):
        return work
    skeleton = np.zeros_like(work)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(work):
        eroded = cv2.erode(work, cross)
        opened = cv2.dilate(eroded, cross)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = eroded
    width = max(1, int(line_width_px))
    if width == 1:
        return skeleton
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.dilate(skeleton, kernel)


@dataclass(frozen=True)
class BevGeometry:
    source_points: np.ndarray
    matrix: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class CanonicalRenderConfig:
    lateral_m_per_px: float
    forward_m_per_px: float
    lateral_range_m: float
    forward_range_m: float
    output_width: int
    output_height: int
    background_gray: int
    line_width_px: int
    white_fit_enabled: bool
    white_fit_window_count: int
    white_fit_margin_px: int
    white_fit_min_pixels: int
    white_fit_min_centers: int
    white_fit_min_span_px: int
    white_fit_residual_px: float
    white_fit_line_width_px: int
    yellow_divider_enabled: bool
    yellow_divider_min_pixels: int
    yellow_divider_residual_px: float
    yellow_divider_line_width_px: int
    yellow_normalize_enabled: bool
    yellow_normalize_line_width_px: int
    yellow_normalize_min_area_px: int
    yellow_normalize_smoothing_rows: int


@dataclass(frozen=True)
class CanonicalRenderResult:
    stages: CanonicalRoadStages
    raw_stages: CanonicalRoadStages
    white_fit: WhiteLaneFitResult | None
    yellow_reference: YellowCenterlineReference | None


def build_bev_geometry(
    image_width: int,
    image_height: int,
    *,
    source_ratios: tuple[float, float, float, float, float, float, float, float],
    destination_ratios: tuple[float, float, float, float],
    bev_width: int,
    bev_height: int,
) -> BevGeometry:
    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = source_ratios
    dst_left, dst_right, dst_top, dst_bottom = destination_ratios
    source = np.float32(
        [
            [tl_x * image_width, tl_y * image_height],
            [tr_x * image_width, tr_y * image_height],
            [br_x * image_width, br_y * image_height],
            [bl_x * image_width, bl_y * image_height],
        ]
    )
    destination = np.float32(
        [
            [dst_left * bev_width, dst_top * bev_height],
            [dst_right * bev_width, dst_top * bev_height],
            [dst_right * bev_width, dst_bottom * bev_height],
            [dst_left * bev_width, dst_bottom * bev_height],
        ]
    )
    return BevGeometry(
        source_points=source,
        matrix=cv2.getPerspectiveTransform(source, destination),
        width=int(bev_width),
        height=int(bev_height),
    )


def build_bev_valid_mask(
    input_shape: tuple[int, int],
    geometry: BevGeometry,
    *,
    valid_lateral_margin_px: int,
    valid_erode_px: int,
    clip_to_source_polygon: bool,
) -> np.ndarray:
    output_size = (geometry.width, geometry.height)
    if clip_to_source_polygon:
        source_valid = np.zeros(input_shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            source_valid,
            np.rint(geometry.source_points).astype(np.int32),
            255,
            lineType=cv2.LINE_8,
        )
        valid = cv2.warpPerspective(
            source_valid,
            geometry.matrix,
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        valid = np.full(
            (geometry.height, geometry.width), 255, dtype=np.uint8
        )
    margin = max(0, int(valid_lateral_margin_px))
    if margin > 0:
        valid = cv2.dilate(
            valid, np.ones((1, margin * 2 + 1), dtype=np.uint8)
        )
    erode = max(0, int(valid_erode_px))
    if erode > 0:
        size = erode * 2 + 1
        valid = cv2.erode(valid, np.ones((size, size), dtype=np.uint8))
    return valid


def warp_semantic_masks_only(
    white_mask: np.ndarray,
    yellow_mask: np.ndarray,
    geometry: BevGeometry,
    *,
    valid_lateral_margin_px: int,
    valid_erode_px: int,
    clip_to_source_polygon: bool = True,
    precomputed_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if white_mask.shape != yellow_mask.shape:
        raise ValueError(
            f"semantic mask dimensions differ: {white_mask.shape} "
            f"!= {yellow_mask.shape}"
        )
    output_size = (geometry.width, geometry.height)
    if precomputed_valid is None:
        valid = build_bev_valid_mask(
            white_mask.shape,
            geometry,
            valid_lateral_margin_px=valid_lateral_margin_px,
            valid_erode_px=valid_erode_px,
            clip_to_source_polygon=clip_to_source_polygon,
        )
    else:
        valid = precomputed_valid
        expected_shape = (geometry.height, geometry.width)
        if valid.shape != expected_shape:
            raise ValueError(
                f"precomputed valid mask has shape {valid.shape}, "
                f"expected {expected_shape}"
            )
    bev_white = cv2.warpPerspective(
        (white_mask > 0).astype(np.uint8) * 255,
        geometry.matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    bev_yellow = cv2.warpPerspective(
        (yellow_mask > 0).astype(np.uint8) * 255,
        geometry.matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (
        cv2.bitwise_and(bev_white, valid),
        cv2.bitwise_and(bev_yellow, valid),
        valid,
    )


def warp_semantic_masks(
    image: np.ndarray,
    white_mask: np.ndarray,
    yellow_mask: np.ndarray,
    geometry: BevGeometry,
    *,
    valid_lateral_margin_px: int,
    valid_erode_px: int,
    clip_to_source_polygon: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    output_size = (geometry.width, geometry.height)
    bev_white, bev_yellow, valid = warp_semantic_masks_only(
        white_mask,
        yellow_mask,
        geometry,
        valid_lateral_margin_px=valid_lateral_margin_px,
        valid_erode_px=valid_erode_px,
        clip_to_source_polygon=clip_to_source_polygon,
    )

    bev_image = cv2.warpPerspective(
        image,
        geometry.matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(70, 70, 70),
    )
    return bev_image, bev_white, bev_yellow, valid


def render_canonical_from_bev_masks(
    bev_white: np.ndarray,
    bev_yellow: np.ndarray,
    valid: np.ndarray,
    config: CanonicalRenderConfig,
) -> CanonicalRenderResult:
    metric_white = _metric_crop(
        (bev_white > 0).astype(np.uint8) * 255,
        lateral_m_per_px=config.lateral_m_per_px,
        forward_m_per_px=config.forward_m_per_px,
        lateral_range_m=config.lateral_range_m,
        forward_range_m=config.forward_range_m,
    )
    metric_yellow = _metric_crop(
        (bev_yellow > 0).astype(np.uint8) * 255,
        lateral_m_per_px=config.lateral_m_per_px,
        forward_m_per_px=config.forward_m_per_px,
        lateral_range_m=config.lateral_range_m,
        forward_range_m=config.forward_range_m,
    )
    metric_valid = _metric_crop(
        (valid > 0).astype(np.uint8) * 255,
        lateral_m_per_px=config.lateral_m_per_px,
        forward_m_per_px=config.forward_m_per_px,
        lateral_range_m=config.lateral_range_m,
        forward_range_m=config.forward_range_m,
    )
    metric_white[metric_yellow > 0] = 0
    metric_white = cv2.bitwise_and(metric_white, metric_valid)
    metric_yellow = cv2.bitwise_and(metric_yellow, metric_valid)
    close_kernel = np.ones((3, 3), dtype=np.uint8)
    metric_white = cv2.morphologyEx(
        metric_white,
        cv2.MORPH_CLOSE,
        close_kernel,
    )
    metric_yellow = cv2.morphologyEx(
        metric_yellow,
        cv2.MORPH_CLOSE,
        close_kernel,
    )
    output_size = (config.output_width, config.output_height)
    white_mask = cv2.resize(
        metric_white,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    yellow_mask = cv2.resize(
        metric_yellow,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    output_valid = cv2.resize(
        metric_valid,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    pre_geometry_white = white_mask.copy()
    pre_geometry_yellow = yellow_mask.copy()
    white_mask = _fixed_width_mask(white_mask, config.line_width_px)
    yellow_mask = _fixed_width_mask(yellow_mask, config.line_width_px)
    white_mask = cv2.bitwise_and(white_mask, output_valid)
    yellow_mask = cv2.bitwise_and(yellow_mask, output_valid)
    white_mask[yellow_mask > 0] = 0
    canonical = np.full(
        (config.output_height, config.output_width, 3),
        int(np.clip(config.background_gray, 0, 255)),
        dtype=np.uint8,
    )
    canonical[white_mask > 0] = (255, 255, 255)
    canonical[yellow_mask > 0] = (0, 220, 255)
    stages = CanonicalRoadStages(
        road_image=canonical,
        white_mask=white_mask,
        yellow_mask=yellow_mask,
        pre_geometry_white_mask=pre_geometry_white,
        pre_geometry_yellow_mask=pre_geometry_yellow,
        post_geometry_white_mask=pre_geometry_white.copy(),
        post_geometry_yellow_mask=pre_geometry_yellow.copy(),
        valid_mask=output_valid,
    )

    raw_stages = stages
    white_fit = None
    yellow_reference = None
    if config.white_fit_enabled:
        if config.yellow_divider_enabled:
            yellow_reference = fit_yellow_centerline_reference(
                raw_stages.yellow_mask,
                min_pixels=config.yellow_divider_min_pixels,
                residual_threshold_px=config.yellow_divider_residual_px,
                line_width_px=config.yellow_divider_line_width_px,
            )
        divider = (
            yellow_reference.x_by_y
            if yellow_reference is not None and yellow_reference.valid
            else None
        )
        white_fit = fit_white_lane_boundaries(
            raw_stages.white_mask,
            window_count=config.white_fit_window_count,
            window_margin_px=config.white_fit_margin_px,
            min_pixels_per_window=config.white_fit_min_pixels,
            min_centers=config.white_fit_min_centers,
            min_span_px=config.white_fit_min_span_px,
            residual_threshold_px=config.white_fit_residual_px,
            line_width_px=config.white_fit_line_width_px,
            divider_x_by_y=divider,
        )
        output_yellow = raw_stages.yellow_mask
        if config.yellow_normalize_enabled:
            output_yellow = normalize_yellow_fragments(
                output_yellow,
                line_width_px=config.yellow_normalize_line_width_px,
                min_component_area_px=config.yellow_normalize_min_area_px,
                smoothing_window_rows=config.yellow_normalize_smoothing_rows,
            )
        fitted_road, fitted_white = compose_fitted_canonical(
            white_fit.mask,
            output_yellow,
            raw_stages.valid_mask,
            background_gray=config.background_gray,
        )
        stages = replace(
            raw_stages,
            road_image=fitted_road,
            white_mask=fitted_white,
            yellow_mask=output_yellow,
        )
    return CanonicalRenderResult(
        stages=stages,
        raw_stages=raw_stages,
        white_fit=white_fit,
        yellow_reference=yellow_reference,
    )


class CanonicalAdapterNode(Node):
    """Preserve the canonical image contract while replacing lane semantics."""

    def __init__(self) -> None:
        super().__init__("lane_seg_canonical_adapter")
        self.declare_parameter("image_topic", "/wide_camera/rect/image_raw")
        self.declare_parameter(
            "white_mask_topic", "/lane_seg/white_boundary_mask"
        )
        self.declare_parameter(
            "yellow_mask_topic", "/lane_seg/yellow_centerline_mask"
        )
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
        self.declare_parameter("debug_topic", "/perception/debug_image")
        self.declare_parameter(
            "canonical_white_fit_debug_topic",
            "/perception/canonical_white_fit_debug",
        )
        self.declare_parameter("bev_color_topic", "/lane_seg_bev/color")
        self.declare_parameter("bev_white_topic", "/lane_seg_bev/white_mask")
        self.declare_parameter("bev_yellow_topic", "/lane_seg_bev/yellow_mask")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("input_is_bev", False)

        self.declare_parameter("src_tl_x_ratio", 0.437418509)
        self.declare_parameter("src_tl_y_ratio", 0.476969898)
        self.declare_parameter("src_tr_x_ratio", 0.662089539)
        self.declare_parameter("src_tr_y_ratio", 0.478676707)
        self.declare_parameter("src_br_x_ratio", 0.916280746)
        self.declare_parameter("src_br_y_ratio", 0.614458919)
        self.declare_parameter("src_bl_x_ratio", 0.214667964)
        self.declare_parameter("src_bl_y_ratio", 0.597002029)
        self.declare_parameter("dst_left_ratio", 0.15)
        self.declare_parameter("dst_right_ratio", 0.85)
        self.declare_parameter("dst_top_y_ratio", 0.0)
        self.declare_parameter("dst_bottom_y_ratio", 0.666666667)
        self.declare_parameter("bev_width", 640)
        self.declare_parameter("bev_height", 220)
        self.declare_parameter("bev_valid_lateral_margin_px", 6)
        self.declare_parameter("bev_valid_erode_px", 4)
        self.declare_parameter("bev_clip_to_source_polygon", True)
        self.declare_parameter("lateral_m_per_px", 0.0021875)
        self.declare_parameter("forward_m_per_px", 0.003125)
        self.declare_parameter("canonical_width", 256)
        self.declare_parameter("canonical_height", 144)
        self.declare_parameter("canonical_lateral_range_m", 1.4)
        self.declare_parameter("canonical_forward_range_m", 1.5)
        self.declare_parameter("canonical_background_gray", 36)
        self.declare_parameter("canonical_line_width_px", 5)
        self.declare_parameter("canonical_white_fit_enabled", False)
        self.declare_parameter("canonical_white_fit_window_count", 9)
        self.declare_parameter("canonical_white_fit_margin_px", 24)
        self.declare_parameter("canonical_white_fit_min_pixels", 4)
        self.declare_parameter("canonical_white_fit_min_centers", 2)
        self.declare_parameter("canonical_white_fit_min_span_px", 8)
        self.declare_parameter("canonical_white_fit_residual_px", 6.0)
        self.declare_parameter("canonical_white_fit_line_width_px", 5)
        self.declare_parameter("canonical_yellow_divider_enabled", False)
        self.declare_parameter("canonical_yellow_divider_min_pixels", 3)
        self.declare_parameter("canonical_yellow_divider_residual_px", 6.0)
        self.declare_parameter("canonical_yellow_divider_line_width_px", 5)
        self.declare_parameter("canonical_yellow_normalize_enabled", False)
        self.declare_parameter("canonical_yellow_normalize_line_width_px", 5)
        self.declare_parameter("canonical_yellow_normalize_min_area_px", 3)
        self.declare_parameter("canonical_yellow_normalize_smoothing_rows", 5)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_qos_depth", 1)
        self.declare_parameter("sync_slop_sec", 0.08)
        self.declare_parameter("debug_rate_hz", 1.0)

        self.bridge = CvBridge()
        self.input_is_bev = bool(self.get_parameter("input_is_bev").value)
        self.geometry: BevGeometry | None = None
        self.geometry_input_size: tuple[int, int] | None = None
        self.base_frame_id = str(self.get_parameter("base_frame_id").value)
        self.debug_rate_hz = float(self.get_parameter("debug_rate_hz").value)
        self.last_debug_bucket: int | None = None
        self.last_canonical_fit_debug_bucket: int | None = None
        self.render_config = self.make_render_config()

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.canonical_pub = self.create_publisher(
            Image, str(self.get_parameter("canonical_topic").value), output_qos
        )
        self.white_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_white_topic").value),
            output_qos,
        )
        self.yellow_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_yellow_topic").value),
            output_qos,
        )
        self.valid_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_valid_topic").value),
            output_qos,
        )
        self.debug_pub = self.create_publisher(
            Image, str(self.get_parameter("debug_topic").value), output_qos
        )
        self.canonical_fit_debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("canonical_white_fit_debug_topic").value),
            output_qos,
        )
        self.bev_color_pub = None
        self.bev_white_pub = None
        self.bev_yellow_pub = None
        if not self.input_is_bev:
            self.bev_color_pub = self.create_publisher(
                Image,
                str(self.get_parameter("bev_color_topic").value),
                output_qos,
            )
            self.bev_white_pub = self.create_publisher(
                Image,
                str(self.get_parameter("bev_white_topic").value),
                output_qos,
            )
            self.bev_yellow_pub = self.create_publisher(
                Image,
                str(self.get_parameter("bev_yellow_topic").value),
                output_qos,
            )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, int(self.get_parameter("sync_qos_depth").value)),
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.image_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("image_topic").value),
            qos_profile=sensor_qos,
        )
        self.white_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("white_mask_topic").value),
            qos_profile=sensor_qos,
        )
        self.yellow_sub = message_filters.Subscriber(
            self,
            Image,
            str(self.get_parameter("yellow_mask_topic").value),
            qos_profile=sensor_qos,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.white_sub, self.yellow_sub],
            queue_size=max(2, int(self.get_parameter("sync_queue_size").value)),
            slop=max(0.001, float(self.get_parameter("sync_slop_sec").value)),
        )
        self.synchronizer.registerCallback(self.on_frame)
        self.get_logger().info(
            "lane_seg canonical adapter ready: semantic masks -> "
            "/perception/canonical_road_image (256x144)"
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
        if self.geometry is not None and self.geometry_input_size == (width, height):
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
        return self.geometry

    def on_frame(
        self, image_message: Image, white_message: Image, yellow_message: Image
    ) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(
                image_message, desired_encoding="bgr8"
            )
            white = self.bridge.imgmsg_to_cv2(
                white_message, desired_encoding="mono8"
            )
            yellow = self.bridge.imgmsg_to_cv2(
                yellow_message, desired_encoding="mono8"
            )
        except Exception as exc:
            self.get_logger().error(f"lane mask conversion failed: {exc}")
            return
        if image.shape[:2] != white.shape or white.shape != yellow.shape:
            self.get_logger().error(
                f"lane input dimensions differ: image={image.shape[:2]}, "
                f"white={white.shape}, yellow={yellow.shape}"
            )
            return

        if self.input_is_bev:
            bev_image = image
            bev_white = (white > 0).astype(np.uint8) * 255
            bev_yellow = (yellow > 0).astype(np.uint8) * 255
            valid = np.full(white.shape, 255, dtype=np.uint8)
        else:
            geometry = self.ensure_geometry(image.shape[1], image.shape[0])
            bev_image, bev_white, bev_yellow, valid = warp_semantic_masks(
                image,
                white,
                yellow,
                geometry,
                valid_lateral_margin_px=int(
                    self.get_parameter("bev_valid_lateral_margin_px").value
                ),
                valid_erode_px=int(
                    self.get_parameter("bev_valid_erode_px").value
                ),
                clip_to_source_polygon=bool(
                    self.get_parameter("bev_clip_to_source_polygon").value
                ),
            )
        result = render_canonical_from_bev_masks(
            bev_white,
            bev_yellow,
            valid,
            self.render_config,
        )
        stages = result.stages
        raw_stages = result.raw_stages
        white_fit = result.white_fit
        yellow_reference = result.yellow_reference

        header = image_message.header
        header.frame_id = self.base_frame_id
        outputs = [(self.canonical_pub, stages.road_image, "bgr8")]
        for publisher, frame, encoding in (
            (self.white_pub, stages.white_mask, "mono8"),
            (self.yellow_pub, stages.yellow_mask, "mono8"),
            (self.valid_pub, stages.valid_mask, "mono8"),
        ):
            if publisher.get_subscription_count() > 0:
                outputs.append((publisher, frame, encoding))
        if self.bev_color_pub is not None:
            for publisher, frame, encoding in (
                (self.bev_color_pub, bev_image, "bgr8"),
                (self.bev_white_pub, bev_white, "mono8"),
                (self.bev_yellow_pub, bev_yellow, "mono8"),
            ):
                if publisher.get_subscription_count() > 0:
                    outputs.append((publisher, frame, encoding))
        for publisher, frame, encoding in outputs:
            output_message = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
            output_message.header = header
            publisher.publish(output_message)

        stamp_ns = (
            int(image_message.header.stamp.sec) * 1_000_000_000
            + int(image_message.header.stamp.nanosec)
        )
        debug_bucket = (
            int(stamp_ns * self.debug_rate_hz / 1_000_000_000)
            if stamp_ns > 0 and self.debug_rate_hz > 0.0
            else None
        )
        if (
            self.debug_pub.get_subscription_count() > 0
            and debug_bucket is not None
            and debug_bucket != self.last_debug_bucket
        ):
            debug = bev_image.copy()
            overlay = np.zeros_like(debug)
            overlay[bev_white > 0] = (255, 255, 255)
            overlay[bev_yellow > 0] = (0, 220, 255)
            selected = (bev_white > 0) | (bev_yellow > 0)
            blended = cv2.addWeighted(debug, 0.55, overlay, 0.45, 0.0)
            debug[selected] = blended[selected]
            debug_message = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_message.header = header
            self.debug_pub.publish(debug_message)
            self.last_debug_bucket = debug_bucket

        if (
            white_fit is not None
            and self.canonical_fit_debug_pub.get_subscription_count() > 0
            and debug_bucket is not None
            and debug_bucket != self.last_canonical_fit_debug_bucket
        ):
            canonical_debug = render_white_lane_fit_debug(
                raw_stages.road_image,
                raw_stages.white_mask,
                raw_stages.yellow_mask,
                white_fit,
                yellow_reference,
            )
            debug_message = self.bridge.cv2_to_imgmsg(
                canonical_debug, encoding="bgr8"
            )
            debug_message.header = header
            self.canonical_fit_debug_pub.publish(debug_message)
            self.last_canonical_fit_debug_bucket = debug_bucket


def main() -> None:
    rclpy.init()
    node = CanonicalAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
