"""Sliding-window fitting for left and right canonical white boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SlidingWindow:
    x_low: int
    x_high: int
    y_low: int
    y_high: int
    pixel_count: int
    center: tuple[float, float] | None


@dataclass(frozen=True)
class LaneSideFit:
    side: str
    windows: tuple[SlidingWindow, ...]
    centers: np.ndarray
    coefficients: np.ndarray | None
    degree: int
    rmse_px: float
    curve_points: np.ndarray
    used_fallback: bool

    @property
    def valid(self) -> bool:
        return self.coefficients is not None and self.curve_points.shape[0] >= 2


@dataclass(frozen=True)
class WhiteLaneFitResult:
    mask: np.ndarray
    left: LaneSideFit
    right: LaneSideFit


@dataclass(frozen=True)
class YellowCenterlineReference:
    mask: np.ndarray
    x_by_y: np.ndarray | None
    coefficients: np.ndarray | None
    component_count: int
    pixel_count: int
    rmse_px: float

    @property
    def valid(self) -> bool:
        return self.x_by_y is not None and self.coefficients is not None


def _robust_polyfit(
    centers: np.ndarray,
    degree: int,
    residual_threshold_px: float,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    if centers.shape[0] < degree + 1:
        return None, np.zeros(centers.shape[0], dtype=bool), float("inf")
    y = centers[:, 1].astype(np.float64)
    x = centers[:, 0].astype(np.float64)
    keep = np.ones(centers.shape[0], dtype=bool)
    coefficients: np.ndarray | None = None
    for _ in range(4):
        if int(np.count_nonzero(keep)) < degree + 1:
            break
        try:
            coefficients = np.polyfit(y[keep], x[keep], degree)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None, keep, float("inf")
        residuals = np.abs(x - np.polyval(coefficients, y))
        median = float(np.median(residuals[keep]))
        mad = float(np.median(np.abs(residuals[keep] - median)))
        robust_limit = max(
            float(residual_threshold_px),
            2.5 * 1.4826 * mad,
        )
        next_keep = residuals <= robust_limit
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    if coefficients is None or int(np.count_nonzero(keep)) < degree + 1:
        return None, keep, float("inf")
    try:
        coefficients = np.polyfit(y[keep], x[keep], degree)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, keep, float("inf")
    fitted = np.polyval(coefficients, y[keep])
    rmse = float(np.sqrt(np.mean(np.square(x[keep] - fitted))))
    return coefficients, keep, rmse


def _least_squares_polyfit(
    centers: np.ndarray,
    degree: int,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """Fit every accepted white-lane center without dropping fragments."""
    keep = np.ones(centers.shape[0], dtype=bool)
    if centers.shape[0] < degree + 1:
        return None, keep, float("inf")
    y = centers[:, 1].astype(np.float64)
    x = centers[:, 0].astype(np.float64)
    try:
        coefficients = np.polyfit(y, x, degree)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, keep, float("inf")
    fitted = np.polyval(coefficients, y)
    rmse = float(np.sqrt(np.mean(np.square(x - fitted))))
    return coefficients, keep, rmse


def _fit_side(
    white_mask: np.ndarray,
    side: str,
    *,
    window_count: int,
    window_margin_px: int,
    min_pixels_per_window: int,
    min_centers: int,
    min_span_px: int,
    residual_threshold_px: float,
    divider_x_by_y: np.ndarray | None,
) -> LaneSideFit:
    height, width = white_mask.shape
    midpoint = width // 2
    ys, xs = np.nonzero(white_mask > 0)
    if divider_x_by_y is None:
        side_min, side_max = (
            (0, midpoint) if side == "left" else (midpoint, width)
        )
        side_selection = (xs >= side_min) & (xs < side_max)
    else:
        side_min, side_max = 0, width
        divider_at_pixels = divider_x_by_y[ys]
        side_selection = (
            xs < divider_at_pixels if side == "left" else xs > divider_at_pixels
        )
    ys = ys[side_selection]
    xs = xs[side_selection]
    window_height = max(1, int(np.ceil(height / float(window_count))))

    if xs.size:
        weights = 1.0 + ys.astype(np.float64) / max(float(height), 1.0)
        histogram = np.bincount(
            xs - side_min,
            weights=weights,
            minlength=side_max - side_min,
        )
        if histogram.size >= 9:
            histogram = np.convolve(histogram, np.ones(9) / 9.0, mode="same")
        current_x = side_min + int(np.argmax(histogram))
    else:
        if divider_x_by_y is None:
            current_x = (side_min + side_max) // 2
        else:
            bottom_divider = int(round(float(divider_x_by_y[-1])))
            current_x = (
                bottom_divider // 2
                if side == "left"
                else (bottom_divider + width) // 2
            )

    windows: list[SlidingWindow] = []
    centers: list[tuple[float, float]] = []
    for index in range(window_count):
        y_high = height - index * window_height
        y_low = max(0, height - (index + 1) * window_height)
        x_low = max(side_min, current_x - window_margin_px)
        x_high = min(side_max, current_x + window_margin_px + 1)
        selected = (
            (ys >= y_low)
            & (ys < y_high)
            & (xs >= x_low)
            & (xs < x_high)
        )
        count = int(np.count_nonzero(selected))
        center: tuple[float, float] | None = None
        if count >= min_pixels_per_window:
            center_x = float(np.median(xs[selected]))
            center_y = float(np.median(ys[selected]))
            center = (center_x, center_y)
            centers.append(center)
            current_x = int(round(center_x))
        windows.append(
            SlidingWindow(
                x_low=x_low,
                x_high=x_high,
                y_low=y_low,
                y_high=y_high,
                pixel_count=count,
                center=center,
            )
        )

    # A disconnected lane fragment can lie outside the margin after the
    # windows lock onto another fragment. Add only those unrepresented
    # same-side component centers so every accepted fragment contributes to
    # the one final curve for this side.
    side_mask = np.zeros_like(white_mask, dtype=np.uint8)
    side_mask[ys, xs] = 255
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        side_mask,
        connectivity=8,
    )
    for component_index in range(1, component_count):
        component_area = int(stats[component_index, cv2.CC_STAT_AREA])
        if component_area < min_pixels_per_window:
            continue
        component_ys, component_xs = np.nonzero(labels == component_index)
        center_x = float(np.median(component_xs))
        center_y = float(np.median(component_ys))
        represented = any(
            window.center is not None
            and window.x_low <= center_x < window.x_high
            and window.y_low <= center_y < window.y_high
            for window in windows
        )
        if not represented:
            centers.append((center_x, center_y))

    centers.sort(key=lambda point: point[1], reverse=True)
    center_array = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
    invalid = LaneSideFit(
        side=side,
        windows=tuple(windows),
        centers=center_array,
        coefficients=None,
        degree=0,
        rmse_px=float("inf"),
        curve_points=np.empty((0, 2), dtype=np.int32),
        used_fallback=True,
    )
    fit_centers = center_array
    used_fallback = False
    center_span = (
        float(np.ptp(center_array[:, 1]))
        if center_array.shape[0] >= 2
        else 0.0
    )
    if center_array.shape[0] < min_centers or center_span < float(min_span_px):
        # Two fragments can fall on opposite sides of a window boundary while
        # their window medians remain too close for a stable slope. Recover the
        # full same-side center axis from all occupied rows and still fit only
        # one line for this side.
        row_centers = np.asarray(
            [
                (float(np.median(xs[ys == row])), float(row))
                for row in np.unique(ys)
            ],
            dtype=np.float32,
        ).reshape(-1, 2)
        row_span = (
            float(np.ptp(row_centers[:, 1]))
            if row_centers.shape[0] >= 2
            else 0.0
        )
        if row_centers.shape[0] < min_centers or row_span < float(min_span_px):
            return invalid
        fit_centers = row_centers
        used_fallback = True

    y_span = float(np.ptp(fit_centers[:, 1]))

    linear, linear_keep, linear_rmse = _least_squares_polyfit(fit_centers, 1)
    if linear is None or int(np.count_nonzero(linear_keep)) < min_centers:
        return invalid

    degree = 1
    coefficients = linear
    keep = linear_keep
    rmse = linear_rmse
    if not used_fallback and fit_centers.shape[0] >= max(8, min_centers + 3):
        quadratic, quadratic_keep, quadratic_rmse = _least_squares_polyfit(
            fit_centers,
            2,
        )
        if quadratic is not None:
            quadratic_count = int(np.count_nonzero(quadratic_keep))
            curvature_shift = abs(float(quadratic[0])) * y_span * y_span
            meaningful_improvement = (
                quadratic_rmse <= linear_rmse * 0.82
                and linear_rmse - quadratic_rmse >= 0.25
            )
            meaningful_curvature = curvature_shift >= 1.5
            plausible_curvature = curvature_shift <= width * 0.45
            if (
                quadratic_count >= min_centers
                and meaningful_improvement
                and meaningful_curvature
                and plausible_curvature
            ):
                degree = 2
                coefficients = quadratic
                keep = quadratic_keep
                rmse = quadratic_rmse

    kept_centers = fit_centers[keep]
    if kept_centers.shape[0] < min_centers:
        return invalid
    y_min = 0
    y_max = min(
        height - 1,
        int(np.ceil(np.max(kept_centers[:, 1]) + window_height / 2)),
    )
    sample_count = max(2, y_max - y_min + 1)
    curve_y = np.linspace(y_min, y_max, sample_count)
    curve_x = np.polyval(coefficients, curve_y)
    observed_y_min = float(np.min(kept_centers[:, 1]))
    if degree == 2:
        top_selection = curve_y < observed_y_min
        top_anchor_x = float(np.polyval(coefficients, observed_y_min))
        top_slope = float(
            2.0 * coefficients[0] * observed_y_min + coefficients[1]
        )
        curve_x[top_selection] = top_anchor_x + top_slope * (
            curve_y[top_selection] - observed_y_min
        )
    if divider_x_by_y is None:
        curve_x = np.clip(curve_x, side_min, side_max - 1)
    else:
        divider_at_curve = divider_x_by_y[
            np.rint(curve_y).astype(np.int32)
        ]
        if side == "left":
            curve_x = np.minimum(curve_x, divider_at_curve - 1.0)
        else:
            curve_x = np.maximum(curve_x, divider_at_curve + 1.0)
        curve_x = np.clip(curve_x, 0, width - 1)
    curve_points = np.column_stack(
        (np.rint(curve_x).astype(np.int32), np.rint(curve_y).astype(np.int32))
    )
    return LaneSideFit(
        side=side,
        windows=tuple(windows),
        centers=center_array,
        coefficients=coefficients.astype(np.float32),
        degree=degree,
        rmse_px=rmse,
        curve_points=curve_points,
        used_fallback=used_fallback,
    )


def fit_white_lane_boundaries(
    white_mask: np.ndarray,
    *,
    window_count: int = 9,
    window_margin_px: int = 24,
    min_pixels_per_window: int = 4,
    min_centers: int = 2,
    min_span_px: int = 8,
    residual_threshold_px: float = 6.0,
    line_width_px: int = 5,
    divider_x_by_y: np.ndarray | None = None,
) -> WhiteLaneFitResult:
    """Fit independent smooth boundaries in canonical image coordinates."""
    if white_mask.ndim != 2:
        raise ValueError(f"white mask must be two-dimensional, got {white_mask.shape}")
    binary = (white_mask > 0).astype(np.uint8) * 255
    divider = None
    if divider_x_by_y is not None:
        divider = np.asarray(divider_x_by_y, dtype=np.float32).reshape(-1)
        if divider.shape[0] != binary.shape[0]:
            raise ValueError(
                "divider must contain one x coordinate per canonical row"
            )
        divider = np.clip(divider, 0, binary.shape[1] - 1)
    elif np.count_nonzero(binary) > 0:
        # Without yellow, this frame is assumed to contain one white lane.
        # Put every white fragment on the side selected by its global median.
        _, white_xs = np.nonzero(binary)
        one_side_is_left = float(np.median(white_xs)) < binary.shape[1] / 2.0
        divider_value = float(binary.shape[1] if one_side_is_left else -1)
        divider = np.full(binary.shape[0], divider_value, dtype=np.float32)
    fits = [
        _fit_side(
            binary,
            side,
            window_count=max(2, int(window_count)),
            window_margin_px=max(1, int(window_margin_px)),
            min_pixels_per_window=max(1, int(min_pixels_per_window)),
            min_centers=max(2, int(min_centers)),
            min_span_px=max(1, int(min_span_px)),
            residual_threshold_px=max(1.0, float(residual_threshold_px)),
            divider_x_by_y=divider,
        )
        for side in ("left", "right")
    ]
    output = np.zeros_like(binary)
    for fit in fits:
        if fit.valid:
            cv2.polylines(
                output,
                [fit.curve_points],
                False,
                255,
                thickness=max(1, int(line_width_px)),
                lineType=cv2.LINE_8,
            )
        # An invalid side stays empty. Copying raw fragments here would create
        # multiple apparent lane boundaries on the same side.
    return WhiteLaneFitResult(mask=output, left=fits[0], right=fits[1])


def fit_yellow_centerline_reference(
    yellow_mask: np.ndarray,
    *,
    min_pixels: int = 3,
    residual_threshold_px: float = 6.0,
    line_width_px: int = 5,
) -> YellowCenterlineReference:
    """Fit one full-height straight divider through canonical yellow marks."""
    if yellow_mask.ndim != 2:
        raise ValueError(
            f"yellow mask must be two-dimensional, got {yellow_mask.shape}"
        )
    binary = (yellow_mask > 0).astype(np.uint8) * 255
    component_count = max(
        0,
        int(cv2.connectedComponents(binary, connectivity=8)[0]) - 1,
    )
    ys, xs = np.nonzero(binary)
    pixel_count = int(xs.size)
    invalid = YellowCenterlineReference(
        mask=np.zeros_like(binary),
        x_by_y=None,
        coefficients=None,
        component_count=component_count,
        pixel_count=pixel_count,
        rmse_px=float("inf"),
    )
    if pixel_count < max(1, int(min_pixels)):
        return invalid

    row_values = np.unique(ys)
    row_centers = np.asarray(
        [
            (float(np.median(xs[ys == row])), float(row))
            for row in row_values
        ],
        dtype=np.float32,
    )
    if row_centers.shape[0] >= 2 and np.ptp(row_centers[:, 1]) >= 1.0:
        coefficients, keep, rmse = _robust_polyfit(
            row_centers,
            1,
            max(1.0, float(residual_threshold_px)),
        )
        if coefficients is None or int(np.count_nonzero(keep)) < 2:
            return invalid
    else:
        coefficients = np.asarray(
            [0.0, float(np.median(xs))], dtype=np.float64
        )
        rmse = 0.0

    height, width = binary.shape
    anchor_y = float(np.median(row_centers[:, 1]))
    anchor_x = float(np.polyval(coefficients, anchor_y))
    anchor_x = float(np.clip(anchor_x, 0, width - 1))
    slope_min = float("-inf")
    slope_max = float("inf")
    for edge_y in (0.0, float(height - 1)):
        delta_y = edge_y - anchor_y
        if abs(delta_y) < 1e-6:
            continue
        bounds = sorted(
            (
                (0.0 - anchor_x) / delta_y,
                (float(width - 1) - anchor_x) / delta_y,
            )
        )
        slope_min = max(slope_min, bounds[0])
        slope_max = min(slope_max, bounds[1])
    constrained_slope = float(
        np.clip(float(coefficients[0]), slope_min, slope_max)
    )
    coefficients = np.asarray(
        [constrained_slope, anchor_x - constrained_slope * anchor_y],
        dtype=np.float64,
    )
    fitted_centers = np.polyval(coefficients, row_centers[:, 1])
    rmse = float(
        np.sqrt(np.mean(np.square(row_centers[:, 0] - fitted_centers)))
    )
    full_y = np.arange(height, dtype=np.float32)
    full_x = np.clip(np.polyval(coefficients, full_y), 0, width - 1)
    points = np.column_stack(
        (np.rint(full_x).astype(np.int32), full_y.astype(np.int32))
    )
    connected = np.zeros_like(binary)
    cv2.polylines(
        connected,
        [points],
        False,
        255,
        thickness=max(1, int(line_width_px)),
        lineType=cv2.LINE_8,
    )
    return YellowCenterlineReference(
        mask=connected,
        x_by_y=full_x.astype(np.float32),
        coefficients=coefficients.astype(np.float32),
        component_count=component_count,
        pixel_count=pixel_count,
        rmse_px=rmse,
    )


def normalize_yellow_fragments(
    yellow_mask: np.ndarray,
    *,
    line_width_px: int = 5,
    min_component_area_px: int = 3,
    smoothing_window_rows: int = 5,
) -> np.ndarray:
    """Give observed yellow fragments a stable width without filling gaps."""
    if yellow_mask.ndim != 2:
        raise ValueError(
            f"yellow mask must be two-dimensional, got {yellow_mask.shape}"
        )
    binary = (yellow_mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    output = np.zeros_like(binary, dtype=np.uint8)
    smoothing_window = max(1, int(smoothing_window_rows))
    if smoothing_window % 2 == 0:
        smoothing_window += 1

    for component_index in range(1, component_count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < max(1, int(min_component_area_px)):
            continue
        component_ys, component_xs = np.nonzero(labels == component_index)
        occupied_rows = np.unique(component_ys)
        if occupied_rows.size == 0:
            continue
        centers_x = np.asarray(
            [
                float(np.median(component_xs[component_ys == row]))
                for row in occupied_rows
            ],
            dtype=np.float32,
        )
        if smoothing_window > 1 and centers_x.size >= smoothing_window:
            padding = smoothing_window // 2
            kernel = np.full(
                smoothing_window,
                1.0 / float(smoothing_window),
                dtype=np.float32,
            )
            centers_x = np.convolve(
                np.pad(centers_x, (padding, padding), mode="edge"),
                kernel,
                mode="valid",
            )
        points = np.column_stack(
            (
                np.rint(centers_x).astype(np.int32),
                occupied_rows.astype(np.int32),
            )
        )
        if points.shape[0] == 1:
            cv2.circle(
                output,
                tuple(points[0]),
                max(1, int(line_width_px)) // 2,
                255,
                -1,
                cv2.LINE_8,
            )
        else:
            cv2.polylines(
                output,
                [points],
                False,
                255,
                thickness=max(1, int(line_width_px)),
                lineType=cv2.LINE_8,
            )
    return output


def compose_fitted_canonical(
    fitted_white: np.ndarray,
    yellow: np.ndarray,
    valid: np.ndarray,
    *,
    background_gray: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose canonical colors while preserving the yellow mask unchanged."""
    if fitted_white.shape != yellow.shape or yellow.shape != valid.shape:
        raise ValueError("canonical white, yellow, and valid masks must match")
    output_white = cv2.bitwise_and(
        (fitted_white > 0).astype(np.uint8) * 255,
        (valid > 0).astype(np.uint8) * 255,
    )
    output_white[yellow > 0] = 0
    gray = int(np.clip(background_gray, 0, 255))
    road = np.full((*yellow.shape, 3), gray, dtype=np.uint8)
    road[output_white > 0] = (255, 255, 255)
    road[yellow > 0] = (0, 220, 255)
    return road, output_white


def render_white_lane_fit_debug(
    canonical_image: np.ndarray,
    raw_white: np.ndarray,
    yellow: np.ndarray,
    result: WhiteLaneFitResult,
    yellow_reference: YellowCenterlineReference | None = None,
) -> np.ndarray:
    """Overlay raw masks, search windows, centers, and final white curves."""
    debug = canonical_image.copy()
    raw_overlay = np.zeros_like(debug)
    raw_overlay[raw_white > 0] = (255, 120, 40)
    raw_overlay[yellow > 0] = (0, 220, 255)
    selected = (raw_white > 0) | (yellow > 0)
    blended = cv2.addWeighted(debug, 0.62, raw_overlay, 0.38, 0.0)
    debug[selected] = blended[selected]

    if yellow_reference is not None and yellow_reference.valid:
        reference_points = np.column_stack(
            (
                np.rint(yellow_reference.x_by_y).astype(np.int32),
                np.arange(debug.shape[0], dtype=np.int32),
            )
        )
        cv2.polylines(
            debug,
            [reference_points],
            False,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    compact = debug.shape[1] < 400
    center_radius = 2 if compact else 4
    curve_thickness = 2 if compact else 3
    font_scale = 0.32 if compact else 0.55
    text_thickness = 1 if compact else 2
    for fit, color in (
        (result.left, (70, 220, 70)),
        (result.right, (70, 220, 70)),
    ):
        for window in fit.windows:
            window_color = color if window.center is not None else (70, 70, 120)
            cv2.rectangle(
                debug,
                (window.x_low, window.y_low),
                (
                    max(window.x_low, window.x_high - 1),
                    max(window.y_low, window.y_high - 1),
                ),
                window_color,
                1,
            )
        for center_x, center_y in fit.centers:
            cv2.circle(
                debug,
                (int(round(center_x)), int(round(center_y))),
                center_radius,
                (255, 0, 255),
                -1,
                cv2.LINE_AA,
            )
        if fit.valid:
            cv2.polylines(
                debug,
                [fit.curve_points],
                False,
                (255, 255, 255),
                curve_thickness,
                cv2.LINE_AA,
            )
        fit_mode = "F" if fit.used_fallback else "P"
        status = (
            f"{fit.side[0].upper()}:{fit_mode}{fit.degree} "
            f"n={fit.centers.shape[0]} rmse={fit.rmse_px:.1f}"
            if fit.valid
            else f"{fit.side[0].upper()}:RAW n={fit.centers.shape[0]}"
        )
        x = 4 if fit.side == "left" else debug.shape[1] // 2 + 4
        cv2.putText(
            debug,
            status,
            (x, 14 if compact else 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    if yellow_reference is not None and yellow_reference.valid:
        divider_status = f"DIV:YELLOW c={yellow_reference.component_count}"
    elif result.left.valid and not result.right.valid:
        divider_status = "DIV:SINGLE-LEFT"
    elif result.right.valid and not result.left.valid:
        divider_status = "DIV:SINGLE-RIGHT"
    else:
        divider_status = "DIV:SINGLE-LANE"
    cv2.putText(
        debug,
        divider_status,
        (4, debug.shape[0] - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 0),
        text_thickness,
        cv2.LINE_AA,
    )
    return debug
