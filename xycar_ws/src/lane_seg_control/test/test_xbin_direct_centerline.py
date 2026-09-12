import numpy as np
import pytest

from lane_seg_control.xbin_direct_centerline import (
    extract_anchor_centers,
    project_points_to_metric,
)


def test_extract_anchor_centers_uses_xbin_rows_and_weighted_center():
    mask = np.zeros((12, 16), dtype=np.uint8)
    probability = np.zeros_like(mask, dtype=np.float32)
    mask[2, 4:7] = 255
    probability[2, 4:7] = (0.5, 1.0, 0.5)
    mask[6, 9:12] = 255
    probability[6, 9:12] = (0.25, 1.0, 0.75)
    mask[7, 1:3] = 255
    probability[7, 1:3] = 1.0

    points, confidence = extract_anchor_centers(mask, probability)

    assert points.shape == (2, 2)
    assert points[:, 1].tolist() == [2.0, 6.0]
    assert points[0, 0] == pytest.approx(5.0)
    assert points[1, 0] == pytest.approx(10.25)
    assert confidence.tolist() == pytest.approx([1.0, 1.0])


def test_project_points_to_metric_without_rendering_bev():
    camera_points = np.asarray(
        [[40.0, 80.0], [60.0, 60.0], [50.0, 100.0]],
        dtype=np.float32,
    )

    metric, indices = project_points_to_metric(
        camera_points,
        np.eye(3),
        bev_width=100,
        bev_height=101,
        lateral_m_per_px=0.01,
        forward_m_per_px=0.01,
        minimum_forward_m=0.03,
        maximum_forward_m=0.50,
        maximum_abs_lateral_m=0.20,
    )

    assert indices.tolist() == [0, 1]
    np.testing.assert_allclose(
        metric, np.asarray([[0.20, 0.10], [0.40, -0.10]])
    )


def test_project_points_filters_nonmetric_extrapolation():
    camera_points = np.asarray(
        [[50.0, -100.0], [200.0, 80.0]], dtype=np.float32
    )

    metric, indices = project_points_to_metric(
        camera_points,
        np.eye(3),
        bev_width=100,
        bev_height=101,
        lateral_m_per_px=0.01,
        forward_m_per_px=0.01,
        minimum_forward_m=0.03,
        maximum_forward_m=0.50,
        maximum_abs_lateral_m=0.20,
    )

    assert metric.shape == (0, 2)
    assert indices.shape == (0,)
