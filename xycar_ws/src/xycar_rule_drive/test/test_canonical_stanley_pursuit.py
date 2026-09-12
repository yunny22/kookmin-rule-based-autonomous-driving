import math
import unittest

import cv2
import numpy as np

try:
    from xycar_rule_drive.canonical_stanley_pursuit_driver import (
        adaptive_smooth_steering_command,
        anticipatory_center_corridor_error,
        apply_turn_transition_recovery,
        blend_pursuit_stanley,
        canonical_class_masks,
        command_during_lane_loss,
        connect_yellow_centerline,
        compute_departure_guard_pure_pursuit_weight,
        fuse_lane_center_paths,
        fused_stanley_pursuit,
        latency_compensated_lookahead,
        lead_compensated_steering_command,
        offset_path_left,
        offset_path_right,
        outer_white_is_consistent,
        path_heading_change_per_m,
        predict_path_in_delayed_vehicle_frame,
        pursuit_requests_command_reversal,
        smooth_target_path,
        steering_term_requests_command_reversal,
        update_heading_recovery_latch,
        white_boundary_to_target_offset,
    )
except ModuleNotFoundError as error:
    EXTERNAL_MESSAGE_DEPENDENCY = str(error)
else:
    EXTERNAL_MESSAGE_DEPENDENCY = ""


@unittest.skipIf(
    EXTERNAL_MESSAGE_DEPENDENCY,
    "requires the external kaiev26_msgs ROS message package: "
    + EXTERNAL_MESSAGE_DEPENDENCY,
)
class CanonicalStanleyPursuitTest(unittest.TestCase):
    def test_latency_preview_adds_distance_travelled_before_actuation(self):
        self.assertAlmostEqual(
            latency_compensated_lookahead(1.0, 1.36, 0.25),
            1.34,
        )
        self.assertAlmostEqual(
            latency_compensated_lookahead(1.0, 1.36, 0.0),
            1.0,
        )

    def test_delayed_frame_predictor_turns_a_straight_path_right(self):
        path = np.column_stack(
            (np.linspace(0.05, 1.5, 32), np.zeros(32))
        )
        predicted = predict_path_in_delayed_vehicle_frame(
            path,
            speed_mps=1.4,
            curvature_per_m=0.8,
            latency_sec=0.25,
        )
        self.assertGreaterEqual(predicted.shape[0], 3)
        self.assertTrue(np.all(predicted[:, 1] < 0.0))
        self.assertLess(float(np.max(predicted[:, 0])), 1.5)

    def test_steering_lead_advances_a_releasing_curve(self):
        self.assertAlmostEqual(
            lead_compensated_steering_command(
                -12.0,
                -24.0,
                0.1,
                0.1,
                10.0,
            ),
            -2.0,
        )

    def test_steering_lead_is_limited_against_frame_noise(self):
        self.assertAlmostEqual(
            lead_compensated_steering_command(
                20.0,
                -20.0,
                0.01,
                0.1,
                6.0,
            ),
            26.0,
        )

    def test_steering_lead_does_not_amplify_curve_entry(self):
        self.assertAlmostEqual(
            lead_compensated_steering_command(
                20.0,
                10.0,
                0.1,
                0.1,
                10.0,
            ),
            20.0,
        )

    def test_steering_release_does_not_create_a_false_reversal(self):
        self.assertAlmostEqual(
            lead_compensated_steering_command(
                5.0,
                20.0,
                0.1,
                0.1,
                20.0,
            ),
            0.0,
        )

    def test_pursuit_detects_a_real_xycar_command_reversal(self):
        self.assertTrue(
            pursuit_requests_command_reversal(-20.0, -0.08, 0.02)
        )
        self.assertFalse(
            pursuit_requests_command_reversal(-20.0, 0.08, 0.02)
        )
        self.assertFalse(
            pursuit_requests_command_reversal(-20.0, -0.01, 0.02)
        )

    def test_stanley_reversal_requires_a_large_opposed_heading_term(self):
        self.assertTrue(
            steering_term_requests_command_reversal(
                -20.0,
                -0.30,
                0.20,
            )
        )
        self.assertFalse(
            steering_term_requests_command_reversal(
                -20.0,
                -0.10,
                0.20,
            )
        )
        self.assertFalse(
            steering_term_requests_command_reversal(
                -20.0,
                0.30,
                0.20,
            )
        )

    def test_heading_recovery_latch_persists_after_command_sign_changes(self):
        sign, misses, active = update_heading_recovery_latch(
            0.0,
            0,
            trigger_command_sign=1.0,
            pure_pursuit_rad=-0.08,
            stanley_rad=-0.30,
            support_activation_rad=0.04,
            release_frames=3,
        )
        self.assertEqual(sign, 1.0)
        self.assertEqual(misses, 0)
        self.assertTrue(active)
        sign, misses, active = update_heading_recovery_latch(
            sign,
            misses,
            trigger_command_sign=0.0,
            pure_pursuit_rad=-0.12,
            stanley_rad=-0.25,
            support_activation_rad=0.04,
            release_frames=3,
        )
        self.assertEqual(sign, 1.0)
        self.assertTrue(active)

    def test_heading_recovery_latch_releases_after_three_misses(self):
        state = (1.0, 0, True)
        for _ in range(3):
            state = update_heading_recovery_latch(
                state[0],
                state[1],
                trigger_command_sign=0.0,
                pure_pursuit_rad=0.10,
                stanley_rad=0.10,
                support_activation_rad=0.04,
                release_frames=3,
            )
        self.assertEqual(state, (0.0, 0, False))

    def test_turn_transition_recovery_holds_a_confirmed_reversal(self):
        command, sign, until, armed = apply_turn_transition_recovery(
            -20.0,
            now=9.9,
            armed_turn_sign=0.0,
            recovery_sign=0.0,
            recovery_until=0.0,
            trigger_previous_command=12.0,
            trigger_new_command=2.0,
            minimum_recovery_command=18.0,
            hold_sec=0.45,
            cancel_opposed_command=24.0,
        )
        self.assertEqual(command, -20.0)
        self.assertEqual(armed, -1.0)
        command, sign, until, armed = apply_turn_transition_recovery(
            4.0,
            now=10.0,
            armed_turn_sign=armed,
            recovery_sign=0.0,
            recovery_until=0.0,
            trigger_previous_command=12.0,
            trigger_new_command=2.0,
            minimum_recovery_command=18.0,
            hold_sec=0.45,
            cancel_opposed_command=24.0,
        )
        self.assertEqual(command, 18.0)
        self.assertEqual(sign, 1.0)
        self.assertAlmostEqual(until, 10.45)
        command, sign, until, armed = apply_turn_transition_recovery(
            -5.0,
            now=10.2,
            armed_turn_sign=armed,
            recovery_sign=sign,
            recovery_until=until,
            trigger_previous_command=12.0,
            trigger_new_command=2.0,
            minimum_recovery_command=18.0,
            hold_sec=0.45,
            cancel_opposed_command=24.0,
        )
        self.assertEqual(command, 18.0)
        self.assertEqual(sign, 1.0)

    def test_canonical_exact_colors_are_split(self):
        image = np.full((8, 12, 3), 36, dtype=np.uint8)
        image[2:5, 2] = (255, 255, 255)
        image[3:7, 8] = (0, 220, 255)

        white, yellow = canonical_class_masks(image)

        self.assertEqual(int(np.count_nonzero(white)), 3)
        self.assertEqual(int(np.count_nonzero(yellow)), 4)
        self.assertEqual(int(np.count_nonzero(white & yellow)), 0)

    def test_fused_lane_center_uses_long_white_geometry_beyond_yellow(self):
        yellow = np.asarray([[0.1, 0.04], [0.7, 0.10]])
        white = np.asarray([[0.1, 0.00], [1.5, -0.20]])
        fused = fuse_lane_center_paths(
            yellow,
            white,
            yellow_weight=0.6,
            point_count=16,
        )
        self.assertEqual(fused.shape, (16, 2))
        self.assertAlmostEqual(float(fused[-1, 1]), -0.20)
        overlap_index = int(np.argmin(np.abs(fused[:, 0] - 0.5)))
        self.assertGreater(float(fused[overlap_index, 1]), 0.0)

    def test_bounded_fusion_never_extends_beyond_observed_yellow(self):
        yellow = np.asarray([[0.3, 0.04], [0.8, 0.10]])
        white = np.asarray([[0.1, 0.00], [1.5, -0.20]])

        fused = fuse_lane_center_paths(
            yellow,
            white,
            yellow_weight=0.65,
            point_count=16,
            extend_to_union=False,
        )

        self.assertAlmostEqual(float(fused[0, 0]), 0.3)
        self.assertAlmostEqual(float(fused[-1, 0]), 0.8)

    def test_final_target_path_is_smoothed_across_source_changes(self):
        forward = np.linspace(0.1, 1.5, 8)
        previous = np.column_stack((forward, np.full(8, 0.20)))
        target = np.column_stack((forward, np.full(8, -0.20)))
        smoothed = smooth_target_path(target, previous, 0.25)
        np.testing.assert_allclose(smoothed[:, 1], -0.10)

    def test_dashes_within_38cm_become_one_continuous_path(self):
        height, width = 144, 256
        mask = np.zeros((height, width), dtype=np.uint8)
        for row_start, row_end in ((116, 140), (70, 92), (28, 50)):
            points = []
            for row in range(row_start, row_end + 1):
                forward = (height - 1 - row) * 1.5 / (height - 1)
                column = int(round(128 - (0.03 + 0.10 * forward) * width / 1.4))
                points.append((column, row))
            cv2.polylines(
                mask,
                [np.asarray(points, dtype=np.int32)],
                False,
                255,
                5,
            )

        connected = connect_yellow_centerline(mask, max_gap_m=0.38)

        self.assertIsNotNone(connected)
        assert connected is not None
        self.assertEqual(connected.points.shape, (32, 2))
        self.assertGreater(connected.forward_span_m, 1.0)
        self.assertLessEqual(connected.max_observation_gap_m, 0.38)
        self.assertTrue(np.all(np.diff(connected.points[:, 0]) > 0.0))

    def test_far_dash_is_extrapolated_only_five_centimetres_toward_vehicle(self):
        mask = np.zeros((144, 256), dtype=np.uint8)
        cv2.line(mask, (132, 58), (140, 24), 255, 5)

        connected = connect_yellow_centerline(mask, max_gap_m=0.38)

        self.assertIsNotNone(connected)
        assert connected is not None
        observed_near_m = (143 - 58) * 1.5 / 143
        self.assertGreaterEqual(
            float(connected.points[0, 0]),
            observed_near_m - 0.07,
        )
        self.assertGreater(float(connected.points[0, 0]), 0.75)

    def test_fused_controller_steers_toward_left_path(self):
        forward = np.linspace(0.05, 1.3, 32)
        path = np.column_stack((forward, 0.04 + 0.08 * forward))

        terms = fused_stanley_pursuit(
            path,
            lookahead_m=0.65,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.62,
        )

        self.assertGreater(terms.pure_pursuit_rad, 0.0)
        self.assertGreater(terms.stanley_rad, 0.0)
        self.assertGreater(terms.fused_rad, 0.0)
        self.assertTrue(math.isfinite(terms.fused_rad))

    def test_straight_path_uses_stanley_as_primary_controller(self):
        forward = np.linspace(0.05, 1.5, 32)
        path = np.column_stack((forward, np.full_like(forward, 0.10)))

        terms = fused_stanley_pursuit(
            path,
            lookahead_m=1.5,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
            straight_stanley_enabled=True,
            straight_path_curvature_threshold=0.16,
            straight_pure_pursuit_weight=0.20,
        )

        expected = 0.20 * terms.pure_pursuit_rad + 0.80 * terms.stanley_rad
        self.assertAlmostEqual(terms.fused_rad, expected)
        self.assertLess(
            abs(terms.fused_rad - terms.stanley_rad),
            abs(terms.fused_rad - terms.pure_pursuit_rad),
        )

    def test_straight_specific_gain_reduces_lateral_overcorrection(self):
        forward = np.linspace(0.05, 1.5, 32)
        path = np.column_stack((forward, np.full_like(forward, 0.12)))
        common = dict(
            lookahead_m=1.5,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
            straight_stanley_enabled=True,
            straight_path_curvature_threshold=0.16,
            straight_pure_pursuit_weight=0.10,
        )

        aggressive = fused_stanley_pursuit(
            path,
            **common,
            straight_stanley_gain=1.15,
            straight_stanley_softening_mps=0.35,
        )
        damped = fused_stanley_pursuit(
            path,
            **common,
            straight_stanley_gain=0.65,
            straight_stanley_softening_mps=0.65,
        )

        self.assertLess(abs(damped.stanley_rad), abs(aggressive.stanley_rad))
        self.assertLess(abs(damped.fused_rad), abs(aggressive.fused_rad))

    def test_curved_path_keeps_curve_controller_blend(self):
        forward = np.linspace(0.05, 1.5, 32)
        path = np.column_stack((forward, 0.34 * forward * forward))
        common = dict(
            lookahead_m=1.5,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
        )

        baseline = fused_stanley_pursuit(
            path,
            **common,
            straight_stanley_enabled=False,
        )
        adaptive = fused_stanley_pursuit(
            path,
            **common,
            straight_stanley_enabled=True,
            straight_path_curvature_threshold=0.16,
            straight_pure_pursuit_weight=0.20,
        )

        self.assertGreater(
            path_heading_change_per_m(
                path,
                near_x_m=0.16,
                far_x_m=1.5,
            ),
            0.16,
        )
        self.assertAlmostEqual(adaptive.fused_rad, baseline.fused_rad)

    def test_straight_center_corridor_reverses_before_crossing(self):
        forward = np.linspace(0.05, 1.5, 32)
        path = np.column_stack((forward, 0.08 - 0.15 * forward))
        common = dict(
            lookahead_m=1.5,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
            straight_stanley_enabled=True,
            straight_path_curvature_threshold=0.16,
            straight_pure_pursuit_weight=0.20,
        )

        baseline = fused_stanley_pursuit(
            path,
            **common,
            straight_center_anticipation_m=0.0,
        )
        anticipated = fused_stanley_pursuit(
            path,
            **common,
            straight_center_anticipation_m=0.10,
            straight_center_minimum_approach_heading_rad=0.02,
        )

        self.assertGreater(anticipated.cross_track_error_m, 0.0)
        self.assertLess(anticipated.heading_error_rad, 0.0)
        self.assertLess(anticipated.fused_rad, baseline.fused_rad)
        self.assertLess(
            anticipatory_center_corridor_error(
                0.06,
                -0.10,
                boundary_m=0.10,
                minimum_approach_heading_rad=0.02,
            ),
            0.0,
        )

    def test_center_corridor_does_not_flip_while_moving_away(self):
        self.assertEqual(
            anticipatory_center_corridor_error(
                0.06,
                0.10,
                boundary_m=0.10,
                minimum_approach_heading_rad=0.02,
            ),
            0.06,
        )

    def test_centered_aligned_straight_reduces_steering_magnitude(self):
        forward = np.linspace(0.05, 1.5, 32)
        path = np.column_stack((forward, np.full_like(forward, 0.02)))
        common = dict(
            lookahead_m=1.5,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
            straight_stanley_enabled=True,
            straight_path_curvature_threshold=0.16,
            straight_pure_pursuit_weight=0.20,
            straight_center_anticipation_m=0.10,
        )

        full = fused_stanley_pursuit(
            path,
            **common,
            straight_center_steering_scale=1.0,
        )
        reduced = fused_stanley_pursuit(
            path,
            **common,
            straight_center_steering_scale=0.70,
            straight_center_scale_heading_limit_rad=0.20,
        )

        self.assertLess(abs(reduced.fused_rad), abs(full.fused_rad))
        self.assertGreater(abs(reduced.fused_rad), 0.0)

    def test_opposed_stanley_term_releases_curve_early(self):
        forward = np.linspace(0.05, 1.3, 32)
        path = np.column_stack((forward, 0.18 - 0.28 * forward))

        terms = fused_stanley_pursuit(
            path,
            lookahead_m=1.0,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
            opposed_stanley_weight=0.70,
        )

        self.assertLess(terms.stanley_rad, 0.0)
        self.assertLess(terms.fused_rad, terms.pure_pursuit_rad)

    def test_departure_guard_starts_before_departure_threshold(self):
        centered_weight, centered_risk = (
            compute_departure_guard_pure_pursuit_weight(
                0.95,
                cross_track_error_m=0.04,
                heading_error_rad=0.03,
                lateral_start_m=0.06,
                lateral_full_m=0.16,
                heading_start_rad=0.10,
                heading_full_rad=0.35,
                guarded_weight=0.35,
            )
        )
        warning_weight, warning_risk = (
            compute_departure_guard_pure_pursuit_weight(
                0.95,
                cross_track_error_m=0.11,
                heading_error_rad=0.03,
                lateral_start_m=0.06,
                lateral_full_m=0.16,
                heading_start_rad=0.10,
                heading_full_rad=0.35,
                guarded_weight=0.35,
            )
        )
        departed_weight, departed_risk = (
            compute_departure_guard_pure_pursuit_weight(
                0.95,
                cross_track_error_m=0.18,
                heading_error_rad=0.03,
                lateral_start_m=0.06,
                lateral_full_m=0.16,
                heading_start_rad=0.10,
                heading_full_rad=0.35,
                guarded_weight=0.35,
            )
        )

        self.assertEqual((centered_weight, centered_risk), (0.95, 0.0))
        self.assertAlmostEqual(warning_weight, 0.65)
        self.assertAlmostEqual(warning_risk, 0.5)
        self.assertEqual((departed_weight, departed_risk), (0.35, 1.0))

    def test_departure_guard_strengthens_early_offset_correction(self):
        forward = np.linspace(0.05, 1.5, 32)
        path = np.column_stack((forward, np.full_like(forward, 0.12)))
        common = dict(
            lookahead_m=1.5,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
        )

        baseline = fused_stanley_pursuit(path, **common)
        guarded = fused_stanley_pursuit(
            path,
            **common,
            departure_guard_enabled=True,
            departure_guard_lateral_start_m=0.06,
            departure_guard_lateral_full_m=0.16,
            departure_guard_heading_start_rad=0.10,
            departure_guard_heading_full_rad=0.35,
            departure_guard_pure_pursuit_weight=0.35,
        )

        self.assertGreater(guarded.fused_rad, baseline.fused_rad)

    def test_heading_uses_nearest_visible_path_endpoint(self):
        forward = np.linspace(0.85, 1.35, 16)
        path = np.column_stack((forward, 0.10 + 0.30 * forward))

        terms = fused_stanley_pursuit(
            path,
            lookahead_m=1.0,
            wheelbase_m=0.32,
            pure_pursuit_control_x_m=-0.08,
            stanley_control_x_m=0.16,
            speed_mps=1.5,
            stanley_gain=1.15,
            stanley_softening_mps=0.35,
            pure_pursuit_weight=0.95,
        )

        self.assertGreater(terms.heading_error_rad, 0.20)

    def test_dominant_opposed_stanley_cannot_reverse_pursuit(self):
        fused = blend_pursuit_stanley(
            -0.04,
            0.33,
            pure_pursuit_weight=0.95,
            opposed_stanley_weight=0.70,
        )

        self.assertAlmostEqual(fused, -0.04)

    def test_right_offset_moves_straight_yellow_to_lane_center(self):
        yellow = np.asarray(
            [[0.05, 0.14], [0.50, 0.14], [1.00, 0.14]],
            dtype=np.float64,
        )

        target = offset_path_right(yellow, 0.20)

        np.testing.assert_allclose(target[:, 0], yellow[:, 0])
        np.testing.assert_allclose(target[:, 1], -0.06)

    def test_right_offset_preserves_forward_stations_on_a_curve(self):
        yellow = np.asarray(
            [[0.20, -0.10], [0.45, 0.25], [0.80, 0.40]],
            dtype=np.float64,
        )

        target = offset_path_right(yellow, 0.20)

        np.testing.assert_allclose(target[:, 0], yellow[:, 0])
        np.testing.assert_allclose(target[:, 1], yellow[:, 1] - 0.20)
        self.assertTrue(np.all(np.diff(target[:, 0]) > 0.0))

    def test_left_offset_moves_outer_white_to_lane_center(self):
        outer_white = np.asarray(
            [[0.05, -0.26], [0.50, -0.26], [1.00, -0.26]],
            dtype=np.float64,
        )

        target = offset_path_left(outer_white, 0.20)

        np.testing.assert_allclose(target[:, 0], outer_white[:, 0])
        np.testing.assert_allclose(target[:, 1], -0.06)

    def test_white_fallback_reconstructs_yellow_centerline_target(self):
        outer_white = np.asarray(
            [[0.05, -0.40], [0.50, -0.40], [1.00, -0.40]],
            dtype=np.float64,
        )

        target = offset_path_left(
            outer_white,
            white_boundary_to_target_offset(0.20, 0.0),
        )

        np.testing.assert_allclose(target[:, 1], 0.0)
        self.assertAlmostEqual(
            white_boundary_to_target_offset(0.20, 0.20),
            0.20,
        )

    def test_outer_white_is_accepted_at_measured_lane_width(self):
        forward = np.linspace(0.10, 1.20, 16)
        yellow = np.column_stack((forward, 0.08 + 0.05 * forward))
        outer_white = np.column_stack((forward, yellow[:, 1] - 0.40))

        accepted = outer_white_is_consistent(
            outer_white,
            yellow_path=yellow,
            previous_white_path=None,
            expected_yellow_to_white_m=0.40,
            yellow_tolerance_m=0.16,
            temporal_jump_m=0.18,
            minimum_overlap_m=0.12,
        )

        self.assertTrue(accepted)

    def test_white_on_opposite_side_of_yellow_is_rejected(self):
        forward = np.linspace(0.10, 1.20, 16)
        yellow = np.column_stack((forward, np.full(16, -0.20)))
        wrong_white = np.column_stack((forward, np.full(16, 0.08)))

        accepted = outer_white_is_consistent(
            wrong_white,
            yellow_path=yellow,
            previous_white_path=None,
            expected_yellow_to_white_m=0.40,
            yellow_tolerance_m=0.16,
            temporal_jump_m=0.18,
            minimum_overlap_m=0.12,
        )

        self.assertFalse(accepted)

    def test_white_only_fallback_rejects_large_temporal_jump(self):
        forward = np.linspace(0.10, 1.20, 16)
        previous = np.column_stack((forward, np.full(16, -0.32)))
        adjacent_road_white = np.column_stack((forward, np.full(16, 0.10)))

        accepted = outer_white_is_consistent(
            adjacent_road_white,
            yellow_path=None,
            previous_white_path=previous,
            expected_yellow_to_white_m=0.40,
            yellow_tolerance_m=0.16,
            temporal_jump_m=0.18,
            minimum_overlap_m=0.12,
        )

        self.assertFalse(accepted)

    def test_white_only_fallback_keeps_a_continuous_outer_boundary(self):
        forward = np.linspace(0.10, 1.20, 16)
        previous = np.column_stack((forward, -0.32 + 0.06 * forward))
        current = np.column_stack((forward, previous[:, 1] + 0.04))

        accepted = outer_white_is_consistent(
            current,
            yellow_path=None,
            previous_white_path=previous,
            expected_yellow_to_white_m=0.40,
            yellow_tolerance_m=0.16,
            temporal_jump_m=0.18,
            minimum_overlap_m=0.12,
        )

        self.assertTrue(accepted)

    def test_lane_loss_holds_last_command_without_time_limit(self):
        command = (-17.5, 19.0)
        for _ in range(1000):
            command = command_during_lane_loss(
                has_valid_command=True,
                last_angle_command=command[0],
                last_speed_command=command[1],
                lane_loss_speed_command=4.0,
                hold_last_steering=True,
                hold_last_speed=False,
            )

        self.assertEqual(command, (-17.5, 4.0))
        self.assertEqual(
            command_during_lane_loss(
                has_valid_command=False,
                last_angle_command=-17.5,
                last_speed_command=19.0,
                lane_loss_speed_command=4.0,
                hold_last_steering=True,
                hold_last_speed=True,
            ),
            (0.0, 0.0),
        )

    def test_tight_curve_reversal_is_not_delayed_by_straight_filter(self):
        command = adaptive_smooth_steering_command(
            last_command=-32.0,
            raw_command=20.0,
            dt=1.0 / 7.0,
            straight_current_weight=0.35,
            curve_current_weight=1.0,
            straight_rate_limit=300.0,
            curve_rate_limit=600.0,
            curve_activation_command=12.0,
            curve_full_command=24.0,
        )

        self.assertAlmostEqual(command, 20.0)

    def test_direction_reversal_releases_filter_before_full_lock(self):
        command = adaptive_smooth_steering_command(
            last_command=-14.0,
            raw_command=8.0,
            dt=1.0 / 7.0,
            straight_current_weight=0.35,
            curve_current_weight=1.0,
            straight_rate_limit=300.0,
            curve_rate_limit=600.0,
            curve_activation_command=12.0,
            curve_full_command=24.0,
        )

        self.assertGreater(command, 5.0)

    def test_small_straight_correction_remains_smoothed(self):
        command = adaptive_smooth_steering_command(
            last_command=1.0,
            raw_command=5.0,
            dt=1.0 / 7.0,
            straight_current_weight=0.35,
            curve_current_weight=1.0,
            straight_rate_limit=300.0,
            curve_rate_limit=600.0,
            curve_activation_command=12.0,
            curve_full_command=24.0,
        )

        self.assertAlmostEqual(command, 2.4)

    def test_small_straight_sign_change_does_not_release_filter(self):
        command = adaptive_smooth_steering_command(
            last_command=-5.0,
            raw_command=5.0,
            dt=1.0 / 7.0,
            straight_current_weight=0.25,
            curve_current_weight=0.55,
            straight_rate_limit=180.0,
            curve_rate_limit=300.0,
            curve_activation_command=12.0,
            curve_full_command=24.0,
        )

        self.assertAlmostEqual(command, -2.5)


if __name__ == "__main__":
    unittest.main()
