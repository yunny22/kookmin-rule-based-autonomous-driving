from xycar_map_nav.lidar_obstacle import LidarPathObstacle
from xycar_map_nav.yolo_lidar_avoidance import (
    ShortcutAvoidanceSuppression,
    ShortcutAvoidanceSuppressionConfig,
    YoloLidarAvoidanceConfig,
    YoloLidarAvoidanceController,
    YoloLidarAvoidanceMode,
)
from xycar_map_nav.sequential_hybrid_driver import is_avoidance_detection
from xycar_map_nav.sequential_hybrid_driver import (
    avoidance_speed_limit_for_vehicle_class,
)
from xycar_map_nav.sequential_hybrid_driver import (
    right_avoidance_settling_speed_limit,
)
from xycar_map_nav.yolo_lidar_avoidance import (
    green_car_retrigger_suppressed,
    update_green_car_retrigger_confirmation,
)
from xycar_map_nav.yolo_lidar_avoidance import (
    preferred_avoidance_mode_from_image_center,
)
from xycar_map_nav.sequential_hybrid_driver import (
    preferred_avoidance_mode_from_yellow_reference,
)
from xycar_map_nav.sequential_hybrid_driver import (
    shortcut_suppresses_vehicle_class,
)
from xycar_map_nav.sequential_hybrid_driver import (
    selected_external_lateral_offset,
)
from xycar_map_nav.sequential_hybrid_driver import (
    shortcut_preposition_requested,
)
from xycar_map_nav.traffic_light_control import TrafficLightAction
from xycar_map_nav.sequential_hybrid_driver import (
    straight_road_side_decision_allowed,
)
from xycar_map_nav.sequential_hybrid_driver import (
    update_return_center_confirmation_frames,
)


def obstacle(left=1.2, right=0.5, distance=1.0, lateral=0.0):
    return LidarPathObstacle(
        distance_m=distance,
        lateral_m=lateral,
        width_m=0.35,
        point_count=6,
        x_vehicle_m=distance,
        y_vehicle_m=lateral,
        left_clearance_m=left,
        right_clearance_m=right,
    )


def test_shortcut_suppression_requires_explicit_s_curve_release():
    suppression = ShortcutAvoidanceSuppression(
        ShortcutAvoidanceSuppressionConfig(
            release_left_angle_command=-8.0,
            release_required_frames=2,
        )
    )

    suppression.start_shortcut()
    assert suppression.active
    assert not suppression.observe_rule_angle(-20.0)
    assert suppression.left_frames == 0

    suppression.start_rule_handoff()
    assert not suppression.observe_rule_angle(-9.0)
    assert suppression.left_frames == 1
    assert not suppression.observe_rule_angle(-7.9)
    assert suppression.left_frames == 0
    assert not suppression.observe_rule_angle(-12.0)
    assert not suppression.observe_rule_angle(-10.0)
    assert suppression.active
    assert suppression.release()
    assert not suppression.active
    assert not suppression.release()


def test_shortcut_suppression_can_be_disabled():
    suppression = ShortcutAvoidanceSuppression(
        ShortcutAvoidanceSuppressionConfig(enabled=False)
    )
    suppression.start_shortcut()
    suppression.start_rule_handoff()
    assert not suppression.active
    assert not suppression.observe_rule_angle(-42.0)


def test_shortcut_left_lane_offset_overrides_avoidance_until_release():
    assert selected_external_lateral_offset(
        shortcut_left_lane_active=True,
        shortcut_left_lane_offset_m=0.10,
        avoidance_offset_m=-0.31,
    ) == 0.10
    assert selected_external_lateral_offset(
        shortcut_left_lane_active=False,
        shortcut_left_lane_offset_m=0.10,
        avoidance_offset_m=-0.31,
    ) == -0.31


def test_shortcut_preposition_ignores_initial_green_release():
    assert not shortcut_preposition_requested(
        action=TrafficLightAction.LEFT_APPROACH,
        signal_name="green_4",
        shortcut_class_name="left_4",
        green_class_name="green_4",
        suppress_initial_green=True,
    )
    assert shortcut_preposition_requested(
        action=TrafficLightAction.LEFT_APPROACH,
        signal_name="green_4",
        shortcut_class_name="left_4",
        green_class_name="green_4",
        suppress_initial_green=False,
    )
    assert not shortcut_preposition_requested(
        action=TrafficLightAction.LEFT_APPROACH,
        signal_name="left_4",
        shortcut_class_name="left_4",
        green_class_name="green_4",
        suppress_initial_green=True,
    )
    assert shortcut_preposition_requested(
        action=TrafficLightAction.LEFT_APPROACH,
        signal_name="left_4",
        shortcut_class_name="left_4",
        green_class_name="green_4",
        suppress_initial_green=False,
    )


def test_shortcut_suppresses_every_vehicle_class():
    assert shortcut_suppresses_vehicle_class(
        "green_car",
        suppression_active=True,
    )
    assert shortcut_suppresses_vehicle_class(
        "red_car",
        suppression_active=True,
    )
    assert shortcut_suppresses_vehicle_class(
        "obstacle_vehicle",
        suppression_active=True,
    )
    assert not shortcut_suppresses_vehicle_class(
        "green_car",
        suppression_active=False,
    )


def test_completed_green_car_is_the_only_retrigger_suppressed_class():
    assert green_car_retrigger_suppressed(
        "green_car", blocked_until_s_curve=True
    )
    assert not green_car_retrigger_suppressed(
        "red_car", blocked_until_s_curve=True
    )
    assert not green_car_retrigger_suppressed(
        "green_car", blocked_until_s_curve=False
    )


def test_green_car_retrigger_requires_two_frames_during_s_entry():
    frames, confirmed = update_green_car_retrigger_confirmation(
        0,
        green_car_detected=True,
        blocked_until_s_curve=True,
        s_curve_entry_guard_active=True,
        required_frames=2,
    )
    assert frames == 1
    assert not confirmed
    frames, confirmed = update_green_car_retrigger_confirmation(
        frames,
        green_car_detected=True,
        blocked_until_s_curve=True,
        s_curve_entry_guard_active=True,
        required_frames=2,
    )
    assert frames == 2
    assert confirmed


def test_green_car_retrigger_resets_on_detection_gap_or_inactive_guard():
    assert update_green_car_retrigger_confirmation(
        1,
        green_car_detected=False,
        blocked_until_s_curve=True,
        s_curve_entry_guard_active=True,
        required_frames=2,
    ) == (0, False)
    assert update_green_car_retrigger_confirmation(
        1,
        green_car_detected=True,
        blocked_until_s_curve=True,
        s_curve_entry_guard_active=False,
        required_frames=2,
    ) == (0, False)


def test_dynamic_car_camera_center_fallback_avoids_opposite_side():
    assert preferred_avoidance_mode_from_image_center(
        object_center_x=100.0,
        image_width=640.0,
    ) == YoloLidarAvoidanceMode.AVOID_RIGHT
    assert preferred_avoidance_mode_from_image_center(
        object_center_x=500.0,
        image_width=640.0,
    ) == YoloLidarAvoidanceMode.AVOID_LEFT


def test_red_and_green_car_use_separate_avoidance_speed_limits():
    common = {
        "default_speed_limit_command": 8.0,
        "red_car_speed_limit_command": 8.0,
        "green_car_speed_limit_command": 15.0,
    }
    assert avoidance_speed_limit_for_vehicle_class(
        "red_car",
        **common,
    ) == 8.0
    assert avoidance_speed_limit_for_vehicle_class(
        "green_car",
        **common,
    ) == 15.0
    assert avoidance_speed_limit_for_vehicle_class(
        "obstacle_vehicle",
        **common,
    ) == 8.0


def test_selected_vehicle_class_speed_persists_during_avoidance():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            yolo_required_frames=1,
            speed_limit_command=8.0,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        target_class_name="green_car",
        speed_limit_command=15.0,
    )
    state = controller.step(
        now_sec=0.1,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_LEFT
    assert state.target_class_name == "green_car"
    assert state.speed_limit_command == 15.0


def test_red_car_yolo_expires_sooner_without_changing_green_car_timeout():
    config = YoloLidarAvoidanceConfig(
        yolo_timeout_sec=1.0,
        red_car_yolo_timeout_sec=0.5,
        yolo_required_frames=1,
        immediate_on_yolo=True,
        preferred_side_required_frames=1,
        minimum_avoid_sec=0.0,
        clear_hold_sec=0.0,
        return_hold_sec=10.0,
    )

    red = YoloLidarAvoidanceController(config)
    red.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=float("inf"),
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        target_class_name="red_car",
    )
    assert red.step(
        now_sec=0.50,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.AVOID_LEFT
    red.step(
        now_sec=0.51,
        dt_sec=0.01,
        obstacle=None,
        cone_active=False,
    )
    assert red.step(
        now_sec=0.52,
        dt_sec=0.01,
        obstacle=None,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.RETURN_CENTER

    green = YoloLidarAvoidanceController(config)
    green.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=float("inf"),
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        target_class_name="green_car",
    )
    assert green.step(
        now_sec=0.52,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.AVOID_LEFT


def test_green_car_yolo_loss_returns_even_with_stale_lidar_cluster():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            yolo_timeout_sec=0.10,
            yolo_required_frames=1,
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            minimum_avoid_sec=0.0,
            clear_hold_sec=0.0,
            return_hold_sec=10.0,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=1.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        target_class_name="green_car",
    )
    stale_cluster = obstacle(distance=1.0)
    assert controller.step(
        now_sec=0.11,
        dt_sec=0.1,
        obstacle=stale_cluster,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.AVOID_LEFT
    assert controller.step(
        now_sec=0.12,
        dt_sec=0.01,
        obstacle=stale_cluster,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.RETURN_CENTER


def test_green_car_can_hold_the_avoidance_lane_longer_than_other_classes():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            yolo_timeout_sec=0.10,
            yolo_required_frames=1,
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            minimum_avoid_sec=0.0,
            clear_hold_sec=0.0,
            green_car_clear_hold_sec=0.60,
            return_hold_sec=10.0,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=1.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        target_class_name="green_car",
    )
    assert controller.step(
        now_sec=0.11,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.AVOID_LEFT
    assert controller.step(
        now_sec=0.70,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.AVOID_LEFT
    assert controller.step(
        now_sec=0.72,
        dt_sec=0.02,
        obstacle=None,
        cone_active=False,
    ).mode == YoloLidarAvoidanceMode.RETURN_CENTER


def test_cone_is_only_an_avoidance_candidate_in_temporary_test_mode():
    common = {
        "class_name": "cone",
        "confidence": 0.8,
        "vehicle_names": {"car", "obstacle_vehicle"},
        "vehicle_min_confidence": 0.45,
        "cone_min_confidence": 0.50,
    }
    assert not is_avoidance_detection(
        **common,
        cone_as_vehicle_obstacle=False,
    )
    assert is_avoidance_detection(
        **common,
        cone_as_vehicle_obstacle=True,
    )


def test_cone_avoidance_uses_its_own_confidence_threshold():
    assert not is_avoidance_detection(
        class_name="cone",
        confidence=0.49,
        vehicle_names={"car"},
        vehicle_min_confidence=0.20,
        cone_as_vehicle_obstacle=True,
        cone_min_confidence=0.50,
    )


def test_yellow_reference_maps_object_to_opposite_avoidance_side():
    yellow_x_by_y = [100.0 + 0.25 * row for row in range(144)]
    assert preferred_avoidance_mode_from_yellow_reference(
        object_x=80.0,
        object_y=100.0,
        yellow_x_by_y=yellow_x_by_y,
        deadband_px=6.0,
    ) == YoloLidarAvoidanceMode.AVOID_RIGHT
    assert preferred_avoidance_mode_from_yellow_reference(
        object_x=140.0,
        object_y=100.0,
        yellow_x_by_y=yellow_x_by_y,
        deadband_px=6.0,
    ) == YoloLidarAvoidanceMode.AVOID_LEFT
    assert preferred_avoidance_mode_from_yellow_reference(
        object_x=128.0,
        object_y=100.0,
        yellow_x_by_y=yellow_x_by_y,
        deadband_px=6.0,
    ) is None


def test_side_decision_requires_fresh_straight_rule_and_yellow_geometry():
    common = dict(
        rule_command_age_sec=0.05,
        yellow_reference_age_sec=0.05,
        yellow_reference_rmse_px=1.5,
        yellow_reference_span_ratio=0.50,
        maximum_rule_angle_command=8.0,
        rule_command_timeout_sec=0.25,
        yellow_reference_timeout_sec=0.40,
        maximum_yellow_rmse_px=3.0,
        minimum_yellow_span_ratio=0.20,
    )
    assert straight_road_side_decision_allowed(
        rule_angle_command=5.0,
        **common,
    )
    assert not straight_road_side_decision_allowed(
        rule_angle_command=12.0,
        **common,
    )
    assert not straight_road_side_decision_allowed(
        rule_angle_command=5.0,
        **{**common, "yellow_reference_rmse_px": 4.0},
    )


def test_curve_gate_clears_uncommitted_avoidance_side():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(preferred_side_required_frames=1)
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
    )
    assert controller.preferred_mode == YoloLidarAvoidanceMode.AVOID_LEFT
    controller.observe_yolo(
        now_sec=0.1,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.0,
        preferred_mode=None,
        side_decision_allowed=False,
    )
    assert controller.preferred_mode is None


def test_curve_gate_does_not_flip_an_active_avoidance_side():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            yolo_required_frames=1,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
    )
    state = controller.step(
        now_sec=0.0,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_LEFT
    controller.observe_yolo(
        now_sec=0.1,
        detected=True,
        confidence=0.9,
        lidar_distance_m=1.5,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        side_decision_allowed=False,
    )
    assert controller.preferred_mode == YoloLidarAvoidanceMode.AVOID_LEFT


def test_oscillating_camera_side_is_not_accepted_from_one_frame():
    controller = YoloLidarAvoidanceController(YoloLidarAvoidanceConfig())
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
    )
    controller.observe_yolo(
        now_sec=0.1,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
    )
    assert controller.preferred_mode is None


def arm(
    controller,
    distance=1.0,
    preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
):
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=distance,
        preferred_mode=preferred_mode,
    )
    controller.observe_yolo(
        now_sec=0.1,
        detected=True,
        confidence=0.9,
        lidar_distance_m=distance,
        preferred_mode=preferred_mode,
    )


def test_yolo_tracks_before_entry_distance_then_chooses_clear_side():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(entry_distance_m=1.2)
    )
    arm(controller, distance=2.0)
    state = controller.step(
        now_sec=0.1,
        dt_sec=0.1,
        obstacle=obstacle(distance=2.0),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.YOLO_TRACKING
    assert not state.controls_vehicle

    controller.update_lidar_distance(1.0)
    state = controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(left=1.4, right=0.4),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_LEFT
    assert state.lateral_offset_m > 0.0
    assert state.controls_vehicle


def test_left_camera_obstacle_keeps_right_avoidance_when_right_is_clear():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(entry_distance_m=1.2)
    )
    for now in (0.0, 0.1):
        controller.observe_yolo(
            now_sec=now,
            detected=True,
            confidence=0.9,
            lidar_distance_m=1.0,
            preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        )
    state = controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(left=1.4, right=0.8),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_RIGHT


def test_left_camera_obstacle_waits_when_right_side_is_too_narrow():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(entry_distance_m=1.2)
    )
    for now in (0.0, 0.1):
        controller.observe_yolo(
            now_sec=now,
            detected=True,
            confidence=0.9,
            lidar_distance_m=1.0,
            preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        )
    state = controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(left=1.4, right=0.5),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.WAIT_SIDE_CLEAR


def test_lidar_lateral_position_does_not_override_yellow_reference():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(entry_distance_m=1.2)
    )
    for now in (0.0, 0.1):
        controller.observe_yolo(
            now_sec=now,
            detected=True,
            confidence=0.9,
            lidar_distance_m=1.0,
            preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        )
    state = controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(left=1.2, right=1.2, lateral=-0.12),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_RIGHT


def test_missing_camera_side_waits_even_when_lidar_space_is_clear():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(entry_distance_m=1.2)
    )
    for now in (0.0, 0.1):
        controller.observe_yolo(
            now_sec=now,
            detected=True,
            confidence=0.9,
            lidar_distance_m=1.0,
            preferred_mode=None,
        )
    state = controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(left=1.2, right=1.2, lateral=0.2),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.WAIT_SIDE_CLEAR


def test_immediate_yolo_avoidance_uses_camera_preferred_side():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(immediate_on_yolo=True)
    )
    for now in (0.0, 0.1):
        controller.observe_yolo(
            now_sec=now,
            detected=True,
            confidence=0.9,
            lidar_distance_m=6.0,
            preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        )
    state = controller.step(
        now_sec=0.1,
        dt_sec=0.1,
        obstacle=obstacle(left=1.2, right=1.2, distance=6.0),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_LEFT
    assert state.lateral_offset_m > 0.0
    assert state.controls_vehicle


def test_immediate_yolo_works_without_matching_lidar_return():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            yolo_required_frames=1,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=float("inf"),
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
    )
    assert controller.state().mode == YoloLidarAvoidanceMode.AVOID_LEFT
    state = controller.step(
        now_sec=0.0,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_LEFT


def test_immediate_yolo_rearms_during_return_without_distance_gate():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            yolo_required_frames=1,
            minimum_avoid_sec=0.0,
            clear_hold_sec=0.0,
            return_hold_sec=1.0,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=float("inf"),
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
    )
    assert controller.state().mode == YoloLidarAvoidanceMode.AVOID_RIGHT
    controller.step(
        now_sec=1.2,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    state = controller.step(
        now_sec=1.3,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.RETURN_CENTER

    controller.observe_yolo(
        now_sec=1.4,
        detected=True,
        confidence=0.9,
        lidar_distance_m=8.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
    )
    state = controller.step(
        now_sec=1.4,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_RIGHT


def test_immediate_yolo_does_not_wait_for_lidar_side_clearance():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            yolo_required_frames=1,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=0.8,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
    )
    state = controller.step(
        now_sec=0.0,
        dt_sec=0.1,
        obstacle=obstacle(left=0.2, right=0.2),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_RIGHT


def test_immediate_yolo_keeps_driving_until_straight_side_is_decided():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            preferred_side_required_frames=1,
            yolo_required_frames=1,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=0.8,
        preferred_mode=None,
        side_decision_allowed=False,
    )
    state = controller.step(
        now_sec=0.0,
        dt_sec=0.1,
        obstacle=obstacle(left=1.2, right=1.2),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.YOLO_TRACKING
    assert not state.controls_vehicle
    assert state.lateral_offset_m == 0.0

    controller.observe_yolo(
        now_sec=0.1,
        detected=True,
        confidence=0.9,
        lidar_distance_m=0.7,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        side_decision_allowed=True,
    )
    state = controller.step(
        now_sec=0.1,
        dt_sec=0.1,
        obstacle=obstacle(left=1.2, right=1.2),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_RIGHT
    assert state.controls_vehicle
    assert state.lateral_offset_m < 0.0


def test_immediate_avoidance_reselects_side_after_confirmed_frames():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            yolo_timeout_sec=2.5,
            active_side_reselection_required_frames=2,
        )
    )
    for now in (0.0, 0.1):
        controller.observe_yolo(
            now_sec=now,
            detected=True,
            confidence=0.9,
            lidar_distance_m=6.0,
            preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        )
    controller.step(
        now_sec=0.1,
        dt_sec=0.1,
        obstacle=obstacle(left=1.2, right=1.2, distance=6.0),
        cone_active=False,
    )
    state = controller.step(
        now_sec=2.2,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_LEFT

    controller.observe_yolo(
        now_sec=2.2,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.3,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
    )
    state = controller.step(
        now_sec=2.3,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_LEFT

    controller.observe_yolo(
        now_sec=2.3,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.2,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
    )
    state = controller.step(
        now_sec=2.4,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_RIGHT


def test_confirmed_side_reselection_uses_faster_offset_rate_until_settled():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            immediate_on_yolo=True,
            yolo_required_frames=1,
            preferred_side_required_frames=1,
            active_side_reselection_required_frames=2,
            left_offset_m=0.13,
            right_offset_m=0.13,
            offset_rate_mps=0.65,
            active_side_reselection_offset_rate_mps=1.30,
            yolo_timeout_sec=2.5,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=2.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
    )
    state = controller.step(
        now_sec=0.0,
        dt_sec=0.2,
        obstacle=None,
        cone_active=False,
    )
    assert abs(state.lateral_offset_m - 0.13) < 1.0e-9

    for now in (0.1, 0.2):
        controller.observe_yolo(
            now_sec=now,
            detected=True,
            confidence=0.9,
            lidar_distance_m=1.5,
            preferred_mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        )
    state = controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.AVOID_RIGHT
    assert abs(state.lateral_offset_m) < 1.0e-9

    state = controller.step(
        now_sec=0.3,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert abs(state.lateral_offset_m + 0.13) < 1.0e-9


def test_right_avoidance_speed_cap_only_applies_before_offset_settles():
    assert right_avoidance_settling_speed_limit(
        mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        lateral_offset_m=0.02,
        right_offset_m=0.13,
        tolerance_m=0.02,
        speed_limit_command=14.0,
    ) == 14.0
    assert right_avoidance_settling_speed_limit(
        mode=YoloLidarAvoidanceMode.AVOID_RIGHT,
        lateral_offset_m=-0.12,
        right_offset_m=0.13,
        tolerance_m=0.02,
        speed_limit_command=14.0,
    ) is None
    assert right_avoidance_settling_speed_limit(
        mode=YoloLidarAvoidanceMode.AVOID_LEFT,
        lateral_offset_m=0.0,
        right_offset_m=0.13,
        tolerance_m=0.02,
        speed_limit_command=14.0,
    ) is None


def test_both_sides_blocked_waits_instead_of_turning():
    controller = YoloLidarAvoidanceController(YoloLidarAvoidanceConfig())
    arm(controller)
    state = controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(left=0.4, right=0.5),
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.WAIT_SIDE_CLEAR
    assert state.speed_limit_command == 0.0
    assert state.lateral_offset_m == 0.0


def test_clear_hold_returns_to_center_and_releases_to_base_policy():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            yolo_timeout_sec=0.2,
            minimum_avoid_sec=0.1,
            clear_hold_sec=0.2,
            return_hold_sec=0.1,
            offset_rate_mps=1.0,
        )
    )
    arm(controller)
    controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(),
        cone_active=False,
    )
    controller.step(
        now_sec=0.4,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    state = controller.step(
        now_sec=0.61,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.RETURN_CENTER
    for index in range(10):
        state = controller.step(
            now_sec=0.72 + 0.1 * index,
            dt_sec=0.1,
            obstacle=None,
            cone_active=False,
        )
        if state.mode == YoloLidarAvoidanceMode.IDLE:
            break
    assert state.mode == YoloLidarAvoidanceMode.IDLE


def test_return_center_waits_for_physical_lane_confirmation():
    controller = YoloLidarAvoidanceController(
        YoloLidarAvoidanceConfig(
            yolo_timeout_sec=0.1,
            yolo_required_frames=1,
            preferred_side_required_frames=1,
            immediate_on_yolo=True,
            minimum_avoid_sec=0.0,
            clear_hold_sec=0.0,
            return_hold_sec=0.0,
            offset_rate_mps=10.0,
        )
    )
    controller.observe_yolo(
        now_sec=0.0,
        detected=True,
        confidence=0.9,
        lidar_distance_m=1.0,
        preferred_mode=YoloLidarAvoidanceMode.AVOID_LEFT,
    )
    controller.step(
        now_sec=0.0,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
    )
    controller.step(
        now_sec=0.11,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
        return_center_confirmed=False,
    )
    state = controller.step(
        now_sec=0.12,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
        return_center_confirmed=False,
    )
    assert state.mode == YoloLidarAvoidanceMode.RETURN_CENTER
    assert state.lateral_offset_m == 0.0

    state = controller.step(
        now_sec=0.13,
        dt_sec=0.1,
        obstacle=None,
        cone_active=False,
        return_center_confirmed=True,
    )
    assert state.mode == YoloLidarAvoidanceMode.IDLE


def test_return_center_confirmation_requires_three_valid_cte_frames():
    frames = 0
    for error in (0.12, 0.07, 0.06, 0.05):
        frames = update_return_center_confirmation_frames(
            frames,
            return_center_active=True,
            path_valid=True,
            cross_track_error_m=error,
            maximum_abs_error_m=0.08,
        )
    assert frames == 3
    assert update_return_center_confirmation_frames(
        frames,
        return_center_active=True,
        path_valid=False,
        cross_track_error_m=0.01,
        maximum_abs_error_m=0.08,
    ) == 0


def test_cone_has_priority_and_cancels_obstacle_override():
    controller = YoloLidarAvoidanceController(YoloLidarAvoidanceConfig())
    arm(controller)
    controller.step(
        now_sec=0.2,
        dt_sec=0.1,
        obstacle=obstacle(),
        cone_active=False,
    )
    state = controller.step(
        now_sec=0.3,
        dt_sec=0.1,
        obstacle=obstacle(),
        cone_active=True,
    )
    assert state.mode == YoloLidarAvoidanceMode.IDLE
    assert state.lateral_offset_m == 0.0
