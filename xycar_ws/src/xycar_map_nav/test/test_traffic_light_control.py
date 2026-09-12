from xycar_map_nav.traffic_light_control import SignalObservation
from xycar_map_nav.traffic_light_control import TrafficLightAction
from xycar_map_nav.traffic_light_control import TrafficLightConfig
from xycar_map_nav.traffic_light_control import TrafficLightController
from xycar_map_nav.traffic_light_control import TrafficLightFrame
from xycar_map_nav.traffic_light_control import (
    left_signal_approach_speed_limit,
)


def observation(confidence=0.9, area=0.01):
    return SignalObservation(
        confidence=float(confidence),
        box_area_ratio=float(area),
    )


def make_controller(**overrides):
    values = dict(
        minimum_confidence=0.50,
        left_minimum_confidence=0.50,
        stop_min_box_area_ratio=0.01,
        go_min_box_area_ratio=0.01,
        stop_required_frames=2,
        go_required_frames=2,
        left_required_frames=2,
        left_absence_frames=2,
        left_start_delay_sec=0.75,
    )
    values.update(overrides)
    return TrafficLightController(TrafficLightConfig(**values))


def test_left_signal_caps_speed_from_first_frame_until_shortcut_takes_over():
    assert left_signal_approach_speed_limit(
        left_detection_frames=1,
        decision_action=TrafficLightAction.CLEAR,
        maximum_speed_command=9.0,
        blocked_by_active_mission=False,
    ) == 9.0
    assert left_signal_approach_speed_limit(
        left_detection_frames=2,
        decision_action=TrafficLightAction.LEFT_APPROACH,
        maximum_speed_command=9.0,
        blocked_by_active_mission=False,
    ) == 9.0
    assert left_signal_approach_speed_limit(
        left_detection_frames=2,
        decision_action=TrafficLightAction.LEFT_APPROACH,
        maximum_speed_command=9.0,
        blocked_by_active_mission=True,
    ) is None
    assert left_signal_approach_speed_limit(
        left_detection_frames=0,
        decision_action=TrafficLightAction.CLEAR,
        maximum_speed_command=9.0,
        blocked_by_active_mission=False,
    ) is None


def test_default_stop_trigger_does_not_require_minimum_box_area():
    controller = TrafficLightController(TrafficLightConfig())
    frame = TrafficLightFrame(red=observation(area=0.0001))

    controller.observe(now_sec=0.0, frame=frame)
    decision = controller.observe(now_sec=0.1, frame=frame)

    assert decision.action == TrafficLightAction.STOP


def test_default_green_release_does_not_require_minimum_box_area():
    controller = TrafficLightController(TrafficLightConfig())
    red = TrafficLightFrame(red=observation(area=0.0001))
    green = TrafficLightFrame(green=observation(area=0.0001))

    controller.observe(now_sec=0.0, frame=red)
    controller.observe(now_sec=0.1, frame=red)
    controller.observe(now_sec=0.2, frame=green)
    decision = controller.observe(now_sec=0.3, frame=green)

    assert decision.action == TrafficLightAction.LEFT_APPROACH
    assert decision.signal_name == "green_4"

    finalized = controller.observe(
        now_sec=0.4,
        frame=TrafficLightFrame(),
    )
    assert finalized.action == TrafficLightAction.CLEAR
    assert finalized.direction_finalized
    assert finalized.cancel_shortcut


def test_far_red_box_does_not_stop_vehicle():
    controller = make_controller()

    for now in (1.0, 1.3, 1.6):
        decision = controller.observe(
            now_sec=now,
            frame=TrafficLightFrame(red=observation(area=0.009)),
        )

    assert decision.action == TrafficLightAction.CLEAR
    assert not controller.stop_latched


def test_supplied_close_box_ratio_clears_a_two_point_five_percent_gate():
    controller = make_controller(
        stop_min_box_area_ratio=0.025,
        go_min_box_area_ratio=0.025,
    )

    controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(red=observation(area=0.025898)),
    )
    close = controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(red=observation(area=0.025898)),
    )

    assert close.action == TrafficLightAction.STOP


def test_box_just_below_two_point_five_percent_remains_far():
    controller = make_controller(
        stop_min_box_area_ratio=0.025,
        go_min_box_area_ratio=0.025,
    )

    for now in (1.0, 1.3, 1.6):
        decision = controller.observe(
            now_sec=now,
            frame=TrafficLightFrame(red=observation(area=0.0249)),
        )

    assert decision.action == TrafficLightAction.CLEAR


def test_close_red_latches_after_two_frames_and_survives_dropout():
    controller = make_controller()

    first = controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(red=observation()),
    )
    second = controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(red=observation()),
    )
    dropout = controller.observe(
        now_sec=1.6,
        frame=TrafficLightFrame(),
    )

    assert first.action == TrafficLightAction.CLEAR
    assert second.action == TrafficLightAction.STOP
    assert second.signal_name == "red_4"
    assert dropout.action == TrafficLightAction.STOP


def test_close_yellow_uses_the_same_stop_latch():
    controller = make_controller()

    controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(yellow=observation()),
    )
    decision = controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(yellow=observation()),
    )

    assert decision.action == TrafficLightAction.STOP
    assert decision.signal_name == "yellow_4"


def test_two_close_green_frames_release_a_stop():
    controller = make_controller()
    controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(red=observation()),
    )
    controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(red=observation()),
    )

    first_green = controller.observe(
        now_sec=2.0,
        frame=TrafficLightFrame(green=observation()),
    )
    second_green = controller.observe(
        now_sec=2.3,
        frame=TrafficLightFrame(green=observation()),
    )

    assert first_green.action == TrafficLightAction.STOP
    assert second_green.action == TrafficLightAction.LEFT_APPROACH
    assert second_green.signal_name == "green_4"


def test_far_green_does_not_release_a_close_red_stop():
    controller = make_controller()
    controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(red=observation()),
    )
    controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(red=observation()),
    )

    for now in (2.0, 2.3, 2.6):
        decision = controller.observe(
            now_sec=now,
            frame=TrafficLightFrame(green=observation(area=0.009)),
        )

    assert decision.action == TrafficLightAction.STOP


def test_left_stays_in_approach_while_visible_then_starts_after_absence():
    controller = make_controller()

    first = controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(left=observation(area=0.001)),
    )
    confirmed = controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(left=observation(area=0.001)),
    )
    still_visible = controller.observe(
        now_sec=2.0,
        frame=TrafficLightFrame(left=observation(area=0.002)),
    )
    first_absence = controller.observe(
        now_sec=2.4,
        frame=TrafficLightFrame(),
    )
    second_absence = controller.observe(
        now_sec=2.8,
        frame=TrafficLightFrame(),
    )
    just_before_delay = controller.update(now_sec=3.14)
    after_delay = controller.update(now_sec=3.16)

    assert first.action == TrafficLightAction.LEFT_APPROACH
    assert confirmed.action == TrafficLightAction.LEFT_APPROACH
    assert not confirmed.shortcut_start
    assert still_visible.action == TrafficLightAction.LEFT_APPROACH
    assert not still_visible.shortcut_start
    assert not first_absence.shortcut_start
    assert not second_absence.shortcut_start
    assert not just_before_delay.shortcut_start
    assert after_delay.shortcut_start


def test_last_yolo_direction_green_cancels_fluctuating_left_sequence():
    controller = make_controller(left_absence_frames=1, left_start_delay_sec=0.0)

    observations = [
        TrafficLightFrame(left=observation(confidence=0.82)),
        TrafficLightFrame(green=observation(confidence=0.76)),
        TrafficLightFrame(left=observation(confidence=0.79)),
        TrafficLightFrame(green=observation(confidence=0.91)),
    ]
    for index, frame in enumerate(observations):
        decision = controller.observe(now_sec=0.1 * index, frame=frame)
        assert decision.action == TrafficLightAction.LEFT_APPROACH
        assert not decision.shortcut_start

    final = controller.observe(now_sec=0.5, frame=TrafficLightFrame())

    assert final.action == TrafficLightAction.CLEAR
    assert final.signal_name == "green_4"
    assert final.direction_finalized
    assert final.cancel_shortcut
    assert not final.shortcut_start
    assert not controller.left_confirmed


def test_last_yolo_direction_left_starts_after_fluctuation():
    controller = make_controller(left_absence_frames=1, left_start_delay_sec=0.0)

    for index, frame in enumerate(
        [
            TrafficLightFrame(green=observation(confidence=0.85)),
            TrafficLightFrame(left=observation(confidence=0.80)),
            TrafficLightFrame(green=observation(confidence=0.77)),
            TrafficLightFrame(left=observation(confidence=0.88)),
        ]
    ):
        decision = controller.observe(now_sec=0.1 * index, frame=frame)
        assert decision.action == TrafficLightAction.LEFT_APPROACH
        assert not decision.shortcut_start

    final = controller.observe(now_sec=0.5, frame=TrafficLightFrame())

    assert final.action == TrafficLightAction.LEFT_APPROACH
    assert final.signal_name == "left_4"
    assert final.shortcut_start
    assert final.direction_finalized
    assert not final.cancel_shortcut


def test_final_green_can_cancel_a_previous_left_start_request():
    controller = make_controller(left_absence_frames=1, left_start_delay_sec=0.0)
    controller.observe(
        now_sec=0.0,
        frame=TrafficLightFrame(left=observation()),
    )
    controller.observe(
        now_sec=0.1,
        frame=TrafficLightFrame(left=observation()),
    )
    assert controller.observe(
        now_sec=0.2,
        frame=TrafficLightFrame(),
    ).shortcut_start

    controller.observe(
        now_sec=0.3,
        frame=TrafficLightFrame(green=observation()),
    )
    final = controller.observe(now_sec=0.4, frame=TrafficLightFrame())

    assert final.action == TrafficLightAction.CLEAR
    assert final.signal_name == "green_4"
    assert final.cancel_shortcut


def test_half_second_delay_is_measured_from_first_missing_frame():
    controller = make_controller(left_start_delay_sec=0.50)
    controller.observe(
        now_sec=0.0,
        frame=TrafficLightFrame(left=observation()),
    )
    controller.observe(
        now_sec=0.3,
        frame=TrafficLightFrame(left=observation()),
    )
    controller.observe(now_sec=0.6, frame=TrafficLightFrame())
    disappeared = controller.observe(
        now_sec=0.9,
        frame=TrafficLightFrame(),
    )

    assert not disappeared.shortcut_start
    assert not controller.update(now_sec=1.09).shortcut_start
    assert controller.update(now_sec=1.101).shortcut_start


def test_three_quarter_second_delay_uses_the_same_control_loop_timer():
    controller = make_controller(left_start_delay_sec=0.75)
    controller.observe(
        now_sec=0.0,
        frame=TrafficLightFrame(left=observation()),
    )
    controller.observe(
        now_sec=0.3,
        frame=TrafficLightFrame(left=observation()),
    )
    controller.observe(now_sec=0.6, frame=TrafficLightFrame())
    controller.observe(now_sec=0.9, frame=TrafficLightFrame())

    assert not controller.update(now_sec=1.34).shortcut_start
    assert controller.update(now_sec=1.351).shortcut_start


def test_elapsed_time_without_detector_frames_cannot_start_shortcut():
    controller = make_controller()
    controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(left=observation()),
    )
    confirmed = controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(left=observation()),
    )

    assert confirmed.action == TrafficLightAction.LEFT_APPROACH
    assert not confirmed.shortcut_start
    assert controller.left_absence_frames == 0


def test_close_red_cancels_an_unstarted_left_sequence():
    controller = make_controller()
    controller.observe(
        now_sec=1.0,
        frame=TrafficLightFrame(left=observation()),
    )
    controller.observe(
        now_sec=1.3,
        frame=TrafficLightFrame(left=observation()),
    )
    controller.observe(
        now_sec=1.6,
        frame=TrafficLightFrame(red=observation()),
    )
    decision = controller.observe(
        now_sec=1.9,
        frame=TrafficLightFrame(red=observation()),
    )

    assert decision.action == TrafficLightAction.STOP
    assert not decision.shortcut_start
    assert not controller.left_confirmed


def test_active_shortcut_ignores_background_signal_updates():
    controller = make_controller()

    decision = controller.observe(
        now_sec=5.0,
        frame=TrafficLightFrame(red=observation()),
        shortcut_active=True,
    )

    assert decision.action == TrafficLightAction.CLEAR
    assert decision.signal_name == "left_4"
    assert not controller.stop_latched
