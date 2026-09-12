import pytest

from xycar_map_nav.space_drive_gate_core import GateCandidateMode
from xycar_map_nav.space_drive_gate_core import RuleToConeSteeringBlend
from xycar_map_nav.space_drive_gate_core import SpaceDriveGateController
from xycar_map_nav.space_drive_gate_core import steering_speed_limit


def test_space_toggles_continuous_fixed_speed_and_stop():
    controller = SpaceDriveGateController(speed_command=5.0)
    stopped = controller.command(
        candidate_fresh=True,
        candidate_angle_command=8.0,
        candidate_speed_command=20.0,
    )
    assert (stopped.angle_command, stopped.speed_command) == (0.0, 0.0)

    assert controller.toggle() is True
    running = controller.command(
        candidate_fresh=True,
        candidate_angle_command=8.0,
        candidate_speed_command=20.0,
    )
    assert (running.angle_command, running.speed_command) == (8.0, 5.0)

    assert controller.toggle() is False
    stopped_again = controller.command(
        candidate_fresh=True,
        candidate_angle_command=8.0,
        candidate_speed_command=20.0,
    )
    assert (stopped_again.angle_command, stopped_again.speed_command) == (
        0.0,
        0.0,
    )


def test_gate_does_not_override_selector_stop_or_stale_command():
    controller = SpaceDriveGateController(speed_command=4.0)
    controller.toggle()
    stale = controller.command(
        candidate_fresh=False,
        candidate_angle_command=15.0,
        candidate_speed_command=6.0,
    )
    selector_stop = controller.command(
        candidate_fresh=True,
        candidate_angle_command=15.0,
        candidate_speed_command=0.0,
    )
    assert stale.speed_command == 0.0
    assert stale.reason == "CANDIDATE_STALE"
    assert selector_stop.speed_command == 0.0
    assert selector_stop.reason == "SELECTOR_STOP"


def test_steering_only_keeps_candidate_angle_and_forces_zero_speed():
    controller = SpaceDriveGateController(
        speed_command=0.0,
        steering_only=True,
    )
    controller.toggle()
    output = controller.command(
        candidate_fresh=True,
        candidate_angle_command=-31.0,
        candidate_speed_command=0.0,
    )
    assert output.angle_command == -31.0
    assert output.speed_command == 0.0
    assert output.reason == "SPACE_RUN"


def test_speed_and_angle_are_clamped():
    controller = SpaceDriveGateController(
        speed_command=20.0,
        maximum_speed_command=10.0,
        maximum_abs_angle_command=42.0,
        adaptive_steering_speed_enabled=False,
    )
    controller.toggle()
    output = controller.command(
        candidate_fresh=True,
        candidate_angle_command=80.0,
        candidate_speed_command=20.0,
    )
    assert output.angle_command == 42.0
    assert output.speed_command == 10.0


def test_selector_can_apply_a_lower_avoidance_speed_cap():
    controller = SpaceDriveGateController(speed_command=8.0)
    controller.toggle()
    output = controller.command(
        candidate_fresh=True,
        candidate_angle_command=-12.0,
        candidate_speed_command=4.0,
    )
    assert output.angle_command == -12.0
    assert output.speed_command == 4.0


def test_default_hard_limit_allows_command_30():
    controller = SpaceDriveGateController(speed_command=30.0)
    controller.toggle()
    output = controller.command(
        candidate_fresh=True,
        candidate_angle_command=0.0,
        candidate_speed_command=30.0,
    )
    assert output.speed_command == 30.0


def test_steering_speed_limit_holds_cap_then_slows_to_8():
    common = dict(
        speed_cap_command=10.0,
        turn_speed_command=8.0,
        slowdown_start_angle_command=20.0,
        full_slowdown_angle_command=42.0,
    )
    assert steering_speed_limit(0.0, **common) == 10.0
    assert steering_speed_limit(20.0, **common) == 10.0
    assert steering_speed_limit(31.0, **common) == 9.0
    assert steering_speed_limit(-42.0, **common) == 8.0


def test_integrated_gate_applies_steering_speed_limit():
    controller = SpaceDriveGateController(speed_command=18.0)
    controller.toggle()
    output = controller.command(
        candidate_fresh=True,
        candidate_angle_command=31.0,
        candidate_speed_command=18.0,
    )
    assert output.speed_command == pytest.approx(14.75)


def test_curve_candidate_slows_from_14_to_12_between_18_and_42_degrees():
    controller = SpaceDriveGateController(
        speed_command=18.0,
        turn_speed_command=12.0,
    )
    controller.toggle()

    at_18 = controller.command(
        candidate_fresh=True,
        candidate_angle_command=18.0,
        candidate_speed_command=14.0,
    )
    at_30 = controller.command(
        candidate_fresh=True,
        candidate_angle_command=30.0,
        candidate_speed_command=14.0,
    )
    at_42 = controller.command(
        candidate_fresh=True,
        candidate_angle_command=42.0,
        candidate_speed_command=14.0,
    )

    assert at_18.speed_command == 14.0
    assert at_30.speed_command == pytest.approx(13.0)
    assert at_42.speed_command == 12.0


def test_lower_cone_or_avoidance_speed_still_wins():
    controller = SpaceDriveGateController(speed_command=10.0)
    controller.toggle()
    output = controller.command(
        candidate_fresh=True,
        candidate_angle_command=31.0,
        candidate_speed_command=6.0,
    )
    assert output.speed_command == 6.0


def test_adaptive_steering_speed_can_be_disabled():
    controller = SpaceDriveGateController(
        speed_command=10.0,
        adaptive_steering_speed_enabled=False,
    )
    controller.toggle()
    output = controller.command(
        candidate_fresh=True,
        candidate_angle_command=42.0,
        candidate_speed_command=10.0,
    )
    assert output.speed_command == 10.0


def test_final_gate_rate_limits_only_rule_to_cone_handoff():
    blend = RuleToConeSteeringBlend(maximum_rate_command_per_sec=60.0)

    assert blend.apply(
        -42.0,
        candidate_mode=GateCandidateMode.RULE,
        now_sec=1.00,
        output_enabled=True,
    ) == -42.0
    first_cone = blend.apply(
        -17.5,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.05,
        output_enabled=True,
    )
    second_cone = blend.apply(
        -17.5,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.10,
        output_enabled=True,
    )

    assert first_cone == pytest.approx(-39.0)
    assert second_cone == pytest.approx(-36.0)
    assert blend.active

    for step in range(3, 11):
        connected = blend.apply(
            -17.5,
            candidate_mode=GateCandidateMode.CONE,
            now_sec=1.00 + step * 0.05,
            output_enabled=True,
        )
    assert connected == pytest.approx(-17.5)
    assert not blend.active
    assert blend.apply(
        3.6,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.55,
        output_enabled=True,
    ) == pytest.approx(3.6)


def test_precomputed_cone_handoff_starts_from_stopped_wheel_angle():
    blend = RuleToConeSteeringBlend(maximum_rate_command_per_sec=60.0)

    assert blend.apply(
        -42.0,
        candidate_mode=GateCandidateMode.RULE,
        now_sec=1.00,
        output_enabled=False,
    ) == 0.0
    assert blend.apply(
        -17.5,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.05,
        output_enabled=False,
    ) == 0.0
    first_armed = blend.apply(
        -17.5,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.10,
        output_enabled=True,
    )

    assert first_armed == pytest.approx(-3.0)
    assert blend.active


def test_gate_starting_after_precompute_also_blends_from_zero():
    blend = RuleToConeSteeringBlend(maximum_rate_command_per_sec=60.0)

    assert blend.apply(
        -17.5,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.00,
        output_enabled=False,
    ) == 0.0
    assert blend.active
    assert blend.apply(
        -17.5,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.05,
        output_enabled=True,
    ) == pytest.approx(-3.0)


def test_other_mission_to_cone_is_outside_rule_handoff_blend():
    blend = RuleToConeSteeringBlend(maximum_rate_command_per_sec=60.0)

    assert blend.apply(
        12.0,
        candidate_mode=GateCandidateMode.OTHER,
        now_sec=1.00,
        output_enabled=True,
    ) == 12.0
    assert blend.apply(
        -17.5,
        candidate_mode=GateCandidateMode.CONE,
        now_sec=1.05,
        output_enabled=True,
    ) == -17.5


def test_non_transition_steering_is_not_rate_limited():
    blend = RuleToConeSteeringBlend(maximum_rate_command_per_sec=60.0)

    assert blend.apply(
        35.0,
        candidate_mode=GateCandidateMode.RULE,
        now_sec=1.00,
        output_enabled=True,
    ) == 35.0
    assert blend.apply(
        -20.0,
        candidate_mode=GateCandidateMode.RULE,
        now_sec=1.05,
        output_enabled=True,
    ) == -20.0
