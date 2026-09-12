from xycar_map_nav.race_lap_policy import RaceLapEvent
from xycar_map_nav.race_lap_policy import RaceLapPolicy
from xycar_map_nav.race_lap_policy import RaceLapPolicyConfig


def policy() -> RaceLapPolicy:
    return RaceLapPolicy(
        RaceLapPolicyConfig(
            enabled=True,
            total_laps=3,
            signal_release_frames=2,
        )
    )


def release_signal_session(target: RaceLapPolicy) -> None:
    target.observe_direction_signal(present=False)
    target.observe_direction_signal(present=False)


def test_first_lap_is_forced_straight_after_initial_green() -> None:
    target = policy()

    assert target.force_straight
    assert (
        target.observe_start_gate(green_release_confirmed=True)
        == RaceLapEvent.RACE_STARTED
    )
    assert target.current_lap == 1
    assert target.force_straight
    assert target.ignore_stop_signals


def test_s_curve_arms_only_one_transition_per_signal_session() -> None:
    target = policy()
    target.observe_direction_signal(present=True)
    target.observe_start_gate(green_release_confirmed=True)
    release_signal_session(target)

    assert target.arm_next_lap() == RaceLapEvent.NEXT_LAP_ARMED
    assert (
        target.observe_direction_signal(present=True)
        == RaceLapEvent.LAP_STARTED
    )
    assert target.current_lap == 2
    assert target.shortcut_allowed
    assert (
        target.observe_direction_signal(present=True)
        == RaceLapEvent.NONE
    )
    assert target.current_lap == 2


def test_lap_three_is_forced_straight_after_lap_two_shortcut_success() -> None:
    target = policy()
    target.observe_start_gate(green_release_confirmed=True)
    target.arm_next_lap()
    target.observe_direction_signal(present=True)
    assert target.current_lap == 2
    assert target.shortcut_allowed

    assert (
        target.mark_shortcut_completed()
        == RaceLapEvent.SHORTCUT_COMPLETED
    )
    release_signal_session(target)
    target.arm_next_lap()
    target.observe_direction_signal(present=True)

    assert target.current_lap == 3
    assert target.force_straight


def test_lap_three_uses_perception_when_lap_two_was_straight() -> None:
    target = policy()
    target.observe_start_gate(green_release_confirmed=True)
    target.arm_next_lap()
    target.observe_direction_signal(present=True)
    release_signal_session(target)
    target.arm_next_lap()
    target.observe_direction_signal(present=True)

    assert target.current_lap == 3
    assert target.shortcut_allowed


def test_disabled_policy_preserves_existing_shortcut_and_stop_behavior() -> None:
    target = RaceLapPolicy(RaceLapPolicyConfig(enabled=False))

    assert target.shortcut_allowed
    assert not target.force_straight
    assert not target.ignore_stop_signals
