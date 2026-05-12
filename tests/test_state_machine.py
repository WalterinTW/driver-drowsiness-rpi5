from driver_drowsiness.decision.fatigue_score import FatigueFeatures
from driver_drowsiness.decision.state_machine import FatigueState, FatigueStateMachine


def normal_features() -> FatigueFeatures:
    return FatigueFeatures(
        ear=0.32,
        eye_closed=False,
        eye_closed_duration=0.0,
        perclos=0.05,
        mar=0.25,
        yawning_detected=False,
        head_pitch=0.0,
        head_yaw=0.0,
    )


def test_normal_driver_sequence_stays_normal():
    machine = FatigueStateMachine()

    states = [machine.update(normal_features()).state for _ in range(8)]

    assert states[-1] is FatigueState.NORMAL


def test_short_blink_does_not_jump_to_warning():
    machine = FatigueStateMachine()
    sequence = [
        normal_features(),
        FatigueFeatures(
            ear=0.12,
            eye_closed=True,
            eye_closed_duration=0.15,
            perclos=0.08,
            mar=0.25,
            yawning_detected=False,
            head_pitch=0.0,
            head_yaw=0.0,
        ),
        normal_features(),
        normal_features(),
    ]

    states = [machine.update(features).state for features in sequence]

    assert FatigueState.WARNING not in states
    assert FatigueState.DANGER not in states


def test_long_eye_closure_sequence_reaches_danger():
    machine = FatigueStateMachine()
    sequence = [
        FatigueFeatures(
            ear=0.14,
            eye_closed=True,
            eye_closed_duration=duration,
            perclos=0.55,
            mar=0.25,
            yawning_detected=False,
            head_pitch=0.0,
            head_yaw=0.0,
        )
        for duration in (1.0, 1.6, 2.2, 2.8, 3.2)
    ]

    states = [machine.update(features).state for features in sequence]

    assert states[-1] is FatigueState.DANGER


def test_high_perclos_sequence_reaches_warning():
    machine = FatigueStateMachine()
    sequence = [
        FatigueFeatures(
            ear=0.23,
            eye_closed=False,
            eye_closed_duration=0.0,
            perclos=0.50,
            mar=0.25,
            yawning_detected=False,
            head_pitch=0.0,
            head_yaw=0.0,
        )
        for _ in range(5)
    ]

    states = [machine.update(features).state for features in sequence]

    assert states[-1] in {FatigueState.WARNING, FatigueState.DANGER}


def test_repeated_yawning_sequence_reaches_watch_or_warning():
    machine = FatigueStateMachine()
    sequence = [
        FatigueFeatures(
            ear=0.30,
            eye_closed=False,
            eye_closed_duration=0.0,
            perclos=0.10,
            mar=0.78,
            yawning_detected=True,
            head_pitch=0.0,
            head_yaw=0.0,
        )
        for _ in range(5)
    ]

    states = [machine.update(features).state for features in sequence]

    assert states[-1] in {FatigueState.WATCH, FatigueState.WARNING}


def test_head_down_sequence_reaches_watch_or_warning():
    machine = FatigueStateMachine()
    sequence = [
        FatigueFeatures(
            ear=0.30,
            eye_closed=False,
            eye_closed_duration=0.0,
            perclos=0.10,
            mar=0.25,
            yawning_detected=False,
            head_pitch=32.0,
            head_yaw=8.0,
        )
        for _ in range(5)
    ]

    states = [machine.update(features).state for features in sequence]

    assert states[-1] in {FatigueState.WATCH, FatigueState.WARNING}
