import json

import numpy as np
import pytest

from offtheball.analysis import AnalysisSession
from offtheball.calibration_state import TemporalCalibrationState
from offtheball.detection import Detection
from offtheball.inputs import Frame
from offtheball.keypoint_calibration import KeypointCalibration


SIZE = (640, 480)
FIELD = np.asarray([
    [0, 0], [52.5, 0], [105, 0],
    [0, 34], [52.5, 34], [105, 34],
    [0, 68], [52.5, 68], [105, 68],
], dtype=float)
BASE_MATRIX = np.asarray([
    [4.6, 0.35, 70],
    [0.12, 3.9, 75],
    [0.0012, 0.0007, 1],
], dtype=float)


def image_points(shift=(0.0, 0.0)):
    homogeneous = np.column_stack((FIELD, np.ones(len(FIELD)))) @ BASE_MATRIX.T
    points = homogeneous[:, :2] / homogeneous[:, 2, None]
    return points + np.asarray(shift, dtype=float)


def candidate(name, shift=(0.0, 0.0)):
    return KeypointCalibration.fit(
        image_points(shift), FIELD, image_size=SIZE, frame_id=name,
        min_inlier_ratio=.9,
    )


def test_initial_candidate_small_motion_and_diagnostics_are_accepted():
    state = TemporalCalibrationState(
        image_size=SIZE, frame_id="video", max_projection_jump_m=3.0)
    assert state.status == "not_calibrated"
    assert state.observe(candidate("frame-0"))
    assert state.status == "active"
    assert state.observe(candidate("frame-1", shift=(2, 1)))
    report = state.diagnostics()
    assert report["status"] == "active"
    assert report["source"] == "pitch_keypoints_temporal"
    assert report["age_frames"] == 0
    assert report["projection_jump_m"] < 3.0
    json.dumps(report, allow_nan=False)

    session = AnalysisSession(object(), object(), calibration=state)
    assert session.calibration_diagnostics()["projection_jump_m"] == report["projection_jump_m"]


def test_large_jump_is_rejected_while_previous_transform_is_fresh():
    state = TemporalCalibrationState(
        image_size=SIZE, frame_id="video", max_reuse_frames=3,
        max_projection_jump_m=3.0)
    first = candidate("frame-0")
    jumped = candidate("frame-1", shift=(90, 50))
    assert state.observe(first)
    before = state.project([image_points()[4]], image_size=SIZE, frame_id="video")
    assert not state.observe(jumped)
    assert state.current is first
    assert state.age_frames == 1
    assert state.diagnostics()["last_update"] == "rejected_projection_jump"
    after = state.project([image_points()[4]], image_size=SIZE, frame_id="video")
    np.testing.assert_allclose(after, before)


def test_missing_candidates_expire_coordinates_and_fresh_candidate_recovers():
    state = TemporalCalibrationState(
        image_size=SIZE, frame_id="video", max_reuse_frames=2,
        max_projection_jump_m=3.0)
    assert state.observe(candidate("frame-0"))
    point = image_points()[4]
    for expected_age in (1, 2):
        assert not state.observe(None)
        assert state.age_frames == expected_age
        assert state.status == "active"
        assert np.isfinite(state.project([point], image_size=SIZE, frame_id="video")).all()
    assert not state.observe(None)
    assert state.status == "stale"
    assert np.isnan(state.project([point], image_size=SIZE, frame_id="video")).all()

    # Once stale, a geometrically valid candidate can establish a new view
    # even if a scene cut made it very different from the old transform.
    replacement = candidate("frame-after-cut", shift=(90, 50))
    assert state.observe(replacement)
    assert state.current is replacement
    assert state.status == "active"
    new_point = image_points((90, 50))[4]
    projected = state.project([new_point], image_size=SIZE, frame_id="video")
    np.testing.assert_allclose(projected, [FIELD[4]], atol=1e-5)


def test_insufficient_overlap_is_rejected_until_old_state_expires():
    state = TemporalCalibrationState(
        image_size=SIZE, frame_id="video", max_reuse_frames=0,
        max_projection_jump_m=20.0, min_shared_points=9)
    assert state.observe(candidate("frame-0"))
    # The shifted hull does not contain all nine of the new candidate anchors.
    replacement = candidate("frame-1", shift=(90, 50))
    assert not state.observe(replacement)
    assert state.status == "stale"
    assert state.diagnostics()["last_update"] == "rejected_insufficient_overlap"
    assert state.observe(replacement)


def test_reset_and_input_identity_fail_safely():
    state = TemporalCalibrationState(image_size=SIZE, frame_id="video")
    state.observe(candidate("frame-0"))
    state.reset()
    assert state.status == "not_calibrated"
    assert np.isnan(state.project([[100, 100]], image_size=SIZE, frame_id="video")).all()
    with pytest.raises(ValueError, match="recalibration"):
        state.project([[100, 100]], image_size=(320, 240), frame_id="video")
    with pytest.raises(ValueError, match="recalibration"):
        state.project([[100, 100]], image_size=SIZE, frame_id="other")
    with pytest.raises(TypeError):
        state.observe(object())


def test_candidate_with_changed_image_size_is_rejected():
    small_field = FIELD.copy()
    small_image = image_points() * .5
    changed = KeypointCalibration.fit(
        small_image, small_field, image_size=(320, 240), frame_id="small",
        min_inlier_ratio=.9,
    )
    state = TemporalCalibrationState(image_size=SIZE, frame_id="video")
    with pytest.raises(ValueError, match="image size changed"):
        state.observe(changed)


def test_analysis_session_emits_coordinates_only_while_state_is_active():
    foot = image_points()[4]

    class Detector:
        def detect(self, _pixels):
            x, y = foot
            return [Detection((x - 5, y - 20, x + 5, y), .9)]

    class Teams:
        ready = False
        def update(self, _pixels, detections):
            return [None] * len(detections)

    state = TemporalCalibrationState(
        image_size=SIZE, frame_id="video", max_reuse_frames=0)
    state.observe(candidate("frame-0"))
    session = AnalysisSession(Detector(), Teams(), calibration=state)
    pixels = np.full((SIZE[1], SIZE[0], 3), 80, np.uint8)
    active = session.process(Frame(pixels, 0, 0.0, 0.0))[0]
    np.testing.assert_allclose(active.field_xy, FIELD[4], atol=1e-5)

    state.observe(None)
    stale = session.process(Frame(pixels, 1, 0.1, 0.1))[0]
    assert stale.field_xy is None
    assert session.calibration_diagnostics()["status"] == "stale"


@pytest.mark.parametrize("kwargs", [
    {"image_size": (0, 480), "frame_id": "video"},
    {"image_size": SIZE, "frame_id": ""},
    {"image_size": SIZE, "frame_id": "video", "max_reuse_frames": -1},
    {"image_size": SIZE, "frame_id": "video", "max_projection_jump_m": 0},
    {"image_size": SIZE, "frame_id": "video", "min_shared_points": 0},
])
def test_invalid_state_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TemporalCalibrationState(**kwargs)
