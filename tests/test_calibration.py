import json
import numpy as np
import pytest
from offtheball.calibration import AutoPitchCalibration, ManualCalibration

SIZE = (640, 480)
FRAME = "test-frame"
SOURCE = np.array([[50, 50], [550, 60], [500, 400], [80, 390]], dtype=float)
MATRIX = np.array([[0.2, 0.015, -8], [-0.01, 0.21, -4], [0.0002, 0.0001, 1]])


def transform(points):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ MATRIX.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def calibration():
    return ManualCalibration.fit(SOURCE, transform(SOURCE), image_size=SIZE, frame_id=FRAME)


def test_perspective_mapping_and_no_extrapolation():
    points = np.array([[150, 100], [450, 300], [0, 0], [600, 470]])
    result = calibration().project(points, image_size=SIZE, frame_id=FRAME)
    np.testing.assert_allclose(result[:2], transform(points[:2]), atol=1e-5)
    assert np.isnan(result[2:]).all()


@pytest.mark.parametrize("points", [
    [[10,10], [20,20], [30,30], [40,40]],
    [[10,10], [20,20], [30,30], [40,40.0001]],
    [[10,10], [10,10], [20,30], [40,20]],
    [[10,10], [20,20], [30,float("nan")], [40,50]],
    [[10,10], [20,20], [30,40]],
])
def test_degenerate_points_rejected(points):
    with pytest.raises(ValueError):
        ManualCalibration.fit(points, points, image_size=SIZE, frame_id=FRAME)


@pytest.mark.parametrize("size", [(0,480), (640,-1), (640.0,480), (True,480), (640,)])
def test_invalid_image_size_rejected(size):
    with pytest.raises(ValueError):
        ManualCalibration.fit(SOURCE, transform(SOURCE), image_size=size, frame_id=FRAME)


def test_changed_frame_or_resolution_rejected():
    instance = calibration()
    for size, frame in [((320,240), FRAME), (SIZE, "next-frame")]:
        with pytest.raises(ValueError):
            instance.project([[150,100]], image_size=size, frame_id=frame)


def test_roundtrip_revalidates_points(tmp_path):
    path = tmp_path / "calibration.json"
    calibration().save(path)
    loaded = ManualCalibration.load(path)
    point = [[250,200]]
    np.testing.assert_allclose(loaded.project(point, image_size=SIZE, frame_id=FRAME), transform(point), atol=1e-5)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_points"][0] = [-1,10]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        ManualCalibration.load(path)


def test_held_out_point_and_distance_errors():
    points = np.array([[150,100], [450,300], [0,0]])
    report = calibration().evaluate(points, transform(points), image_size=SIZE, frame_id=FRAME, segments=[(0,1),(1,2)])
    assert report["valid_count"] == 2
    assert report["max_position_error_m"] < 1e-5
    assert report["position_errors_m"][2] is None
    assert report["median_relative_distance_error_ge_10m"] < 1e-6
    assert report["segments"][1]["measured_m"] is None
    json.dumps(report, allow_nan=False)


def test_fitted_points_cannot_be_claimed_as_held_out():
    with pytest.raises(ValueError, match="held-out"):
        calibration().evaluate(SOURCE, transform(SOURCE), image_size=SIZE, frame_id=FRAME)


def test_reference_points_are_copied():
    source = SOURCE.copy()
    instance = ManualCalibration.fit(source, transform(source), image_size=SIZE, frame_id=FRAME)
    source[:] = 0
    np.testing.assert_array_equal(instance.image_points, SOURCE)


def test_projective_horizon_rejected():
    # A finite but physically unusable mapping whose denominator changes sign
    # across the source polygon must not yield apparently valid metre values.
    matrix = np.array([[1,0,0], [0,1,0], [0.01,0,-3]])
    h = np.column_stack((SOURCE, np.ones(4))) @ matrix.T
    with pytest.raises(ValueError):
        ManualCalibration.fit(SOURCE, h[:,:2]/h[:,2,None], image_size=SIZE, frame_id=FRAME)


def test_three_collinear_of_four_points_rejected():
    points = [[10,10], [20,20], [30,30], [40,10]]
    with pytest.raises(ValueError, match="unique homography"):
        ManualCalibration.fit(points, points, image_size=SIZE, frame_id=FRAME)


def test_manual_calibration_reports_stable_diagnostics():
    report = calibration().diagnostics()
    assert report["status"] == "active"
    assert report["source"] == "manual_fixed_frame"
    assert report["inlier_count"] == 4


def test_auto_calibration_reports_staleness_without_hiding_legacy_projection():
    auto = AutoPitchCalibration(calibration(), stale_after_frames=2)
    blank = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    for _ in range(3):
        assert not auto.update(blank)
    assert auto.diagnostics()["status"] == "stale"
    assert auto.diagnostics()["age_frames"] == 3
    # Diagnostics are additive in this feature; enforcing expiry belongs to
    # the validated automatic-calibration stage.
    point = [[250, 200]]
    np.testing.assert_allclose(auto.project(point, image_size=SIZE, frame_id=FRAME), transform(point), atol=1e-5)
