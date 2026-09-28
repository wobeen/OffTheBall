import json

import cv2
import numpy as np
import pytest

from offtheball.keypoint_calibration import KeypointCalibration


SIZE = (640, 480)
FRAME_ID = "synthetic-keypoints"
FIELD_POINTS = np.asarray([
    [0, 0], [52.5, 0], [105, 0],
    [0, 34], [52.5, 34], [105, 34],
    [0, 68], [52.5, 68], [105, 68],
    [16.5, 13.84], [88.5, 54.16], [52.5, 24.85],
], dtype=float)
FIELD_TO_IMAGE = np.asarray([
    [4.6, 0.35, 70],
    [0.12, 3.9, 75],
    [0.0012, 0.0007, 1],
], dtype=float)


def transform(points, matrix=FIELD_TO_IMAGE):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def observations(*, with_outliers=True):
    image = transform(FIELD_POINTS)
    noise = np.asarray([
        [0.2, -0.2], [-0.3, 0.1], [0.1, 0.3], [0.3, 0.2],
        [-0.2, -0.1], [0.1, -0.2], [-0.3, 0.2], [0.2, 0.1],
        [0.1, -0.3], [-0.2, 0.2], [0.2, -0.1], [-0.1, 0.1],
    ])
    image = image + noise
    if with_outliers:
        image[-2:] = [[580, 70], [80, 410]]
    return image


def test_ransac_rejects_outliers_and_projects_held_out_point():
    cv2.setRNGSeed(7)
    calibration = KeypointCalibration.fit(
        observations(), FIELD_POINTS, image_size=SIZE, frame_id=FRAME_ID,
        min_inliers=8, min_inlier_ratio=.65,
    )
    assert calibration.correspondence_count == 12
    assert calibration.inlier_count == 10
    assert calibration.inlier_ratio == pytest.approx(10 / 12)
    assert calibration.median_reprojection_error_px < .5

    expected_field = np.asarray([[30.0, 30.0]])
    held_out_image = transform(expected_field)
    projected = calibration.project(held_out_image, image_size=SIZE, frame_id=FRAME_ID)
    np.testing.assert_allclose(projected, expected_field, atol=.2)

    report = calibration.diagnostics()
    assert report["source"] == "pitch_keypoints_ransac"
    assert report["inlier_count"] == 10
    json.dumps(report, allow_nan=False)


def test_projection_is_limited_to_observed_keypoint_hull():
    calibration = KeypointCalibration.fit(
        observations(with_outliers=False), FIELD_POINTS,
        image_size=SIZE, frame_id=FRAME_ID,
    )
    inside = transform(np.asarray([[52.5, 34.0]]))[0]
    result = calibration.project([inside, [5, 5]], image_size=SIZE, frame_id=FRAME_ID)
    assert np.isfinite(result[0]).all()
    assert np.isnan(result[1]).all()


@pytest.mark.parametrize("image_points, field_points, message", [
    (observations()[:5], FIELD_POINTS[:5], "at least 6"),
    (observations(), FIELD_POINTS[:-1], "equal shapes"),
    (np.column_stack((np.linspace(100, 120, 12), np.linspace(100, 120, 12))),
     FIELD_POINTS, "too little area"),
])
def test_invalid_or_poorly_distributed_correspondences_are_rejected(
        image_points, field_points, message):
    with pytest.raises(ValueError, match=message):
        KeypointCalibration.fit(
            image_points, field_points, image_size=SIZE, frame_id=FRAME_ID,
        )


def test_low_inlier_ratio_is_rejected():
    image = observations(with_outliers=False)
    image[5:] = np.asarray([
        [30, 400], [600, 400], [40, 40], [590, 60],
        [310, 440], [620, 240], [20, 230],
    ])
    cv2.setRNGSeed(11)
    with pytest.raises(ValueError, match="inlier ratio|inliers"):
        KeypointCalibration.fit(
            image, FIELD_POINTS, image_size=SIZE, frame_id=FRAME_ID,
            min_inliers=6, min_inlier_ratio=.75,
        )


def test_inputs_are_copied_and_frame_identity_is_enforced():
    image = observations(with_outliers=False)
    original = image.copy()
    calibration = KeypointCalibration.fit(
        image, FIELD_POINTS, image_size=SIZE, frame_id=FRAME_ID,
    )
    image[:] = 0
    assert np.any(calibration.image_points != 0)
    assert not np.shares_memory(calibration.image_points, image)
    with pytest.raises(ValueError, match="recalibration"):
        calibration.project([original[0]], image_size=(320, 240), frame_id=FRAME_ID)
    with pytest.raises(ValueError, match="recalibration"):
        calibration.project([original[0]], image_size=SIZE, frame_id="other-frame")


@pytest.mark.parametrize("kwargs", [
    {"min_correspondences": 3},
    {"min_inliers": 3},
    {"min_inliers": 13},
    {"min_inlier_ratio": 0},
    {"ransac_threshold_px": float("nan")},
    {"max_median_reprojection_error_px": 0},
])
def test_invalid_quality_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        KeypointCalibration.fit(
            observations(with_outliers=False), FIELD_POINTS,
            image_size=SIZE, frame_id=FRAME_ID, **kwargs,
        )
