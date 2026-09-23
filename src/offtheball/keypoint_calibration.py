"""Validated pitch homography fitting from labelled keypoint correspondences.

This module deliberately does not load a keypoint model.  It is the small,
deterministic boundary between any future model adapter and the rest of the
application: labelled image/field pairs go in, and either a validated
calibration or a clear rejection comes out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


PITCH_SIZE_M = (105.0, 68.0)


def _points(value: Any, name: str, minimum: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < minimum:
        raise ValueError(f"{name} requires at least {minimum} coordinate pairs")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _image_size(value: Any) -> tuple[int, int]:
    if (not hasattr(value, "__len__") or len(value) != 2 or
            any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) or item <= 0
                for item in value)):
        raise ValueError("image_size must be positive integer (width, height)")
    return int(value[0]), int(value[1])


def _validate_distribution(points: np.ndarray, name: str, *, area_scale: float,
                           minimum_area_ratio: float) -> None:
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError(f"{name} contains duplicate points")
    hull = cv2.convexHull(points.astype(np.float32))
    area_ratio = abs(float(cv2.contourArea(hull))) / area_scale
    if area_ratio < minimum_area_ratio:
        raise ValueError(f"{name} points cover too little area")


def _transform(points: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    valid = np.abs(homogeneous[:, 2]) > 1e-9
    output = np.full(points.shape, np.nan, dtype=np.float64)
    output[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
    return output, valid


@dataclass(frozen=True)
class KeypointCalibration:
    """An image-to-pitch transform accepted by explicit quality gates."""

    image_points: np.ndarray
    field_points: np.ndarray
    image_size: tuple[int, int]
    frame_id: str
    matrix: np.ndarray
    hull: np.ndarray
    correspondence_count: int
    inlier_count: int
    inlier_ratio: float
    median_reprojection_error_px: float
    max_reprojection_error_px: float

    @classmethod
    def fit(cls, image_points: Any, field_points: Any, *, image_size: Any, frame_id: str,
            min_correspondences: int = 6, min_inliers: int = 6,
            min_inlier_ratio: float = 0.65, ransac_threshold_px: float = 5.0,
            max_median_reprojection_error_px: float = 3.0,
            min_image_hull_ratio: float = 0.01,
            min_field_hull_ratio: float = 0.01) -> "KeypointCalibration":
        if not isinstance(min_correspondences, int) or isinstance(min_correspondences, bool) or min_correspondences < 4:
            raise ValueError("min_correspondences must be an integer of at least 4")
        if not isinstance(min_inliers, int) or isinstance(min_inliers, bool) or min_inliers < 4:
            raise ValueError("min_inliers must be an integer of at least 4")
        if not 0 < float(min_inlier_ratio) <= 1:
            raise ValueError("min_inlier_ratio must be in (0, 1]")
        for value, name in (
            (ransac_threshold_px, "ransac_threshold_px"),
            (max_median_reprojection_error_px, "max_median_reprojection_error_px"),
            (min_image_hull_ratio, "min_image_hull_ratio"),
            (min_field_hull_ratio, "min_field_hull_ratio"),
        ):
            if not np.isfinite(value) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")

        size = _image_size(image_size)
        source = _points(image_points, "image_points", min_correspondences)
        target = _points(field_points, "field_points", min_correspondences)
        if source.shape != target.shape:
            raise ValueError("image_points and field_points must have equal shapes")
        if min_inliers > len(source):
            raise ValueError("min_inliers cannot exceed the number of correspondences")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        if (np.any(source < 0) or np.any(source[:, 0] >= size[0]) or
                np.any(source[:, 1] >= size[1])):
            raise ValueError("image_points must lie inside the image")
        pitch_length, pitch_width = PITCH_SIZE_M
        if (np.any(target < 0) or np.any(target[:, 0] > pitch_length) or
                np.any(target[:, 1] > pitch_width)):
            raise ValueError("field_points must lie inside the 105m by 68m pitch")

        image_area = float(size[0] * size[1])
        field_area = pitch_length * pitch_width
        _validate_distribution(source, "image_points", area_scale=image_area,
                               minimum_area_ratio=float(min_image_hull_ratio))
        _validate_distribution(target, "field_points", area_scale=field_area,
                               minimum_area_ratio=float(min_field_hull_ratio))

        # Fit field -> image so OpenCV's RANSAC threshold is expressed in
        # pixels.  The inverse is what the application uses for player feet.
        field_to_image, mask = cv2.findHomography(
            target, source, cv2.RANSAC, float(ransac_threshold_px),
            maxIters=3000, confidence=0.995,
        )
        if field_to_image is None or mask is None:
            raise ValueError("RANSAC could not estimate a pitch homography")
        inliers = mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        inlier_ratio = inlier_count / len(source)
        if inlier_count < min_inliers:
            raise ValueError(f"RANSAC accepted only {inlier_count} inliers; at least {min_inliers} required")
        if inlier_ratio < float(min_inlier_ratio):
            raise ValueError(f"RANSAC inlier ratio {inlier_ratio:.3f} is below {float(min_inlier_ratio):.3f}")

        inlier_image = source[inliers].copy()
        inlier_field = target[inliers].copy()
        _validate_distribution(inlier_image, "inlier image_points", area_scale=image_area,
                               minimum_area_ratio=float(min_image_hull_ratio))
        _validate_distribution(inlier_field, "inlier field_points", area_scale=field_area,
                               minimum_area_ratio=float(min_field_hull_ratio))

        # Refit without RANSAC using only accepted correspondences.  This
        # removes dependence on rejected points and gives reproducible errors.
        field_to_image, _ = cv2.findHomography(inlier_field, inlier_image, method=0)
        if (field_to_image is None or not np.isfinite(field_to_image).all() or
                np.linalg.matrix_rank(field_to_image) != 3):
            raise ValueError("Inlier points do not define an invertible homography")
        projected, valid = _transform(inlier_field, field_to_image)
        if not valid.all():
            raise ValueError("Homography sends an inlier through the projective horizon")
        errors = np.linalg.norm(projected - inlier_image, axis=1)
        median_error = float(np.median(errors))
        max_error = float(errors.max())
        if median_error > float(max_median_reprojection_error_px):
            raise ValueError(
                f"Median reprojection error {median_error:.3f}px exceeds "
                f"{float(max_median_reprojection_error_px):.3f}px"
            )

        try:
            image_to_field = np.linalg.inv(field_to_image)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Homography is not invertible") from exc
        hull = cv2.convexHull(inlier_image.astype(np.float32))
        denominator = np.column_stack((hull[:, 0, :], np.ones(len(hull)))) @ image_to_field[2]
        if (np.any(np.abs(denominator) < 1e-9) or
                not (np.all(denominator > 0) or np.all(denominator < 0))):
            raise ValueError("Projective horizon crosses the calibrated image area")

        arrays = (inlier_image, inlier_field, image_to_field.copy(), hull.copy())
        for array in arrays:
            array.setflags(write=False)
        return cls(
            image_points=arrays[0], field_points=arrays[1], image_size=size,
            frame_id=frame_id, matrix=arrays[2], hull=arrays[3],
            correspondence_count=len(source), inlier_count=inlier_count,
            inlier_ratio=float(inlier_ratio), median_reprojection_error_px=median_error,
            max_reprojection_error_px=max_error,
        )

    def project(self, points: Any, *, image_size: Any, frame_id: str) -> np.ndarray:
        if _image_size(image_size) != self.image_size or frame_id != self.frame_id:
            raise ValueError("Source frame or image size changed; recalibration required")
        coordinates = _points(points, "points", 0)
        output = np.full(coordinates.shape, np.nan, dtype=np.float64)
        if not len(coordinates):
            return output
        inside = np.asarray([
            cv2.pointPolygonTest(self.hull, (float(x), float(y)), False) >= 0
            for x, y in coordinates
        ])
        transformed, valid = _transform(coordinates, self.matrix)
        valid &= inside
        output[valid] = transformed[valid]
        return output

    def diagnostics(self) -> dict[str, Any]:
        return {
            "status": "active",
            "source": "pitch_keypoints_ransac",
            "age_frames": 0,
            "stale_after_frames": None,
            "last_update": "keypoints",
            "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_error_px": self.median_reprojection_error_px,
        }
