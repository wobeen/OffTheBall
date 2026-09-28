"""Manual planar calibration in metres, bound to a single still frame."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np


def suggest_pitch_corners(pixels):
    """Suggest four image corners from the largest green field-like region."""
    image = np.asarray(pixels)
    if image.ndim != 3 or image.shape[2] != 3:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 35, 20), (95, 255, 245))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .12:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if w < image.shape[1] * .35 or h < image.shape[0] * .25:
        return None
    return [[float(x), float(y)], [float(x+w-1), float(y)],
            [float(x+w-1), float(y+h-1)], [float(x), float(y+h-1)]]

class AutoPitchCalibration:
    def __init__(self, calibration, *, stale_after_frames=60):
        if not isinstance(stale_after_frames, int) or isinstance(stale_after_frames, bool) or stale_after_frames < 1:
            raise ValueError("stale_after_frames must be a positive integer")
        self.current = calibration
        self.frame_id = calibration.frame_id
        self.image_size = calibration.image_size
        self._corners = np.asarray(calibration.image_points[:4], dtype=float)
        self.stale_after_frames = stale_after_frames
        self._age_frames = 0
        self._last_update = "manual_seed"

    def update(self, pixels):
        self._age_frames += 1
        corners = suggest_pitch_corners(pixels)
        if corners is None:
            self._last_update = "field_region_not_found"
            return False
        corners = np.asarray(corners, dtype=float)
        if np.mean(np.linalg.norm(corners - self._corners, axis=1)) < 12:
            self._age_frames = 0
            self._last_update = "field_region_confirmed"
            return False
        self.current = ManualCalibration.fit(corners, [[0,0],[105,0],[105,68],[0,68]], image_size=self.image_size, frame_id=self.frame_id)
        self._corners = corners
        self._age_frames = 0
        self._last_update = "field_region_updated"
        return True

    def project(self, points, *, image_size, frame_id):
        return self.current.project(points, image_size=image_size, frame_id=frame_id)

    def diagnostics(self):
        return {
            "status": "stale" if self._age_frames > self.stale_after_frames else "active",
            "source": "green_region_bbox",
            "age_frames": self._age_frames,
            "stale_after_frames": self.stale_after_frames,
            "last_update": self._last_update,
            "inlier_count": None,
            "inlier_ratio": None,
            "reprojection_error_px": None,
        }


def _points(value, name, minimum=0):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < minimum or not np.isfinite(result).all():
        raise ValueError(f"{name} requires at least {minimum} finite coordinate pairs")
    return result


def _size(value):
    if len(value) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x <= 0 for x in value):
        raise ValueError("image_size must be positive integer (width, height)")
    return tuple(int(x) for x in value)


def _distribution(points):
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Duplicate calibration points")
    spread = np.ptp(points, axis=0)
    if np.any(spread <= 1e-8):
        raise ValueError("Calibration points must span an area")
    normalized = (points - points.mean(axis=0)) / max(spread)
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular[1] / singular[0] < 0.01 or cv2.contourArea(cv2.convexHull(normalized.astype(np.float32))) < 0.005:
        raise ValueError("Calibration points are too close to a line")


@dataclass(frozen=True)
class ManualCalibration:
    """Construct with fit/load. No moving-camera validity is implied."""
    image_points: np.ndarray
    field_points: np.ndarray
    image_size: tuple[int, int]
    frame_id: str
    matrix: np.ndarray
    hull: np.ndarray

    @classmethod
    def fit(cls, image_points, field_points, *, image_size, frame_id):
        source = _points(image_points, "image_points", 4)
        target = _points(field_points, "field_points", 4)
        size = _size(image_size)
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("A nonempty frame_id is required")
        if source.shape != target.shape:
            raise ValueError("Point arrays must have equal lengths")
        if np.any(source < 0) or np.any(source[:, 0] >= size[0]) or np.any(source[:, 1] >= size[1]):
            raise ValueError("Calibration points must lie inside the image")
        _distribution(source)
        _distribution(target)
        # Rank-eight normalized DLT constraints are necessary for a unique fit.
        # This also catches exactly four points with three collinear points.
        source_normal = (source-source.mean(axis=0))/np.max(np.ptp(source, axis=0))
        target_normal = (target-target.mean(axis=0))/np.max(np.ptp(target, axis=0))
        constraints = []
        for (x, y), (u, v) in zip(source_normal, target_normal):
            constraints.extend([[-x,-y,-1,0,0,0,u*x,u*y,u], [0,0,0,-x,-y,-1,v*x,v*y,v]])
        singular = np.linalg.svd(np.asarray(constraints), compute_uv=False)
        if singular[7] < singular[0] * 1e-8:
            raise ValueError("Point correspondences do not constrain a unique homography")
        matrix, _ = cv2.findHomography(source, target, method=0)
        if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
            raise ValueError("Points do not define an invertible homography")
        hull = cv2.convexHull(source.astype(np.float32))
        denominator = np.column_stack((hull[:, 0, :], np.ones(len(hull)))) @ matrix[2]
        if np.any(np.abs(denominator) < 1e-9) or not (np.all(denominator > 0) or np.all(denominator < 0)):
            raise ValueError("Projective horizon crosses the calibrated area")
        arrays = [source.copy(), target.copy(), matrix.copy(), hull.copy()]
        for array in arrays:
            array.setflags(write=False)
        return cls(arrays[0], arrays[1], size, frame_id, arrays[2], arrays[3])

    def project(self, points, *, image_size, frame_id):
        """Return Nx2 metres; outside-hull points become NaN. Wrong frame fails."""
        if _size(image_size) != self.image_size or frame_id != self.frame_id:
            raise ValueError("Source frame or image size changed; recalibration required")
        coordinates = _points(points, "points")
        output = np.full(coordinates.shape, np.nan)
        if not len(coordinates):
            return output
        inside = np.array([cv2.pointPolygonTest(self.hull, (float(x), float(y)), False) >= 0 for x, y in coordinates])
        homogeneous = np.column_stack((coordinates, np.ones(len(coordinates)))) @ self.matrix.T
        valid = inside & (np.abs(homogeneous[:, 2]) > 1e-9)
        output[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
        return output

    def evaluate(self, image_points, field_points, *, image_size, frame_id, segments=()):
        """Held-out point errors and optional distance errors for index pairs."""
        source = _points(image_points, "evaluation image_points", 1)
        expected = _points(field_points, "evaluation field_points", 1)
        if source.shape != expected.shape:
            raise ValueError("Evaluation arrays must have equal lengths")
        if np.any(np.linalg.norm(source[:, None, :] - self.image_points[None, :, :], axis=2) < 1e-6):
            raise ValueError("Evaluation requires held-out points")
        projected = self.project(source, image_size=image_size, frame_id=frame_id)
        valid = np.isfinite(projected).all(axis=1)
        errors = np.linalg.norm(projected - expected, axis=1)
        distances = []
        for pair in segments:
            if len(pair) != 2 or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= len(source) for i in pair):
                raise ValueError("Segments require two valid indices")
            a, b = pair
            reference = float(np.linalg.norm(expected[a] - expected[b]))
            if reference <= 0:
                raise ValueError("Reference segment must have positive length")
            measured = float(np.linalg.norm(projected[a] - projected[b])) if valid[a] and valid[b] else None
            distances.append({"indices": [int(a), int(b)], "reference_m": reference, "measured_m": measured,
                              "relative_error": abs(measured-reference)/reference if measured is not None else None})
        benchmark = [s["relative_error"] for s in distances if s["reference_m"] >= 10 and s["relative_error"] is not None]
        return {"count": len(source), "valid_count": int(valid.sum()),
                "position_errors_m": [float(e) if ok else None for e, ok in zip(errors, valid)],
                "median_position_error_m": float(np.median(errors[valid])) if valid.any() else None,
                "max_position_error_m": float(errors[valid].max()) if valid.any() else None,
                "segments": distances,
                "median_relative_distance_error_ge_10m": float(np.median(benchmark)) if benchmark else None}

    def diagnostics(self):
        return {
            "status": "active",
            "source": "manual_fixed_frame",
            "age_frames": 0,
            "stale_after_frames": None,
            "last_update": "manual",
            "inlier_count": len(self.image_points),
            "inlier_ratio": 1.0,
            "reprojection_error_px": None,
        }

    def save(self, path):
        payload = {"version": 1, "scope": "single_frame", "units": "metres", "image_size": list(self.image_size),
                   "frame_id": self.frame_id, "image_points": self.image_points.tolist(), "field_points": self.field_points.tolist()}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("scope") != "single_frame" or payload.get("units") != "metres":
            raise ValueError("Unsupported calibration version, scope or units")
        try:
            return cls.fit(payload["image_points"], payload["field_points"], image_size=payload["image_size"], frame_id=payload["frame_id"])
        except KeyError as exc:
            raise ValueError(f"Missing calibration field: {exc.args[0]}") from exc

