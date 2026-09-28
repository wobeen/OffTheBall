"""Temporal acceptance and expiry rules for per-frame pitch calibrations."""
from __future__ import annotations

from typing import Any

import numpy as np

from .keypoint_calibration import KeypointCalibration, _image_size, _points


class TemporalCalibrationState:
    """Keep only plausible calibration updates and expire unsupported state.

    A future model adapter should create a :class:`KeypointCalibration` for a
    frame and pass it to :meth:`observe`.  Missing or rejected observations
    age the last accepted transform.  Once it becomes stale, :meth:`project`
    returns NaNs instead of presenting old coordinates as current facts.
    """

    def __init__(self, *, image_size: Any, frame_id: str,
                 max_reuse_frames: int = 30, max_projection_jump_m: float = 8.0,
                 min_shared_points: int = 4) -> None:
        self.image_size = _image_size(image_size)
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a nonempty string")
        if (not isinstance(max_reuse_frames, int) or isinstance(max_reuse_frames, bool)
                or max_reuse_frames < 0):
            raise ValueError("max_reuse_frames must be a nonnegative integer")
        if not np.isfinite(max_projection_jump_m) or float(max_projection_jump_m) <= 0:
            raise ValueError("max_projection_jump_m must be positive and finite")
        if (not isinstance(min_shared_points, int) or isinstance(min_shared_points, bool)
                or min_shared_points < 1):
            raise ValueError("min_shared_points must be a positive integer")
        self.frame_id = frame_id
        self.max_reuse_frames = max_reuse_frames
        self.max_projection_jump_m = float(max_projection_jump_m)
        self.min_shared_points = min_shared_points
        self.current: KeypointCalibration | None = None
        self.age_frames: int | None = None
        self.last_update = "waiting_for_keypoints"
        self.last_projection_jump_m: float | None = None

    @property
    def status(self) -> str:
        if self.current is None:
            return "not_calibrated"
        if self.age_frames is None or self.age_frames > self.max_reuse_frames:
            return "stale"
        return "active"

    def reset(self) -> None:
        self.current = None
        self.age_frames = None
        self.last_update = "reset"
        self.last_projection_jump_m = None

    def _miss(self, reason: str) -> bool:
        if self.current is not None:
            self.age_frames = (self.age_frames or 0) + 1
        self.last_update = reason
        return False

    def _projection_jump(self, candidate: KeypointCalibration) -> float | None:
        if self.current is None:
            return None
        # Candidate inlier pixels sample only the part of the image supported
        # by the new observation.  Requiring validity in both hulls avoids
        # comparing extrapolated coordinates.
        anchors = candidate.image_points
        previous = self.current.project(
            anchors, image_size=self.image_size, frame_id=self.current.frame_id)
        proposed = candidate.project(
            anchors, image_size=self.image_size, frame_id=candidate.frame_id)
        valid = np.isfinite(previous).all(axis=1) & np.isfinite(proposed).all(axis=1)
        if int(valid.sum()) < self.min_shared_points:
            return None
        differences = np.linalg.norm(previous[valid] - proposed[valid], axis=1)
        return float(np.median(differences))

    def observe(self, candidate: KeypointCalibration | None) -> bool:
        """Accept one frame's validated candidate, returning whether it won."""
        if candidate is None:
            self.last_projection_jump_m = None
            return self._miss("missing_candidate")
        if not isinstance(candidate, KeypointCalibration):
            raise TypeError("candidate must be a KeypointCalibration or None")
        if candidate.image_size != self.image_size:
            raise ValueError("candidate image size changed; reset the calibration state")

        # A stale transform must not permanently block recovery after a cut.
        # The candidate has already passed its own geometric quality gates.
        if self.status == "active":
            jump = self._projection_jump(candidate)
            self.last_projection_jump_m = jump
            if jump is None:
                return self._miss("rejected_insufficient_overlap")
            if jump > self.max_projection_jump_m:
                return self._miss("rejected_projection_jump")
        else:
            self.last_projection_jump_m = None

        self.current = candidate
        self.age_frames = 0
        self.last_update = "accepted_keypoints"
        return True

    def project(self, points: Any, *, image_size: Any, frame_id: str) -> np.ndarray:
        if _image_size(image_size) != self.image_size or frame_id != self.frame_id:
            raise ValueError("Source frame or image size changed; recalibration required")
        coordinates = _points(points, "points", 0)
        if self.status != "active" or self.current is None:
            return np.full(coordinates.shape, np.nan, dtype=np.float64)
        return self.current.project(
            coordinates, image_size=self.image_size, frame_id=self.current.frame_id)

    def diagnostics(self) -> dict[str, Any]:
        report = self.current.diagnostics() if self.current is not None else {}
        return {
            "status": self.status,
            "source": "pitch_keypoints_temporal",
            "age_frames": self.age_frames,
            "stale_after_frames": self.max_reuse_frames,
            "last_update": self.last_update,
            "inlier_count": report.get("inlier_count"),
            "inlier_ratio": report.get("inlier_ratio"),
            "reprojection_error_px": report.get("reprojection_error_px"),
            "projection_jump_m": self.last_projection_jump_m,
        }
