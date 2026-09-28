"""Per-input tracking, scene lifecycle, team labels, and screen trails."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Any

import cv2
import numpy as np

from .detection import Detection


class AnalysisSession:
    """Own all state that must not leak between input sources.

    A session is deliberately short lived: construct one for each video file,
    screen capture, or tablet connection.  ByteTrack IDs are meaningful only
    inside the current ``scene_id`` and are saved with that scene ID.
    """

    def __init__(self, detector: Any, team_classifier: Any | None = None,
                 *, calibration: Any | None = None, max_gap_s: float = 2.0,
                 trail_length: int = 15, pitch_polygon: Any | None = None) -> None:
        self.detector = detector
        self.team_classifier = team_classifier if team_classifier is not None else self._new_team_classifier()
        self.max_gap_s = float(max_gap_s)
        self.trail_length = max(1, int(trail_length))
        self.calibration = calibration
        self.pitch_polygon = pitch_polygon
        self._pitch_polygon_size: tuple[int, int] | None = None
        self.scene_id = 0
        self._last_timestamp: float | None = None
        self._last_size: tuple[int, int] | None = None
        self._last_small: np.ndarray | None = None
        self._had_valid_frame = False
        self._last_hist = None
        self._trails: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
        self._last_seen = {}
        self._reset_detector_tracking()

    @staticmethod
    def _new_team_classifier():
        # The team module is intentionally optional for compatibility with
        # raw detection callers and old installations during an upgrade.
        try:
            from .teams import TeamClassifier
        except ImportError:
            return None
        return TeamClassifier()

    @property
    def ready(self) -> bool:
        return bool(self.team_classifier is not None and
                    getattr(self.team_classifier, "ready", False))

    def calibration_diagnostics(self) -> dict[str, Any]:
        """Return a JSON-safe, stable view of the active calibration state."""
        if self.calibration is None:
            return {
                "status": "not_calibrated", "source": None, "age_frames": None,
                "stale_after_frames": None, "last_update": None,
                "inlier_count": None, "inlier_ratio": None,
                "reprojection_error_px": None, "projection_jump_m": None,
            }
        getter = getattr(self.calibration, "diagnostics", None)
        raw = getter() if callable(getter) else {}
        defaults = {
            "status": "active", "source": "fixed_calibration", "age_frames": 0,
            "stale_after_frames": None, "last_update": None,
            "inlier_count": None, "inlier_ratio": None,
            "reprojection_error_px": None, "projection_jump_m": None,
        }
        if isinstance(raw, dict):
            defaults.update({key: raw.get(key) for key in defaults if key in raw})
        if defaults["status"] not in {"active", "stale", "invalid", "not_calibrated"}:
            defaults["status"] = "invalid"
        for key in ("inlier_ratio", "reprojection_error_px", "projection_jump_m"):
            value = defaults[key]
            if value is not None and not np.isfinite(value):
                defaults[key] = None
        return defaults

    def _reset_detector_tracking(self) -> None:
        reset = getattr(self.detector, "reset_tracking", None)
        if callable(reset):
            reset()

    def _reset_team(self) -> None:
        reset = getattr(self.team_classifier, "reset_tracks", None) or getattr(self.team_classifier, "reset", None)
        if callable(reset):
            reset()

    def reset_scene(self) -> None:
        """Start a new scene and clear all identity/trail/team state."""
        self.scene_id += 1
        self._last_timestamp = None
        self._last_small = None
        self._had_valid_frame = False
        self._trails.clear()
        self._last_seen.clear()
        self._last_hist = None
        self._reset_detector_tracking()
        self._reset_team()

    def reset(self) -> None:
        """Reset a source session, including underlying ByteTrack state."""
        self.reset_scene()
        self._last_size = None
        reset = getattr(self.team_classifier, "reset", None)
        if callable(reset):
            reset()

    @staticmethod
    def _small_frame(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    @staticmethod
    def _histogram(frame):
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist)

    def _is_scene_cut(self, frame: np.ndarray) -> bool:
        if self._last_small is None:
            return False
        diff = np.abs(self._small_frame(frame) - self._last_small)
        strong_change = float(diff.mean()) >= 62.0 and float((diff >= 42.0).mean()) >= .78
        histogram_change = self._last_hist is not None and cv2.compareHist(
            self._histogram(frame), self._last_hist, cv2.HISTCMP_BHATTACHARYYA) > .33
        return bool(strong_change or (float(diff.mean()) > 18.0 and histogram_change))

    def _should_reset(self, frame) -> bool:
        timestamp = float(frame.timestamp_s)
        size = (int(frame.width), int(frame.height))
        if self._last_size is not None and size != self._last_size:
            return True
        if self._last_timestamp is not None:
            delta = timestamp - self._last_timestamp
            if delta < 0 or delta > self.max_gap_s:
                return True
        return self._is_scene_cut(frame.pixels)

    def _call_detector(self, pixels):
        track = getattr(self.detector, "track", None)
        if not callable(track):
            return list(self.detector.detect(pixels))
        polygon = self.pitch_polygon
        size = (int(pixels.shape[1]), int(pixels.shape[0]))
        if polygon is not None:
            polygon_size = getattr(polygon, "width", None), getattr(polygon, "height", None)
            polygon_points = getattr(polygon, "points", polygon)
            if polygon_size[0] is not None:
                if polygon_size != size:
                    raise ValueError(
                        "pitch polygon dimensions do not match the input frame; recalibrate the playing area"
                    )
                else:
                    polygon = polygon_points
            else:
                if self._pitch_polygon_size is None:
                    self._pitch_polygon_size = size
                elif self._pitch_polygon_size != size:
                    raise ValueError(
                        "pitch polygon dimensions do not match the input frame; recalibrate the playing area"
                    )
        try:
            return list(track(pixels, pitch_polygon=polygon))
        except TypeError as exc:
            # Keep lightweight/legacy detector doubles working while making
            # the production detector receive the pre-association filter.
            if "pitch_polygon" not in str(exc):
                raise
            return list(track(pixels))

    def _team_labels(self, frame, detections):
        if self.team_classifier is None:
            return [None] * len(detections)
        update = getattr(self.team_classifier, "update", None)
        if not callable(update):
            return [None] * len(detections)
        result = update(frame.pixels, detections)
        if isinstance(result, dict):
            return [result.get(item.track_id) for item in detections]
        if isinstance(result, (list, tuple)):
            labels = []
            for index, item in enumerate(detections):
                value = result[index] if index < len(result) else None
                labels.append(getattr(value, "team", value))
            return labels
        return [getattr(item, "team", None) for item in detections]

    @staticmethod
    def _normalise_team(value: Any) -> str | None:
        if value is None:
            return None
        value = str(value).strip().upper()
        return value if value in {"A", "B"} else None

    def process(self, frame, *, black: bool = False) -> list[Detection]:
        """Process one input frame and return IDs, teams, and pixel trails."""
        if black:
            if self._had_valid_frame:
                self.reset_scene()
            self._last_timestamp = float(frame.timestamp_s)
            self._last_size = (int(frame.width), int(frame.height))
            return []

        if self._should_reset(frame):
            self.reset_scene()
        self._last_size = (int(frame.width), int(frame.height))
        self._last_timestamp = float(frame.timestamp_s)
        self._last_small = self._small_frame(frame.pixels)
        self._last_hist = self._histogram(frame.pixels)
        self._had_valid_frame = True
        updater = getattr(self.calibration, "update", None)
        if callable(updater):
            try:
                updater(frame.pixels)
            except (ValueError, cv2.error):
                pass

        now = float(frame.timestamp_s)
        for key in list(self._last_seen):
            if now - self._last_seen[key] > self.max_gap_s:
                self._last_seen.pop(key, None)
                self._trails.pop(key, None)
        raw = self._call_detector(frame.pixels)
        labels = self._team_labels(frame, raw)
        output: list[Detection] = []
        for index, item in enumerate(raw):
            team = self._normalise_team(labels[index] if index < len(labels) else item.team)
            track_id = item.track_id
            trail: tuple[tuple[float, float], ...] = ()
            if track_id is not None:
                self._last_seen[int(track_id)] = now
                history = self._trails[int(track_id)]
                x1, y1, x2, y2 = item.xyxy
                history.append(((x1 + x2) / 2.0, y2))
                trail = tuple(history)
            field_xy = None
            if self.calibration is not None:
                x1, y1, x2, y2 = item.xyxy
                try:
                    projected = self.calibration.project(
                        [[(x1 + x2) / 2.0, y2]],
                        image_size=(frame.width, frame.height),
                        frame_id=self.calibration.frame_id,
                    )[0]
                    if np.isfinite(projected).all():
                        field_xy = (float(projected[0]), float(projected[1]))
                except (ValueError, AttributeError):
                    # A stale calibration must not stop detection; its status is
                    # reported by the caller and field_xy remains unavailable.
                    field_xy = None
            output.append(replace(item, team=team, trail=trail,
                                  scene_id=self.scene_id, field_xy=field_xy))
        while len(self._last_seen) > 256:
            oldest = min(self._last_seen, key=self._last_seen.get)
            self._last_seen.pop(oldest)
            self._trails.pop(oldest, None)
        return output


# Descriptive alias for callers that prefer the implementation's role.
TrackingSession = AnalysisSession
