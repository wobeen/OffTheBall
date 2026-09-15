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
                 *, max_gap_s: float = 2.0, trail_length: int = 15) -> None:
        self.detector = detector
        self.team_classifier = team_classifier if team_classifier is not None else self._new_team_classifier()
        self.max_gap_s = float(max_gap_s)
        self.trail_length = max(1, int(trail_length))
        self.scene_id = 0
        self._last_timestamp: float | None = None
        self._last_size: tuple[int, int] | None = None
        self._last_small: np.ndarray | None = None
        self._had_valid_frame = False
        self._trails: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )
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

    def _reset_detector_tracking(self) -> None:
        reset = getattr(self.detector, "reset_tracking", None)
        if callable(reset):
            reset()

    def _reset_team(self) -> None:
        reset = getattr(self.team_classifier, "reset", None)
        if callable(reset):
            reset()

    def reset_scene(self) -> None:
        """Start a new scene and clear all identity/trail/team state."""
        self.scene_id += 1
        self._last_timestamp = None
        self._last_small = None
        self._had_valid_frame = False
        self._trails.clear()
        self._reset_detector_tracking()
        self._reset_team()

    def reset(self) -> None:
        """Reset a source session, including underlying ByteTrack state."""
        self.reset_scene()
        self._last_size = None

    @staticmethod
    def _small_frame(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _is_scene_cut(self, frame: np.ndarray) -> bool:
        if self._last_small is None:
            return False
        current = self._small_frame(frame)
        diff = np.abs(current - self._last_small)
        # Require a large change across most of the image.  This avoids
        # resetting on normal pans and players moving through the camera.
        return float(diff.mean()) >= 62.0 and float((diff >= 42.0).mean()) >= 0.78

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
        return list(track(pixels) if callable(track) else self.detector.detect(pixels))

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
        self._had_valid_frame = True

        raw = self._call_detector(frame.pixels)
        labels = self._team_labels(frame, raw)
        output: list[Detection] = []
        for index, item in enumerate(raw):
            team = self._normalise_team(labels[index] if index < len(labels) else item.team)
            track_id = item.track_id
            trail: tuple[tuple[float, float], ...] = ()
            if track_id is not None:
                history = self._trails[int(track_id)]
                x1, y1, x2, y2 = item.xyxy
                history.append(((x1 + x2) / 2.0, y2))
                trail = tuple(history)
            output.append(replace(item, team=team, trail=trail, scene_id=self.scene_id))
        return output


# Descriptive alias for callers that prefer the implementation's role.
TrackingSession = AnalysisSession
