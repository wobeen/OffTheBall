"""Conservative two-kit jersey classification.

The detector and tracker deliberately remain independent of this module.  A
classifier update accepts any objects with an ``xyxy`` attribute and, when
available, a ``track_id`` attribute.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class _Observation:
    feature: np.ndarray
    track_id: Any


class TeamClassifier:
    """Classify detections as stable scene teams ``"A"``/``"B"`` or unknown.

    The first usable observations establish two kit centers.  Centers are
    fixed for the lifetime of a scene, while each tracked player gets a small
    majority vote to suppress frame-to-frame color noise.  A detection is
    returned as ``None`` until the centers are reliable or when it is an
    outlier (for example a referee or goalkeeper in a third kit).
    """

    def __init__(
        self,
        *,
        min_samples_per_team: int = 2,
        vote_window: int = 7,
        min_crop_width: int = 12,
        min_crop_height: int = 20,
        outlier_distance: float = 26.0,
        separation: float = 32.0,
    ) -> None:
        if not isinstance(min_samples_per_team, int) or isinstance(min_samples_per_team, bool) or min_samples_per_team < 1:
            raise ValueError("min_samples_per_team must be a positive integer")
        if not isinstance(vote_window, int) or isinstance(vote_window, bool) or vote_window < 1:
            raise ValueError("vote_window must be a positive integer")
        if min_crop_width < 4 or min_crop_height < 4:
            raise ValueError("crop dimensions are too small")
        if outlier_distance <= 0 or separation <= 0:
            raise ValueError("distance thresholds must be positive")
        self.min_samples_per_team = min_samples_per_team
        self.vote_window = vote_window
        self.min_crop_width = int(min_crop_width)
        self.min_crop_height = int(min_crop_height)
        self.outlier_distance = float(outlier_distance)
        self.separation = float(separation)
        self._samples: list[np.ndarray] = []
        self._centers: tuple[np.ndarray, np.ndarray] | None = None
        self._votes: dict[Any, deque[str]] = defaultdict(lambda: deque(maxlen=self.vote_window))

    @property
    def ready(self) -> bool:
        """Whether two separated, supported kit centers have been learned."""

        return self._centers is not None

    @property
    def centers(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """Read-only Lab centers, useful for diagnostics and UI overlays."""

        if self._centers is None:
            return None
        return tuple(tuple(float(x) for x in center) for center in self._centers)  # type: ignore[return-value]

    def reset_tracks(self) -> None:
        """Clear temporal player votes while retaining this scene's kit centers."""

        self._votes.clear()

    def reset(self) -> None:
        """Forget kit centers and player history for a new scene or source."""

        self._samples.clear()
        self._centers = None
        self.reset_tracks()

    def update(self, frame_bgr: np.ndarray, detections: Any) -> list[str | None]:
        """Return one team label per detection for a BGR frame.

        ``detections`` may be any iterable.  Invalid, tiny, or background-only
        crops produce ``None``.  Labels are ``"A"`` and ``"B"`` once enough
        evidence exists; centers are assigned deterministically by the first
        two-color scene split and never relabelled during that scene.
        """

        self._validate_frame(frame_bgr)
        items = list(detections)
        observations: list[_Observation | None] = [
            self._observe(frame_bgr, item) for item in items
        ]
        if self._centers is None:
            for observation in observations:
                if observation is not None:
                    self._samples.append(observation.feature)
            # A stream with only one visible kit must not grow without bound
            # while waiting for the opposing team to enter the scene.
            if len(self._samples) > 256:
                self._samples = self._samples[-256:]
            self._try_fit_centers()

        if self._centers is None:
            return [None] * len(items)

        result: list[str | None] = []
        for observation in observations:
            if observation is None:
                result.append(None)
                continue
            team, distance = self._nearest(observation.feature)
            if distance > self.outlier_distance:
                result.append(None)
                continue
            # Untracked detections still receive a frame-local classification;
            # only real track IDs are voted over time.
            if observation.track_id is None:
                result.append(team)
                continue
            if observation.track_id not in self._votes and len(self._votes) >= 256:
                self._votes.pop(next(iter(self._votes)))
            votes = self._votes[observation.track_id]
            votes.append(team)
            counts = {label: votes.count(label) for label in ("A", "B")}
            if counts["A"] == counts["B"]:
                # Keep the previous vote on a one-frame tie.  This prevents a
                # single occlusion or color cast from flipping a track.
                result.append(votes[-2] if len(votes) > 1 else team)
            else:
                result.append("A" if counts["A"] > counts["B"] else "B")
        return result

    @staticmethod
    def _validate_frame(frame_bgr: np.ndarray) -> None:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 array")
        if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
            raise ValueError("frame_bgr must not be empty")

    def _observe(self, frame: np.ndarray, detection: Any) -> _Observation | None:
        try:
            box = np.asarray(detection.xyxy, dtype=float).reshape(4)
        except (AttributeError, TypeError, ValueError):
            return None
        if not np.isfinite(box).all():
            return None
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        height, width = frame.shape[:2]
        left, top = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        right, bottom = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if right - left < self.min_crop_width or bottom - top < self.min_crop_height:
            return None

        # Keep a central upper-torso patch.  The margin avoids turf outside a
        # person box while retaining sleeves and both white/dark kits.
        crop = frame[top:bottom, left:right]
        ch, cw = crop.shape[:2]
        xa, xb = int(cw * .25), max(int(cw * .75), int(cw * .25) + 1)
        ya, yb = int(ch * .18), max(int(ch * .48), int(ch * .18) + 1)
        torso = crop[ya:yb, xa:xb]
        if torso.size == 0:
            return None
        pixels = torso.reshape(-1, 3)
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        ycrcb = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        # Turf is commonly the dominant green background in loose boxes.  The
        # center-patch strategy handles green kits too; only reject a sample
        # when most of the patch is turf-like and no chromatic alternative is
        # present.
        green = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 45) & (hsv[:, 2] >= 35)
        # Skin-colored hands/arms can occupy the lower edge of a loose box.
        # This conservative chroma test leaves saturated red/orange kits in
        # place while removing the common peach skin range.
        skin = ((ycrcb[:, 1] >= 135) & (ycrcb[:, 1] <= 180) &
                (ycrcb[:, 2] >= 80) & (ycrcb[:, 2] <= 135) &
                (ycrcb[:, 1] > ycrcb[:, 2] + 8) &
                (hsv[:, 0] <= 25) & (hsv[:, 1] < 155))
        usable = pixels[~(green | skin)]
        if len(usable) < max(6, len(pixels) // 12):
            # A strongly green patch is likely field, but retain a green kit
            # when it is spatially uniform and therefore not grass texture.
            if len(pixels) < 16 or float(np.mean(np.std(pixels.astype(float), axis=0))) > 18:
                return None
            usable = pixels
        # Prefer shirt chroma over white numbers/shorts when it is well supported.
        usable_hsv = cv2.cvtColor(usable.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        colored = usable_hsv[:, 1] >= 55
        if np.count_nonzero(colored) >= max(6, int(len(usable) * .3)):
            usable = usable[colored]
        lab = cv2.cvtColor(usable.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        # Coordinate-wise medians are stable under small skin/number/logo
        # contamination, including very dark and nearly white shirts.
        feature = np.median(lab, axis=0)
        feature[0] *= .25  # Lighting changes must not turn a blue kit into a red kit.
        if not np.isfinite(feature).all():
            return None
        try:
            track_id = getattr(detection, "track_id", None)
            hash(track_id)
        except (TypeError, ValueError):
            track_id = None
        return _Observation(feature, track_id)

    def _try_fit_centers(self) -> None:
        if len(self._samples) < self.min_samples_per_team * 2:
            return
        data = np.asarray(self._samples, dtype=np.float32)
        # Evaluate every separated pair as deterministic seeds.  Keeping the
        # pair that explains the most samples within the outlier radius makes a
        # lone referee/keeper kit unable to become one of the two team centers.
        best: tuple[int, float, np.ndarray, np.ndarray] | None = None
        _, candidates = np.unique(np.round(data / 12).astype(int), axis=0, return_index=True)
        candidates = np.sort(candidates)
        if len(candidates) > 16:
            candidates = candidates[np.linspace(0, len(candidates)-1, 16).astype(int)]
        seeds = data[candidates]
        for i in range(len(seeds) - 1):
            for j in range(i + 1, len(seeds)):
                if float(np.linalg.norm(seeds[i] - seeds[j])) < self.separation:
                    continue
                centers = np.stack((seeds[i], seeds[j]))
                for _ in range(12):
                    distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
                    labels = np.argmin(distances, axis=1)
                    updated = np.array([
                        data[labels == group].mean(axis=0) if np.any(labels == group) else centers[group]
                        for group in (0, 1)
                    ], dtype=np.float32)
                    if np.allclose(updated, centers, atol=.01):
                        break
                    centers = updated
                distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
                labels = np.argmin(distances, axis=1)
                nearest = distances[np.arange(len(data)), labels]
                inliers = nearest <= self.outlier_distance
                counts = np.bincount(labels[inliers], minlength=2)
                if np.min(counts) < self.min_samples_per_team:
                    continue
                separation = float(np.linalg.norm(centers[0] - centers[1]))
                if separation < self.separation:
                    continue
                score = (int(np.count_nonzero(inliers)), float(np.sum(nearest[inliers])))
                if best is None or score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
                    best = (score[0], score[1], centers.copy(), inliers.copy())
        if best is None:
            return
        _, _, centers, inliers = best
        # A is anchored to the first usable observation in this scene.  The
        # assignment is then fixed; later frames cannot swap the two labels.
        final_labels = np.argmin(np.linalg.norm(data[inliers, None, :] - centers[None, :, :], axis=2), axis=1)
        first_group = int(final_labels[0])
        order = (first_group, 1 - first_group)
        self._centers = (centers[order[0]], centers[order[1]])
        self._samples.clear()

    def _nearest(self, feature: np.ndarray) -> tuple[str, float]:
        assert self._centers is not None
        distances = [float(np.linalg.norm(feature - center)) for center in self._centers]
        index = int(np.argmin(distances))
        if abs(distances[0] - distances[1]) < 8:
            return (None, float("inf"))
        return ("A", distances[0]) if index == 0 else ("B", distances[1])
