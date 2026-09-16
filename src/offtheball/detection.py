"""Person detection and ByteTrack integration.

The detector still exposes the original ``detect`` method for callers that
only need raw person boxes.  ``track`` uses Ultralytics' official ByteTrack
implementation and is used by the analysis session for video inputs.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from . import ROOT


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None
    team: str | None = None
    trail: tuple[tuple[float, float], ...] = ()
    scene_id: int | None = None
    field_xy: tuple[float, float] | None = None

    def to_dict(self):
        x1, y1, x2, y2 = self.xyxy
        return {"xyxy": list(self.xyxy), "confidence": self.confidence,
                "foot_pixel": [(x1 + x2) / 2, y2],
                "team": self.team, "track_id": self.track_id,
                "scene_id": self.scene_id,
                "trail": [list(point) for point in self.trail],
                "field_xy": list(self.field_xy) if self.field_xy is not None else None}


class PersonDetector:
    def __init__(self, model=None, confidence=0.25, image_size=960, cpu_threads=4):
        path = Path(model) if model else ROOT / "models/yolo11n.pt"
        if not path.is_file():
            raise FileNotFoundError("탐지 모델이 없습니다. 먼저 설치.cmd를 실행하세요.")
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if not isinstance(cpu_threads, int) or isinstance(cpu_threads, bool) or cpu_threads < 1:
            raise ValueError('cpu_threads must be a positive integer')
        import torch
        torch.set_num_threads(cpu_threads)
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, pixels):
        """Return raw person detections without tracker state."""
        result = self.model.predict(pixels, classes=[0], conf=self.confidence,
                                    imgsz=self.image_size, device="cpu",
                                    verbose=False, save=False)[0]
        return self._detections_from_result(result)

    def track(self, pixels):
        """Return detections with IDs supplied by Ultralytics ByteTrack.

        ``persist=True`` is the documented Ultralytics API for keeping the
        tracker between calls.  The owning :class:`AnalysisSession` resets
        the predictor tracker whenever a source or scene ends.
        """
        result = self.model.track(pixels, persist=True, tracker='bytetrack.yaml', classes=[0],
                                  conf=self.confidence, imgsz=self.image_size,
                                  device="cpu", verbose=False, save=False)[0]
        detections = self._detections_from_result(result)
        if result.boxes is None or result.boxes.id is None:
            return detections
        ids = result.boxes.id.cpu().numpy().astype(int).tolist()
        return [Detection(item.xyxy, item.confidence, track_id=track_id)
                for item, track_id in zip(detections, ids)]

    def reset_tracking(self):
        """Clear the underlying Ultralytics tracker state for a new scene."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None)
        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()
            if hasattr(predictor, "vid_path"):
                predictor.vid_path = [None] * len(trackers)

    @staticmethod
    def _detections_from_result(result):
        if result.boxes is None:
            return []
        return [Detection(tuple(float(v) for v in box), float(score))
                for box, score in zip(result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy())]


def annotate(pixels, detections):
    out = pixels.copy()
    h, w = out.shape[:2]
    # Draw only a few nearest same-team links so a full-pitch view remains readable.
    measured = []
    for i, left in enumerate(detections):
        if left.team not in {"A", "B"} or left.field_xy is None:
            continue
        for j in range(i + 1, len(detections)):
            right = detections[j]
            if right.team != left.team or right.field_xy is None:
                continue
            distance = float(np.linalg.norm(np.asarray(left.field_xy) - np.asarray(right.field_xy)))
            measured.append((distance, left, right))
    used = set()
    for distance, left, right in sorted(measured, key=lambda item: item[0]):
        key = (id(left), id(right))
        if key in used or sum(id(item) in used for item in (left, right)):
            continue
        if len(used) >= 8:
            break
        def foot(item):
            x1, _, x2, y2 = item.xyxy
            return (max(0, min(w - 1, round((x1 + x2) / 2))), max(0, min(h - 1, round(y2))))
        pa, pb = foot(left), foot(right)
        cv2.line(out, pa, pb, (230, 220, 120), 1, cv2.LINE_AA)
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
        cv2.putText(out, f"{distance:.1f}m", (mid[0] + 3, mid[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 240, 150), 1, cv2.LINE_AA)
        used.update((id(left), id(right)))
    for item in detections:
        x1, y1, x2, y2 = item.xyxy
        a = (max(0, min(w-1, round(x1))), max(0, min(h-1, round(y1))))
        b = (max(0, min(w-1, round(x2))), max(0, min(h-1, round(y2))))
        color = {"A": (255, 150, 40), "B": (40, 80, 255)}.get(item.team, (145, 205, 145))
        if len(item.trail) >= 2:
            points = np.asarray(item.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [points], False, color, 2, cv2.LINE_AA)
        cv2.rectangle(out, a, b, color, 2)
        cv2.circle(out, (round((a[0]+b[0])/2), b[1]), 3, color, -1)
        label = item.team if item.team in {"A", "B"} else "?"
        suffix = f" #{item.track_id}" if item.track_id is not None else ""
        cv2.putText(out, f"{label}{suffix} {item.confidence:.2f}", (a[0], max(16, a[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return out
