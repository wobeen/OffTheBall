"""Structural contracts for replaceable analysis components.

The production classes do not need to inherit from these protocols.  Keeping
the boundaries structural lets experimental OSS-backed adapters live beside
the current lightweight implementations without coupling the core session to
one framework.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from .detection import Detection


class Detector(Protocol):
    def detect(self, pixels: np.ndarray) -> Sequence[Detection]: ...


class Tracker(Protocol):
    def track(self, pixels: np.ndarray, pitch_polygon: Any | None = None) -> Sequence[Detection]: ...

    def reset_tracking(self) -> None: ...


class TeamAssigner(Protocol):
    @property
    def ready(self) -> bool: ...

    def update(self, pixels: np.ndarray, detections: Sequence[Detection]) -> Any: ...


class PitchCalibrator(Protocol):
    frame_id: str

    def project(self, points: Any, *, image_size: tuple[int, int], frame_id: str) -> np.ndarray: ...

    def diagnostics(self) -> dict[str, Any]: ...
