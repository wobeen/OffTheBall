"""Validated playing-area polygons used to scope person tracking."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class PitchPolygon:
    """A user-confirmed polygon in source pixel coordinates.

    The dimensions are part of the value because pixel coordinates cannot be
    safely reused after a source resize.
    """

    points: tuple[tuple[float, float], ...]
    width: int
    height: int

    @classmethod
    def from_points(cls, points, *, width: int, height: int):
        values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("pitch polygon requires at least three finite points")
        if abs(float(cv2.contourArea(values.reshape(-1, 1, 2)))) < 1e-3:
            raise ValueError("pitch polygon must enclose an area")
        if width < 1 or height < 1:
            raise ValueError("pitch polygon dimensions must be positive")
        return cls(tuple((float(x), float(y)) for x, y in values), int(width), int(height))

    def matches(self, width: int, height: int) -> bool:
        return (int(width), int(height)) == (self.width, self.height)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.points, dtype=np.float32)

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.as_array().reshape(-1, 1, 2),
                                    (float(x), float(y)), False) >= 0
