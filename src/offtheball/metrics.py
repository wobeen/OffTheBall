"""Small, explainable tactical metrics from calibrated player coordinates."""
from __future__ import annotations
import math
import numpy as np


def team_metrics(detections):
    result = {}
    for team in ("A", "B"):
        def get(item, key):
            return item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        points = [get(d, "field_xy") for d in detections
                  if get(d, "team") == team and get(d, "field_xy") is not None]
        if len(points) < 2:
            result[team] = {"count": len(points), "width_m": None, "nearest_m": None}
            continue
        data = np.asarray(points, dtype=float)
        distances = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
        distances += np.eye(len(data)) * 1e9
        result[team] = {"count": len(points), "width_m": float(np.ptp(data[:, 0])),
                        "nearest_m": float(distances.min())}
    return result


def metric_log(detections):
    metrics = team_metrics(detections)
    parts = []
    for team in ("A", "B"):
        value = metrics[team]
        if value["width_m"] is None:
            parts.append(f"{team}: 좌표 부족")
        else:
            parts.append(f"{team} 폭 {value['width_m']:.1f}m · 최소 {value['nearest_m']:.1f}m")
    return " / ".join(parts)
