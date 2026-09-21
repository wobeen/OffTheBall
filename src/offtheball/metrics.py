"""Small, explainable tactical metrics from calibrated player coordinates."""
from __future__ import annotations
import math
import numpy as np


def _formation_points(detections, formation):
    """Return calibrated points for the team represented by a formation.

    A saved formation is a single team's eleven reference slots.  Older files
    did not contain a team field, so when both teams are present we prefer A
    (the UI's default team) and otherwise retain the legacy all-points
    behaviour for callers that provide unlabelled samples.
    """
    target = formation.get("team") if isinstance(formation, dict) else None
    labelled = []
    unlabelled = []
    for item in detections:
        point = item.get("field_xy") if isinstance(item, dict) else getattr(item, "field_xy", None)
        if point is None:
            continue
        team = item.get("team") if isinstance(item, dict) else getattr(item, "team", None)
        if team is not None:
            labelled.append((team, point))
        else:
            unlabelled.append(point)
    if target is None and labelled:
        target = "A"
    if target is not None and labelled:
        selected = [point for team, point in labelled if team == target]
        if selected:
            return selected
    return unlabelled if unlabelled else [point for _, point in labelled]


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

def formation_deviation(detections, formation):
    if not formation:
        return None
    refs = np.asarray(formation.get("players", []), dtype=float)
    points = np.asarray(_formation_points(detections, formation), dtype=float)
    if len(refs) == 0 or len(points) == 0:
        return None
    observed = points - points.mean(axis=0)
    reference = refs - refs.mean(axis=0)
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(observed[:, None, :] - reference[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    distances = cost[rows, cols]
    return {"count": int(len(points)), "mean_m": float(distances.mean()), "max_m": float(distances.max())}


def formation_message(detections, formation):
    if not formation:
        return None
    deviation = formation_deviation(detections, formation)
    if not deviation:
        return None
    points = _formation_points(detections, formation)
    if len(points) < 2:
        return None
    width = max(p[0] for p in points) - min(p[0] for p in points)
    baseline = float(formation.get("team_width_m", 0.0))
    if baseline <= 0:
        return f"기준 편차 평균 {deviation['mean_m']:.1f}m"
    delta = width - baseline
    direction = "넓음" if delta > 0.5 else "좁음" if delta < -0.5 else "유사"
    return f"기준보다 {abs(delta):.1f}m {direction} · 위치 편차 평균 {deviation['mean_m']:.1f}m"
