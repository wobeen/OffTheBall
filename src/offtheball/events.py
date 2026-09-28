"""Manual tactical event tags and small before/after summaries.

Events are deliberately independent from the video UI: a tag is a timestamp
and a label, while analysis rows provide the measured player positions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from .metrics import formation_deviation, team_metrics

EVENT_LABELS = ("실점", "위험 장면", "빌드업 시작", "빌드업 성공", "빌드업 실패", "기타")


@dataclass(frozen=True)
class Event:
    timestamp_s: float
    label: str
    note: str = ""

    def __post_init__(self):
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("이벤트 시점은 0 이상의 유한한 초여야 합니다.")
        if self.label not in EVENT_LABELS:
            raise ValueError(f"지원하지 않는 이벤트 라벨입니다: {self.label}")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "note", str(self.note or "").strip())


class EventTimeline:
    """Sorted, JSON-serializable collection of manually tagged moments."""

    def __init__(self, events=()):
        self._events = []
        for event in events:
            self.add(event)

    @property
    def events(self):
        return tuple(self._events)

    def add(self, event_or_timestamp, label=None, note=""):
        event = event_or_timestamp if isinstance(event_or_timestamp, Event) else Event(event_or_timestamp, label, note)
        self._events.append(event)
        self._events.sort(key=lambda item: item.timestamp_s)
        return event

    def remove(self, index):
        return self._events.pop(index)

    def clear(self):
        self._events.clear()

    def to_list(self):
        return [asdict(event) for event in self._events]

    def save(self, path):
        target = Path(path)
        target.write_text(json.dumps(self.to_list(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("이벤트 파일 형식은 목록이어야 합니다.")
        return cls(Event(item["timestamp_s"], item["label"], item.get("note", "")) for item in data)


def summarize_event(event, rows, *, before_s=5.0, after_s=5.0, formation=None):
    """Summarize measured rows surrounding an event timestamp.

    Rows may be JSONL dictionaries or any iterable with ``source_time_s`` and
    ``detections``. Missing/invalid timestamps are skipped safely.
    """
    if not isinstance(event, Event):
        event = Event(event["timestamp_s"], event["label"], event.get("note", ""))
    before_s, after_s = float(before_s), float(after_s)
    if before_s < 0 or after_s < 0 or not math.isfinite(before_s + after_s):
        raise ValueError("이벤트 분석 범위는 0 이상의 유한한 초여야 합니다.")
    start, end = event.timestamp_s - before_s, event.timestamp_s + after_s
    selected = []
    for row in rows:
        try:
            stamp = float(row.get("source_time_s"))
        except (AttributeError, TypeError, ValueError):
            continue
        if start <= stamp <= end:
            selected.append((stamp, row.get("detections", [])))
    selected.sort(key=lambda item: item[0])
    widths = {team: [] for team in ("A", "B")}
    deviations = []
    for stamp, detections in selected:
        metrics = team_metrics(detections)
        for team in widths:
            if metrics[team]["width_m"] is not None:
                widths[team].append(metrics[team]["width_m"])
        if formation:
            deviation = formation_deviation(detections, formation)
            if deviation:
                deviations.append((stamp, deviation))
    result = {
        "timestamp_s": event.timestamp_s,
        "label": event.label,
        "note": event.note,
        "window_s": [start, end],
        "samples": len(selected),
        "teams": {},
    }
    for team in widths:
        values = widths[team]
        result["teams"][team] = {
            "samples": len(values),
            "mean_width_m": sum(values) / len(values) if values else None,
            "max_width_m": max(values) if values else None,
        }
    if deviations:
        peak_stamp, peak = max(deviations, key=lambda item: item[1]["mean_m"])
        result["formation"] = {"peak_time_s": peak_stamp, "mean_m": peak["mean_m"], "max_m": peak["max_m"]}
    else:
        result["formation"] = None
    return result

