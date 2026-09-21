import json
import pytest
from offtheball.events import Event, EventTimeline, summarize_event
from offtheball.metrics import formation_deviation


def test_timeline_sorts_and_serializes():
    timeline = EventTimeline()
    timeline.add(12, "실점", "전환 수비")
    timeline.add(3.5, "위험 장면")
    data = timeline.to_list()
    assert [event["timestamp_s"] for event in data] == [3.5, 12.0]
    assert data[1]["note"] == "전환 수비"
    assert json.loads(json.dumps(data, ensure_ascii=False))[0]["label"] == "위험 장면"


def test_event_rejects_invalid_values():
    with pytest.raises(ValueError):
        Event(-1, "실점")
    with pytest.raises(ValueError):
        Event(1, "없는 라벨")


def test_summary_uses_before_after_window():
    rows = [
        {"source_time_s": 4, "detections": [{"team": "A", "field_xy": [10, 20]}, {"team": "A", "field_xy": [30, 20]}]},
        {"source_time_s": 10, "detections": [{"team": "A", "field_xy": [10, 20]}, {"team": "A", "field_xy": [50, 20]}]},
        {"source_time_s": 16, "detections": [{"team": "A", "field_xy": [10, 20]}, {"team": "A", "field_xy": [90, 20]}]},
    ]
    result = summarize_event(Event(10, "위험 장면"), rows, before_s=5, after_s=2)
    assert result["samples"] == 1
    assert result["teams"]["A"]["mean_width_m"] == 40


def test_formation_comparison_uses_selected_team_only():
    formation = {"team": "A", "players": [[10, 10], [20, 10]]}
    detections = [
        {"team": "A", "field_xy": [10, 10]},
        {"team": "A", "field_xy": [20, 10]},
        # The opposing team must not distort the selected team's comparison.
        {"team": "B", "field_xy": [95, 60]},
    ]
    result = formation_deviation(detections, formation)
    assert result["count"] == 2
    assert result["mean_m"] == 0
