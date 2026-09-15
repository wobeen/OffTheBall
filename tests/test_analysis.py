import json

import numpy as np

from offtheball.analysis import AnalysisSession
from offtheball.detection import Detection
from offtheball.inputs import Frame


class FakeTeams:
    def __init__(self):
        self.ready = False
        self.resets = 0

    def reset(self):
        self.resets += 1

    def update(self, _frame, detections):
        return ["A" if detections else None for _ in detections]


class FakeDetector:
    def __init__(self):
        self.resets = 0
        self.calls = 0

    def reset_tracking(self):
        self.resets += 1

    def track(self, _pixels):
        self.calls += 1
        x = 8.0 + self.calls
        return [Detection((x, 5.0, x + 10.0, 25.0), .9, track_id=7)]


def frame(value=30, timestamp=0.0, size=(80, 60)):
    width, height = size
    pixels = np.full((height, width, 3), value, np.uint8)
    return Frame(pixels, int(timestamp * 10), timestamp, timestamp)


def test_trails_are_bounded_and_team_labels_are_attached():
    detector = FakeDetector()
    teams = FakeTeams()
    session = AnalysisSession(detector, teams, trail_length=3)
    rows = [session.process(frame(timestamp=i * .1))[0] for i in range(6)]
    assert [row.track_id for row in rows] == [7] * 6
    assert rows[-1].team == "A"
    assert len(rows[-1].trail) == 3
    payload = rows[-1].to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["scene_id"] == 0
    assert payload["track_id"] == 7


def test_timestamp_gap_black_input_and_dimension_change_start_new_scenes():
    detector = FakeDetector()
    teams = FakeTeams()
    session = AnalysisSession(detector, teams)
    first = session.process(frame(timestamp=0.0))[0]
    assert first.scene_id == 0
    gap = session.process(frame(timestamp=3.0))[0]
    assert gap.scene_id == 1
    session.process(frame(value=0, timestamp=3.1), black=True)
    after_black = session.process(frame(timestamp=3.2))[0]
    assert after_black.scene_id == 2
    changed = session.process(frame(timestamp=3.3, size=(100, 60)))[0]
    assert changed.scene_id == 3
    assert detector.resets >= 4
    assert teams.resets >= 3


def test_backward_timestamp_starts_new_scene():
    session = AnalysisSession(FakeDetector(), FakeTeams())
    assert session.process(frame(timestamp=2.0))[0].scene_id == 0
    assert session.process(frame(timestamp=1.0))[0].scene_id == 1
