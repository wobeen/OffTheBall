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

def test_scene_reset_keeps_kit_centers_but_clears_identity():
    class PersistentTeams(FakeTeams):
        def __init__(self):
            super().__init__()
            self.track_resets = 0
        def reset_tracks(self):
            self.track_resets += 1
    teams = PersistentTeams()
    session = AnalysisSession(FakeDetector(), teams)
    session.process(frame(value=30, timestamp=0))
    cut = session.process(frame(value=230, timestamp=.1))[0]
    assert cut.scene_id == 1
    assert len(cut.trail) == 1
    assert teams.track_resets == 1
    assert teams.resets == 0
    session.reset()
    assert teams.resets == 1


def test_absent_track_histories_expire_in_long_scene():
    class NewIds(FakeDetector):
        def track(self, pixels):
            self.calls += 1
            return [Detection((10.,10.,30.,40.), .9, track_id=self.calls)]
    session = AnalysisSession(NewIds(), FakeTeams())
    for i in range(100):
        session.process(frame(timestamp=i*.1))
    assert len(session._trails) <= 22
    assert 1 not in session._trails


def test_fixed_calibration_adds_field_coordinates_to_detections():
    class FixedCalibration:
        frame_id = "fixed-camera"

        def project(self, points, *, image_size, frame_id):
            assert image_size == (80, 60)
            assert frame_id == self.frame_id
            return np.asarray(points, dtype=float) / 10.0

    detection = AnalysisSession(FakeDetector(), FakeTeams(), calibration=FixedCalibration()).process(frame())[0]
    assert detection.field_xy == (1.4, 2.5)
    assert detection.to_dict()["field_xy"] == [1.4, 2.5]
