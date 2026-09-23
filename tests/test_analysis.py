import json

import numpy as np
import pytest

from offtheball.analysis import AnalysisSession
from offtheball.detection import Detection
from offtheball.inputs import Frame
from offtheball.pitch import PitchPolygon


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

    def track(self, _pixels, pitch_polygon=None):
        self.calls += 1
        x = 8.0 + self.calls
        return [Detection((x, 5.0, x + 10.0, 25.0), .9, track_id=7)]


class PolygonDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.polygons = []

    def track(self, pixels, pitch_polygon=None):
        self.polygons.append(pitch_polygon)
        # This models a tracker that only ever receives the playing-area
        # detections. A spectator outside the polygon must not get an ID.
        if pitch_polygon is None:
            return [Detection((70., 10., 78., 20.), .9, track_id=99)]
        return [Detection((20., 10., 30., 30.), .9, track_id=7)]


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


def test_calibration_diagnostics_support_legacy_and_missing_calibrators():
    empty = AnalysisSession(FakeDetector(), FakeTeams())
    assert empty.calibration_diagnostics()["status"] == "not_calibrated"

    class LegacyCalibration:
        frame_id = "legacy"
        def project(self, points, *, image_size, frame_id):
            return np.asarray(points, dtype=float)

    legacy = AnalysisSession(FakeDetector(), FakeTeams(), calibration=LegacyCalibration())
    report = legacy.calibration_diagnostics()
    assert report["status"] == "active"
    assert report["source"] == "fixed_calibration"


def test_pitch_polygon_is_passed_before_tracking_and_excludes_after_resize():
    detector = PolygonDetector()
    polygon = [(5, 5), (55, 5), (55, 50), (5, 50)]
    session = AnalysisSession(detector, FakeTeams(), pitch_polygon=polygon)
    inside = session.process(frame(size=(80, 60), timestamp=0))[0]
    assert inside.track_id == 7
    assert detector.polygons[-1] == polygon

    # Pixel coordinates are invalid after a source dimension change. Failing
    # loudly prevents silently claiming that spectators were excluded.
    with pytest.raises(ValueError, match="pitch polygon dimensions"):
        session.process(frame(size=(100, 60), timestamp=.1))


def test_confirmed_pitch_polygon_carries_dimensions():
    polygon = PitchPolygon.from_points([(5, 5), (55, 5), (55, 50), (5, 50)], width=80, height=60)
    assert polygon.matches(80, 60)
    assert not polygon.matches(100, 60)
    assert polygon.contains(20, 20)
    assert not polygon.contains(70, 20)
