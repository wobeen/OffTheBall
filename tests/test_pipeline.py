import json
from threading import Event
import cv2
import numpy as np
import pytest
from offtheball.detection import Detection, annotate
from offtheball.pipeline import analyze_video


class Detector:
    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on

    def detect(self, pixels):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("injected detector failure")
        return [Detection((10., 10., 30., 38.), .9)]


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "입력.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((48, 64, 3), 40+i*20, np.uint8))
    writer.release()
    return path


def test_encoded_result_and_timestamped_records(clip, tmp_path):
    directory, summary = analyze_video(clip, Detector(), output_dir=tmp_path/"result")
    assert summary["status"] == "completed"
    assert summary["frames"] == 6
    rows = [json.loads(line) for line in (directory/"frames.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert [r["frame_index"] for r in rows] == list(range(6))
    assert rows[-1]["source_time_s"] > rows[0]["source_time_s"]
    assert rows[0]["detections"][0]["field_xy"] is None
    assert rows[0]["schema_version"] == 2
    assert rows[0]["calibration_status"] == "not_calibrated"
    assert rows[0]["calibration"]["source"] is None
    assert summary["schema_version"] == 2
    cap = cv2.VideoCapture(str(directory/"analysis.mp4"))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)
    assert frames[0][:, :, 1].max() > 100


def test_cancel_preserves_partial_output(clip, tmp_path):
    stop = Event()
    output, summary = analyze_video(clip, Detector(), stop=stop,
                 on_frame=lambda image, row, raw: stop.set(), output_dir=tmp_path/"cancel")
    assert summary["status"] == "cancelled"
    assert summary["frames"] == 1
    assert (output/"analysis.mp4").stat().st_size > 0


def test_failure_records_error_and_releases_encoder(clip, tmp_path):
    output = tmp_path/"failure"
    with pytest.raises(RuntimeError, match="injected"):
        analyze_video(clip, Detector(fail_on=2), output_dir=output)
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["frames"] == 1
    cap = cv2.VideoCapture(str(output/"analysis.mp4"))
    assert cap.read()[0]
    cap.release()


def test_existing_output_is_not_overwritten(clip, tmp_path):
    output = tmp_path/"existing"
    output.mkdir()
    data = output/"frames.jsonl"
    data.write_text("keep")
    with pytest.raises(FileExistsError):
        analyze_video(clip, Detector(), output_dir=output)
    assert data.read_text(encoding="utf-8") == "keep"


def test_annotation_does_not_modify_input():
    raw = np.zeros((40, 40, 3), np.uint8)
    out = annotate(raw, [Detection((3., 3., 30., 30.), .9)])
    assert not raw.any()
    assert out.any()

def test_resolution_change_fails_before_mismatched_frame_is_written(tmp_path, monkeypatch):
    from offtheball.inputs import Frame
    class ChangingSource:
        fps = 10
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            yield Frame(np.zeros((48,64,3), np.uint8), 0, 0, 0)
            yield Frame(np.zeros((60,80,3), np.uint8), 1, .1, .1)
    monkeypatch.setattr("offtheball.pipeline.VideoSource", ChangingSource)
    output = tmp_path/"change"
    with pytest.raises(ValueError):
        analyze_video("ignored", Detector(), output_dir=output)
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["frames"] == 1


def test_odd_frame_size_is_padded_without_cropping(tmp_path, monkeypatch):
    from offtheball.inputs import Frame
    class OddSource:
        fps = 10
        def __init__(self, path): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            yield Frame(np.full((49,65,3), 100, np.uint8), 0, 0, 0)
    monkeypatch.setattr("offtheball.pipeline.VideoSource", OddSource)
    output, summary = analyze_video("ignored", Detector(), output_dir=tmp_path/"odd")
    row = json.loads((output/"frames.jsonl").read_text(encoding="utf-8"))
    assert row["image_size"] == [65,49]
    assert row["encoded_image_size"] == [66,50]
    cap = cv2.VideoCapture(str(output/"analysis.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (50,66)

def test_black_input_is_marked_and_not_sent_to_detector(tmp_path):
    source = tmp_path/"black.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64,48))
    assert writer.isOpened()
    writer.write(np.zeros((48,64,3), np.uint8))
    writer.release()
    detector = Detector(fail_on=1)
    output, summary = analyze_video(source, detector, output_dir=tmp_path/"black-result")
    assert summary["black_frames"] == 1
    assert summary["warnings"]
    assert detector.calls == 0
    row = json.loads((output/"frames.jsonl").read_text(encoding="utf-8"))
    assert row["input_status"] == "black_frame"

