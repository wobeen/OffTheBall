"""Real-model integration smoke, explicitly not a football accuracy benchmark."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from offtheball import ROOT
from offtheball.detection import PersonDetector, annotate
from offtheball.pipeline import analyze_video
from offtheball.calibration import ManualCalibration
import ultralytics

work = ROOT / "work/smoke"
work.mkdir(parents=True, exist_ok=True)
asset = Path(ultralytics.__file__).parent / "assets/bus.jpg"
image = cv2.imdecode(np.fromfile(str(asset), dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("Bundled person test image missing")
image = cv2.resize(image, (480, 640))
detector = PersonDetector()
start = time.perf_counter()
detections = detector.detect(image)
first_s = time.perf_counter() - start
assert len(detections) > 0, "Actual detector must detect a person in the bundled test fixture"
source = work/"person-fixture.avi"
writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 6, (480, 640))
assert writer.isOpened()
for _ in range(12):
    writer.write(image)
writer.release()
output, summary = analyze_video(source, detector)
ok, encoded = cv2.imencode(".jpg", annotate(image, detections))
assert ok
encoded.tofile(str(output/"preview.jpg"))
calibration = ManualCalibration.fit([[10,10],[210,10],[210,110],[10,110]],
    [[0,0],[20,0],[20,10],[0,10]], image_size=(240,140), frame_id="synthetic-static")
evaluation = calibration.evaluate([[30,30],[190,90]], [[2,2],[18,8]],
    image_size=(240,140), frame_id="synthetic-static", segments=[(0,1)])
report = {"test": "bundled non-football person image repeated as 12-frame video",
          "football_accuracy_validated": False, "device": "cpu", "model_image_size": 960,
          "first_inference_s": first_s, "detected_people": len(detections),
          "output": str(output), "video_summary": summary,
          "calibration": {"test": "synthetic mathematical transform", **evaluation}}
(work/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
