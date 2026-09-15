"""Download the official small baseline model into the project only."""
import os
from pathlib import Path
from offtheball import ROOT
from ultralytics import YOLO, settings

(ROOT / "models").mkdir(parents=True, exist_ok=True)
settings.update({"sync": False, "weights_dir": str(ROOT / "models"),
                 "runs_dir": str(ROOT / "work/runs"),
                 "datasets_dir": str(ROOT / "data")})
path = ROOT / "models/yolo11n.pt"
os.chdir(ROOT / "models")
model = YOLO(str(path))
print(f"Model ready: {path}")
