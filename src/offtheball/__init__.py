"""OffTheBall: local analysis, explicit uncertainty."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
for key, suffix in {
    "YOLO_CONFIG_DIR": "work/yolo",
    "MPLCONFIGDIR": "work/matplotlib",
    "TORCH_HOME": "work/torch",
}.items():
    os.environ[key] = str(ROOT / suffix)
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_VERBOSE"] = "false"
__version__ = "0.1.0"
