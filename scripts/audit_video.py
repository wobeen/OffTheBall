"""Read-only per-frame input quality audit; no model required."""
import json
from pathlib import Path
import sys
from offtheball.inputs import VideoSource
from offtheball.quality import is_black_frame

path = Path(sys.argv[1])
states = []
with VideoSource(path) as source:
    fps = source.fps
    for frame in source:
        states.append((frame.index, frame.timestamp_s, is_black_frame(frame.pixels)))
segments = []
begin = None
for index, seconds, black in states:
    if black and begin is None:
        begin = seconds
    if not black and begin is not None:
        segments.append([begin, seconds])
        begin = None
if begin is not None:
    segments.append([begin, states[-1][1]+1/fps])
report = {"source": str(path.resolve()), "frames": len(states),
          "black_frames": sum(s[2] for s in states),
          "black_fraction": sum(s[2] for s in states)/len(states) if states else None,
          "black_segments_s": segments,
          "note": "Pixel quality check only; no claim about the cause of black capture."}
destination = Path("work/football/input-audit.json")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
